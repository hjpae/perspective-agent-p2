## --- this includes Dreamer-like policy optimization, involves "actor" unit --- 

# cear_pilot/training/train.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from cear_pilot.envs.nzone_grid import NZoneConfig, NZoneGridEnv
from cear_pilot.models.agent import CEARAgent, AgentConfig
from cear_pilot.models.decoder import ObsDecoder, DecoderConfig

import pandas as pd

# ---------------------------
# utils
# ---------------------------

def onehot(indices: torch.Tensor, n: int) -> torch.Tensor:
    return F.one_hot(indices.long(), num_classes=n).float()


@torch.no_grad()
def make_proprio_from_last_action(last_action: int, n_actions: int, device: torch.device) -> torch.Tensor:
    a = torch.tensor([last_action], device=device)
    return onehot(a, n_actions)


def timestamp_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def save_meta(run_dir: Path, meta: Dict) -> None:
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """
    Proper seeding for reproducibility.
    - random / numpy / torch
    - CuDNN deterministic where applicable
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # makes runs more repeatable (esp. GPU); may reduce speed a bit.
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EMAMeanVar:
    """EMA mean/std tracker for stabilizing advantage scaling."""
    def __init__(self, beta: float = 0.99, eps: float = 1e-8):
        self.beta = beta
        self.eps = eps
        self.mean = None
        self.var = None

    def update(self, x: float) -> Tuple[float, float]:
        if self.mean is None:
            self.mean = x
            self.var = 0.0
        else:
            m = self.mean
            self.mean = self.beta * self.mean + (1 - self.beta) * x
            # EMA variance update (approx)
            self.var = self.beta * self.var + (1 - self.beta) * (x - m) * (x - m)
        std = float(np.sqrt(max(self.var, 0.0) + self.eps))
        return float(self.mean), std


# ---------------------------
# main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--steps", type=int, default=48000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")

    ap.add_argument("--w_smooth", type=float, default=0.25)
    ap.add_argument("--w_entropy", type=float, default=0.01)

    ap.add_argument("--width", type=int, default=15)
    ap.add_argument("--height", type=int, default=9)
    ap.add_argument("--obs_dim", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=240)
    
    ap.add_argument("--mirror_x", action="store_true")
    ap.add_argument("--mirror_actions", action="store_true")

    # ---------------------------
    # MINIMAL TRAJ LOGGING (NEW)
    # ---------------------------
    ap.add_argument("--log_traj", action="store_true", help="Save per-step training trajectory to train_traj.parquet")
    ap.add_argument("--log_every", type=int, default=1, help="Log every N steps (1 = log all steps)")

    # actor term (Dreamer-like; internal cost, not env reward)
    ap.add_argument(
        "--w_actor",
        type=float,
        default=0.5,
        help="Weight for actor loss based on prediction error (internal cost).",
    )
    ap.add_argument(
        "--actor_b",
        type=float,
        default=0.98,
        help="EMA momentum for actor baseline (0 disables baseline).",
    )

    # ---- live viewer flags ----
    ap.add_argument("--view", action="store_true", help="Show live pygame viewer during training")
    ap.add_argument("--view_every", type=int, default=2, help="Render every N training steps")
    ap.add_argument("--view_fps", type=int, default=20, help="Viewer FPS cap")
    ap.add_argument("--view_cell_px", type=int, default=42, help="Cell size in pixels")

    # ---- Phase 2 env toggles ----
    ap.add_argument("--use_slip", action="store_true")
    ap.add_argument("--use_drift", action="store_true")
    ap.add_argument("--use_volatility", action="store_true")
    ap.add_argument("--use_hazard", action="store_true")

    ap.add_argument("--p_slip", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    ap.add_argument("--p_drift", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    ap.add_argument("--drift_vec", type=int, nargs=6, default=(0, 0, 0, 0, 0, 0))  # z0dx z0dy z1dx z1dy z2dx z2dy

    ap.add_argument("--volatile_zone", type=int, default=0)
    ap.add_argument("--volatile_period", type=int, default=40)
    ap.add_argument("--volatile_strength", type=float, default=0.0)

    ap.add_argument("--hazard_mode", type=str, default="teleport")
    ap.add_argument("--p_hazard", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    ap.add_argument("--hazard_teleport_to", type=int, nargs=2, default=(0, 0))
    ap.add_argument("--hazard_blackout_steps", type=int, default=6)

    args = ap.parse_args()

    # ---------------------------
    # SEED
    # ---------------------------
    seed_everything(args.seed, deterministic=True)

    device = torch.device(args.device)

    dv = args.drift_vec
    drift_vec = ((dv[0], dv[1]), (dv[2], dv[3]), (dv[4], dv[5]))

    env_cfg = NZoneConfig(
        width=args.width,
        height=args.height,
        obs_dim=args.obs_dim,
        max_steps=args.max_steps,

        use_slip=args.use_slip,
        use_drift=args.use_drift,
        use_volatility=args.use_volatility,
        use_hazard=args.use_hazard,

        p_slip=tuple(args.p_slip),
        p_drift=tuple(args.p_drift),
        drift_vec=drift_vec,

        volatile_zone=args.volatile_zone,
        volatile_period=args.volatile_period,
        volatile_strength=args.volatile_strength,

        hazard_mode=args.hazard_mode,
        p_hazard=tuple(args.p_hazard),
        hazard_teleport_to=tuple(args.hazard_teleport_to),
        hazard_blackout_steps=args.hazard_blackout_steps,
    )
    env = NZoneGridEnv(config=env_cfg)

    # make sure env RNG is seeded too
    obs, info = env.reset(seed=args.seed)
    try:
        # gymnasium-style
        env.action_space.seed(args.seed)
        env.observation_space.seed(args.seed)
    except Exception:
        pass

    n_actions = int(env.action_space.n)

    # ---- agent config (wire dims)
    agent_cfg = AgentConfig(device=args.device)
    agent_cfg.encoder.obs_dim = args.obs_dim
    agent_cfg.encoder.proprio_dim = n_actions

    agent_cfg.world.z_dim = agent_cfg.encoder.z_dim
    agent_cfg.world.p_dim = agent_cfg.encoder.p_dim

    agent_cfg.state.z_dim = agent_cfg.encoder.z_dim
    agent_cfg.state.p_dim = agent_cfg.encoder.p_dim
    agent_cfg.state.g_dim = agent_cfg.world.g_dim

    agent_cfg.policy.n_actions = n_actions
    agent_cfg.policy.s_dim = agent_cfg.state.s_dim

    agent = CEARAgent(agent_cfg).to(device)

    dec_cfg = DecoderConfig(
        g_dim=agent_cfg.world.g_dim,
        n_actions=n_actions,
        obs_dim=args.obs_dim,
        hidden=64,
        dropout=0.0,
    )
    decoder = ObsDecoder(dec_cfg).to(device)

    # one optimizer is fine (models untouched), but we stabilize gradients below
    params = list(agent.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    run_dir = Path("outputs") / "runs" / timestamp_id()
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- training schedule (no new argparse):
    # warmup: learn world model + g dynamics first, then turn on actor
    warmup_steps = max(2000, min(args.steps // 4, 20000))

    meta = {
        "seed": args.seed,
        "steps": args.steps,
        "lr": args.lr,
        "device": args.device,
        "loss_weights": {"w_smooth": args.w_smooth, "w_entropy": args.w_entropy, "w_actor": args.w_actor},
        "actor_b": args.actor_b,
        "stopgrad_default": {
            "policy_state_detach": True,   # action policy uses s.detach()
            "pred_pi_detach": True,        # pred mixture uses detached pi
            "actor_cost_detach": True,     # actor uses detached cost
        },
        "warmup_steps": warmup_steps,
        "env_cfg": asdict(env_cfg),
        "agent_cfg": {
            "encoder": asdict(agent_cfg.encoder),
            "world": asdict(agent_cfg.world),
            "state": asdict(agent_cfg.state),
            "policy": asdict(agent_cfg.policy),
        },
        "decoder_cfg": asdict(dec_cfg),
        "viewer": {
            "enabled": bool(args.view),
            "view_every": int(args.view_every),
            "view_fps": int(args.view_fps),
            "view_cell_px": int(args.view_cell_px),
        },
        "traj_logging": {
            "enabled": bool(args.log_traj),
            "log_every": int(max(1, args.log_every)),
        },
    }
    save_meta(run_dir, meta)

    # ---------------------------
    # MINIMAL TRAJ LOGGING (NEW)
    # ---------------------------
    log_rows = []
    log_every = int(max(1, args.log_every))

    # ---- live viewer init ----
    viewer = None
    if args.view:
        from cear_pilot.training.pygame_viewer import PygameGridViewer
        viewer = PygameGridViewer(
            width=args.width,
            height=args.height,
            cell_px=args.view_cell_px,
            fps=args.view_fps,
            title="Live Training (SPACE=Pause, Close=Stop)",
        )

    # ---- state init
    agent.reset(batch_size=1)
    last_action = 4  # stay (assumed)
    g_prev = agent.get_latents()["g"].detach().clone()

    # ---- logs / diagnostics
    ema_world = None
    pi_prev = None
    kl_ema = None
    maxpi_ema = None

    # actor baseline + scaling
    b = None
    err_stats = EMAMeanVar(beta=0.99)

    # histograms over the last window (for sanity)
    window = 2000
    act_hist = np.zeros(n_actions, dtype=np.int64)
    zone_hist = np.zeros(3, dtype=np.int64)

    # extra diagnostics
    logits_norm_ema = None

    t0 = time.time()
    episode = 0
    t_in_ep = 0  # (NEW) track step within episode

    try:
        for step in range(args.steps):
            # ---------- forward
            x_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # (1, obs_dim)
            p_t = make_proprio_from_last_action(last_action, n_actions, device=device)  # (1, n_actions)

            out = agent.forward_step(x_t, p_t, ablate_g=False)
            g_t = out["g"]
            s_t = out["s"]

            # logits produced internally (often from policy head)
            logits_pred = out["logits"]

            # ---- STOP-GRAD DEFAULT:
            # action policy uses detached state -> prevents leakage from actor/policy into representation learning
            logits_act = agent.policy(s_t.detach())

            # action distribution (learned by actor + entropy only)
            pi_act = torch.softmax(logits_act, dim=-1)

            # pred mixture policy is detached so pred loss doesn't update policy weights
            pi_pred = torch.softmax(logits_pred, dim=-1).detach()

            # ---------- sample + env step (must use action logits!)
            a_t = agent.policy.sample_action(logits_act, greedy=False)
            a_int = int(a_t.item())

            obs_next, _, terminated, truncated, info = env.step(a_int)
            x_next = torch.tensor(obs_next, dtype=torch.float32, device=device).unsqueeze(0)

            # ---------- decoder predictions
            xhat_all = decoder.predict_all_actions(g_t)  # (1, A, obs_dim)
            xhat_exp = torch.sum(pi_pred.unsqueeze(-1) * xhat_all, dim=1)  # (1, obs_dim)

            # ---------- world-model losses
            loss_pred = F.mse_loss(xhat_exp, x_next)
            loss_smooth = torch.mean((g_t - g_prev) ** 2)

            # ---------- entropy (computed on action policy)
            entropy = -torch.sum(pi_act * torch.log(pi_act + 1e-9), dim=-1).mean()

            # ---------- actor loss (REINFORCE on chosen-action internal cost)
            # per-action prediction error (cost signal)
            per_a_err = torch.mean((xhat_all - x_next.unsqueeze(1)) ** 2, dim=-1).squeeze(0)  # (A,)
            e_chosen = per_a_err[a_int]  # scalar tensor

            # baseline + normalize
            with torch.no_grad():
                e_val = float(e_chosen.detach().item())
                m, s = err_stats.update(e_val)
                if b is None:
                    b = e_val
                if args.actor_b > 0.0:
                    b = float(args.actor_b * b + (1.0 - args.actor_b) * e_val)
                baseline = float(b) if (args.actor_b > 0.0) else 0.0

                # normalized advantage (negative centered error)
                adv = -(e_val - baseline)
                adv = adv / (s + 1e-8)
                # clip to avoid rare spikes exploding training
                adv = float(np.clip(adv, -5.0, 5.0))

            logp = F.log_softmax(logits_act, dim=-1)[0, a_int]
            loss_actor = -(torch.tensor(adv, device=device) * logp)

            # ---------- warmup schedule
            phase = "A" if step < warmup_steps else "B"
            w_actor_eff = 0.0 if step < warmup_steps else args.w_actor

            # ---------- adaptive entropy coefficient eps (tiny, but prevents "sudden collapse")
            with torch.no_grad():
                H = float(entropy.item())
                H_target = 1.0  # heuristic: keep some diversity for g trajectory richness
                bump = max(0.0, (H_target - H) / max(H_target, 1e-6))
                ent_coef = args.w_entropy * (1.0 + 2.0 * bump)

            # total loss
            loss_world = loss_pred + args.w_smooth * loss_smooth
            loss = loss_world + w_actor_eff * loss_actor - ent_coef * entropy

            # ---------- optimize
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            # ---------- step state
            g_prev = g_t.detach().clone()
            obs = obs_next
            last_action = a_int

            # ---------- MINIMAL TRAJ LOGGING (NEW)
            if args.log_traj and ((step % log_every) == 0):
                # zone index usually in info; fallback robustly
                z = info.get("zone", None)
                if z is None:
                    z = info.get("zone_id", None)
                if isinstance(z, (int, np.integer)):
                    z = int(z)
                else:
                    z = -1  # unknown

                with torch.no_grad():
                    g_norm = float(torch.linalg.vector_norm(g_t).item())
                    ent_val = float(entropy.item())

                log_rows.append({
                    "t_global": int(step),
                    "episode": int(episode),
                    "t_in_ep": int(t_in_ep),
                    "zone_id": int(z),
                    "action": int(a_int),
                    "entropy": float(ent_val),
                    "g_norm": float(g_norm),
                    "loss_pred": float(loss_pred.item()),
                    "loss_smooth": float(loss_smooth.item()),
                })

            # ---------- viewer draw
            if viewer is not None and (step % max(1, args.view_every) == 0):
                g_norm = float(torch.linalg.vector_norm(g_t.detach()).item())
                ok = viewer.draw(
                    env=env,
                    step=step + 1,
                    episode=episode,
                    last_action=last_action,
                    loss=float(loss.item()),
                    loss_pred=float(loss_pred.item()),
                    loss_smooth=float(loss_smooth.item()),
                    entropy=float(entropy.item()),
                    g_norm=g_norm,
                )
                if ok is False:
                    print("Viewer closed. Stopping training.")
                    break

            # ---------- episode reset
            t_in_ep += 1
            if truncated or terminated:
                obs, info = env.reset(seed=args.seed + episode + 1)
                agent.reset(batch_size=1)
                last_action = 4
                g_prev = agent.get_latents()["g"].detach().clone()
                episode += 1
                t_in_ep = 0

            # ---------- rolling stats
            with torch.no_grad():
                maxpi = float(pi_act.max(dim=-1).values.mean().item())
                if pi_prev is None:
                    kl = 0.0
                else:
                    kl_t = torch.sum(
                        pi_act * (torch.log(pi_act + 1e-9) - torch.log(pi_prev + 1e-9)),
                        dim=-1,
                    )
                    kl = float(kl_t.mean().item())
                pi_prev = pi_act.detach()

                maxpi_ema = maxpi if (maxpi_ema is None) else (0.98 * maxpi_ema + 0.02 * maxpi)
                kl_ema = kl if (kl_ema is None) else (0.98 * kl_ema + 0.02 * kl)

                ln = float(torch.mean(torch.abs(logits_act)).item())
                logits_norm_ema = ln if (logits_norm_ema is None) else (0.98 * logits_norm_ema + 0.02 * ln)

            act_hist[a_int] += 1
            z = info.get("zone", None)
            if z is None:
                z = info.get("zone_id", None)
            if isinstance(z, (int, np.integer)) and 0 <= int(z) <= 2:
                zone_hist[int(z)] += 1

            lw = float(loss_world.item())
            ema_world = lw if ema_world is None else 0.98 * ema_world + 0.02 * lw

            # ---------- log every 2000
            if (step + 1) % 2000 == 0:
                dt = time.time() - t0

                with torch.no_grad():
                    e_det = per_a_err.detach().float().cpu().numpy()
                    e_min, e_max, e_std = float(e_det.min()), float(e_det.max()), float(e_det.std())

                act_prob = (act_hist / max(act_hist.sum(), 1)).tolist()
                zone_prob = (zone_hist / max(zone_hist.sum(), 1)).tolist()

                act_hist[:] = 0
                zone_hist[:] = 0

                print(
                    f"[{step+1:>7}/{args.steps}] "
                    f"phase={phase} "
                    f"world={lw:.4f} w_ema={float(ema_world):.4f} pred={float(loss_pred.item()):.4f} "
                    f"smooth={float(loss_smooth.item()):.4f} | "
                    f"actor={float(loss_actor.item()):.4f} b={0.0 if b is None else float(b):.4f} "
                    f"H={float(entropy.item()):.3f} maxpi={float(maxpi_ema):.3f} KL={float(kl_ema):.6f} "
                    f"logits|.|={float(logits_norm_ema):.3f} "
                    f"e[min,max,std]={e_min:.3f},{e_max:.3f},{e_std:.3f} "
                    f"zone={[round(x,2) for x in zone_prob]} act={[round(x,2) for x in act_prob]} "
                    f"(ep={episode}, {dt:.1f}s)"
                )
                t0 = time.time()

    finally:
        if viewer is not None:
            viewer.close()

    # ---------------------------
    # SAVE TRAJ (NEW)
    # ---------------------------
    if args.log_traj and len(log_rows) > 0:
        df = pd.DataFrame(log_rows)
    
        out_parquet = run_dir / "train_traj.parquet"
        out_csv = run_dir / "train_traj.csv"
    
        try:
            df.to_parquet(out_parquet, index=False)
            print(f"Saved training trajectory to: {out_parquet}")
        except Exception as e:
            print(f"[WARN] Parquet failed ({type(e).__name__}: {e}). Falling back to CSV.")
            df.to_csv(out_csv, index=False)
            print(f"Saved training trajectory to: {out_csv}")

    ckpt = {
        "agent_state": agent.state_dict(),
        "decoder_state": decoder.state_dict(),   # IMPORTANT: keep for run_collect.py
        "meta": meta,
    }
    torch.save(ckpt, run_dir / "ckpt.pt")
    print(f"Saved checkpoint to: {run_dir / 'ckpt.pt'}")


if __name__ == "__main__":
    main()
