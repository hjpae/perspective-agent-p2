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
import pandas as pd
import torch
import torch.nn.functional as F

from cear_pilot.envs.nzone_grid import NZoneConfig, NZoneGridEnv
from cear_pilot.models.agent import CEARAgent, AgentConfig
from cear_pilot.models.decoder import ObsDecoder, DecoderConfig


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
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EMAMeanVar:
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
            self.var = self.beta * self.var + (1 - self.beta) * (x - m) * (x - m)
        std = float(np.sqrt(max(self.var, 0.0) + self.eps))
        return float(self.mean), std


def save_checkpoint(run_dir: Path, tag: str, agent: CEARAgent, decoder: ObsDecoder, meta: Dict) -> None:
    ckpt = {
        "agent_state": agent.state_dict(),
        "decoder_state": decoder.state_dict(),
        "meta": meta,
    }
    torch.save(ckpt, run_dir / f"ckpt_{tag}.pt")


def main():
    ap = argparse.ArgumentParser()

    # -------------------------
    # core training
    # -------------------------
    ap.add_argument("--steps", type=int, default=48000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")

    ap.add_argument("--w_smooth", type=float, default=0.05)
    ap.add_argument("--w_entropy", type=float, default=0.003)
    ap.add_argument("--w_actor", type=float, default=0.5)
    ap.add_argument("--actor_b", type=float, default=0.98)

    # explicit, short warmup for Phase 2
    ap.add_argument("--warmup_steps", type=int, default=2000)

    # default: keep g across episode boundaries
    ap.add_argument(
        "--reset_g_every_episode",
        action="store_true",
        help="If set, reset g to zero at every episode boundary (old Phase-1 behavior). "
             "Default is False, i.e. carry g across episodes.",
    )

    # -------------------------
    # env base
    # -------------------------
    ap.add_argument("--width", type=int, default=15)
    ap.add_argument("--height", type=int, default=9)
    ap.add_argument("--obs_dim", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=240)

    ap.add_argument("--mirror_x", action="store_true")
    ap.add_argument("--mirror_actions", action="store_true")

    # -------------------------
    # logging / saving
    # -------------------------
    ap.add_argument("--log_traj", action="store_true")
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--save_ckpt_every", type=int, default=12000)

    # -------------------------
    # optional live viewer
    # -------------------------
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--view_every", type=int, default=2)
    ap.add_argument("--view_fps", type=int, default=20)
    ap.add_argument("--view_cell_px", type=int, default=42)

    # -------------------------
    # ecology toggles (continuous from the start)
    # -------------------------
    ap.add_argument("--use_slip", action="store_true")
    ap.add_argument("--use_drift", action="store_true")
    ap.add_argument("--use_volatility", action="store_true")
    ap.add_argument("--use_hazard", action="store_true")
    ap.add_argument("--use_encounter", action="store_true")

    ap.add_argument("--p_slip", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    ap.add_argument("--p_drift", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    ap.add_argument("--drift_vec", type=int, nargs=6, default=(0, 0, 0, 0, 0, 0))

    ap.add_argument("--volatile_zone", type=int, default=0)
    ap.add_argument("--volatile_period", type=int, default=40)
    ap.add_argument("--volatile_strength", type=float, default=0.0)

    ap.add_argument("--hazard_mode", type=str, default="teleport")
    ap.add_argument("--p_hazard", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    ap.add_argument("--hazard_teleport_to", type=int, nargs=2, default=(0, 0))
    ap.add_argument("--hazard_blackout_steps", type=int, default=6)

    # encounter / hidden context
    ap.add_argument("--encounter_signal", type=float, default=1.25)
    ap.add_argument("--encounter_delay_min", type=int, default=2)
    ap.add_argument("--encounter_delay_max", type=int, default=5)

    ap.add_argument("--fragility_init", type=float, default=0.10)
    ap.add_argument("--fragility_decay", type=float, default=0.00)
    ap.add_argument("--rupture_memory_init", type=float, default=0.00)
    ap.add_argument("--rupture_memory_decay", type=float, default=0.95)

    ap.add_argument("--zone_fragility_delta", type=float, nargs=3, default=(0.00, 0.03, 0.08))

    ap.add_argument("--rupture_base_prob", type=float, default=0.55)
    ap.add_argument("--rupture_fragility_weight", type=float, default=1.20)
    ap.add_argument("--rupture_memory_weight", type=float, default=0.50)

    ap.add_argument("--rupture_obs_corrupt_steps", type=int, default=8)
    ap.add_argument("--rupture_obs_sigma", type=float, default=5.0)
    ap.add_argument("--rupture_action_slip_prob", type=float, default=0.65)
    ap.add_argument("--rupture_memory_increment", type=float, default=0.50)
    ap.add_argument("--no_rupture_memory_delta", type=float, default=-0.10)

    args = ap.parse_args()

    seed_everything(args.seed, deterministic=True)
    device = torch.device(args.device)

    dv = args.drift_vec
    drift_vec = ((dv[0], dv[1]), (dv[2], dv[3]), (dv[4], dv[5]))

    env_cfg = NZoneConfig(
        width=args.width,
        height=args.height,
        obs_dim=args.obs_dim,
        max_steps=args.max_steps,
        mirror_x=args.mirror_x,
        mirror_actions=args.mirror_actions,

        use_slip=args.use_slip,
        use_drift=args.use_drift,
        use_volatility=args.use_volatility,
        use_hazard=args.use_hazard,
        use_encounter=args.use_encounter,

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

        encounter_signal=args.encounter_signal,
        encounter_delay_min=args.encounter_delay_min,
        encounter_delay_max=args.encounter_delay_max,

        fragility_init=args.fragility_init,
        fragility_decay=args.fragility_decay,
        rupture_memory_init=args.rupture_memory_init,
        rupture_memory_decay=args.rupture_memory_decay,
        zone_fragility_delta=tuple(args.zone_fragility_delta),

        rupture_base_prob=args.rupture_base_prob,
        rupture_fragility_weight=args.rupture_fragility_weight,
        rupture_memory_weight=args.rupture_memory_weight,
        rupture_obs_corrupt_steps=args.rupture_obs_corrupt_steps,
        rupture_obs_sigma=args.rupture_obs_sigma,
        rupture_action_slip_prob=args.rupture_action_slip_prob,
        rupture_memory_increment=args.rupture_memory_increment,
        no_rupture_memory_delta=args.no_rupture_memory_delta,
    )
    env = NZoneGridEnv(config=env_cfg)

    obs, info = env.reset(seed=args.seed)
    try:
        env.action_space.seed(args.seed)
        env.observation_space.seed(args.seed)
    except Exception:
        pass

    n_actions = int(env.action_space.n)

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

    params = list(agent.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    run_dir = Path("outputs") / "runs" / timestamp_id()
    run_dir.mkdir(parents=True, exist_ok=True)

    warmup_steps = int(max(0, args.warmup_steps))

    meta = {
        "seed": args.seed,
        "steps": args.steps,
        "lr": args.lr,
        "device": args.device,
        "loss_weights": {
            "w_smooth": args.w_smooth,
            "w_entropy": args.w_entropy,
            "w_actor": args.w_actor,
        },
        "actor_b": args.actor_b,
        "warmup_steps": warmup_steps,
        "reset_g_every_episode": bool(args.reset_g_every_episode),
        "env_cfg": asdict(env_cfg),
        "agent_cfg": {
            "encoder": asdict(agent_cfg.encoder),
            "world": asdict(agent_cfg.world),
            "state": asdict(agent_cfg.state),
            "policy": asdict(agent_cfg.policy),
        },
        "decoder_cfg": asdict(dec_cfg),
    }
    save_meta(run_dir, meta)

    log_rows = []
    log_every = int(max(1, args.log_every))

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

    # initialize agent ONCE
    agent.reset(batch_size=1)
    last_action = 4
    g_prev = agent.get_latents()["g"].detach().clone()

    ema_world = None
    pi_prev = None
    kl_ema = None
    maxpi_ema = None
    logits_norm_ema = None

    b = None
    err_stats = EMAMeanVar(beta=0.99)

    act_hist = np.zeros(n_actions, dtype=np.int64)
    zone_hist = np.zeros(3, dtype=np.int64)
    enc_hist = 0
    rup_hist = 0
    slip_hist = 0
    drift_hist = 0

    t0 = time.time()
    episode = 0
    t_in_ep = 0

    try:
        for step in range(args.steps):
            x_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            p_t = make_proprio_from_last_action(last_action, n_actions, device=device)

            out = agent.forward_step(x_t, p_t, ablate_g=False)
            g_t = out["g"]
            s_t = out["s"]
            logits_pred = out["logits"]

            # actor uses detached s
            logits_act = agent.policy(s_t.detach())
            pi_act = torch.softmax(logits_act, dim=-1)
            pi_pred = torch.softmax(logits_pred, dim=-1).detach()

            a_t = agent.policy.sample_action(logits_act, greedy=False)
            a_int = int(a_t.item())

            obs_next, _, terminated, truncated, info = env.step(a_int)
            x_next = torch.tensor(obs_next, dtype=torch.float32, device=device).unsqueeze(0)

            xhat_all = decoder.predict_all_actions(g_t)
            xhat_exp = torch.sum(pi_pred.unsqueeze(-1) * xhat_all, dim=1)

            loss_pred = F.mse_loss(xhat_exp, x_next)
            loss_smooth = torch.mean((g_t - g_prev) ** 2)

            entropy = -torch.sum(pi_act * torch.log(pi_act + 1e-9), dim=-1).mean()

            per_a_err = torch.mean((xhat_all - x_next.unsqueeze(1)) ** 2, dim=-1).squeeze(0)
            e_chosen = per_a_err[a_int]

            with torch.no_grad():
                e_val = float(e_chosen.detach().item())
                _, s = err_stats.update(e_val)

                if b is None:
                    b = e_val
                if args.actor_b > 0.0:
                    b = float(args.actor_b * b + (1.0 - args.actor_b) * e_val)

                baseline = float(b) if (args.actor_b > 0.0) else 0.0
                adv = -(e_val - baseline)
                adv = adv / (s + 1e-8)
                adv = float(np.clip(adv, -5.0, 5.0))

            logp = F.log_softmax(logits_act, dim=-1)[0, a_int]
            loss_actor = -(torch.tensor(adv, device=device) * logp)

            learn_phase = "warmup" if step < warmup_steps else "full"
            w_actor_eff = 0.0 if step < warmup_steps else args.w_actor

            loss_world = loss_pred + args.w_smooth * loss_smooth
            loss = loss_world + w_actor_eff * loss_actor - args.w_entropy * entropy

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            g_prev = g_t.detach().clone()
            obs = obs_next
            last_action = a_int

            # logging counters
            enc_hist += int(bool(info.get("encounter_event", False)))
            rup_hist += int(bool(info.get("rupture", False)))
            slip_hist += int(bool(info.get("slip", False)))
            drift_hist += int(bool(info.get("drift", False)))

            if args.log_traj and ((step % log_every) == 0):
                z = info.get("zone_id", -1)
                with torch.no_grad():
                    g_np = g_t.detach().squeeze(0).float().cpu().numpy()
                    g_norm = float(torch.linalg.vector_norm(g_t).item())
                    ent_val = float(entropy.item())

                row = {
                    "t_global": int(step),
                    "episode": int(episode),
                    "t_in_ep": int(t_in_ep),
                    "learn_phase": str(learn_phase),
                    "zone_id": int(z),
                    "x": int(info.get("x", -1)),
                    "y": int(info.get("y", -1)),
                    "action": int(a_int),
                    "entropy": float(ent_val),
                    "g_norm": float(g_norm),
                    "loss_pred": float(loss_pred.item()),
                    "loss_smooth": float(loss_smooth.item()),
                    "fragility": float(info.get("fragility", np.nan)),
                    "rupture_memory": float(info.get("rupture_memory", np.nan)),
                    "on_encounter": int(bool(info.get("on_encounter", False))),
                    "encounter_event": int(bool(info.get("encounter_event", False))),
                    "rupture": int(bool(info.get("rupture", False))),
                    "pending_ruptures": int(info.get("pending_ruptures", 0)),
                    "blackout_timer": int(info.get("blackout_timer", 0)),
                    "rupture_obs_timer": int(info.get("rupture_obs_timer", 0)),
                    "rupture_action_timer": int(info.get("rupture_action_timer", 0)),
                    "slip": int(bool(info.get("slip", False))),
                    "drift": int(bool(info.get("drift", False))),
                    "hazard": int(bool(info.get("hazard", False))),
                }
                for i, gv in enumerate(g_np):
                    row[f"g_{i}"] = float(gv)
                log_rows.append(row)

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

            t_in_ep += 1
            if truncated or terminated:
                obs, info = env.reset(seed=args.seed + episode + 1)

                if args.reset_g_every_episode:
                    agent.reset(batch_size=1)

                # if carrying g across episodes, keep current latent
                g_prev = agent.get_latents()["g"].detach().clone()

                last_action = 4
                episode += 1
                t_in_ep = 0

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
            z = info.get("zone_id", -1)
            if isinstance(z, (int, np.integer)) and 0 <= int(z) <= 2:
                zone_hist[int(z)] += 1

            lw = float(loss_world.item())
            ema_world = lw if ema_world is None else 0.98 * ema_world + 0.02 * lw

            if (step + 1) % 2000 == 0:
                dt = time.time() - t0
                with torch.no_grad():
                    e_det = per_a_err.detach().float().cpu().numpy()
                    e_min, e_max, e_std = float(e_det.min()), float(e_det.max()), float(e_det.std())

                act_prob = (act_hist / max(act_hist.sum(), 1)).tolist()
                zone_prob = (zone_hist / max(zone_hist.sum(), 1)).tolist()

                print(
                    f"[{step+1:>7}/{args.steps}] "
                    f"phase={learn_phase} "
                    f"world={lw:.4f} w_ema={float(ema_world):.4f} pred={float(loss_pred.item()):.4f} "
                    f"smooth={float(loss_smooth.item()):.4f} | "
                    f"actor={float(loss_actor.item()):.4f} b={0.0 if b is None else float(b):.4f} "
                    f"H={float(entropy.item()):.3f} maxpi={float(maxpi_ema):.3f} KL={float(kl_ema):.6f} "
                    f"logits|.|={float(logits_norm_ema):.3f} "
                    f"e[min,max,std]={e_min:.3f},{e_max:.3f},{e_std:.3f} "
                    f"zone={[round(x,2) for x in zone_prob]} act={[round(x,2) for x in act_prob]} "
                    f"enc_win={enc_hist} rup_win={rup_hist} slip_win={slip_hist} drift_win={drift_hist} "
                    f"frag={float(info.get('fragility', np.nan)):.2f} "
                    f"rmem={float(info.get('rupture_memory', np.nan)):.2f} "
                    f"carry_g={int(not args.reset_g_every_episode)} "
                    f"(ep={episode}, {dt:.1f}s)"
                )

                act_hist[:] = 0
                zone_hist[:] = 0
                enc_hist = 0
                rup_hist = 0
                slip_hist = 0
                drift_hist = 0
                t0 = time.time()

            if args.save_ckpt_every > 0 and ((step + 1) % args.save_ckpt_every == 0):
                save_checkpoint(run_dir, f"step{step+1}", agent, decoder, meta)

    finally:
        if viewer is not None:
            viewer.close()

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

    save_checkpoint(run_dir, "final", agent, decoder, meta)
    print(f"Saved final checkpoint to: {run_dir / 'ckpt_final.pt'}")


if __name__ == "__main__":
    main()