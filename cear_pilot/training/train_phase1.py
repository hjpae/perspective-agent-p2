# cear_pilot/training/train_phase1.py
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

from cear_pilot.envs.nzone_phase1 import NZonePhase1Config, NZonePhase1Env
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
            self.mean = self.beta * self.mean + (1.0 - self.beta) * x
            self.var = self.beta * self.var + (1.0 - self.beta) * (x - m) * (x - m)
        std = float(np.sqrt(max(self.var, 0.0) + self.eps))
        return float(self.mean), std


def save_checkpoint(run_dir: Path, tag: str, agent: CEARAgent, decoder: ObsDecoder, meta: Dict) -> None:
    ckpt = {
        "agent_state": agent.state_dict(),
        "decoder_state": decoder.state_dict(),
        "meta": meta,
    }
    torch.save(ckpt, run_dir / f"ckpt_{tag}.pt")


def try_save_table(rows, out_path: Path) -> Path:
    df = pd.DataFrame(rows)
    try:
        p = out_path.with_suffix(".parquet")
        df.to_parquet(p, index=False)
        return p
    except Exception:
        p = out_path.with_suffix(".csv")
        df.to_csv(p, index=False)
        return p


def action_name(a: int) -> str:
    return ["UP", "DOWN", "LEFT", "RIGHT", "STAY"][int(a)]


def main():
    ap = argparse.ArgumentParser()

    # -------------------------
    # core training
    # -------------------------
    ap.add_argument("--steps", type=int, default=48000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")

    ap.add_argument("--w_smooth", type=float, default=0.25)
    ap.add_argument("--w_entropy", type=float, default=0.01)
    ap.add_argument("--w_actor", type=float, default=0.5)
    ap.add_argument("--actor_b", type=float, default=0.98)
    ap.add_argument("--warmup_steps", type=int, default=12000)

    # default: keep g across episode boundaries
    ap.add_argument(
        "--reset_g_every_episode",
        action="store_true",
        help="If set, reset g to zero at every episode boundary. Default is False (carry g).",
    )

    # -------------------------
    # phase-1 env base
    # -------------------------
    ap.add_argument("--width", type=int, default=23)
    ap.add_argument("--height", type=int, default=7)
    ap.add_argument("--obs_dim", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=240)

    ap.add_argument("--phase1_sigma_left", type=float, default=0.60)
    ap.add_argument("--phase1_sigma_center", type=float, default=0.30)
    ap.add_argument("--phase1_sigma_right", type=float, default=0.03)
    ap.add_argument("--phase1_left_power", type=float, default=0.90)
    ap.add_argument("--phase1_right_power", type=float, default=1.85)

    # -------------------------
    # logging / saving
    # -------------------------
    ap.add_argument("--log_traj", action="store_true")
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--save_ckpt_every", type=int, default=12000)
    ap.add_argument("--print_every", type=int, default=1200)

    args = ap.parse_args()

    seed_everything(args.seed, deterministic=True)
    device = torch.device(args.device)

    env_cfg = NZonePhase1Config(
        width=args.width,
        height=args.height,
        obs_dim=args.obs_dim,
        max_steps=args.max_steps,
        phase1_sigma_left=args.phase1_sigma_left,
        phase1_sigma_center=args.phase1_sigma_center,
        phase1_sigma_right=args.phase1_sigma_right,
        phase1_left_power=args.phase1_left_power,
        phase1_right_power=args.phase1_right_power,
    )
    env = NZonePhase1Env(config=env_cfg)

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
    agent_cfg.world.update_mode = "fixed"
    agent_cfg.world.alpha_fixed = agent_cfg.world.g_damping

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
        "env_type": "phase1_common",
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

    # initialize agent once
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
    zone_hist = np.zeros(5, dtype=np.int64)

    t0 = time.time()
    episode = 0
    t_in_ep = 0

    try:
        for step in range(args.steps):
            x_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            p_t = make_proprio_from_last_action(last_action, n_actions, device=device)

            out = agent.forward_step(x_t, p_t, ablate_g=False, err_t=None)
            g_t = out["g"]
            s_t = out["s"]
            logits_pred = out["logits"]
            alpha_t = out["alpha"]

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

            if args.log_traj and ((step % log_every) == 0):
                z = int(info.get("zone_id", -1))
                with torch.no_grad():
                    g_np = g_t.detach().squeeze(0).float().cpu().numpy()
                    g_norm = float(torch.linalg.vector_norm(g_t).item())
                    ent_val = float(entropy.item())
                    alpha_val = float(alpha_t.mean().item())

                row = {
                    "t_global": int(step),
                    "episode": int(episode),
                    "t_in_ep": int(t_in_ep),
                    "learn_phase": str(learn_phase),
                    "zone_id": int(z),
                    "x": int(info.get("x", -1)),
                    "y": int(info.get("y", -1)),
                    "action": int(a_int),
                    "action_name": action_name(a_int),
                    "current_sigma": float(info.get("current_sigma", np.nan)),
                    "entropy": float(ent_val),
                    "g_norm": float(g_norm),
                    "alpha": float(alpha_val),
                    "loss": float(loss.item()),
                    "loss_world": float(loss_world.item()),
                    "loss_pred": float(loss_pred.item()),
                    "loss_smooth": float(loss_smooth.item()),
                    "loss_actor": float(loss_actor.item()),
                    "baseline": float(baseline),
                    "adv": float(adv),
                    "w_actor_eff": float(w_actor_eff),
                }
                for i, gv in enumerate(g_np):
                    row[f"g_{i}"] = float(gv)
                log_rows.append(row)

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
            if isinstance(z, (int, np.integer)) and 0 <= int(z) <= 4:
                zone_hist[int(z)] += 1

            lw = float(loss_world.item())
            ema_world = lw if ema_world is None else 0.98 * ema_world + 0.02 * lw

            if (step + 1) % args.print_every == 0:
                dt = time.time() - t0
                with torch.no_grad():
                    e_det = per_a_err.detach().float().cpu().numpy()
                    e_min, e_max, e_std = float(e_det.min()), float(e_det.max()), float(e_det.std())
                    alpha_mean = float(alpha_t.mean().item())
                    g_norm = float(torch.linalg.vector_norm(g_t.detach()).item())

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
                    f"zone={[round(x, 2) for x in zone_prob]} act={[round(x, 2) for x in act_prob]} "
                    f"carry_g={int(not args.reset_g_every_episode)} "
                    f"| adv={float(adv):.4f} w_actor={float(w_actor_eff):.3f} "
                    f"sigma={float(info.get('current_sigma', np.nan)):.3f} "
                    f"alpha={alpha_mean:.3f} ||g||={g_norm:.3f} "
                    f"(ep={episode}, {dt:.1f}s)"
                )

                act_hist[:] = 0
                zone_hist[:] = 0
                t0 = time.time()

            if args.save_ckpt_every > 0 and ((step + 1) % args.save_ckpt_every == 0):
                save_checkpoint(run_dir, f"step{step+1}", agent, decoder, meta)

    finally:
        if args.log_traj and len(log_rows) > 0:
            out_path = try_save_table(log_rows, run_dir / "train_traj")
            print(f"Saved training trajectory to: {out_path}")

    save_checkpoint(run_dir, "final", agent, decoder, meta)
    print(f"Saved final checkpoint to: {run_dir / 'ckpt_final.pt'}")


if __name__ == "__main__":
    main()