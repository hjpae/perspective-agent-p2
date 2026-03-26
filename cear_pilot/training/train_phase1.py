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
            self.mean = float(x)
            self.var = 0.0
        else:
            prev_mean = float(self.mean)
            self.mean = self.beta * float(self.mean) + (1.0 - self.beta) * float(x)
            self.var = self.beta * float(self.var) + (1.0 - self.beta) * (float(x) - prev_mean) ** 2

        std = float(np.sqrt(max(float(self.var), 0.0) + self.eps))
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
    ap.add_argument("--steps", type=int, default=36000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")

    ap.add_argument("--w_smooth", type=float, default=0.25)
    ap.add_argument("--w_actor", type=float, default=0.5)
    ap.add_argument("--warmup_steps", type=int, default=12000)

    # actor advantage stabilization
    ap.add_argument("--actor_baseline_beta", type=float, default=0.98)
    ap.add_argument("--actor_std_beta", type=float, default=0.99)
    ap.add_argument("--adv_clip", type=float, default=3.0)

    # adaptive entropy
    ap.add_argument("--w_entropy_init", type=float, default=0.01)
    ap.add_argument("--w_entropy_min", type=float, default=0.002)
    ap.add_argument("--w_entropy_max", type=float, default=0.05)
    ap.add_argument("--entropy_target_ratio", type=float, default=0.60)
    ap.add_argument("--entropy_adapt_rate", type=float, default=0.02)

    # default: reset g every episode
    ap.add_argument(
        "--carry_g_across_episodes",
        action="store_true",
        help="If set, keep g across episode boundaries. Default is False (reset g every episode).",
    )

    # default: fixed world after first reset(seed=args.seed)
    ap.add_argument(
        "--resample_env_every_episode",
        action="store_true",
        help="If set, rebuild phase1 static maps at every episode boundary using seed+episode. Default is False.",
    )

    # -------------------------
    # phase-1 env
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

    reset_g_every_episode = not bool(args.carry_g_across_episodes)

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
        static_map_seed=args.seed,
        resample_static_maps_on_reset=bool(args.resample_env_every_episode),
    )
    env = NZonePhase1Env(config=env_cfg)

    # first reset initializes the rollout RNG and, if configured, the static world
    obs, info = env.reset(seed=args.seed)
    try:
        env.action_space.seed(args.seed)
        env.observation_space.seed(args.seed)
    except Exception:
        pass

    n_actions = int(env.action_space.n)

    # -------------------------
    # agent / decoder configs
    # -------------------------
    agent_cfg = AgentConfig(device=args.device)

    agent_cfg.encoder.obs_dim = args.obs_dim
    agent_cfg.encoder.proprio_dim = n_actions

    agent_cfg.world.update_mode = "fixed"
    agent_cfg.world.alpha_fixed = agent_cfg.world.g_damping
    agent_cfg.world.z_dim = agent_cfg.encoder.z_dim
    agent_cfg.world.p_dim = agent_cfg.encoder.p_dim
    agent_cfg.world.err_dim = 1

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
    entropy_weight = float(args.w_entropy_init)
    entropy_target = float(np.log(n_actions) * args.entropy_target_ratio)

    meta = {
        "env_type": "phase1_common",
        "seed": args.seed,
        "steps": args.steps,
        "lr": args.lr,
        "device": args.device,
        "warmup_steps": warmup_steps,
        "reset_g_every_episode": bool(reset_g_every_episode),
        "carry_g_across_episodes": bool(args.carry_g_across_episodes),
        "resample_env_every_episode": bool(args.resample_env_every_episode),
        "fixed_world_seed": int(args.seed),
        "loss_weights": {
            "w_smooth": args.w_smooth,
            "w_actor": args.w_actor,
            "w_entropy_init": args.w_entropy_init,
            "w_entropy_min": args.w_entropy_min,
            "w_entropy_max": args.w_entropy_max,
        },
        "actor_baseline_beta": args.actor_baseline_beta,
        "actor_std_beta": args.actor_std_beta,
        "adv_clip": args.adv_clip,
        "entropy_target_ratio": args.entropy_target_ratio,
        "entropy_adapt_rate": args.entropy_adapt_rate,
        "env_cfg": asdict(env_cfg),
        "agent_cfg": {
            "encoder": asdict(agent.cfg.encoder),
            "world": asdict(agent.cfg.world),
            "state": asdict(agent.cfg.state),
            "policy": asdict(agent.cfg.policy),
        },
        "decoder_cfg": asdict(decoder.cfg),
    }
    save_meta(run_dir, meta)

    log_rows = []
    log_every = int(max(1, args.log_every))

    agent.reset(batch_size=1)
    last_action = 4
    g_prev = agent.get_latents()["g"].detach().clone()

    ema_world = None
    pi_prev = None
    kl_ema = None
    maxpi_ema = None
    logits_norm_ema = None
    entropy_coef_ema = None

    baseline_stats = EMAMeanVar(beta=args.actor_baseline_beta)
    adv_stats = EMAMeanVar(beta=args.actor_std_beta)

    act_hist = np.zeros(n_actions, dtype=np.int64)
    zone_hist = np.zeros(5, dtype=np.int64)

    t0 = time.time()
    episode = 0
    t_in_ep = 0

    print(
        f"[phase1:init] "
        f"carry_g={int(args.carry_g_across_episodes)} "
        f"reset_g={int(reset_g_every_episode)} "
        f"resample_env={int(args.resample_env_every_episode)} "
        f"fixed_world_seed={args.seed}"
    )

    try:
        for step in range(args.steps):
            x_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            p_t = make_proprio_from_last_action(last_action, n_actions, device=device)

            out = agent.forward_step(x_t, p_t, ablate_g=False, err_t=None)
            g_t = out["g"]
            s_t = out["s"]
            logits_pred = out["logits"]
            alpha_t = out["alpha"]

            # actor uses detached state; predictor uses its own logits
            logits_act = agent.policy(s_t.detach())
            pi_act = torch.softmax(logits_act, dim=-1)
            pi_pred = torch.softmax(logits_pred, dim=-1).detach()

            a_t = agent.policy.sample_action(logits_act, greedy=False)
            a_int = int(a_t.item())

            obs_next, _, terminated, truncated, info2 = env.step(a_int)
            x_next = torch.tensor(obs_next, dtype=torch.float32, device=device).unsqueeze(0)

            # detached policy mixture for prediction target
            xhat_all = decoder.predict_all_actions(g_t)
            xhat_exp = torch.sum(pi_pred.unsqueeze(-1) * xhat_all, dim=1)

            loss_pred = F.mse_loss(xhat_exp, x_next)
            loss_smooth = F.mse_loss(g_t, g_prev.detach())

            entropy = -(pi_act * torch.log(pi_act + 1e-9)).sum(dim=-1).mean()

            per_a_err = torch.mean((xhat_all - x_next.unsqueeze(1)) ** 2, dim=-1).squeeze(0)
            e_chosen = per_a_err[a_int]
            c_t = float(e_chosen.detach().item())

            b_mean, _ = baseline_stats.update(c_t)
            adv_raw = float(b_mean - c_t)
            _, adv_std = adv_stats.update(adv_raw)
            adv_norm = adv_raw / max(adv_std, 1e-6)
            adv_clip = float(np.clip(adv_norm, -float(args.adv_clip), float(args.adv_clip)))

            logp_a = torch.log_softmax(logits_act, dim=-1)[0, a_int]
            loss_actor = -(adv_clip * logp_a)

            if step < warmup_steps:
                w_actor_now = 0.0
                phase_name = "warmup"
            else:
                w_actor_now = float(args.w_actor)
                phase_name = "full"

            loss = (
                loss_pred
                + float(args.w_smooth) * loss_smooth
                - float(entropy_weight) * entropy
                + w_actor_now * loss_actor
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            # adaptive entropy coefficient
            ent_val = float(entropy.detach().item())
            if ent_val < entropy_target:
                entropy_weight *= (1.0 + float(args.entropy_adapt_rate))
            else:
                entropy_weight *= (1.0 - 0.5 * float(args.entropy_adapt_rate))
            entropy_weight = float(np.clip(entropy_weight, args.w_entropy_min, args.w_entropy_max))

            act_hist[a_int] += 1
            z = info2.get("zone_id", -1)
            if isinstance(z, (int, np.integer)) and 0 <= int(z) <= 4:
                zone_hist[int(z)] += 1

            with torch.no_grad():
                lw = float(loss_pred.item())
                ema_world = lw if ema_world is None else 0.98 * ema_world + 0.02 * lw

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

                ln = float(torch.mean(torch.abs(logits_act)).item())
                maxpi_ema = maxpi if maxpi_ema is None else (0.98 * maxpi_ema + 0.02 * maxpi)
                kl_ema = kl if kl_ema is None else (0.98 * kl_ema + 0.02 * kl)
                logits_norm_ema = ln if logits_norm_ema is None else (0.98 * logits_norm_ema + 0.02 * ln)
                entropy_coef_ema = (
                    entropy_weight if entropy_coef_ema is None else (0.98 * entropy_coef_ema + 0.02 * entropy_weight)
                )

                g_norm = float(torch.linalg.vector_norm(g_t.detach()).item())
                e_det = per_a_err.detach().float().cpu().numpy()
                e_min, e_max, e_std = float(e_det.min()), float(e_det.max()), float(e_det.std())

            if args.log_traj and ((step % log_every) == 0):
                with torch.no_grad():
                    g_np = g_t.detach().squeeze(0).float().cpu().numpy()
                    s_np = s_t.detach().squeeze(0).float().cpu().numpy()

                row = {
                    "t_global": int(step),
                    "episode": int(episode),
                    "t_in_ep": int(t_in_ep),
                    "phase": str(phase_name),
                    "episode_seed": int(args.seed if not args.resample_env_every_episode else (args.seed + episode)),
                    "world_seed": int(args.seed if not args.resample_env_every_episode else (args.seed + episode)),
                    "resample_env_every_episode": int(args.resample_env_every_episode),
                    "carry_g_across_episodes": int(args.carry_g_across_episodes),
                    "reset_g_every_episode": int(reset_g_every_episode),

                    "zone_id": int(info2.get("zone_id", -1)),
                    "x": int(info2.get("x", -1)),
                    "y": int(info2.get("y", -1)),

                    "action": int(a_int),
                    "action_name": action_name(a_int),

                    "loss_pred": float(loss_pred.item()),
                    "loss_smooth": float(loss_smooth.item()),
                    "loss_actor": float(loss_actor.item()),
                    "cost_chosen": float(c_t),
                    "baseline": float(b_mean),
                    "adv_raw": float(adv_raw),
                    "adv_std": float(adv_std),
                    "adv_clip": float(adv_clip),
                    "w_actor_now": float(w_actor_now),

                    "entropy": float(entropy.item()),
                    "entropy_weight": float(entropy_weight),
                    "maxpi": float(maxpi),
                    "kl_to_prev_pi": float(kl),
                    "logits_abs_mean": float(ln),

                    "current_sigma": float(info2.get("current_sigma", np.nan)),
                    "alpha": float(alpha_t.mean().item()),
                    "g_norm": float(g_norm),
                    "e_min": float(e_min),
                    "e_max": float(e_max),
                    "e_std": float(e_std),
                }

                for i, gv in enumerate(g_np):
                    row[f"g_{i}"] = float(gv)
                for i, sv in enumerate(s_np):
                    row[f"s_{i}"] = float(sv)

                log_rows.append(row)

            if (step + 1) % args.print_every == 0:
                act_prob = (act_hist / max(act_hist.sum(), 1)).tolist()
                zone_prob = (zone_hist / max(zone_hist.sum(), 1)).tolist()

                print(
                    f"[{step+1:>7}/{args.steps}] "
                    f"phase={phase_name} "
                    f"world={float(loss_pred.item()):.4f} w_ema={0.0 if ema_world is None else float(ema_world):.4f} "
                    f"pred={float(loss_pred.item()):.4f} smooth={float(loss_smooth.item()):.4f} "
                    f"| actor={float(loss_actor.item()):.4f} b={float(b_mean):.4f} "
                    f"H={float(entropy.item()):.3f} Hc={0.0 if entropy_coef_ema is None else float(entropy_coef_ema):.4f} "
                    f"maxpi={0.0 if maxpi_ema is None else float(maxpi_ema):.3f} "
                    f"KL={0.0 if kl_ema is None else float(kl_ema):.6f} "
                    f"logits|.|={0.0 if logits_norm_ema is None else float(logits_norm_ema):.3f} "
                    f"e[min,max,std]={e_min:.3f},{e_max:.3f},{e_std:.3f} "
                    f"zone={[round(x, 2) for x in zone_prob]} "
                    f"act={[round(x, 2) for x in act_prob]} "
                    f"carry_g={int(args.carry_g_across_episodes)} "
                    f"| adv={float(adv_clip):.4f} w_actor={float(w_actor_now):.3f} "
                    f"sigma={float(info2.get('current_sigma', np.nan)):.3f} "
                    f"alpha={float(alpha_t.mean().item()):.3f} "
                    f"||g||={float(g_norm):.3f} "
                    f"(ep={episode}, {time.time() - t0:.1f}s)"
                )

                act_hist[:] = 0
                zone_hist[:] = 0
                t0 = time.time()

            if args.save_ckpt_every > 0 and ((step + 1) % args.save_ckpt_every == 0):
                save_checkpoint(run_dir, f"step{step+1}", agent, decoder, meta)

            obs = obs_next
            last_action = a_int
            g_prev = g_t.detach().clone()
            t_in_ep += 1

            if truncated or terminated:
                if args.resample_env_every_episode:
                    next_seed = args.seed + episode + 1
                    obs, info = env.reset(seed=next_seed)
                else:
                    obs, info = env.reset()

                if reset_g_every_episode:
                    agent.reset(batch_size=1)

                g_prev = agent.get_latents()["g"].detach().clone()
                last_action = 4
                episode += 1
                t_in_ep = 0

    finally:
        if args.log_traj and len(log_rows) > 0:
            out_path = try_save_table(log_rows, run_dir / "train_traj")
            print(f"Saved training trajectory to: {out_path}")

    save_checkpoint(run_dir, "final", agent, decoder, meta)
    print(f"Saved final checkpoint to: {run_dir / 'ckpt_final.pt'}")


if __name__ == "__main__":
    main()