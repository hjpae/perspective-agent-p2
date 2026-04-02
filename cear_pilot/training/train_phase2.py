# cear_pilot/training/train_phase2.py
# -*- coding: utf-8 -*-
"""
Phase 2 training: salience gating + self-modulating perspective.

Supports:
  - Fixed perturbation count: --n_perturbations 4
  - Mixed block schedule:     --mixed_schedule "0:50,4:50,0:50"
  - Alpha mode:               --update_mode adaptive|fixed
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from cear_pilot.envs.nzone_phase2 import NZonePhase2Config, NZonePhase2Env
from cear_pilot.models.agent import CEARAgent, AgentConfig
from cear_pilot.models.encoder import EncoderConfig
from cear_pilot.models.world_latent import WorldLatentConfig
from cear_pilot.models.state_head import StateHeadConfig
from cear_pilot.models.policy import PolicyConfig
from cear_pilot.models.decoder import ObsDecoder, DecoderConfig


def freeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def count_params(module: nn.Module, only_trainable=False):
    return sum(p.numel() for p in module.parameters() if (not only_trainable or p.requires_grad))


ERR_DIM = 6


def build_err_t(pred_err, pred_err_ema_short, pred_err_ema_long,
                pred_err_prev, perturbation_active, perturbation_trace, device):
    ema_safe = max(pred_err_ema_long, 1e-6)
    feats = [
        min(pred_err / ema_safe, 5.0),
        min(pred_err_ema_short / ema_safe, 5.0),
        float(np.log1p(pred_err_ema_long)),
        float(np.tanh((pred_err - pred_err_prev) * 10.0)),
        float(perturbation_active),
        float(perturbation_trace),
    ]
    return torch.tensor([feats], dtype=torch.float32, device=device)


def parse_mixed_schedule(s: str) -> List[Tuple[int, int]]:
    """Parse "0:50,4:50,0:50" → [(0,50),(4,50),(0,50)]"""
    blocks = []
    for part in s.split(","):
        n, eps = part.strip().split(":")
        blocks.append((int(n), int(eps)))
    return blocks


# ── Loading + freezing ──

def load_phase1_and_build(args):
    device = args.device
    ckpt = torch.load(args.phase1_ckpt, map_location=device)
    meta = ckpt["meta"]

    enc_cfg = EncoderConfig()
    enc_cfg.__dict__.update(meta["agent_cfg"]["encoder"])
    enc_cfg.obs_dim = 8
    enc_cfg.g_dim = 12
    enc_cfg.use_salience_gate = True

    world_cfg = WorldLatentConfig()
    world_cfg.g_dim = 12
    world_cfg.z_dim = enc_cfg.z_dim
    world_cfg.p_dim = enc_cfg.p_dim
    world_cfg.update_mode = args.update_mode
    world_cfg.alpha_fixed = args.alpha_fixed
    world_cfg.alpha_min = args.alpha_min
    world_cfg.alpha_max = args.alpha_max
    world_cfg.use_error_feedback = True
    world_cfg.err_dim = ERR_DIM

    state_cfg = StateHeadConfig()
    state_cfg.__dict__.update(meta["agent_cfg"]["state"])
    policy_cfg = PolicyConfig()
    policy_cfg.__dict__.update(meta["agent_cfg"]["policy"])

    agent_cfg = AgentConfig(encoder=enc_cfg, world=world_cfg,
                            state=state_cfg, policy=policy_cfg, device=device)
    agent = CEARAgent(agent_cfg)

    ckpt_state = ckpt["agent_state"]
    model_state = agent.state_dict()
    compatible = {k: v for k, v in ckpt_state.items()
                  if k in model_state and model_state[k].shape == v.shape}
    result = agent.load_state_dict(compatible, strict=False)
    print(f"[load] Phase 1 loaded. New params: {len(result.missing_keys)}")

    dec_cfg = DecoderConfig(**meta["decoder_cfg"])
    dec_cfg.obs_dim = 8
    decoder = ObsDecoder(dec_cfg)
    decoder.load_state_dict(ckpt["decoder_state"], strict=False)

    return agent, decoder, meta


def apply_freezing(agent, decoder):
    freeze(agent.enc.obs_enc.mlp)
    freeze(agent.enc.prop_enc)
    freeze(agent.state)
    freeze(agent.policy)
    freeze(decoder)
    print(f"[freeze] Trainable: {count_params(agent, True)}  Total: {count_params(agent)}")


# ── Training ──

def train(args):
    device = args.device
    agent, decoder, p1_meta = load_phase1_and_build(args)
    agent.to(device)
    decoder.to(device)
    apply_freezing(agent, decoder)

    env_cfg = NZonePhase2Config(
        max_steps=args.max_steps,
        sigma_left=args.sigma_left, sigma_right=args.sigma_right,
        n_perturbations=args.n_perturbations,
        perturbation_duration=args.perturbation_duration,
        perturbation_scale=args.perturbation_scale,
    )
    env = NZonePhase2Env(config=env_cfg)

    trainable_params = [p for p in agent.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    outdir = Path(args.outdir) if args.outdir else Path(f"outputs/phase2_s{args.seed}")
    outdir.mkdir(parents=True, exist_ok=True)

    # Mixed schedule
    mixed = None
    if args.mixed_schedule:
        mixed = parse_mixed_schedule(args.mixed_schedule)
        total_eps = sum(eps for _, eps in mixed)
        print(f"[mixed] Schedule: {mixed} → {total_eps} episodes")
    else:
        total_eps = args.episodes

    traj_rows = []
    n_actions = int(env.action_space.n)
    pe_ema_s, pe_ema_l, pe_prev = 0.05, 0.05, 0.05

    agent.reset(batch_size=1)
    t0 = time.time()
    global_ep = 0

    # Build episode list with (n_perturb, block_id) for each episode
    ep_schedule = []
    if mixed:
        for block_id, (n_p, n_eps) in enumerate(mixed):
            for _ in range(n_eps):
                ep_schedule.append((n_p, block_id))
    else:
        for _ in range(total_eps):
            ep_schedule.append((args.n_perturbations, 0))

    for global_ep, (n_perturb_now, block_id) in enumerate(ep_schedule):
        # Update env perturbation count for this episode
        env.cfg.n_perturbations = n_perturb_now

        obs, info = env.reset(seed=args.seed + global_ep)
        last_action = 4
        ep_pe = []
        done = False

        while not done:
            x_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            p_t = F.one_hot(torch.tensor([last_action], device=device), num_classes=n_actions).float()

            err_t = build_err_t(pe_prev, pe_ema_s, pe_ema_l, pe_prev,
                                int(info.get("perturbation_active", 0)),
                                float(info.get("perturbation_trace", 0.0)), device)

            out = agent.forward_step(x_t, p_t, err_t=err_t)
            action = agent.policy.sample_action(out["logits"], greedy=args.greedy)
            a_int = int(action.item())

            obs_next, _, terminated, truncated, info_next = env.step(a_int)
            x_next = torch.tensor(obs_next, dtype=torch.float32, device=device).unsqueeze(0)
            a_oh = F.one_hot(torch.tensor([a_int], device=device), num_classes=n_actions).float()

            pred_obs = decoder(out["g"], a_oh)
            pe = F.mse_loss(pred_obs, x_next)
            pe_val = float(pe.item())

            optimizer.zero_grad()
            pe.backward()
            if args.clip_grad > 0:
                nn.utils.clip_grad_norm_(trainable_params, args.clip_grad)
            optimizer.step()

            pe_ema_s = 0.1 * pe_val + 0.9 * pe_ema_s
            pe_ema_l = 0.01 * pe_val + 0.99 * pe_ema_l
            pe_prev = pe_val
            ep_pe.append(pe_val)

            if args.save_traj:
                g_np = out["g"].detach().cpu().numpy()[0]
                # FiLM gating: compute gamma/beta for logging
                gamma_np = np.zeros(agent.cfg.encoder.z_dim)
                beta_np = np.zeros(agent.cfg.encoder.z_dim)
                if hasattr(agent.enc.obs_enc, "film"):
                    with torch.no_grad():
                        gb = agent.enc.obs_enc.film(out["g"].detach())
                        gam, bet = gb.chunk(2, dim=-1)
                        gamma_np = gam.cpu().numpy()[0]
                        beta_np = bet.cpu().numpy()[0]

                row = {
                    "episode": global_ep, "t": int(info["t"]),
                    "x": int(info["x"]), "y": int(info["y"]),
                    "zone_id": int(info["zone_id"]), "action": a_int,
                    "pred_err": pe_val, "alpha": float(out["alpha"].item()),
                    "g_norm": float(np.linalg.norm(g_np)),
                    "perturbation_active": int(info.get("perturbation_active", 0)),
                    "perturbation_trace": float(info.get("perturbation_trace", 0.0)),
                    "n_perturb_setting": n_perturb_now,
                    "block_id": block_id,
                }
                for gi in range(len(g_np)):
                    row[f"g_{gi}"] = float(g_np[gi])
                for gi in range(len(gamma_np)):
                    row[f"gamma_{gi}"] = float(gamma_np[gi])
                    row[f"beta_{gi}"] = float(beta_np[gi])
                traj_rows.append(row)

            obs = obs_next
            info = info_next
            last_action = a_int
            done = bool(terminated or truncated)

        if (global_ep + 1) % args.print_every == 0:
            elapsed = time.time() - t0
            print(f"[ep {global_ep+1:4d}/{len(ep_schedule)}] "
                  f"PE={np.mean(ep_pe):.4f} ||g||={out['g'].detach().cpu().norm().item():.3f} "
                  f"α={out['alpha'].item():.3f} nP={n_perturb_now} blk={block_id} "
                  f"({elapsed:.0f}s)")

    # Save
    if args.save_traj and traj_rows:
        traj_df = pd.DataFrame(traj_rows)
        traj_df.to_parquet(outdir / "traj.parquet", index=False)
        print(f"[save] {outdir / 'traj.parquet'} ({len(traj_df)} rows)")

    torch.save({
        "agent_state": agent.state_dict(),
        "decoder_state": decoder.state_dict(),
        "meta": {
            "phase1_ckpt": str(args.phase1_ckpt),
            "agent_cfg": {"encoder": agent.cfg.encoder.__dict__,
                          "world": agent.cfg.world.__dict__,
                          "state": agent.cfg.state.__dict__,
                          "policy": agent.cfg.policy.__dict__},
            "decoder_cfg": decoder.cfg.__dict__,
            "env_cfg": asdict(env_cfg),
            "args": vars(args),
        },
    }, outdir / "ckpt_final.pt")
    print(f"[save] {outdir / 'ckpt_final.pt'}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1_ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip_grad", type=float, default=1.0)
    ap.add_argument("--print_every", type=int, default=10)
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--save_traj", action="store_true")

    ap.add_argument("--update_mode", type=str, default="adaptive", choices=["adaptive", "fixed"])
    ap.add_argument("--alpha_fixed", type=float, default=0.10)
    ap.add_argument("--alpha_min", type=float, default=0.03)
    ap.add_argument("--alpha_max", type=float, default=0.30)

    ap.add_argument("--sigma_left", type=float, default=0.20)
    ap.add_argument("--sigma_right", type=float, default=0.10)
    ap.add_argument("--n_perturbations", type=int, default=4)
    ap.add_argument("--perturbation_duration", type=int, default=15)
    ap.add_argument("--perturbation_scale", type=float, default=0.12)

    ap.add_argument("--mixed_schedule", type=str, default="",
                    help="Block schedule: 'n_perturb:episodes,...' e.g. '0:50,4:50,0:50'")

    return ap.parse_args()


def main():
    train(parse_args())


if __name__ == "__main__":
    main()