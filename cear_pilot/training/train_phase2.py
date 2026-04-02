# cear_pilot/training/train_phase2.py
# -*- coding: utf-8 -*-
"""
Phase 2 training: salience gating + self-modulating perspective.

Loads Phase 1 checkpoint, freezes base components,
trains only FiLM (salience gate) + world_latent (GRU + alpha_net).
Loss: prediction error only.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

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


def freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def count_params(module: nn.Module, only_trainable: bool = False) -> int:
    if only_trainable:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


# ── err_t packaging ──

ERR_DIM = 6

def build_err_t(
    pred_err: float,
    pred_err_ema_short: float,
    pred_err_ema_long: float,
    pred_err_prev: float,
    perturbation_active: int,
    perturbation_trace: float,
    device: torch.device,
) -> torch.Tensor:
    """Package 6 error/perturbation features for world_latent."""
    ema_long_safe = max(pred_err_ema_long, 1e-6)
    feats = [
        min(pred_err / ema_long_safe, 5.0),                # 0: PE ratio
        min(pred_err_ema_short / ema_long_safe, 5.0),       # 1: short/long ratio
        float(np.log1p(pred_err_ema_long)),                  # 2: log long EMA
        float(np.tanh((pred_err - pred_err_prev) * 10.0)),  # 3: signed trend
        float(perturbation_active),                          # 4: perturbation flag
        float(perturbation_trace),                           # 5: perturbation trace
    ]
    return torch.tensor([feats], dtype=torch.float32, device=device)


# ── Phase 1 checkpoint loading ──

def load_phase1_and_build(args: argparse.Namespace):
    device = args.device
    ckpt = torch.load(args.phase1_ckpt, map_location=device)
    meta = ckpt["meta"]

    # Build agent with Phase 2 config
    enc_cfg = EncoderConfig()
    enc_cfg.__dict__.update(meta["agent_cfg"]["encoder"])
    enc_cfg.obs_dim = 8
    enc_cfg.g_dim = 12  # needed for FiLM
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

    agent_cfg = AgentConfig(
        encoder=enc_cfg, world=world_cfg,
        state=state_cfg, policy=policy_cfg, device=device,
    )
    agent = CEARAgent(agent_cfg)

    # Load Phase 1 weights (strict=False: missing keys = new modules)
    ckpt_state = ckpt["agent_state"]
    model_state = agent.state_dict()
    compatible = {}
    for k, v in ckpt_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            compatible[k] = v
    result = agent.load_state_dict(compatible, strict=False)
    print(f"[load] Phase 1 weights loaded. Missing (new): {len(result.missing_keys)}")
    for k in result.missing_keys:
        print(f"  + {k}")

    # Decoder
    dec_cfg = DecoderConfig(**meta["decoder_cfg"])
    dec_cfg.obs_dim = 8
    decoder = ObsDecoder(dec_cfg)
    decoder.load_state_dict(ckpt["decoder_state"], strict=False)

    return agent, decoder, meta


# ── Freeze logic ──

def apply_freezing(agent: CEARAgent, decoder: ObsDecoder):
    """Freeze everything except FiLM + world_latent."""
    # Freeze encoder base MLP (keep FiLM trainable)
    freeze(agent.enc.obs_enc.mlp)
    freeze(agent.enc.prop_enc)
    # Freeze state head
    #freeze(agent.state)
    # Freeze policy
    freeze(agent.policy)
    # Freeze decoder
    freeze(decoder)

    # Trainable: agent.enc.obs_enc.film + agent.world (GRU + alpha_net)
    print(f"[freeze] Trainable params: {count_params(agent, only_trainable=True)}")
    print(f"[freeze] Total params:     {count_params(agent)}")
    print(f"[freeze] Decoder frozen:   {count_params(decoder)}")


# ── Training loop ──

def train(args: argparse.Namespace):
    device = args.device

    # Load Phase 1 + build
    agent, decoder, p1_meta = load_phase1_and_build(args)
    agent.to(device)
    decoder.to(device)
    apply_freezing(agent, decoder)

    # Env
    env_cfg = NZonePhase2Config(
        max_steps=args.max_steps,
        sigma_left=args.sigma_left,
        sigma_right=args.sigma_right,
        n_perturbations=args.n_perturbations,
        perturbation_duration=args.perturbation_duration,
        perturbation_scale=args.perturbation_scale,
    )
    env = NZonePhase2Env(config=env_cfg)

    # Optimizer (only trainable params)
    trainable_params = [p for p in agent.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    # Outdir
    outdir = Path(args.outdir) if args.outdir else Path(f"outputs/phase2_nperturb{args.n_perturbations}_s{args.seed}")
    outdir.mkdir(parents=True, exist_ok=True)

    # Logging
    traj_rows = []
    n_actions = int(env.action_space.n)

    # PE tracking
    pred_err_ema_short = 0.05
    pred_err_ema_long = 0.05
    pred_err_prev = 0.05

    # Training
    agent.reset(batch_size=1)
    t0 = time.time()

    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)
        last_action = 4  # STAY
        ep_pe = []
        done = False

        while not done:
            x_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            p_t = F.one_hot(torch.tensor([last_action], device=device), num_classes=n_actions).float()

            # Build err_t
            err_t = build_err_t(
                pred_err=pred_err_prev,
                pred_err_ema_short=pred_err_ema_short,
                pred_err_ema_long=pred_err_ema_long,
                pred_err_prev=pred_err_prev,
                perturbation_active=int(info.get("perturbation_active", 0)),
                perturbation_trace=float(info.get("perturbation_trace", 0.0)),
                device=device,
            )

            # Forward
            out = agent.forward_step(x_t, p_t, err_t=err_t)
            action = agent.policy.sample_action(out["logits"], greedy=args.greedy)
            a_int = int(action.item())

            # Step env
            obs_next, _, terminated, truncated, info_next = env.step(a_int)

            # Decoder prediction error
            x_next = torch.tensor(obs_next, dtype=torch.float32, device=device).unsqueeze(0)
            a_onehot = F.one_hot(torch.tensor([a_int], device=device), num_classes=n_actions).float()

            pred_obs = decoder(out["g"], a_onehot)
            pe = F.mse_loss(pred_obs, x_next)
            pe_val = float(pe.item())

            # Backward
            optimizer.zero_grad()
            pe.backward()
            if args.clip_grad > 0:
                nn.utils.clip_grad_norm_(trainable_params, args.clip_grad)
            optimizer.step()

            # Update PE tracking
            pred_err_ema_short = 0.1 * pe_val + 0.9 * pred_err_ema_short
            pred_err_ema_long = 0.01 * pe_val + 0.99 * pred_err_ema_long
            pred_err_prev = pe_val
            ep_pe.append(pe_val)

            # Log trajectory
            if args.save_traj:
                g_np = out["g"].detach().cpu().numpy()[0]
                row = {
                    "episode": episode,
                    "t": int(info["t"]),
                    "x": int(info["x"]),
                    "y": int(info["y"]),
                    "zone_id": int(info["zone_id"]),
                    "action": a_int,
                    "pred_err": pe_val,
                    "alpha": float(out["alpha"].item()),
                    "g_norm": float(np.linalg.norm(g_np)),
                    "perturbation_active": int(info.get("perturbation_active", 0)),
                    "perturbation_trace": float(info.get("perturbation_trace", 0.0)),
                }
                for gi in range(len(g_np)):
                    row[f"g_{gi}"] = float(g_np[gi])
                traj_rows.append(row)

            obs = obs_next
            info = info_next
            last_action = a_int
            done = bool(terminated or truncated)

        # Episode summary
        if (episode + 1) % args.print_every == 0:
            elapsed = time.time() - t0
            mean_pe = float(np.mean(ep_pe))
            g_norm = float(out["g"].detach().cpu().norm().item())
            alpha_val = float(out["alpha"].item())
            print(
                f"[ep {episode+1:4d}/{args.episodes}] "
                f"PE={mean_pe:.4f} ||g||={g_norm:.3f} α={alpha_val:.3f} "
                f"({elapsed:.0f}s)"
            )

        # Carry g between episodes (don't reset)
        # agent.reset() is NOT called

    # ── Save ──
    # Trajectory
    if args.save_traj and traj_rows:
        traj_df = pd.DataFrame(traj_rows)
        traj_path = outdir / "traj.parquet"
        traj_df.to_parquet(traj_path, index=False)
        print(f"[save] Trajectory: {traj_path} ({len(traj_df)} rows)")

    # Checkpoint
    ckpt_path = outdir / "ckpt_final.pt"
    torch.save({
        "agent_state": agent.state_dict(),
        "decoder_state": decoder.state_dict(),
        "meta": {
            "phase1_ckpt": str(args.phase1_ckpt),
            "agent_cfg": {
                "encoder": agent.cfg.encoder.__dict__,
                "world": agent.cfg.world.__dict__,
                "state": agent.cfg.state.__dict__,
                "policy": agent.cfg.policy.__dict__,
            },
            "decoder_cfg": decoder.cfg.__dict__,
            "env_cfg": asdict(env_cfg),
            "args": vars(args),
        },
    }, ckpt_path)
    print(f"[save] Checkpoint: {ckpt_path}")


# ── CLI ──

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

    # World latent
    ap.add_argument("--update_mode", type=str, default="adaptive")
    ap.add_argument("--alpha_fixed", type=float, default=0.10)
    ap.add_argument("--alpha_min", type=float, default=0.03)
    ap.add_argument("--alpha_max", type=float, default=0.30)

    # Env
    ap.add_argument("--sigma_left", type=float, default=0.20)
    ap.add_argument("--sigma_right", type=float, default=0.10)
    ap.add_argument("--n_perturbations", type=int, default=4)
    ap.add_argument("--perturbation_duration", type=int, default=15)
    ap.add_argument("--perturbation_scale", type=float, default=0.12)

    return ap.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()