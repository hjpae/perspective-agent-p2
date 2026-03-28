# cear_pilot/training/train_phase2.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

_THIS_FILE = Path(__file__).resolve()
for _p in (_THIS_FILE.parents[2], _THIS_FILE.parents[1], _THIS_FILE.parent):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from cear_pilot.envs.nzone_phase2 import NZonePhase2Config, NZonePhase2Env
from cear_pilot.models.agent import CEARAgent, AgentConfig
from cear_pilot.models.decoder import ObsDecoder, DecoderConfig


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def timestamp_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def onehot(idx: int, n: int, device: torch.device) -> torch.Tensor:
    return F.one_hot(torch.tensor([idx], device=device), num_classes=n).float()


def try_save_table(rows: List[Dict[str, Any]], out_path: Path) -> Path:
    df = pd.DataFrame(rows)
    try:
        p = out_path.with_suffix(".parquet")
        df.to_parquet(p, index=False)
        return p
    except Exception:
        p = out_path.with_suffix(".csv")
        df.to_csv(p, index=False)
        return p


def freeze_module(module: torch.nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_module(module: torch.nn.Module) -> None:
    module.train()
    for p in module.parameters():
        p.requires_grad = True


def grad_norm(params: Iterable[torch.nn.Parameter]) -> float:
    acc = 0.0
    for p in params:
        if p.grad is not None:
            g = p.grad.detach()
            acc += float(torch.sum(g * g).item())
    return float(np.sqrt(max(acc, 0.0)))


def resolve_env_from_phase1_meta(meta: Dict[str, Any], args: argparse.Namespace) -> NZonePhase2Config:
    env_cfg = NZonePhase2Config()
    env_cfg.width = int(args.width)
    env_cfg.height = int(args.height)
    env_cfg.obs_dim = int(args.obs_dim)
    env_cfg.max_steps = int(args.max_steps)
    env_cfg.schedule_pattern = str(args.schedule_pattern)
    env_cfg.valence_sequence = str(args.valence_sequence)
    env_cfg.sigma_left = float(args.sigma_left)
    env_cfg.sigma_right = float(args.sigma_right)
    env_cfg.c_decay = float(args.c_decay)
    env_cfg.supportive_impulse = float(args.supportive_impulse)
    env_cfg.misleading_impulse = float(args.misleading_impulse)
    env_cfg.distortion_scale = float(args.distortion_scale)
    env_cfg.event_delay_steps = int(args.event_delay_steps)
    return env_cfg


def load_phase1_checkpoint(args: argparse.Namespace):
    ckpt = torch.load(args.phase1_ckpt, map_location=args.device)
    meta = ckpt["meta"]

    agent_cfg = AgentConfig(device=args.device)
    agent_cfg.encoder.__dict__.update(meta["agent_cfg"]["encoder"])
    agent_cfg.world.__dict__.update(meta["agent_cfg"]["world"])
    agent_cfg.state.__dict__.update(meta["agent_cfg"]["state"])
    agent_cfg.policy.__dict__.update(meta["agent_cfg"]["policy"])

    agent_cfg.encoder.obs_dim = int(args.obs_dim)
    agent_cfg.world.update_mode = str(args.update_mode)
    agent_cfg.world.alpha_fixed = float(args.alpha_fixed)
    agent_cfg.world.alpha_min = float(args.alpha_min)
    agent_cfg.world.alpha_max = float(args.alpha_max)
    agent_cfg.world.energy_mode = str(args.energy_mode)
    agent_cfg.world.dyn_eta = float(args.dyn_eta)
    agent_cfg.world.confine_lambda = float(args.confine_lambda)
    agent_cfg.world.n_prototypes = int(args.n_prototypes)
    agent_cfg.world.well_depth = float(args.well_depth)
    agent_cfg.world.well_width = float(args.well_width)
    agent_cfg.world.learnable_prototypes = True
    agent_cfg.world.learnable_well_depth = True
    agent_cfg.world.learnable_well_width = bool(args.learnable_well_width)
    agent_cfg.world.temperature = float(args.temperature)

    agent = CEARAgent(agent_cfg)
    agent.load_state_dict(ckpt["agent_state"], strict=False)

    dec_cfg = DecoderConfig(**meta["decoder_cfg"])
    dec_cfg.obs_dim = int(args.obs_dim)
    decoder = ObsDecoder(dec_cfg)
    decoder.load_state_dict(ckpt["decoder_state"], strict=False)
    return agent, decoder, meta


def action_name(a: int) -> str:
    return ["UP", "DOWN", "LEFT", "RIGHT", "STAY"][int(a)]


def pairwise_separation_loss(centers: torch.Tensor) -> torch.Tensor:
    if centers.shape[0] <= 1:
        return centers.new_tensor(0.0)
    dist = torch.cdist(centers, centers, p=2)
    eye = torch.eye(dist.shape[0], device=dist.device, dtype=dist.dtype)
    masked = dist + eye * 1e9
    min_dist = masked.min(dim=1).values
    return torch.exp(-min_dist).mean()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1_ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--clip_grad", type=float, default=1.0)
    ap.add_argument("--print_every", type=int, default=10)
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--carry_g_between_episodes", action="store_true")
    ap.add_argument("--save_traj", action="store_true")

    ap.add_argument("--freeze_encoder", action="store_true")
    ap.add_argument("--freeze_state", action="store_true")
    ap.add_argument("--freeze_policy", action="store_true")
    ap.add_argument("--freeze_decoder", action="store_true")

    ap.add_argument("--width", type=int, default=23)
    ap.add_argument("--height", type=int, default=7)
    ap.add_argument("--obs_dim", type=int, default=8)
    ap.add_argument("--schedule_pattern", type=str, default="1-1-1-1")
    ap.add_argument("--valence_sequence", type=str, default="SSSS")
    ap.add_argument("--sigma_left", type=float, default=0.20)
    ap.add_argument("--sigma_right", type=float, default=0.10)
    ap.add_argument("--c_decay", type=float, default=0.91)
    ap.add_argument("--supportive_impulse", type=float, default=0.45)
    ap.add_argument("--misleading_impulse", type=float, default=-0.45)
    ap.add_argument("--distortion_scale", type=float, default=0.30)
    ap.add_argument("--event_delay_steps", type=int, default=3)

    ap.add_argument("--update_mode", type=str, default="fixed", choices=["fixed", "adaptive"])
    ap.add_argument("--alpha_fixed", type=float, default=0.20)
    ap.add_argument("--alpha_min", type=float, default=0.05)
    ap.add_argument("--alpha_max", type=float, default=0.50)

    ap.add_argument("--energy_mode", type=str, default="prototype_wells", choices=["prototype_wells", "none"])
    ap.add_argument("--dyn_eta", type=float, default=0.10)
    ap.add_argument("--confine_lambda", type=float, default=0.01)
    ap.add_argument("--n_prototypes", type=int, default=2)
    ap.add_argument("--well_depth", type=float, default=0.60)
    ap.add_argument("--well_width", type=float, default=1.25)
    ap.add_argument("--learnable_well_width", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)

    ap.add_argument("--w_pred", type=float, default=1.0)
    ap.add_argument("--w_energy", type=float, default=0.02)
    ap.add_argument("--w_basin", type=float, default=0.02)
    ap.add_argument("--w_sep", type=float, default=0.05)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)

    agent, decoder, meta = load_phase1_checkpoint(args)
    agent.to(device)
    decoder.to(device)

    if args.freeze_encoder:
        freeze_module(agent.enc)
    if args.freeze_state:
        freeze_module(agent.state)
    if args.freeze_policy:
        freeze_module(agent.policy)
    if args.freeze_decoder:
        freeze_module(decoder)

    unfreeze_module(agent.world)
    if not args.freeze_decoder:
        unfreeze_module(decoder)

    params = [p for p in list(agent.parameters()) + list(decoder.parameters()) if p.requires_grad]
    optim = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    env_cfg = resolve_env_from_phase1_meta(meta, args)
    env = NZonePhase2Env(config=env_cfg)

    if args.outdir:
        base_dir = Path(args.outdir)
    else:
        base_dir = Path("outputs") / "runs" / "p2_"
    
    run_dir = base_dir.parent / f"{base_dir.name}_{timestamp_id()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    meta_out = {
        "mode": "train_phase2_landscape",
        "phase1_ckpt": str(Path(args.phase1_ckpt).resolve()),
        "args": vars(args),
        "env_cfg": asdict(env_cfg),
        "base_meta": meta,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta_out, indent=2))

    traj_rows: List[Dict[str, Any]] = []
    ep_rows: List[Dict[str, Any]] = []

    agent.reset(batch_size=1)
    n_actions = int(env.action_space.n)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        if not args.carry_g_between_episodes:
            agent.reset(batch_size=1)

        last_action = 4
        done = False
        t = 0
        loss_sum = 0.0
        pred_sum = 0.0
        energy_sum = 0.0
        alpha_sum = 0.0
        grad_sum = 0.0
        switches = 0
        last_basin = None
        g_start = agent.get_latents()["g"].detach().clone().cpu().numpy()[0]

        while not done:
            x_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            p_t = onehot(last_action, n_actions, device)

            out = agent.forward_step(x_t, p_t, err_t=None)
            logits = out["logits"]
            action = agent.policy.sample_action(logits, greedy=args.greedy)
            a_int = int(action.item())
            a_oh = F.one_hot(action.long(), num_classes=n_actions).float()

            obs_next, _, terminated, truncated, info2 = env.step(a_int)
            x_next = torch.tensor(obs_next, dtype=torch.float32, device=device).unsqueeze(0)

            pred = decoder(out["g"], a_oh)
            loss_pred = F.mse_loss(pred, x_next)
            loss_energy = out["energy"].mean()
            basin_conf = out["basin_probs"].max(dim=-1).values
            loss_basin = (1.0 - basin_conf).mean()
            loss_sep = pairwise_separation_loss(agent.world.prototype_centers)
            loss = (
                float(args.w_pred) * loss_pred
                + float(args.w_energy) * loss_energy
                + float(args.w_basin) * loss_basin
                + float(args.w_sep) * loss_sep
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=float(args.clip_grad))
            gn = grad_norm(params)
            optim.step()

            with torch.no_grad():
                pi = torch.softmax(logits, dim=-1)
                entropy = float((-(pi * torch.log(pi + 1e-9)).sum(dim=-1)).mean().item())
                basin_id = int(out["basin_id"].detach().cpu().item())
                if last_basin is not None and basin_id != last_basin:
                    switches += 1
                last_basin = basin_id

                if args.save_traj:
                    g_np = out["g"].detach().cpu().numpy()[0]
                    row = {
                        "episode": ep,
                        "t": t,
                        "x": int(info2.get("x", -1)),
                        "y": int(info2.get("y", -1)),
                        "action": a_int,
                        "action_name": action_name(a_int),
                        "entropy": entropy,
                        "alpha": float(out["alpha"].mean().item()),
                        "energy": float(out["energy"].mean().item()),
                        "grad_norm": float(out["grad_norm"].mean().item()),
                        "basin_id": basin_id,
                        "basin_conf": float(out["basin_probs"].max(dim=-1).values.mean().item()),
                        "consequence_active": int(abs(float(info2.get("c_state", 0.0))) > 1e-6),
                        "c_state": float(info2.get("c_state", 0.0)),
                        "on_encounter": int(info2.get("on_encounter", 0)),
                        "encounter_outcome": int(info2.get("encounter_outcome", 1)),
                        "loss": float(loss.item()),
                        "loss_pred": float(loss_pred.item()),
                    }
                    for i, v in enumerate(g_np):
                        row[f"g_{i}"] = float(v)
                    traj_rows.append(row)

            loss_sum += float(loss.item())
            pred_sum += float(loss_pred.item())
            energy_sum += float(out["energy"].mean().item())
            alpha_sum += float(out["alpha"].mean().item())
            grad_sum += float(out["grad_norm"].mean().item())

            obs = obs_next
            info = info2
            last_action = a_int
            t += 1
            done = bool(terminated or truncated)

        g_end = agent.get_latents()["g"].detach().cpu().numpy()[0]
        ep_rows.append({
            "episode": ep,
            "mean_loss": loss_sum / max(t, 1),
            "mean_pred_loss": pred_sum / max(t, 1),
            "mean_energy": energy_sum / max(t, 1),
            "mean_alpha": alpha_sum / max(t, 1),
            "mean_grad_norm": grad_sum / max(t, 1),
            "switches": switches,
            "g_start_norm": float(np.linalg.norm(g_start)),
            "g_end_norm": float(np.linalg.norm(g_end)),
            "g_shift": float(np.linalg.norm(g_end - g_start)),
            "final_x": int(info.get("x", -1)),
            "final_c": float(info.get("c_state", 0.0)),
        })

        if (ep + 1) % max(args.print_every, 1) == 0:
            print(
                f"[ep {ep + 1:04d}] "
                f"pred={ep_rows[-1]['mean_pred_loss']:.5f} "
                f"energy={ep_rows[-1]['mean_energy']:.5f} "
                f"alpha={ep_rows[-1]['mean_alpha']:.4f} "
                f"switches={switches}"
            )

    traj_path = try_save_table(traj_rows, run_dir / "traj") if args.save_traj else None
    ep_path = try_save_table(ep_rows, run_dir / "episode_summary")

    ckpt_out = {
        "agent_state": agent.state_dict(),
        "decoder_state": decoder.state_dict(),
        "meta": meta_out,
    }
    torch.save(ckpt_out, run_dir / "ckpt_final.pt")
    print(f"[train_phase2] episode summary: {ep_path}")
    if traj_path is not None:
        print(f"[train_phase2] trajectory: {traj_path}")
    print(f"[train_phase2] checkpoint: {run_dir / 'ckpt_final.pt'}")


if __name__ == "__main__":
    main()