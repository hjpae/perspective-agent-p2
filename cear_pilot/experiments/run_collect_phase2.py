# cear_pilot/experiments/run_collect_phase2.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from cear_pilot.envs.nzone_phase2 import NZonePhase2Config, NZonePhase2Env
from cear_pilot.models.agent import CEARAgent, AgentConfig
from cear_pilot.models.decoder import ObsDecoder, DecoderConfig


def timestamp_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


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


def onehot(idx: int, n: int) -> np.ndarray:
    v = np.zeros((n,), dtype=np.float32)
    v[idx] = 1.0
    return v


def action_name(a: int) -> str:
    return ["UP", "DOWN", "LEFT", "RIGHT", "STAY"][int(a)]


def decision_code(action: int) -> int:
    if int(action) == 2:
        return 0  # avoid / left
    if int(action) == 4:
        return 1  # wait
    if int(action) == 3:
        return 2  # engage / right
    return 3      # vertical shift / sample


def build_agent_and_decoder_from_meta(meta: Dict[str, Any], device: str):
    agent_cfg = AgentConfig(device=device)
    agent_cfg.encoder.__dict__.update(meta["agent_cfg"]["encoder"])
    agent_cfg.world.__dict__.update(meta["agent_cfg"]["world"])
    agent_cfg.state.__dict__.update(meta["agent_cfg"]["state"])
    agent_cfg.policy.__dict__.update(meta["agent_cfg"]["policy"])

    agent = CEARAgent(agent_cfg)

    dec_cfg = DecoderConfig(**meta["decoder_cfg"])
    decoder = ObsDecoder(dec_cfg)
    return agent, decoder, agent_cfg, dec_cfg


def build_phase2_err_scalar(info: Dict[str, Any], device: str) -> torch.Tensor:
    val = float(info.get("fragility", 0.0)) + float(info.get("conflict_load", 0.0))
    return torch.tensor([[val]], dtype=torch.float32, device=device)


def collect_obs_prediction_error(
    decoder: ObsDecoder,
    g_t: torch.Tensor,
    x_next: np.ndarray,
    device: str,
) -> Tuple[float, float, float]:
    with torch.no_grad():
        x_next_t = torch.tensor(x_next, dtype=torch.float32, device=device).unsqueeze(0)
        xhat_all = decoder.predict_all_actions(g_t)
        per_a_err = torch.mean((xhat_all - x_next_t.unsqueeze(1)) ** 2, dim=-1).squeeze(0)
        e_np = per_a_err.detach().float().cpu().numpy()
    return float(e_np.min()), float(e_np.max()), float(e_np.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--ablate_g", action="store_true")

    ap.add_argument("--do_g_at", type=int, default=-1)
    ap.add_argument("--do_g_mode", type=str, default="shock", choices=["shock", "swap", "zero"])
    ap.add_argument("--do_g_scale", type=float, default=1.0)

    ap.add_argument("--carry_g_between_episodes", action="store_true")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device)
    meta = ckpt["meta"]

    env_cfg = NZonePhase2Config(**meta["env_cfg"])
    env = NZonePhase2Env(config=env_cfg)

    agent, decoder, agent_cfg, dec_cfg = build_agent_and_decoder_from_meta(meta, device=args.device)
    agent.load_state_dict(ckpt["agent_state"])
    decoder.load_state_dict(ckpt["decoder_state"])
    agent.to(args.device).eval()
    decoder.to(args.device).eval()

    run_dir = Path(args.outdir) if args.outdir else (Path("outputs") / "runs" / timestamp_id())
    ensure_dir(run_dir)

    run_meta = {
        "mode": "collect_phase2",
        "ckpt": str(Path(args.ckpt).resolve()),
        "episodes": int(args.episodes),
        "seed": int(args.seed),
        "device": str(args.device),
        "greedy": bool(args.greedy),
        "ablate_g": bool(args.ablate_g),
        "do_g_at": int(args.do_g_at),
        "do_g_mode": str(args.do_g_mode),
        "do_g_scale": float(args.do_g_scale),
        "carry_g_between_episodes": bool(args.carry_g_between_episodes),
        "env_cfg": meta["env_cfg"],
        "agent_cfg": meta["agent_cfg"],
        "decoder_cfg": meta["decoder_cfg"],
        "train_meta": meta,
    }
    (run_dir / "meta.json").write_text(json.dumps(run_meta, indent=2))

    n_actions = int(env.action_space.n)
    rows: List[Dict[str, Any]] = []

    agent.reset(batch_size=1)

    for ep in range(args.episodes):
        ep_seed = int(args.seed + ep)
        obs, info = env.reset(seed=ep_seed)

        if not args.carry_g_between_episodes:
            agent.reset(batch_size=1)

        last_action = 4
        done = False
        t = 0
        g_prev = None

        while not done:
            if args.do_g_at >= 0 and t == args.do_g_at:
                agent.apply_perturbation(kind=args.do_g_mode, scale=args.do_g_scale)
                did_do_g = 1
            else:
                did_do_g = 0

            x_t = torch.tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
            p_t = torch.tensor(onehot(last_action, n_actions), dtype=torch.float32, device=args.device).unsqueeze(0)
            err_scalar = build_phase2_err_scalar(info, device=args.device)

            with torch.no_grad():
                action, out = agent.step(
                    x_t,
                    p_t,
                    greedy=args.greedy,
                    ablate_g=args.ablate_g,
                    err_t=err_scalar,
                )

            a_int = int(action.item())
            obs_next, _, terminated, truncated, info2 = env.step(a_int)

            with torch.no_grad():
                logits = out["logits"]
                pi = torch.softmax(logits, dim=-1)
                entropy = float((-(pi * torch.log(pi + 1e-9)).sum(dim=-1)).mean().item())
                action_prob_max = float(pi.max(dim=-1).values.mean().item())
                policy_mode = int(torch.argmax(pi, dim=-1).item())
                alpha = float(out["alpha"].mean().detach().cpu().item())

            g = out["g"].squeeze(0).detach().cpu().numpy()
            s = out["s"].squeeze(0).detach().cpu().numpy()
            z = out["z"].squeeze(0).detach().cpu().numpy()

            if g_prev is None:
                delta_g = 0.0
            else:
                delta_g = float(np.linalg.norm(g - g_prev))
            g_prev = g.copy()

            e_min, e_max, e_std = collect_obs_prediction_error(
                decoder=decoder,
                g_t=out["g"],
                x_next=obs_next,
                device=args.device,
            )

            row: Dict[str, Any] = {
                "episode": int(ep),
                "episode_seed": int(ep_seed),
                "phase": "phase2",
                "t": int(info2.get("t", t)),
                "x": int(info2.get("x", -1)),
                "y": int(info2.get("y", -1)),
                "zone_id": int(info2.get("zone_id", -1)),

                "on_encounter": int(info2.get("on_encounter", 0)),
                "encounter_event": int(info2.get("encounter_event", 0)),
                "encounter_idx": int(info2.get("encounter_idx", -1)),
                "encounter_profile": str(info2.get("encounter_profile", "none")),
                "encounter_outcome": int(info2.get("encounter_outcome", 1)),

                "row_band": str(info2.get("row_band", "balanced")),
                "row_exposure_mult": float(info2.get("row_exposure_mult", 1.0)),
                "current_sigma": float(info2.get("current_sigma", np.nan)),

                "reliability_estimate": float(info2.get("reliability_estimate", 0.0)),
                "recent_reliability": float(info2.get("reliability_estimate", 0.0)),
                "fragility": float(info2.get("fragility", 0.0)),
                "rupture_memory": float(info2.get("rupture_memory", 0.0)),
                "conflict_load": float(info2.get("conflict_load", 0.0)),

                "rupture": int(info2.get("rupture", 0)),
                "pending_ruptures": int(info2.get("pending_ruptures", 0)),
                "blackout_timer": int(info2.get("blackout_timer", 0)),
                "rupture_obs_timer": int(info2.get("rupture_obs_timer", 0)),
                "rupture_action_timer": int(info2.get("rupture_action_timer", 0)),
                "supportive_timer": int(info2.get("supportive_timer", 0)),
                "misleading_timer": int(info2.get("misleading_timer", 0)),

                "action": int(a_int),
                "action_name": str(action_name(a_int)),
                "decision_code": int(decision_code(a_int)),
                "policy_mode": int(policy_mode),
                "action_prob_max": float(action_prob_max),
                "entropy": float(entropy),
                "alpha": float(alpha),
                "delta_g": float(delta_g),
                "did_do_g": int(did_do_g),
                "pred_err_min": float(e_min),
                "pred_err_max": float(e_max),
                "pred_err_std": float(e_std),
            }

            row["engage"] = 1.0 if a_int == 3 else 0.0
            row["wait"] = 1.0 if a_int == 4 else 0.0
            row["avoid"] = 1.0 if a_int == 2 else 0.0
            row["sample"] = 1.0 if a_int in (0, 1) else 0.0

            for i, v in enumerate(g):
                row[f"g_{i}"] = float(v)
            for i, v in enumerate(s):
                row[f"s_{i}"] = float(v)
            for i, v in enumerate(z):
                row[f"z_{i}"] = float(v)

            rows.append(row)

            obs = obs_next
            info = info2
            last_action = a_int
            t += 1
            done = bool(terminated or truncated)

    saved = try_save_table(rows, run_dir / "traj")
    print(f"[collect_phase2] saved: {saved}")


if __name__ == "__main__":
    main()