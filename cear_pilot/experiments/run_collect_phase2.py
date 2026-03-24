# cear_pilot/experiments/run_collect_phase2.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch

from cear_pilot.envs.nzone_phase2 import NZonePhase2Config, NZonePhase2Env
from cear_pilot.models.agent import CEARAgent, AgentConfig
from cear_pilot.models.decoder import ObsDecoder, DecoderConfig


def timestamp_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def try_save_table(rows, out_path: Path) -> Path:
    import pandas as pd
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


def build_agent_from_meta(meta: Dict[str, Any], device: str):
    env_cfg = NZonePhase2Config(**meta["env_cfg"])
    env = NZonePhase2Env(config=env_cfg)

    agent_cfg = AgentConfig(device=device)
    agent_cfg.encoder.__dict__.update(meta["agent_cfg"]["encoder"])
    agent_cfg.world.__dict__.update(meta["agent_cfg"]["world"])
    agent_cfg.state.__dict__.update(meta["agent_cfg"]["state"])
    agent_cfg.policy.__dict__.update(meta["agent_cfg"]["policy"])

    agent = CEARAgent(agent_cfg)
    dec_cfg = DecoderConfig(**meta["decoder_cfg"])
    decoder = ObsDecoder(dec_cfg)
    return agent, decoder, env


def decision_code(action: int) -> int:
    # 0=avoid(left), 1=wait(stay), 2=engage(right), 3=sample(up/down)
    if int(action) == 2:
        return 0
    if int(action) == 4:
        return 1
    if int(action) == 3:
        return 2
    return 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--ablate_g", action="store_true")

    # optional g intervention
    ap.add_argument("--do_g_at", type=int, default=-1)
    ap.add_argument("--do_g_mode", type=str, default="shock", choices=["shock", "swap", "zero"])
    ap.add_argument("--do_g_scale", type=float, default=1.0)

    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device)
    meta = ckpt["meta"]

    agent, decoder, env = build_agent_from_meta(meta, device=args.device)
    agent.load_state_dict(ckpt["agent_state"])
    decoder.load_state_dict(ckpt["decoder_state"])
    agent.to(args.device).eval()
    decoder.to(args.device).eval()

    run_dir = Path(args.outdir) if args.outdir else (Path("outputs") / "runs" / timestamp_id())
    ensure_dir(run_dir)
    ensure_dir(run_dir / "figs")

    run_meta = {
        "mode": "collect_phase2_maturity",
        "ckpt": str(Path(args.ckpt).resolve()),
        "episodes": int(args.episodes),
        "seed": int(args.seed),
        "device": str(args.device),
        "greedy": bool(args.greedy),
        "ablate_g": bool(args.ablate_g),
        "do_g_at": int(args.do_g_at),
        "do_g_mode": str(args.do_g_mode),
        "do_g_scale": float(args.do_g_scale),
        "train_meta": meta,
    }
    (run_dir / "meta.json").write_text(json.dumps(run_meta, indent=2))

    rng = np.random.default_rng(args.seed)
    n_actions = int(env.action_space.n)

    rows: List[Dict[str, Any]] = []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        agent.reset(batch_size=1)
        last_action = 4

        done = False
        t = 0

        while not done:
            if args.do_g_at >= 0 and t == args.do_g_at:
                agent.apply_perturbation(kind=args.do_g_mode, scale=args.do_g_scale)
                did_do_g = 1
            else:
                did_do_g = 0

            x_t = torch.tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
            p_t = torch.tensor(onehot(last_action, n_actions), dtype=torch.float32, device=args.device).unsqueeze(0)

            with torch.no_grad():
                action, out = agent.step(x_t, p_t, greedy=args.greedy, ablate_g=args.ablate_g)

            a_int = int(action.item())
            obs_next, _, terminated, truncated, info2 = env.step(a_int)

            with torch.no_grad():
                logits = out["logits"]
                pi = torch.softmax(logits, dim=-1)
                entropy = float((-(pi * torch.log(pi + 1e-9)).sum(dim=-1)).mean().item())

            g = out["g"].squeeze(0).detach().cpu().numpy()
            s = out["s"].squeeze(0).detach().cpu().numpy()
            z = out["z"].squeeze(0).detach().cpu().numpy()

            row: Dict[str, Any] = {
                "episode": int(ep),
                "t": int(info2.get("t", t)),
                "x": int(info2.get("x", -1)),
                "y": int(info2.get("y", -1)),
                "zone_id": int(info2.get("zone_id", -1)),

                "action_env": int(a_int),
                "decision_code": int(decision_code(a_int)),
                "avoid": int(decision_code(a_int) == 0),
                "wait": int(decision_code(a_int) == 1),
                "engage": int(decision_code(a_int) == 2),
                "sample": int(decision_code(a_int) == 3),

                "pi_entropy": float(entropy),
                "did_do_g": int(did_do_g),

                "on_encounter": int(bool(info2.get("on_encounter", False))),
                "encounter_event": int(bool(info2.get("encounter_event", False))),
                "encounter_type": int(info2.get("encounter_type", 1)),
                "recent_reliability": float(info2.get("recent_reliability", 0.0)),
                "reliability_mode": int(info2.get("reliability_mode", 1)),
                "fragility": float(info2.get("fragility", np.nan)),
                "rupture_memory": float(info2.get("rupture_memory", np.nan)),
                "conflict_load": float(info2.get("conflict_load", np.nan)),
                "rupture": int(bool(info2.get("rupture", False))),
                "pending_ruptures": int(info2.get("pending_ruptures", 0)),
                "slip": int(bool(info2.get("slip", False))),
                "drift": int(bool(info2.get("drift", False))),
            }

            for i, v in enumerate(obs.astype(np.float32)):
                row[f"obs_{i}"] = float(v)
            for i, v in enumerate(z):
                row[f"z_{i}"] = float(v)
            for i, v in enumerate(s):
                row[f"s_{i}"] = float(v)
            for i, v in enumerate(g):
                row[f"g_{i}"] = float(v)

            rows.append(row)

            obs = obs_next
            last_action = a_int
            done = bool(terminated or truncated)
            t += 1

    out_path = try_save_table(rows, run_dir / "traj")
    print(f"Saved trajectories to: {out_path}")
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()