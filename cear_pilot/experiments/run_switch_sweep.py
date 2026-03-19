# cear_pilot/experiments/run_switch_sweep.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from cear_pilot.envs.nzone_grid import NZoneConfig, NZoneGridEnv
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


def build_agent_from_meta(meta: Dict[str, Any], device: str, max_steps_override=None):
    env_cfg = NZoneConfig(**meta["env_cfg"])
    if max_steps_override is not None:
        env_cfg.max_steps = int(max_steps_override)
    env = NZoneGridEnv(config=env_cfg)

    agent_cfg = AgentConfig(device=device)
    agent_cfg.encoder.__dict__.update(meta["agent_cfg"]["encoder"])
    agent_cfg.world.__dict__.update(meta["agent_cfg"]["world"])
    agent_cfg.state.__dict__.update(meta["agent_cfg"]["state"])
    agent_cfg.policy.__dict__.update(meta["agent_cfg"]["policy"])

    agent = CEARAgent(agent_cfg)
    dec_cfg = DecoderConfig(**meta["decoder_cfg"])
    decoder = ObsDecoder(dec_cfg)
    return agent, decoder, env


def set_env_zone_sigma(env: NZoneGridEnv, sigma_triplet: Tuple[float, float, float]) -> None:
    env.set_zone_sigma(tuple(float(x) for x in sigma_triplet))


def policy_stats_from_logits(logits: np.ndarray):
    ex = np.exp(logits - np.max(logits))
    p = ex / (np.sum(ex) + 1e-12)
    p_sorted = np.sort(p)[::-1]
    pi_max = float(p_sorted[0])
    margin = float(p_sorted[0] - (p_sorted[1] if len(p_sorted) > 1 else 0.0))
    ent = float(-np.sum(p * np.log(p + 1e-12)))
    return pi_max, ent, margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--greedy", action="store_true")

    ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--period", type=int, default=20)

    ap.add_argument("--sigma_A", type=float, nargs=3, default=(0.60, 0.30, 0.05))
    ap.add_argument("--sigma_B", type=float, nargs=3, default=(0.05, 0.30, 0.60))

    ap.add_argument("--outdir", type=str, default="")
    args = ap.parse_args()

    device = args.device
    rng = np.random.default_rng(args.seed)

    ckpt = torch.load(args.ckpt, map_location=device)
    meta = ckpt["meta"]

    agent, decoder, env = build_agent_from_meta(meta, device=device, max_steps_override=args.T)
    agent.load_state_dict(ckpt["agent_state"])
    decoder.load_state_dict(ckpt["decoder_state"])
    agent.to(device).eval()
    decoder.to(device).eval()

    run_dir = Path(args.outdir) if args.outdir else (Path("outputs") / "runs" / timestamp_id())
    ensure_dir(run_dir)
    ensure_dir(run_dir / "figs")

    run_meta = {
        "mode": "switch_sweep_phase2",
        "ckpt": str(Path(args.ckpt).resolve()),
        "seed": args.seed,
        "device": args.device,
        "greedy": bool(args.greedy),
        "T": int(args.T),
        "warmup": int(args.warmup),
        "period": int(args.period),
        "sigma_A": list(map(float, args.sigma_A)),
        "sigma_B": list(map(float, args.sigma_B)),
        "train_meta": meta,
    }
    (run_dir / "meta.json").write_text(json.dumps(run_meta, indent=2))

    obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
    agent.reset(batch_size=1)
    last_action = 4
    n_actions = int(env.action_space.n)

    rows: List[Dict[str, Any]] = []

    regime = 0
    set_env_zone_sigma(env, tuple(args.sigma_A))

    for t_global in range(int(args.T)):
        switched = 0
        if t_global >= int(args.warmup):
            k = (t_global - int(args.warmup)) // max(1, int(args.period))
            new_regime = int(k % 2)
            if new_regime != regime:
                regime = new_regime
                switched = 1
                set_env_zone_sigma(env, tuple(args.sigma_A) if regime == 0 else tuple(args.sigma_B))

        x_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        p_t = torch.tensor(onehot(last_action, n_actions), dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action, out = agent.step(x_t, p_t, greedy=args.greedy, ablate_g=False)

        a_int = int(action.item())
        obs_next, _, terminated, truncated, info2 = env.step(a_int)

        g = out["g"].squeeze(0).detach().cpu().numpy()
        logits = out["logits"].squeeze(0).detach().cpu().numpy()
        pi_max, ent, margin = policy_stats_from_logits(logits)

        row = {
            "t": int(info2["t"]),
            "t_global": int(t_global),
            "regime": int(regime),
            "switch": int(switched),
            "a": int(a_int),
            "pi_max": float(pi_max),
            "entropy": float(ent),
            "margin": float(margin),
            "zone_id": int(info2.get("zone_id", -1)),
            "x": int(info2.get("x", -1)),
            "y": int(info2.get("y", -1)),

            # Phase 2 env diagnostics
            "fragility": float(info2.get("fragility", np.nan)),
            "rupture_memory": float(info2.get("rupture_memory", np.nan)),
            "on_encounter": int(bool(info2.get("on_encounter", False))),
            "encounter_event": int(bool(info2.get("encounter_event", False))),
            "rupture": int(bool(info2.get("rupture", False))),
            "pending_ruptures": int(info2.get("pending_ruptures", 0)),
        }
        for i, v in enumerate(g.tolist()):
            row[f"g_{i}"] = float(v)

        rows.append(row)

        obs = obs_next
        last_action = a_int

        if terminated or truncated:
            break

    out_path = try_save_table(rows, run_dir / "traj")
    print(f"[OK] Saved traj: {out_path}")
    print(f"[OK] Run dir: {run_dir}")


if __name__ == "__main__":
    main()