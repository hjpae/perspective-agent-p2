# cear_pilot/experiments/collect_reversal.py
# -*- coding: utf-8 -*-
"""
Reversal experiment: continuous g trajectory across valence switch.

Phase A (warmup):  N episodes in NATIVE valence with carry_g=True
                   → g settles into formation basin
Phase B (reversal): M episodes in OPPOSITE valence, g continues
                   → watch g migrate (or not)

All in one process — g is NEVER reset after initial agent.reset().

Usage:
    python -m cear_pilot.experiments.collect_reversal \
        --ckpt outputs/.../ckpt_final.pt \
        --native_valence SSSS \
        --warmup_episodes 5 \
        --reversal_episodes 10 \
        --outdir outputs/reversal/...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


ERR_FEATURE_NAMES = [
    "pred_err_now_ratio",
    "pred_err_ema_short_ratio",
    "pred_err_ema_long_log",
    "pred_err_rise_ratio",
    "recent_event_trace",
    "recent_consequence_trace",
    "current_event_flag",
    "c_state_signed",
    "supportive_flag",
    "misleading_flag",
]


def timestamp_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


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


def load_agent_decoder(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt["meta"]

    base = meta.get("base_meta", meta)
    agent_cfg = AgentConfig(device=device)
    agent_cfg.encoder.__dict__.update(base["agent_cfg"]["encoder"])
    agent_cfg.world.__dict__.update(base["agent_cfg"]["world"])
    agent_cfg.state.__dict__.update(base["agent_cfg"]["state"])
    agent_cfg.policy.__dict__.update(base["agent_cfg"]["policy"])

    if "args" in meta:
        a = meta["args"]
        agent_cfg.encoder.obs_dim = int(a.get("obs_dim", agent_cfg.encoder.obs_dim))
        agent_cfg.world.update_mode = str(a.get("update_mode", agent_cfg.world.update_mode))
        agent_cfg.world.alpha_fixed = float(a.get("alpha_fixed", agent_cfg.world.alpha_fixed))
        agent_cfg.world.alpha_min = float(a.get("alpha_min", agent_cfg.world.alpha_min))
        agent_cfg.world.alpha_max = float(a.get("alpha_max", agent_cfg.world.alpha_max))
        agent_cfg.world.energy_mode = str(a.get("energy_mode", agent_cfg.world.energy_mode))
        agent_cfg.world.dyn_eta = float(a.get("dyn_eta", agent_cfg.world.dyn_eta))
        agent_cfg.world.confine_lambda = float(a.get("confine_lambda", agent_cfg.world.confine_lambda))
        agent_cfg.world.n_prototypes = int(a.get("n_prototypes", agent_cfg.world.n_prototypes))
        agent_cfg.world.well_depth = float(a.get("well_depth", agent_cfg.world.well_depth))
        agent_cfg.world.well_width = float(a.get("well_width", agent_cfg.world.well_width))
        agent_cfg.world.temperature = float(a.get("temperature", agent_cfg.world.temperature))

    agent_cfg.world.use_error_feedback = True
    agent_cfg.world.err_dim = len(ERR_FEATURE_NAMES)

    agent = CEARAgent(agent_cfg)
    agent.load_state_dict(ckpt["agent_state"], strict=False)

    dec_cfg = DecoderConfig(**base["decoder_cfg"])
    dec_cfg.obs_dim = int(agent_cfg.encoder.obs_dim)
    decoder = ObsDecoder(dec_cfg)
    decoder.load_state_dict(ckpt["decoder_state"], strict=False)
    return agent, decoder, meta


def safe_info_float(info, key, default=0.0):
    try:
        return float(info.get(key, default))
    except Exception:
        return float(default)


def safe_info_int(info, key, default=0):
    try:
        return int(info.get(key, default))
    except Exception:
        return int(default)


def safe_info_str(info, key, default="none"):
    try:
        return str(info.get(key, default))
    except Exception:
        return str(default)


def exp_trace_from_steps(steps, tau):
    if steps is None or steps < 0:
        return 0.0
    return float(np.exp(-float(steps) / max(float(tau), 1e-6)))


def valence_to_flags(valence):
    v = str(valence).strip().lower()
    if v == "supportive":
        return 1.0, 0.0
    if v == "misleading":
        return 0.0, 1.0
    return 0.0, 0.0


def build_err_t(prev_pred_err, err_ema_short, err_ema_long, info, device):
    eps = 1e-3
    long_base = torch.clamp(err_ema_long, min=eps)

    pred_err_now_ratio = torch.log1p(prev_pred_err / long_base)
    pred_err_ema_short_ratio = torch.log1p(err_ema_short / long_base)
    pred_err_ema_long_log = torch.log1p(err_ema_long)
    pred_err_rise_ratio = torch.tanh((err_ema_short - err_ema_long) / (long_base + eps))

    steps_since_event = safe_info_int(info, "steps_since_last_event", -1)
    steps_since_consequence = safe_info_int(info, "steps_since_last_consequence", -1)
    recent_event_trace = exp_trace_from_steps(steps_since_event, tau=4.0)
    recent_consequence_trace = exp_trace_from_steps(steps_since_consequence, tau=6.0)
    current_event_flag = float(max(
        safe_info_int(info, "event_now", 0),
        safe_info_int(info, "on_encounter", 0),
    ))
    c_state_signed = safe_info_float(info, "c_state", 0.0)

    event_val = safe_info_str(info, "event_valence_hidden", "none")
    consequence_val = safe_info_str(info, "consequence_valence_hidden", "none")
    sup_event, mis_event = valence_to_flags(event_val)
    sup_cons, mis_cons = valence_to_flags(consequence_val)
    supportive_flag = max(sup_event, sup_cons)
    misleading_flag = max(mis_event, mis_cons)

    feats = torch.cat([
        pred_err_now_ratio,
        pred_err_ema_short_ratio,
        pred_err_ema_long_log,
        pred_err_rise_ratio,
        torch.tensor([[recent_event_trace]], device=device, dtype=torch.float32),
        torch.tensor([[recent_consequence_trace]], device=device, dtype=torch.float32),
        torch.tensor([[current_event_flag]], device=device, dtype=torch.float32),
        torch.tensor([[c_state_signed]], device=device, dtype=torch.float32),
        torch.tensor([[supportive_flag]], device=device, dtype=torch.float32),
        torch.tensor([[misleading_flag]], device=device, dtype=torch.float32),
    ], dim=-1)
    return feats


def action_name(a):
    return ["UP", "DOWN", "LEFT", "RIGHT", "STAY"][int(a)]


def make_env(valence_sequence, schedule_pattern, max_steps, meta):
    env_cfg_data = meta.get("env_cfg", meta.get("base_meta", {}).get("env_cfg", {}))
    env_cfg = NZonePhase2Config(**env_cfg_data) if env_cfg_data else NZonePhase2Config()
    env_cfg.valence_sequence = valence_sequence
    env_cfg.schedule_pattern = schedule_pattern
    env_cfg.max_steps = max_steps
    return NZonePhase2Env(config=env_cfg), env_cfg


def run_episodes(
    agent, decoder, env, n_episodes, seed_start, device,
    phase_label, greedy=True,
):
    """Run episodes WITHOUT resetting agent (g carries over)."""
    rows = []
    n_actions = int(env.action_space.n)

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed_start + ep)
        # DO NOT reset agent — g carries from previous episode

        last_action = 4
        done = False
        t = 0

        prev_pred_err = torch.zeros((1, 1), device=device, dtype=torch.float32)
        err_ema_short = torch.zeros((1, 1), device=device, dtype=torch.float32)
        err_ema_long = torch.zeros((1, 1), device=device, dtype=torch.float32)

        while not done:
            x_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            p_t = F.one_hot(torch.tensor([last_action], device=device), num_classes=n_actions).float()

            err_t = build_err_t(prev_pred_err, err_ema_short, err_ema_long, info, torch.device(device))

            with torch.no_grad():
                action, out = agent.step(x_t, p_t, greedy=greedy, err_t=err_t)

            a_int = int(action.item())
            obs_next, _, terminated, truncated, info2 = env.step(a_int)

            with torch.no_grad():
                logits = out["logits"]
                pi = torch.softmax(logits, dim=-1)
                entropy = float((-(pi * torch.log(pi + 1e-9)).sum(dim=-1)).mean().item())
                g = out["g"].detach().cpu().numpy()[0]

                # Prediction error
                x_next_t = torch.tensor(obs_next, dtype=torch.float32, device=device).unsqueeze(0)
                xhat_all = decoder.predict_all_actions(out["g"])
                per_a_err = torch.mean((xhat_all - x_next_t.unsqueeze(1)) ** 2, dim=-1).squeeze(0)
                pred_err = float(per_a_err.min().item())

                pred_err_scalar = torch.tensor([[pred_err]], device=device, dtype=torch.float32)
                prev_pred_err = pred_err_scalar
                err_ema_short = 0.85 * err_ema_short + 0.15 * prev_pred_err
                err_ema_long = 0.97 * err_ema_long + 0.03 * prev_pred_err

            row = {
                "phase": phase_label,
                "episode": ep,
                "t": t,
                "global_step": ep * env.max_steps + t,
                "x": int(info2.get("x", -1)),
                "y": int(info2.get("y", -1)),
                "action": a_int,
                "action_name": action_name(a_int),
                "alpha": float(out["alpha"].mean().item()),
                "energy": float(out["energy"].mean().item()),
                "grad_norm": float(out["grad_norm"].mean().item()),
                "basin_id": int(out["basin_id"].detach().cpu().item()),
                "basin_conf": float(out["basin_probs"].max(dim=-1).values.mean().item()),
                "entropy": entropy,
                "pred_err": pred_err,
                "c_state": float(info2.get("c_state", 0.0)),
                "event_now": int(info2.get("event_now", 0)),
                "consequence_now": int(info2.get("consequence_now", 0)),
                "event_valence_hidden": safe_info_str(info2, "event_valence_hidden", "none"),
                "consequence_valence_hidden": safe_info_str(info2, "consequence_valence_hidden", "none"),
            }
            for i, v in enumerate(g):
                row[f"g_{i}"] = float(v)
            rows.append(row)

            obs = obs_next
            info = info2
            last_action = a_int
            t += 1
            done = bool(terminated or truncated)

        g_end = agent.get_latents()["g"].detach().cpu().numpy()[0]
        g_norm = float(np.linalg.norm(g_end))
        print(f"  [{phase_label} ep {ep}] ||g||={g_norm:.3f} c={info.get('c_state',0):.3f}")

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--native_valence", type=str, required=True, choices=["SSSS", "MMMM"])
    ap.add_argument("--warmup_episodes", type=int, default=5)
    ap.add_argument("--reversal_episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=3000)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--schedule_pattern", type=str, default="1-1-1-1")
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--greedy", action="store_true", default=True)
    args = ap.parse_args()

    device = args.device
    native_val = args.native_valence
    opposite_val = "MMMM" if native_val == "SSSS" else "SSSS"

    # Load agent
    agent, decoder, meta = load_agent_decoder(args.ckpt, device)
    agent.to(device).eval()
    decoder.to(device).eval()

    # Output
    outdir = Path(args.outdir) if args.outdir else Path("outputs") / "reversal" / timestamp_id()
    outdir.mkdir(parents=True, exist_ok=True)

    # Save meta
    meta_out = {
        "mode": "collect_reversal",
        "ckpt": str(Path(args.ckpt).resolve()),
        "native_valence": native_val,
        "opposite_valence": opposite_val,
        "warmup_episodes": args.warmup_episodes,
        "reversal_episodes": args.reversal_episodes,
        "seed": args.seed,
        "args": vars(args),
    }
    (outdir / "meta.json").write_text(json.dumps(meta_out, indent=2))

    # ── Phase A: Warmup in native valence ──
    print(f"\n=== Phase A: Warmup in {native_val} ({args.warmup_episodes} episodes) ===")
    env_native, env_cfg_native = make_env(native_val, args.schedule_pattern, args.max_steps, meta)

    agent.reset(batch_size=1)  # g = 0, only reset ONCE here

    warmup_rows = run_episodes(
        agent, decoder, env_native,
        n_episodes=args.warmup_episodes,
        seed_start=args.seed,
        device=device,
        phase_label="warmup",
        greedy=args.greedy,
    )

    g_at_switch = agent.get_latents()["g"].detach().cpu().numpy()[0].copy()
    print(f"\n  G at switch point: ||g||={np.linalg.norm(g_at_switch):.4f}")

    # ── Phase B: Reversal in opposite valence ──
    # DO NOT reset agent — g carries over from warmup
    print(f"\n=== Phase B: Reversal {native_val} → {opposite_val} ({args.reversal_episodes} episodes) ===")
    env_reversal, env_cfg_reversal = make_env(opposite_val, args.schedule_pattern, args.max_steps, meta)

    reversal_rows = run_episodes(
        agent, decoder, env_reversal,
        n_episodes=args.reversal_episodes,
        seed_start=args.seed + 1000,  # different seeds from warmup
        device=device,
        phase_label="reversal",
        greedy=args.greedy,
    )

    g_at_end = agent.get_latents()["g"].detach().cpu().numpy()[0].copy()
    print(f"\n  G at end: ||g||={np.linalg.norm(g_at_end):.4f}")
    print(f"  G shift during reversal: {np.linalg.norm(g_at_end - g_at_switch):.4f}")

    # ── Save ──
    all_rows = warmup_rows + reversal_rows

    # Add formation info
    for row in all_rows:
        row["formation_valence"] = native_val
        row["test_valence"] = native_val if row["phase"] == "warmup" else opposite_val
        row["is_reversal"] = row["phase"] == "reversal"
        row["g_at_switch_norm"] = float(np.linalg.norm(g_at_switch))

    traj_path = try_save_table(all_rows, outdir / "traj")
    print(f"\n  Trajectory: {traj_path}")
    print(f"  Total rows: {len(all_rows)} ({len(warmup_rows)} warmup + {len(reversal_rows)} reversal)")

    # Save switch point g
    np.save(outdir / "g_at_switch.npy", g_at_switch)
    np.save(outdir / "g_at_end.npy", g_at_end)


if __name__ == "__main__":
    main()
