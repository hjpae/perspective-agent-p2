# cear_pilot/experiments/collect_phase2.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
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


def load_agent_decoder(ckpt_path: str, device: str) -> Tuple[CEARAgent, ObsDecoder, Dict[str, Any]]:
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

    # evidence vector is always used in phase2
    agent_cfg.world.use_error_feedback = True
    agent_cfg.world.err_dim = len(ERR_FEATURE_NAMES)

    agent = CEARAgent(agent_cfg)
    agent.load_state_dict(ckpt["agent_state"], strict=False)

    dec_cfg = DecoderConfig(**base["decoder_cfg"])
    dec_cfg.obs_dim = int(agent_cfg.encoder.obs_dim)
    decoder = ObsDecoder(dec_cfg)
    decoder.load_state_dict(ckpt["decoder_state"], strict=False)
    return agent, decoder, meta


def resolve_env(meta: Dict[str, Any], args: argparse.Namespace) -> NZonePhase2Config:
    env_cfg_data = meta.get("env_cfg", meta.get("base_meta", {}).get("env_cfg", {}))
    env_cfg = NZonePhase2Config(**env_cfg_data) if env_cfg_data else NZonePhase2Config()
    if args.schedule_pattern:
        env_cfg.schedule_pattern = str(args.schedule_pattern)
    if args.valence_sequence:
        env_cfg.valence_sequence = str(args.valence_sequence)
    if args.max_steps > 0:
        env_cfg.max_steps = int(args.max_steps)
    return env_cfg


def action_name(a: int) -> str:
    return ["UP", "DOWN", "LEFT", "RIGHT", "STAY"][int(a)]


def collect_obs_prediction_error(decoder: ObsDecoder, g_t: torch.Tensor, x_next: np.ndarray, device: str) -> Tuple[float, float, float]:
    with torch.no_grad():
        x_next_t = torch.tensor(x_next, dtype=torch.float32, device=device).unsqueeze(0)
        xhat_all = decoder.predict_all_actions(g_t)
        per_a_err = torch.mean((xhat_all - x_next_t.unsqueeze(1)) ** 2, dim=-1).squeeze(0)
        e_np = per_a_err.detach().float().cpu().numpy()
        return float(e_np.min()), float(e_np.max()), float(e_np.std())


def safe_info_float(info: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(info.get(key, default))
    except Exception:
        return float(default)


def safe_info_int(info: Dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(info.get(key, default))
    except Exception:
        return int(default)


def exp_trace_from_steps(steps: int, tau: float) -> float:
    if steps is None or steps < 0:
        return 0.0
    return float(np.exp(-float(steps) / max(float(tau), 1e-6)))


def build_err_t(
    prev_pred_err: torch.Tensor,
    err_ema_short: torch.Tensor,
    err_ema_long: torch.Tensor,
    info: Dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    eps = 1e-4
    long_base = torch.clamp(err_ema_long, min=eps)

    pred_err_now_ratio = prev_pred_err / long_base
    pred_err_ema_short_ratio = err_ema_short / long_base
    pred_err_ema_long_log = torch.log1p(err_ema_long)
    pred_err_rise_ratio = (err_ema_short - err_ema_long) / long_base

    steps_since_event = safe_info_int(info, "steps_since_last_event", -1)
    steps_since_consequence = safe_info_int(info, "steps_since_last_consequence", -1)

    recent_event_trace = exp_trace_from_steps(steps_since_event, tau=4.0)
    recent_consequence_trace = exp_trace_from_steps(steps_since_consequence, tau=6.0)

    current_event_flag = float(
        max(
            safe_info_int(info, "on_encounter", 0),
            safe_info_int(info, "event_now", 0),
        )
    )

    c_state_signed = safe_info_float(info, "c_state", 0.0)

    outcome_raw = safe_info_float(info, "encounter_outcome", 1.0)
    supportive_flag = max(outcome_raw - 1.0, 0.0)
    misleading_flag = max(1.0 - outcome_raw, 0.0)

    feats = torch.cat(
        [
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
        ],
        dim=-1,
    )
    return feats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--schedule_pattern", type=str, default="")
    ap.add_argument("--valence_sequence", type=str, default="")
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--carry_g_between_episodes", action="store_true")
    ap.add_argument("--ablate_g", action="store_true")
    ap.add_argument("--do_g_at", type=int, default=-1)
    ap.add_argument("--do_g_mode", type=str, default="shock", choices=["shock", "swap", "zero", "flip"])
    ap.add_argument("--do_g_scale", type=float, default=1.0)
    args = ap.parse_args()

    agent, decoder, meta = load_agent_decoder(args.ckpt, args.device)
    agent.to(args.device).eval()
    decoder.to(args.device).eval()

    env_cfg = resolve_env(meta, args)
    env = NZonePhase2Env(config=env_cfg)

    run_dir = Path(args.outdir) if args.outdir else Path("outputs") / "runs" / f"collect_phase2_{timestamp_id()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    meta_out = {
        "mode": "collect_phase2_landscape",
        "ckpt": str(Path(args.ckpt).resolve()),
        "args": vars(args),
        "env_cfg": env_cfg.__dict__,
        "source_meta": meta,
        "err_feature_names": list(ERR_FEATURE_NAMES),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta_out, indent=2))

    rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    agent.reset(batch_size=1)
    n_actions = int(env.action_space.n)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        if not args.carry_g_between_episodes:
            agent.reset(batch_size=1)

        last_action = 4
        done = False
        t = 0
        g_prev = None
        basin_prev = None
        perturb_step = None
        recovery_anchor = None
        switches = 0

        prev_pred_err = torch.zeros((1, 1), device=args.device, dtype=torch.float32)
        err_ema_short = torch.zeros((1, 1), device=args.device, dtype=torch.float32)
        err_ema_long = torch.zeros((1, 1), device=args.device, dtype=torch.float32)

        ep_alpha_sum = 0.0
        ep_delta_g_sum = 0.0
        ep_entropy_sum = 0.0
        ep_recovery_valid = 0
        ep_recovery_sum = 0.0
        ep_err_feat_sums = {name: 0.0 for name in ERR_FEATURE_NAMES}

        while not done:
            if args.do_g_at >= 0 and t == args.do_g_at:
                recovery_anchor = agent.get_latents()["g"].detach().clone()
                agent.apply_perturbation(kind=args.do_g_mode, scale=args.do_g_scale)
                perturb_step = t
                did_do_g = 1
            else:
                did_do_g = 0

            x_t = torch.tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
            p_t = F.one_hot(torch.tensor([last_action], device=args.device), num_classes=n_actions).float()

            err_t = build_err_t(
                prev_pred_err=prev_pred_err,
                err_ema_short=err_ema_short,
                err_ema_long=err_ema_long,
                info=info,
                device=torch.device(args.device),
            )

            with torch.no_grad():
                action, out = agent.step(x_t, p_t, greedy=args.greedy, ablate_g=args.ablate_g, err_t=err_t)

            a_int = int(action.item())
            obs_next, _, terminated, truncated, info2 = env.step(a_int)

            with torch.no_grad():
                logits = out["logits"]
                pi = torch.softmax(logits, dim=-1)
                entropy = float((-(pi * torch.log(pi + 1e-9)).sum(dim=-1)).mean().item())
                g = out["g"].detach().cpu().numpy()[0]
                basin_id = int(out["basin_id"].detach().cpu().item())
                basin_conf = float(out["basin_probs"].max(dim=-1).values.mean().item())

                if basin_prev is not None and basin_id != basin_prev:
                    switches += 1
                basin_prev = basin_id

                delta_g = 0.0 if g_prev is None else float(np.linalg.norm(g - g_prev))
                g_prev = g.copy()

                if recovery_anchor is not None:
                    recovery_dist = float(torch.norm(out["g"] - recovery_anchor, dim=-1).mean().item())
                    time_since_perturb = t - int(perturb_step)
                    ep_recovery_sum += recovery_dist
                    ep_recovery_valid += 1
                else:
                    recovery_dist = np.nan
                    time_since_perturb = -1

                e_min, e_max, e_std = collect_obs_prediction_error(decoder, out["g"], obs_next, args.device)
                pred_err_scalar = torch.tensor([[e_min]], device=args.device, dtype=torch.float32)

                prev_pred_err = pred_err_scalar
                err_ema_short = 0.85 * err_ema_short + 0.15 * prev_pred_err
                err_ema_long = 0.97 * err_ema_long + 0.03 * prev_pred_err

                ep_alpha_sum += float(out["alpha"].mean().item())
                ep_delta_g_sum += delta_g
                ep_entropy_sum += entropy

                err_vec_np = err_t.detach().cpu().numpy()[0]
                for name, val in zip(ERR_FEATURE_NAMES, err_vec_np.tolist()):
                    ep_err_feat_sums[name] += float(val)

            row = {
                "episode": ep,
                "t": t,
                "x": int(info2.get("x", -1)),
                "y": int(info2.get("y", -1)),
                "action": a_int,
                "action_name": action_name(a_int),
                "alpha": float(out["alpha"].mean().item()),
                "energy": float(out["energy"].mean().item()),
                "grad_norm": float(out["grad_norm"].mean().item()),
                "basin_id": basin_id,
                "basin_conf": basin_conf,
                "entropy": entropy,
                "delta_g": delta_g,
                "did_do_g": did_do_g,
                "time_since_perturb": time_since_perturb,
                "recovery_dist": recovery_dist,
                "pred_err_min": e_min,
                "pred_err_max": e_max,
                "pred_err_std": e_std,
                "on_encounter": int(info2.get("on_encounter", 0)),
                "event_now": int(info2.get("event_now", 0)),
                "consequence_now": int(info2.get("consequence_now", 0)),
                "encounter_outcome": float(info2.get("encounter_outcome", 1.0)),
                "c_state": float(info2.get("c_state", 0.0)),
                "steps_since_last_event": int(info2.get("steps_since_last_event", -1)),
                "steps_since_last_consequence": int(info2.get("steps_since_last_consequence", -1)),
                "supportive_timer": int(info2.get("supportive_timer", 0)),
                "misleading_timer": int(info2.get("misleading_timer", 0)),
            }
            for i, v in enumerate(err_vec_np.tolist()):
                row[f"err_{ERR_FEATURE_NAMES[i]}"] = float(v)
            for i, v in enumerate(g):
                row[f"g_{i}"] = float(v)
            rows.append(row)

            obs = obs_next
            info = info2
            last_action = a_int
            t += 1
            done = bool(terminated or truncated)

        summary_row = {
            "episode": ep,
            "switches": switches,
            "final_x": int(info.get("x", -1)),
            "final_c": float(info.get("c_state", 0.0)),
            "mean_alpha": ep_alpha_sum / max(t, 1),
            "mean_delta_g": ep_delta_g_sum / max(t, 1),
            "mean_entropy": ep_entropy_sum / max(t, 1),
            "mean_recovery_dist": ep_recovery_sum / max(ep_recovery_valid, 1) if ep_recovery_valid > 0 else np.nan,
        }
        for name in ERR_FEATURE_NAMES:
            summary_row[f"mean_err_{name}"] = ep_err_feat_sums[name] / max(t, 1)
        summary_rows.append(summary_row)

    traj_path = try_save_table(rows, run_dir / "traj")
    ep_path = try_save_table(summary_rows, run_dir / "episode_summary")
    print(f"[collect_phase2] trajectory: {traj_path}")
    print(f"[collect_phase2] episode summary: {ep_path}")


if __name__ == "__main__":
    main()