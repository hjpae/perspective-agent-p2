# cear_pilot/experiments/phase2_g_assay.py
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


def build_agent_and_decoder_from_meta(meta: Dict[str, Any], device: str):
    agent_cfg = AgentConfig(device=device)
    agent_cfg.encoder.__dict__.update(meta["agent_cfg"]["encoder"])
    agent_cfg.world.__dict__.update(meta["agent_cfg"]["world"])
    agent_cfg.state.__dict__.update(meta["agent_cfg"]["state"])
    agent_cfg.policy.__dict__.update(meta["agent_cfg"]["policy"])

    decoder_cfg = DecoderConfig(**meta["decoder_cfg"])

    agent = CEARAgent(agent_cfg)
    decoder = ObsDecoder(decoder_cfg)
    return agent, decoder, agent_cfg, decoder_cfg


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


def build_env_config_from_args(args: argparse.Namespace) -> NZonePhase2Config:
    return NZonePhase2Config(
        width=23,
        height=7,
        obs_dim=8,
        max_steps=300,
        include_xy=bool(args.include_xy),
        reward_scale=0.0,
        start_xy=(0, 3),

        sigma_left=0.20,
        sigma_right=0.10,

        schedule_pattern=str(args.schedule_pattern),
        valence_sequence=str(args.valence_sequence),
        schedule_jitter_std=5.0,
        min_event_gap=6,
        event_delay_steps=3,

        use_event_marker=True,
        event_marker_signal=0.25,

        c_init=0.0,
        c_decay=0.91,
        c_max=1.0,
        supportive_impulse=0.45,
        misleading_impulse=-0.45,
        distortion_scale=0.30,
    )


class WindowMeter:
    """Aggregate assay statistics over a fixed global-step window."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.n_steps = 0
        self.n_event_sup = 0
        self.n_event_mis = 0
        self.n_cons_sup = 0
        self.n_cons_mis = 0

        self.sum_c = 0.0
        self.sum_abs_c = 0.0
        self.sum_dg = 0.0
        self.sum_sigma = 0.0
        self.sum_pred_err_min = 0.0
        self.sum_pred_err_max = 0.0
        self.sum_pred_err_std = 0.0
        self.sum_entropy = 0.0

        self.action_counts = np.zeros((5,), dtype=np.int64)

    def update(
        self,
        action: int,
        sigma: float,
        c_state: float,
        delta_g: float,
        pred_err_min: float,
        pred_err_max: float,
        pred_err_std: float,
        entropy: float,
        event_now: int,
        event_valence_hidden: str,
        consequence_now: int,
        consequence_valence_hidden: str,
    ):
        self.n_steps += 1
        self.sum_c += float(c_state)
        self.sum_abs_c += abs(float(c_state))
        self.sum_dg += float(delta_g)
        self.sum_sigma += float(sigma)
        self.sum_pred_err_min += float(pred_err_min)
        self.sum_pred_err_max += float(pred_err_max)
        self.sum_pred_err_std += float(pred_err_std)
        self.sum_entropy += float(entropy)

        self.action_counts[int(action)] += 1

        if int(event_now) == 1:
            if str(event_valence_hidden) == "supportive":
                self.n_event_sup += 1
            elif str(event_valence_hidden) == "misleading":
                self.n_event_mis += 1

        if int(consequence_now) == 1:
            if str(consequence_valence_hidden) == "supportive":
                self.n_cons_sup += 1
            elif str(consequence_valence_hidden) == "misleading":
                self.n_cons_mis += 1

    def summary(self) -> Dict[str, Any]:
        n = max(1, self.n_steps)
        return {
            "n_steps": int(self.n_steps),
            "event_sup": int(self.n_event_sup),
            "event_mis": int(self.n_event_mis),
            "cons_sup": int(self.n_cons_sup),
            "cons_mis": int(self.n_cons_mis),

            "mean_c": float(self.sum_c / n),
            "mean_abs_c": float(self.sum_abs_c / n),
            "mean_dg": float(self.sum_dg / n),
            "mean_sigma": float(self.sum_sigma / n),
            "mean_pred_err_min": float(self.sum_pred_err_min / n),
            "mean_pred_err_max": float(self.sum_pred_err_max / n),
            "mean_pred_err_std": float(self.sum_pred_err_std / n),
            "mean_entropy": float(self.sum_entropy / n),

            "act_up": int(self.action_counts[0]),
            "act_down": int(self.action_counts[1]),
            "act_left": int(self.action_counts[2]),
            "act_right": int(self.action_counts[3]),
            "act_stay": int(self.action_counts[4]),
        }


def print_window_summary(
    start_step: int,
    end_step: int,
    meter: WindowMeter,
):
    s = meter.summary()
    print(
        f"[window {start_step:06d}-{end_step:06d}] "
        f"steps={s['n_steps']} "
        f"event(S/M)={s['event_sup']}/{s['event_mis']} "
        f"cons(S/M)={s['cons_sup']}/{s['cons_mis']} "
        f"mean_c={s['mean_c']:+.3f} "
        f"mean|c|={s['mean_abs_c']:.3f} "
        f"mean_dg={s['mean_dg']:.4f} "
        f"mean_sigma={s['mean_sigma']:.3f} "
        f"err[min/max/std]={s['mean_pred_err_min']:.4f}/{s['mean_pred_err_max']:.4f}/{s['mean_pred_err_std']:.4f} "
        f"entropy={s['mean_entropy']:.4f} "
        f"act[U/D/L/R/S]={s['act_up']}/{s['act_down']}/{s['act_left']}/{s['act_right']}/{s['act_stay']}"
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", type=str, required=True, help="Phase 1 checkpoint path.")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--ablate_g", action="store_true")

    ap.add_argument("--do_g_at", type=int, default=-1)
    ap.add_argument("--do_g_mode", type=str, default="shock", choices=["shock", "swap", "zero"])
    ap.add_argument("--do_g_scale", type=float, default=1.0)

    ap.add_argument("--alpha_fixed", type=float, default=0.10)

    ap.add_argument(
        "--schedule_pattern",
        type=str,
        default="1-1-1-1",
        choices=["1-1-1-1", "2-2", "3-1", "1-3"],
    )
    ap.add_argument("--valence_sequence", type=str, default="SSSS")
    ap.add_argument("--include_xy", action="store_true")

    # Aggregate printing cadence, not line-by-line spam.
    ap.add_argument("--print_every_steps", type=int, default=1000)

    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device)
    meta = ckpt["meta"]

    agent, decoder, agent_cfg, dec_cfg = build_agent_and_decoder_from_meta(meta, device=args.device)

    # Fixed-alpha assay mode.
    # Keep use_error_feedback exactly as stored in the checkpoint so state_dict shapes match.
    agent_cfg.world.update_mode = "fixed"
    agent_cfg.world.alpha_fixed = float(args.alpha_fixed)

    agent = CEARAgent(agent_cfg)
    decoder = ObsDecoder(dec_cfg)

    agent.load_state_dict(ckpt["agent_state"], strict=True)
    decoder.load_state_dict(ckpt["decoder_state"], strict=True)

    agent.to(args.device).eval()
    decoder.to(args.device).eval()

    env_cfg = build_env_config_from_args(args)
    env = NZonePhase2Env(config=env_cfg)

    run_dir = Path(args.outdir) if args.outdir else (
        Path("outputs") / "runs" / f"phase2_g_assay_{timestamp_id()}"
    )
    ensure_dir(run_dir)

    run_meta = {
        "mode": "phase2_g_assay",
        "ckpt": str(Path(args.ckpt).resolve()),
        "episodes": int(args.episodes),
        "seed": int(args.seed),
        "device": str(args.device),
        "greedy": bool(args.greedy),
        "ablate_g": bool(args.ablate_g),
        "do_g_at": int(args.do_g_at),
        "do_g_mode": str(args.do_g_mode),
        "do_g_scale": float(args.do_g_scale),
        "alpha_fixed": float(args.alpha_fixed),
        "print_every_steps": int(args.print_every_steps),
        "env_cfg": env_cfg.__dict__,
        "agent_cfg": {
            "encoder": dict(agent_cfg.encoder.__dict__),
            "world": dict(agent_cfg.world.__dict__),
            "state": dict(agent_cfg.state.__dict__),
            "policy": dict(agent_cfg.policy.__dict__),
        },
        "decoder_cfg": dict(dec_cfg.__dict__),
        "source_train_meta": meta,
    }
    (run_dir / "meta.json").write_text(json.dumps(run_meta, indent=2))

    rows: List[Dict[str, Any]] = []
    n_actions = int(env.action_space.n)

    global_step = 0
    window_start = 0
    meter = WindowMeter()

    for ep in range(args.episodes):
        ep_seed = int(args.seed + ep)

        # Always reset g at the start of every episode.
        agent.reset(batch_size=1)
        obs, info = env.reset(seed=ep_seed)

        last_action = 4
        done = False
        t = 0
        g_prev = None

        event_steps_json = json.dumps(info.get("event_steps", []))
        consequence_steps_json = json.dumps(info.get("consequence_steps", []))
        event_valences_json = json.dumps(info.get("event_valences", []))

        print(
            f"[ep {ep:03d} start] "
            f"pattern={args.schedule_pattern} "
            f"valence={args.valence_sequence} "
            f"event_steps={event_steps_json} "
            f"consequence_steps={consequence_steps_json} "
            f"alpha={args.alpha_fixed:.3f}"
        )

        while not done:
            if args.do_g_at >= 0 and t == args.do_g_at:
                agent.apply_perturbation(kind=args.do_g_mode, scale=args.do_g_scale)
                did_do_g = 1
            else:
                did_do_g = 0

            x_t = torch.tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
            p_t = torch.tensor(onehot(last_action, n_actions), dtype=torch.float32, device=args.device).unsqueeze(0)

            with torch.no_grad():
                action, out = agent.step(
                    x_t,
                    p_t,
                    greedy=args.greedy,
                    ablate_g=args.ablate_g,
                    err_t=None,
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
                "global_step": int(global_step),
                "t": int(info2.get("t", t)),
                "x": int(info2.get("x", -1)),
                "y": int(info2.get("y", -1)),
                "zone_id": int(info2.get("zone_id", -1)),
                "current_sigma": float(info2.get("current_sigma", np.nan)),

                "schedule_pattern": str(info2.get("schedule_pattern", args.schedule_pattern)),
                "valence_sequence": str(info2.get("valence_sequence", args.valence_sequence)),
                "event_steps_json": str(event_steps_json),
                "consequence_steps_json": str(consequence_steps_json),
                "event_valences_json": str(event_valences_json),

                "event_now": int(info2.get("event_now", 0)),
                "event_id": int(info2.get("event_id", -1)),
                "event_valence_hidden": str(info2.get("event_valence_hidden", "none")),

                "consequence_now": int(info2.get("consequence_now", 0)),
                "consequence_id": int(info2.get("consequence_id", -1)),
                "consequence_valence_hidden": str(info2.get("consequence_valence_hidden", "none")),

                "steps_since_last_event": int(info2.get("steps_since_last_event", -1)),
                "steps_since_last_consequence": int(info2.get("steps_since_last_consequence", -1)),
                "pending_consequences": int(info2.get("pending_consequences", 0)),
                "c_state": float(info2.get("c_state", 0.0)),

                "action": int(a_int),
                "action_name": str(action_name(a_int)),
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

            for i, v in enumerate(g):
                row[f"g_{i}"] = float(v)
            for i, v in enumerate(s):
                row[f"s_{i}"] = float(v)
            for i, v in enumerate(z):
                row[f"z_{i}"] = float(v)

            rows.append(row)

            meter.update(
                action=a_int,
                sigma=float(info2.get("current_sigma", np.nan)),
                c_state=float(info2.get("c_state", 0.0)),
                delta_g=float(delta_g),
                pred_err_min=float(e_min),
                pred_err_max=float(e_max),
                pred_err_std=float(e_std),
                entropy=float(entropy),
                event_now=int(info2.get("event_now", 0)),
                event_valence_hidden=str(info2.get("event_valence_hidden", "none")),
                consequence_now=int(info2.get("consequence_now", 0)),
                consequence_valence_hidden=str(info2.get("consequence_valence_hidden", "none")),
            )

            global_step += 1

            if args.print_every_steps > 0 and (global_step % int(args.print_every_steps) == 0):
                print_window_summary(window_start, global_step - 1, meter)
                meter.reset()
                window_start = global_step

            obs = obs_next
            last_action = a_int
            t += 1
            done = bool(terminated or truncated)

        print(
            f"[ep {ep:03d} end] "
            f"last_t={t:03d} "
            f"final_x={int(info2.get('x', -1)):02d} "
            f"final_y={int(info2.get('y', -1)):02d} "
            f"final_c={float(info2.get('c_state', 0.0)):+.3f}"
        )

    if meter.n_steps > 0:
        print_window_summary(window_start, global_step - 1, meter)

    saved = try_save_table(rows, run_dir / "traj")
    print(f"[phase2_g_assay] saved: {saved}")


if __name__ == "__main__":
    main()