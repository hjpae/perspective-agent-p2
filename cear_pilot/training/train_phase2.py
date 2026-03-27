# cear_pilot/training/train_phase2.py
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


# -----------------------------------------------------------------------------
# utils
# -----------------------------------------------------------------------------

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
        max_steps=int(args.max_steps),
        include_xy=bool(args.include_xy),
        reward_scale=0.0,
        start_xy=(0, 3),
        sigma_left=float(args.sigma_left),
        sigma_right=float(args.sigma_right),
        schedule_pattern=str(args.schedule_pattern),
        valence_sequence=str(args.valence_sequence),
        schedule_jitter_std=float(args.schedule_jitter_std),
        min_event_gap=int(args.min_event_gap),
        event_delay_steps=int(args.event_delay_steps),
        use_event_marker=True,
        event_marker_signal=float(args.event_marker_signal),
        c_init=0.0,
        c_decay=float(args.c_decay),
        c_max=1.0,
        supportive_impulse=float(args.supportive_impulse),
        misleading_impulse=float(args.misleading_impulse),
        distortion_scale=float(args.distortion_scale),
    )


def vec_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x))


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Formation-style Phase 2 runner for persistent g evolution. "
            "Loads a Phase 1 checkpoint, forces fixed-alpha update mode, and "
            "rolls a scheduled Phase 2 world while optionally carrying g across episodes."
        )
    )

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

    # The core diagnostic variable.
    ap.add_argument("--alpha_fixed", type=float, default=0.10)

    # By default, DO NOT reset g between episodes.
    ap.add_argument("--reset_g_every_episode", action="store_true")

    # Recommended default formation condition.
    ap.add_argument(
        "--schedule_pattern",
        type=str,
        default="1-1-1-1",
        choices=["1-1-1-1", "2-2", "3-1", "1-3"],
    )
    ap.add_argument("--valence_sequence", type=str, default="SSSS")
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--include_xy", action="store_true")

    # World knobs kept explicit so the same script can probe persistence vs. decay.
    ap.add_argument("--sigma_left", type=float, default=0.20)
    ap.add_argument("--sigma_right", type=float, default=0.10)
    ap.add_argument("--schedule_jitter_std", type=float, default=5.0)
    ap.add_argument("--min_event_gap", type=int, default=6)
    ap.add_argument("--event_delay_steps", type=int, default=3)
    ap.add_argument("--event_marker_signal", type=float, default=0.25)
    ap.add_argument("--c_decay", type=float, default=0.91)
    ap.add_argument("--supportive_impulse", type=float, default=0.45)
    ap.add_argument("--misleading_impulse", type=float, default=-0.45)
    ap.add_argument("--distortion_scale", type=float, default=0.30)

    # Logging / plotting cadence.
    ap.add_argument("--print_every_steps", type=int, default=0)
    ap.add_argument("--print_every_episodes", type=int, default=10)
    ap.add_argument("--save_traj", action="store_true")
    ap.add_argument("--episode_probe_every", type=int, default=10)

    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device)
    meta = ckpt["meta"]

    agent, decoder, agent_cfg, dec_cfg = build_agent_and_decoder_from_meta(meta, device=args.device)

    # Force fixed-alpha mode for interpretable formation rollouts.
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
        Path("outputs") / "runs" / f"phase2_train_{timestamp_id()}"
    )
    ensure_dir(run_dir)

    run_meta = {
        "mode": "phase2_train_persistent_g",
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
        "reset_g_every_episode": bool(args.reset_g_every_episode),
        "episode_probe_every": int(args.episode_probe_every),
        "print_every_steps": int(args.print_every_steps),
        "print_every_episodes": int(args.print_every_episodes),
        "save_traj": bool(args.save_traj),
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

    step_rows: List[Dict[str, Any]] = []
    ep_rows: List[Dict[str, Any]] = []
    n_actions = int(env.action_space.n)

    global_step = 0
    agent.reset(batch_size=1)

    g0_global = agent.get_latents()["g"].detach().cpu().numpy().squeeze(0).copy()
    prev_ep_end_g = g0_global.copy()

    print(
        f"[run start] mode=persistent_g alpha={args.alpha_fixed:.3f} "
        f"carry_g={int(not args.reset_g_every_episode)} pattern={args.schedule_pattern} "
        f"valence={args.valence_sequence} episodes={args.episodes} max_steps={args.max_steps}"
    )

    for ep in range(args.episodes):
        ep_seed = int(args.seed + ep)

        if args.reset_g_every_episode:
            agent.reset(batch_size=1)

        g_start = agent.get_latents()["g"].detach().cpu().numpy().squeeze(0).copy()
        obs, info = env.reset(seed=ep_seed)

        last_action = 4
        done = False
        t = 0
        g_prev_step = g_start.copy()

        event_steps_json = json.dumps(info.get("event_steps", []))
        consequence_steps_json = json.dumps(info.get("consequence_steps", []))
        event_valences_json = json.dumps(info.get("event_valences", []))

        sum_delta_g = 0.0
        sum_entropy = 0.0
        sum_c = 0.0
        sum_abs_c = 0.0
        n_event_sup = 0
        n_event_mis = 0
        n_cons_sup = 0
        n_cons_mis = 0
        max_delta_g = 0.0
        final_info = info

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
            final_info = info2

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
            g_cand = out["g_candidate"].squeeze(0).detach().cpu().numpy()

            delta_g = float(np.linalg.norm(g - g_prev_step))
            g_prev_step = g.copy()

            e_min, e_max, e_std = collect_obs_prediction_error(
                decoder=decoder,
                g_t=out["g"],
                x_next=obs_next,
                device=args.device,
            )

            sum_delta_g += delta_g
            sum_entropy += entropy
            c_state = float(info2.get("c_state", 0.0))
            sum_c += c_state
            sum_abs_c += abs(c_state)
            max_delta_g = max(max_delta_g, delta_g)

            event_now = int(info2.get("event_now", 0))
            consequence_now = int(info2.get("consequence_now", 0))
            event_valence_hidden = str(info2.get("event_valence_hidden", "none"))
            consequence_valence_hidden = str(info2.get("consequence_valence_hidden", "none"))

            if event_now and event_valence_hidden == "supportive":
                n_event_sup += 1
            elif event_now and event_valence_hidden == "misleading":
                n_event_mis += 1

            if consequence_now and consequence_valence_hidden == "supportive":
                n_cons_sup += 1
            elif consequence_now and consequence_valence_hidden == "misleading":
                n_cons_mis += 1

            if args.save_traj:
                row: Dict[str, Any] = {
                    "episode": int(ep),
                    "episode_seed": int(ep_seed),
                    "phase": "phase2_train",
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
                    "event_now": event_now,
                    "event_id": int(info2.get("event_id", -1)),
                    "event_valence_hidden": event_valence_hidden,
                    "consequence_now": consequence_now,
                    "consequence_id": int(info2.get("consequence_id", -1)),
                    "consequence_valence_hidden": consequence_valence_hidden,
                    "steps_since_last_event": int(info2.get("steps_since_last_event", -1)),
                    "steps_since_last_consequence": int(info2.get("steps_since_last_consequence", -1)),
                    "pending_consequences": int(info2.get("pending_consequences", 0)),
                    "c_state": c_state,
                    "action": int(a_int),
                    "action_name": str(action_name(a_int)),
                    "policy_mode": int(policy_mode),
                    "action_prob_max": float(action_prob_max),
                    "entropy": float(entropy),
                    "alpha": float(alpha),
                    "delta_g": float(delta_g),
                    "g_norm": vec_norm(g),
                    "g_cand_norm": vec_norm(g_cand),
                    "did_do_g": int(did_do_g),
                    "pred_err_min": float(e_min),
                    "pred_err_max": float(e_max),
                    "pred_err_std": float(e_std),
                }
                for i, v in enumerate(g):
                    row[f"g_{i}"] = float(v)
                for i, v in enumerate(g_cand):
                    row[f"g_cand_{i}"] = float(v)
                for i, v in enumerate(s):
                    row[f"s_{i}"] = float(v)
                for i, v in enumerate(z):
                    row[f"z_{i}"] = float(v)
                step_rows.append(row)

            if args.print_every_steps > 0 and ((global_step + 1) % int(args.print_every_steps) == 0):
                print(
                    f"[step {global_step + 1:06d}] ep={ep:03d} t={t:03d} "
                    f"x={int(info2.get('x', -1)):02d} y={int(info2.get('y', -1)):02d} "
                    f"c={c_state:+.3f} dg={delta_g:.4f} |g|={vec_norm(g):.4f} alpha={alpha:.3f}"
                )

            global_step += 1
            obs = obs_next
            last_action = a_int
            t += 1
            done = bool(terminated or truncated)

        g_end = agent.get_latents()["g"].detach().cpu().numpy().squeeze(0).copy()
        mean_delta_g = sum_delta_g / max(1, t)
        mean_entropy = sum_entropy / max(1, t)
        mean_c = sum_c / max(1, t)
        mean_abs_c = sum_abs_c / max(1, t)

        ep_row: Dict[str, Any] = {
            "episode": int(ep),
            "episode_seed": int(ep_seed),
            "global_step_end": int(global_step - 1),
            "schedule_pattern": str(args.schedule_pattern),
            "valence_sequence": str(args.valence_sequence),
            "reset_g_every_episode": int(bool(args.reset_g_every_episode)),
            "alpha_fixed": float(args.alpha_fixed),
            "steps": int(t),
            "final_x": int(final_info.get("x", -1)),
            "final_y": int(final_info.get("y", -1)),
            "final_zone_id": int(final_info.get("zone_id", -1)),
            "final_c": float(final_info.get("c_state", 0.0)),
            "g_start_norm": vec_norm(g_start),
            "g_end_norm": vec_norm(g_end),
            "g_delta_within_ep": vec_norm(g_end - g_start),
            "g_delta_from_global_init": vec_norm(g_end - g0_global),
            "g_delta_from_prev_ep_end": vec_norm(g_end - prev_ep_end_g),
            "mean_delta_g": float(mean_delta_g),
            "max_delta_g": float(max_delta_g),
            "mean_entropy": float(mean_entropy),
            "mean_c": float(mean_c),
            "mean_abs_c": float(mean_abs_c),
            "n_event_sup": int(n_event_sup),
            "n_event_mis": int(n_event_mis),
            "n_cons_sup": int(n_cons_sup),
            "n_cons_mis": int(n_cons_mis),
            "event_steps_json": str(event_steps_json),
            "consequence_steps_json": str(consequence_steps_json),
            "event_valences_json": str(event_valences_json),
            "probe_marker": int((ep % max(1, int(args.episode_probe_every))) == 0),
        }
        for i, v in enumerate(g_start):
            ep_row[f"g_start_{i}"] = float(v)
        for i, v in enumerate(g_end):
            ep_row[f"g_end_{i}"] = float(v)
        ep_rows.append(ep_row)

        prev_ep_end_g = g_end.copy()

        if args.print_every_episodes > 0 and (((ep + 1) % int(args.print_every_episodes) == 0) or ep == 0):
            print(
                f"[ep {ep:03d} end] final_x={ep_row['final_x']:02d} final_y={ep_row['final_y']:02d} "
                f"final_c={ep_row['final_c']:+.3f} |g_start|={ep_row['g_start_norm']:.4f} "
                f"|g_end|={ep_row['g_end_norm']:.4f} dg_ep={ep_row['g_delta_within_ep']:.4f} "
                f"dg_prev={ep_row['g_delta_from_prev_ep_end']:.4f} mean_dg={ep_row['mean_delta_g']:.4f}"
            )

    ep_path = try_save_table(ep_rows, run_dir / "episode_summary")
    print(f"[phase2_train] saved episode summary: {ep_path}")

    if args.save_traj and len(step_rows) > 0:
        traj_path = try_save_table(step_rows, run_dir / "traj")
        print(f"[phase2_train] saved trajectory: {traj_path}")

    print(f"[run done] outdir={run_dir}")


if __name__ == "__main__":
    main()
