# cear_pilot/experiments/run_collect.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from cear_pilot.envs.nzone_grid import NZoneGridEnv, NZoneConfig
from cear_pilot.models.agent import CEARAgent, AgentConfig
from cear_pilot.models.decoder import ObsDecoder, DecoderConfig


def timestamp_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def try_save_table(rows: List[Dict[str, Any]], out_path: Path) -> Path:
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


def _tuple3(x) -> Optional[Tuple[float, float, float]]:
    if x is None:
        return None
    return (float(x[0]), float(x[1]), float(x[2]))


def _load_replay_actions(path: str) -> Optional[List[int]]:
    if not str(path).strip():
        return None
    p = Path(path)
    obj = json.loads(p.read_text())
    if isinstance(obj, dict) and "actions" in obj:
        actions = [int(a) for a in obj["actions"]]
    elif isinstance(obj, list):
        actions = [int(a) for a in obj]
    else:
        raise ValueError("replay_actions JSON must be a list or dict with key 'actions'")
    if len(actions) == 0:
        raise ValueError("replay_actions is empty")
    return actions


def build_agent_from_meta(
    meta: Dict[str, Any],
    device: str,
    zone_sigma_override: Optional[Tuple[float, float, float]] = None,
) -> tuple[CEARAgent, ObsDecoder, NZoneGridEnv]:
    env_cfg = NZoneConfig(**meta["env_cfg"])
    if zone_sigma_override is not None:
        env_cfg.zone_sigma = tuple(float(v) for v in zone_sigma_override)
    env = NZoneGridEnv(config=env_cfg)

    agent_cfg = AgentConfig(device=device)

    enc = meta["agent_cfg"]["encoder"]
    world = meta["agent_cfg"]["world"]
    state = meta["agent_cfg"]["state"]
    pol = meta["agent_cfg"]["policy"]

    agent_cfg.encoder.__dict__.update(enc)
    agent_cfg.world.__dict__.update(world)
    agent_cfg.state.__dict__.update(state)
    agent_cfg.policy.__dict__.update(pol)

    agent = CEARAgent(agent_cfg)
    dec_cfg = DecoderConfig(**meta["decoder_cfg"])
    decoder = ObsDecoder(dec_cfg)
    return agent, decoder, env


def _policy_stats_from_s(agent: CEARAgent, s_t: torch.Tensor):
    logits = agent.policy(s_t.detach())  # (1, A)
    pi = torch.softmax(logits, dim=-1)   # (1, A)
    entropy = float((-(pi * torch.log(pi + 1e-9)).sum(dim=-1)).mean().item())
    top2 = torch.topk(pi, k=min(2, pi.shape[-1]), dim=-1).values
    pi_max = float(top2[:, 0].mean().item())
    pi_margin = float((top2[:, 0] - top2[:, 1]).mean().item()) if top2.shape[-1] > 1 else pi_max
    pi_argmax = int(torch.argmax(pi, dim=-1).item())
    return logits, pi, entropy, pi_max, pi_margin, pi_argmax


def maybe_apply_g_intervention(
    agent: CEARAgent,
    t: int,
    do_g_at: int,
    do_g_mode: str,
    do_g_scale: float,
):
    if do_g_at < 0 or t != do_g_at:
        return False
    agent.apply_perturbation(kind=do_g_mode, scale=do_g_scale)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--ablate_g", action="store_true")

    ap.add_argument("--zone_sigma", type=float, nargs=3, default=None)
    ap.add_argument("--replay_actions", type=str, default="")

    ap.add_argument("--t_switch", type=int, default=-1)
    ap.add_argument("--zone_sigma2", type=float, nargs=3, default=None)

    ap.add_argument("--log_policy_full", action="store_true")

    # Phase 2 probe hooks
    ap.add_argument("--do_g_at", type=int, default=-1,
                    help="If >=0, apply g intervention at this rollout step.")
    ap.add_argument("--do_g_mode", type=str, default="shock",
                    choices=["shock", "swap", "zero"])
    ap.add_argument("--do_g_scale", type=float, default=1.0)

    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device)
    meta = ckpt["meta"]

    sigma1 = _tuple3(args.zone_sigma)
    sigma2 = _tuple3(args.zone_sigma2)

    agent, decoder, env = build_agent_from_meta(meta, device=args.device, zone_sigma_override=sigma1)
    agent.load_state_dict(ckpt["agent_state"])
    decoder.load_state_dict(ckpt["decoder_state"])
    agent.to(args.device).eval()
    decoder.to(args.device).eval()

    run_dir = Path(args.outdir) if args.outdir else (Path("outputs") / "runs" / timestamp_id())
    ensure_dir(run_dir)
    ensure_dir(run_dir / "figs")

    run_meta = {
        "mode": "collect_phase2",
        "ckpt": str(Path(args.ckpt).resolve()),
        "episodes": int(args.episodes),
        "seed": int(args.seed),
        "device": str(args.device),
        "greedy": bool(args.greedy),
        "ablate_g": bool(args.ablate_g),
        "zone_sigma": sigma1,
        "t_switch": int(args.t_switch),
        "zone_sigma2": sigma2,
        "replay_actions": str(Path(args.replay_actions).resolve()) if str(args.replay_actions).strip() else "",
        "log_policy_full": bool(args.log_policy_full),
        "do_g_at": int(args.do_g_at),
        "do_g_mode": str(args.do_g_mode),
        "do_g_scale": float(args.do_g_scale),
        "train_meta": meta,
    }
    (run_dir / "meta.json").write_text(json.dumps(run_meta, indent=2))

    rng = np.random.default_rng(args.seed)
    n_actions = int(env.action_space.n)
    replay_actions = _load_replay_actions(args.replay_actions)

    rows: List[Dict[str, Any]] = []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        agent.reset(batch_size=1)
        last_action = 4

        done = False
        t = 0
        switched = False

        while not done:
            if (not switched) and args.t_switch >= 0 and sigma2 is not None and t == args.t_switch:
                env.set_zone_sigma(sigma2)
                switched = True

            did_do_g = maybe_apply_g_intervention(
                agent=agent,
                t=t,
                do_g_at=args.do_g_at,
                do_g_mode=args.do_g_mode,
                do_g_scale=args.do_g_scale,
            )

            x_t = torch.tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
            p_t = torch.tensor(onehot(last_action, n_actions), dtype=torch.float32, device=args.device).unsqueeze(0)

            if replay_actions is None:
                with torch.no_grad():
                    action, out = agent.step(x_t, p_t, greedy=args.greedy, ablate_g=args.ablate_g)
                a_int = int(action.item())
            else:
                if t >= len(replay_actions):
                    break
                a_int = int(replay_actions[t])
                with torch.no_grad():
                    out = agent.forward_step(x_t, p_t, ablate_g=args.ablate_g)

            with torch.no_grad():
                logits_act, pi_act, pi_entropy, pi_max, pi_margin, pi_argmax = _policy_stats_from_s(agent, out["s"])

            obs_next, _, terminated, truncated, info2 = env.step(a_int)

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
                "action_replay": int(a_int) if replay_actions is not None else -1,

                "pi_argmax": int(pi_argmax),
                "pi_max": float(pi_max),
                "pi_margin": float(pi_margin),
                "pi_entropy": float(pi_entropy),

                "switched": int(switched),
                "t_switch": int(args.t_switch),

                "did_do_g": int(did_do_g),

                # Phase 2 env diagnostics
                "fragility": float(info2.get("fragility", np.nan)),
                "rupture_memory": float(info2.get("rupture_memory", np.nan)),
                "on_encounter": int(bool(info2.get("on_encounter", False))),
                "encounter_event": int(bool(info2.get("encounter_event", False))),
                "rupture": int(bool(info2.get("rupture", False))),
                "pending_ruptures": int(info2.get("pending_ruptures", 0)),
                "rupture_obs_timer": int(info2.get("rupture_obs_timer", 0)),
                "rupture_action_timer": int(info2.get("rupture_action_timer", 0)),
                "slip": int(bool(info2.get("slip", False))),
                "drift": int(bool(info2.get("drift", False))),
                "hazard": int(bool(info2.get("hazard", False))),
            }

            if sigma1 is not None:
                row["sigma_0"] = float(sigma1[0])
                row["sigma_1"] = float(sigma1[1])
                row["sigma_2"] = float(sigma1[2])
            if sigma2 is not None:
                row["sigma2_0"] = float(sigma2[0])
                row["sigma2_1"] = float(sigma2[1])
                row["sigma2_2"] = float(sigma2[2])

            if args.log_policy_full:
                la = logits_act.squeeze(0).detach().cpu().numpy()
                pa = pi_act.squeeze(0).detach().cpu().numpy()
                for i in range(n_actions):
                    row[f"logits_act_{i}"] = float(la[i])
                    row[f"pi_act_{i}"] = float(pa[i])

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

    saved_path = try_save_table(rows, run_dir / "traj")
    print(f"Saved trajectories to: {saved_path}")
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()