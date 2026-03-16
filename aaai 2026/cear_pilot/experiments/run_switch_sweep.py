# cear_pilot/experiments/run_switch_sweep.py
# -*- coding: utf-8 -*-
"""
Run one long rollout with regime switching AFTER warmup.
Saves traj.(parquet|csv) with columns:
  t, regime, switch, g_*, pi_max, entropy, margin, (optional logits_*)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def set_env_zone_sigma(env: NZoneGridEnv, sigma_triplet: Tuple[float, float, float]) -> None:
    env._zone_sigma = np.array(list(sigma_triplet), dtype=np.float32)


def policy_stats_from_logits(logits: np.ndarray) -> Tuple[float, float, float]:
    # logits: (A,)
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

    # timeline
    ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--period", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=400)

    # regimes: sigma only (A/B)
    ap.add_argument("--sigma_A", type=float, nargs=3, default=(0.60, 0.30, 0.05))
    ap.add_argument("--sigma_B", type=float, nargs=3, default=(0.05, 0.30, 0.60))

    ap.add_argument("--outdir", type=str, default="")
    args = ap.parse_args()

    device = args.device
    rng = np.random.default_rng(args.seed)

    ckpt = torch.load(args.ckpt, map_location=device)
    meta = ckpt["meta"]
    meta["agent_cfg"]["world"]["g_damping"] = 0.1
    print("[OVERRIDE] set g_damping =", meta["agent_cfg"]["world"]["g_damping"])


    agent, decoder, env = build_agent_from_meta(meta, device=device, max_steps_override=args.T)
    print("[CKPT meta] g_damping =", meta["agent_cfg"]["world"].get("g_damping", None))
    print("[Agent cfg] g_damping =", agent.cfg.world.g_damping)
    print("[WorldLatent cfg] g_damping =", agent.world.cfg.g_damping)

    agent.load_state_dict(ckpt["agent_state"])
    decoder.load_state_dict(ckpt["decoder_state"])
    agent.to(device).eval()
    decoder.to(device).eval()

    run_dir = Path(args.outdir) if args.outdir else (Path("outputs") / "runs" / timestamp_id())
    ensure_dir(run_dir)
    ensure_dir(run_dir / "figs")

    run_meta = {
        "mode": "switch_sweep",
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

    # reset
    obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
    agent.reset(batch_size=1)
    last_action = 4
    n_actions = int(env.action_space.n)

    rows: List[Dict[str, Any]] = []

    # start in regime A
    regime = 0  # 0=A, 1=B
    set_env_zone_sigma(env, tuple(args.sigma_A))

    for t_global in range(int(args.T)):
        # schedule switching after warmup
        switched = 0
        if t_global >= int(args.warmup):
            k = (t_global - int(args.warmup)) // max(1, int(args.period))
            new_regime = int(k % 2)  # 0,1,0,1,...
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
        logits = out["logits"].squeeze(0).detach().cpu().numpy()  # policy logits (from s)

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

#%% old version (01.09-12)
# # cear_pilot/experiments/run_switch_sweep.py
# # -*- coding: utf-8 -*-
# """
# Collect ONE long episode under a regime-switch schedule (A<->B),
# optionally with action replay, and log both g-dynamics and policy snapshots.

# Outputs:
#   outputs/runs/<timestamp>/
#     traj.parquet (or traj.csv)
#     meta.json

# Key idea:
# - "Switch" is about environment statistics (zone_sigma), while keeping actions fixed (replay) if desired.
# - We log policy logits/probs even in replay mode by explicitly calling agent.policy(s_t.detach()).
# """

# from __future__ import annotations

# import argparse
# import json
# import time
# from pathlib import Path
# from typing import Dict, Any, List, Optional, Tuple

# import numpy as np
# import torch
# import torch.nn.functional as F

# from cear_pilot.envs.nzone_grid import NZoneGridEnv, NZoneConfig
# from cear_pilot.models.agent import CEARAgent, AgentConfig
# from cear_pilot.models.decoder import ObsDecoder, DecoderConfig


# def timestamp_id() -> str:
#     return time.strftime("%Y%m%d_%H%M%S")


# def ensure_dir(p: Path) -> None:
#     p.mkdir(parents=True, exist_ok=True)


# def try_save_table(rows: List[Dict[str, Any]], out_path: Path) -> Path:
#     import pandas as pd
#     df = pd.DataFrame(rows)
#     try:
#         p = out_path.with_suffix(".parquet")
#         df.to_parquet(p, index=False)
#         return p
#     except Exception:
#         p = out_path.with_suffix(".csv")
#         df.to_csv(p, index=False)
#         return p


# def onehot(idx: int, n: int) -> np.ndarray:
#     v = np.zeros((n,), dtype=np.float32)
#     v[idx] = 1.0
#     return v


# def set_env_sigma(env: NZoneGridEnv, sigma: Tuple[float, float, float]) -> None:
#     # Prefer a dedicated setter if the env provides it.
#     if hasattr(env, "set_zone_sigma") and callable(getattr(env, "set_zone_sigma")):
#         env.set_zone_sigma(tuple(float(x) for x in sigma))
#         return
#     # Fallbacks (best-effort).
#     if hasattr(env, "config") and hasattr(env.config, "zone_sigma"):
#         env.config.zone_sigma = tuple(float(x) for x in sigma)
#         return
#     raise AttributeError("Env has no set_zone_sigma() and no config.zone_sigma to override.")


# def build_agent_from_meta(meta: Dict[str, Any], device: str, zone_sigma_override=None) -> tuple[CEARAgent, ObsDecoder, NZoneGridEnv]:
#     env_cfg = NZoneConfig(**meta["env_cfg"])
#     if zone_sigma_override is not None:
#         env_cfg.zone_sigma = tuple(float(x) for x in zone_sigma_override)
#     env = NZoneGridEnv(config=env_cfg)

#     agent_cfg = AgentConfig(device=device)

#     enc = meta["agent_cfg"]["encoder"]
#     world = meta["agent_cfg"]["world"]
#     state = meta["agent_cfg"]["state"]
#     pol = meta["agent_cfg"]["policy"]

#     # Wire dims from meta
#     agent_cfg.encoder.obs_dim = enc["obs_dim"]
#     agent_cfg.encoder.proprio_dim = enc["proprio_dim"]
#     agent_cfg.encoder.z_dim = enc["z_dim"]
#     agent_cfg.encoder.p_dim = enc["p_dim"]
#     agent_cfg.encoder.hidden = enc["hidden"]
#     agent_cfg.encoder.dropout = enc["dropout"]

#     agent_cfg.world.z_dim = world["z_dim"]
#     agent_cfg.world.p_dim = world["p_dim"]
#     agent_cfg.world.g_dim = world["g_dim"]
#     agent_cfg.world.g_damping = world["g_damping"]
#     agent_cfg.world.layernorm = world["layernorm"]

#     agent_cfg.state.z_dim = state["z_dim"]
#     agent_cfg.state.p_dim = state["p_dim"]
#     agent_cfg.state.g_dim = state["g_dim"]
#     agent_cfg.state.s_dim = state["s_dim"]
#     agent_cfg.state.hidden = state["hidden"]
#     agent_cfg.state.dropout = state["dropout"]
#     agent_cfg.state.g_influence = state["g_influence"]

#     agent_cfg.policy.s_dim = pol["s_dim"]
#     agent_cfg.policy.hidden = pol["hidden"]
#     agent_cfg.policy.n_actions = pol["n_actions"]
#     agent_cfg.policy.dropout = pol["dropout"]

#     agent = CEARAgent(agent_cfg)

#     dec_cfg = DecoderConfig(**meta["decoder_cfg"])
#     decoder = ObsDecoder(dec_cfg)
#     return agent, decoder, env


# def load_replay_actions(path: str) -> List[int]:
#     p = Path(path)
#     obj = json.loads(p.read_text())
#     if isinstance(obj, dict) and "actions" in obj:
#         acts = [int(a) for a in obj["actions"]]
#     elif isinstance(obj, list):
#         acts = [int(a) for a in obj]
#     else:
#         raise ValueError("replay_actions JSON must be a list or a dict with key 'actions'")
#     if len(acts) == 0:
#         raise ValueError("replay_actions is empty")
#     return acts


# def policy_stats_from_s(agent: CEARAgent, s_t: torch.Tensor) -> Dict[str, Any]:
#     """
#     Compute policy snapshot stats from state s_t.
#     This works even when env actions are forced (replay mode).
#     """
#     logits = agent.policy(s_t.detach())  # [B, A]
#     probs = F.softmax(logits, dim=-1)

#     # Entropy per batch item
#     ent = -(probs * (probs.clamp_min(1e-12)).log()).sum(dim=-1)

#     # Top-1, top-2 margin
#     top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
#     pi_max = top2[:, 0]
#     pi_2 = top2[:, 1] if top2.shape[-1] > 1 else torch.zeros_like(pi_max)
#     margin = pi_max - pi_2

#     argmax = torch.argmax(probs, dim=-1)

#     return {
#         "logits": logits,
#         "probs": probs,
#         "pi_max": float(pi_max.item()),
#         "pi_entropy": float(ent.item()),
#         "pi_margin": float(margin.item()),
#         "pi_argmax": int(argmax.item()),
#     }


# def main():
#     ap = argparse.ArgumentParser()

#     ap.add_argument("--ckpt", type=str, required=True)
#     ap.add_argument("--device", type=str, default="cpu")
#     ap.add_argument("--seed", type=int, default=0)
#     ap.add_argument("--steps", type=int, default=240)
#     ap.add_argument("--greedy", action="store_true")
#     ap.add_argument("--ablate_g", action="store_true")

#     # Base sigma and alternate sigma
#     ap.add_argument("--zone_sigma", type=float, nargs=3, required=True)
#     ap.add_argument("--zone_sigma2", type=float, nargs=3, required=True)

#     # Switching protocol
#     ap.add_argument("--pattern", type=str, default="toggle", choices=["toggle", "hysteresis"])
#     ap.add_argument("--t0", type=int, default=0, help="First time the switching schedule starts.")
#     ap.add_argument("--period", type=int, default=20, help="Toggle period for pattern=toggle.")
#     ap.add_argument("--t_switch", type=int, default=80, help="Switch time for pattern=hysteresis (A->B).")
#     ap.add_argument("--t_back", type=int, default=160, help="Back time for pattern=hysteresis (B->A).")

#     # Optional action replay
#     ap.add_argument("--replay_actions", type=str, default="", help="Path to JSON list/dict of actions.")
#     ap.add_argument("--outdir", type=str, default="")

#     args = ap.parse_args()

#     ckpt = torch.load(args.ckpt, map_location=args.device)
#     meta = ckpt["meta"]

#     agent, decoder, env = build_agent_from_meta(meta, device=args.device, zone_sigma_override=args.zone_sigma)
#     agent.load_state_dict(ckpt["agent_state"])
#     decoder.load_state_dict(ckpt["decoder_state"])
#     agent.to(args.device).eval()
#     decoder.to(args.device).eval()

#     run_dir = Path(args.outdir) if args.outdir else (Path("outputs") / "runs" / timestamp_id())
#     ensure_dir(run_dir)
#     ensure_dir(run_dir / "figs")

#     replay_actions: Optional[List[int]] = None
#     if str(args.replay_actions).strip():
#         replay_actions = load_replay_actions(args.replay_actions)

#     run_meta = {
#         "mode": "switch_sweep",
#         "ckpt": str(Path(args.ckpt).resolve()),
#         "seed": args.seed,
#         "device": args.device,
#         "steps": args.steps,
#         "greedy": bool(args.greedy),
#         "ablate_g": bool(args.ablate_g),
#         "zone_sigma": tuple(float(x) for x in args.zone_sigma),
#         "zone_sigma2": tuple(float(x) for x in args.zone_sigma2),
#         "pattern": args.pattern,
#         "t0": args.t0,
#         "period": args.period,
#         "t_switch": args.t_switch,
#         "t_back": args.t_back,
#         "replay_actions": str(args.replay_actions),
#         "train_meta": meta,
#     }
#     (run_dir / "meta.json").write_text(json.dumps(run_meta, indent=2))

#     rng = np.random.default_rng(args.seed)
#     obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
#     agent.reset(batch_size=1)
#     last_action = 4  # stay
#     n_actions = int(env.action_space.n)

#     rows: List[Dict[str, Any]] = []
#     done = False

#     # Track regime id: 0 => sigma(A), 1 => sigma2(B)
#     regime_id = 0
#     sigma_A = tuple(float(x) for x in args.zone_sigma)
#     sigma_B = tuple(float(x) for x in args.zone_sigma2)

#     def update_regime(t: int) -> int:
#         nonlocal regime_id
#         if args.pattern == "toggle":
#             if t < args.t0:
#                 return regime_id
#             # Toggle every 'period' steps starting at t0
#             k = (t - args.t0) // max(1, args.period)
#             new_id = int(k % 2)
#             if new_id != regime_id:
#                 regime_id = new_id
#                 set_env_sigma(env, sigma_A if regime_id == 0 else sigma_B)
#             return regime_id

#         # hysteresis: A -> B -> A
#         if t == args.t_switch:
#             regime_id = 1
#             set_env_sigma(env, sigma_B)
#         if t == args.t_back:
#             regime_id = 0
#             set_env_sigma(env, sigma_A)
#         return regime_id

#     # Ensure env starts in A
#     set_env_sigma(env, sigma_A)

#     t = 0
#     while (not done) and (t < args.steps):
#         # Apply regime schedule BEFORE consuming current step
#         update_regime(t)

#         x_t = torch.tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
#         p_t = torch.tensor(onehot(last_action, n_actions), dtype=torch.float32, device=args.device).unsqueeze(0)

#         if replay_actions is None:
#             with torch.no_grad():
#                 action, out = agent.step(x_t, p_t, greedy=args.greedy, ablate_g=args.ablate_g)
#             a_int = int(action.item())
#         else:
#             if t >= len(replay_actions):
#                 break
#             a_int = int(replay_actions[t])
#             with torch.no_grad():
#                 out = agent.forward_step(x_t, p_t, ablate_g=args.ablate_g)

#         # Policy snapshot (always computed from s_t)
#         with torch.no_grad():
#             pol = policy_stats_from_s(agent, out["s"])

#         obs_next, _, terminated, truncated, info2 = env.step(a_int)

#         g = out["g"].squeeze(0).cpu().numpy()
#         s = out["s"].squeeze(0).cpu().numpy()
#         z = out["z"].squeeze(0).cpu().numpy()

#         # Action-specific logits/prob for the executed action
#         logits_act = float(pol["logits"][0, a_int].item())
#         pi_act = float(pol["probs"][0, a_int].item())

#         row: Dict[str, Any] = {
#             "episode": 0,
#             "t": int(info2.get("t", t)),
#             "x": int(info2.get("x", -1)),
#             "y": int(info2.get("y", -1)),
#             "zone_id": int(info2.get("zone_id", -1)),
#             "regime_id": int(regime_id),
#             "action": int(a_int),

#             # Policy snapshot stats (requested)
#             "logits_act": logits_act,
#             "pi_act": pi_act,
#             "pi_max": float(pol["pi_max"]),
#             "pi_entropy": float(pol["pi_entropy"]),
#             "pi_argmax": int(pol["pi_argmax"]),
#             "pi_margin": float(pol["pi_margin"]),
#         }

#         for i, v in enumerate(obs.astype(np.float32)):
#             row[f"obs_{i}"] = float(v)
#         for i, v in enumerate(z):
#             row[f"z_{i}"] = float(v)
#         for i, v in enumerate(s):
#             row[f"s_{i}"] = float(v)
#         for i, v in enumerate(g):
#             row[f"g_{i}"] = float(v)

#         rows.append(row)

#         obs = obs_next
#         last_action = a_int
#         done = bool(terminated or truncated)
#         t += 1

#     saved_path = try_save_table(rows, run_dir / "traj")
#     print(f"[OK] Saved switch-sweep traj to: {saved_path}")
#     print(f"[OK] Run dir: {run_dir}")


# if __name__ == "__main__":
#     main()
