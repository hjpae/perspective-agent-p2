# cear_pilot/testing.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cear_pilot.envs.nzone_grid import NZoneConfig, NZoneGridEnv
from cear_pilot.models.agent import CEARAgent, AgentConfig
from cear_pilot.models.decoder import ObsDecoder, DecoderConfig


def onehot(indices: torch.Tensor, n: int) -> torch.Tensor:
    return F.one_hot(indices.long(), num_classes=n).float()


@torch.no_grad()
def make_proprio_from_last_action(last_action: int, n_actions: int, device: torch.device) -> torch.Tensor:
    a = torch.tensor([last_action], device=device)
    return onehot(a, n_actions)


def parse_sigma_list(sigmas: list[str]) -> list[tuple[float, float, float]]:
    out = []
    for s in sigmas:
        parts = s.split(",")
        if len(parts) != 3:
            raise ValueError(f"--sigmas expects 'a,b,c' but got: {s}")
        out.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return out


def build_agent_and_decoder_from_meta(meta: dict, device: torch.device):
    # Rebuild AgentConfig from meta
    agent_cfg = AgentConfig(device=str(device))
    for k, v in meta["agent_cfg"]["encoder"].items():
        setattr(agent_cfg.encoder, k, v)
    for k, v in meta["agent_cfg"]["world"].items():
        setattr(agent_cfg.world, k, v)
    for k, v in meta["agent_cfg"]["state"].items():
        setattr(agent_cfg.state, k, v)
    for k, v in meta["agent_cfg"]["policy"].items():
        setattr(agent_cfg.policy, k, v)

    agent = CEARAgent(agent_cfg).to(device)

    dec_cfg = DecoderConfig(**meta["decoder_cfg"])
    decoder = ObsDecoder(dec_cfg).to(device)
    return agent, decoder


def load_ckpt(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt.get("meta", None)
    if meta is None:
        raise ValueError("ckpt.meta not found. Your train.py saves meta; did you load the right ckpt?")

    agent, decoder = build_agent_and_decoder_from_meta(meta, device)
    agent.load_state_dict(ckpt["agent_state"], strict=True)
    decoder.load_state_dict(ckpt["decoder_state"], strict=True)

    agent.eval()
    decoder.eval()
    for p in agent.parameters():
        p.requires_grad_(False)
    for p in decoder.parameters():
        p.requires_grad_(False)

    return agent, decoder, meta


@torch.no_grad()
def rollout_greedy_actions(agent: CEARAgent, env: NZoneGridEnv, device: torch.device, steps: int, seed: int):
    """Make a reference action sequence using greedy policy on this env."""
    obs, info = env.reset(seed=seed)
    n_actions = int(env.action_space.n)

    agent.reset(batch_size=1)
    last_action = 4  # stay

    actions = []
    for t in range(steps):
        x_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        p_t = make_proprio_from_last_action(last_action, n_actions, device=device)

        out = agent.forward_step(x_t, p_t, ablate_g=False)
        s_t = out["s"]
        logits = agent.policy(s_t.detach())

        a_int = int(torch.argmax(logits, dim=-1).item())
        actions.append(a_int)

        obs, _, terminated, truncated, info = env.step(a_int)
        last_action = a_int
        if terminated or truncated:
            break

    return actions


@torch.no_grad()
def rollout_replay(agent: CEARAgent, env: NZoneGridEnv, device: torch.device, actions: list[int], seed: int):
    """Replay given actions, while agent updates g via forward_step from observations."""
    obs, info = env.reset(seed=seed)
    n_actions = int(env.action_space.n)

    agent.reset(batch_size=1)
    last_action = 4

    traj = {
        "a": [],
        "x": [],
        "y": [],
        "zone": [],
        "g_norm": [],
        "g": [],
    }

    for a_int in actions:
        # update agent with current obs + proprio
        x_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        p_t = make_proprio_from_last_action(last_action, n_actions, device=device)

        out = agent.forward_step(x_t, p_t, ablate_g=False)
        g_t = out["g"]

        # step env with replay action
        obs, _, terminated, truncated, info = env.step(int(a_int))

        g_vec = g_t.squeeze(0).detach().cpu().numpy()

        traj["a"].append(int(a_int))
        traj["x"].append(int(info["x"]))
        traj["y"].append(int(info["y"]))
        traj["zone"].append(int(info["zone_id"]))
        traj["g_norm"].append(float(np.linalg.norm(g_vec)))
        traj["g"].append(g_vec.tolist())

        last_action = int(a_int)
        if terminated or truncated:
            break

    return traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=240)

    # sigma conditions: pass multiple triplets like "0.60, 0.30, 0.05"
    ap.add_argument(
        "--sigmas",
        type=str,
        nargs="+",
        required=True,
        help='Sigma triplets: e.g. --sigmas "0.60, 0.30, 0.05" "0.05, 0.30, 0.60" "0.30, 0.30, 0.30"',
    )
    ap.add_argument("--outdir", type=str, default="outputs/tests_sigma_demo")

    args = ap.parse_args()
    device = torch.device(args.device)

    ckpt_path = Path(args.ckpt)
    agent, decoder, meta = load_ckpt(ckpt_path, device)

    sigma_list = parse_sigma_list(args.sigmas)

    # Use meta env cfg as base, but override zone_sigma per condition
    base_env_cfg = meta["env_cfg"]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) build actions_ref from the *first* sigma condition (canonical A)
    sigma_A = sigma_list[0]
    cfgA = NZoneConfig(**base_env_cfg)
    cfgA.zone_sigma = sigma_A
    envA = NZoneGridEnv(cfgA)

    actions_ref = rollout_greedy_actions(agent, envA, device, steps=args.steps, seed=args.seed)

    (outdir / "actions_ref.json").write_text(json.dumps({"sigma_A": sigma_A, "actions": actions_ref}, indent=2))

    # 2) replay the same actions on each sigma condition and save trajectories
    index = []
    for sig in sigma_list:
        cfg = NZoneConfig(**base_env_cfg)
        cfg.zone_sigma = sig
        env = NZoneGridEnv(cfg)

        traj = rollout_replay(agent, env, device, actions_ref, seed=args.seed)

        tag = f"sigma_{sig[0]:.3f}_{sig[1]:.3f}_{sig[2]:.3f}"
        (outdir / f"{tag}.json").write_text(json.dumps(traj, indent=2))

        index.append(
            {
                "tag": tag,
                "sigma": sig,
                "T": len(traj["a"]),
                "g_norm_mean": float(np.mean(traj["g_norm"])) if len(traj["g_norm"]) else None,
                "g_norm_max": float(np.max(traj["g_norm"])) if len(traj["g_norm"]) else None,
                "zone_frac": (
                    np.bincount(np.array(traj["zone"], dtype=np.int64), minlength=3) / max(1, len(traj["zone"]))
                ).tolist(),
            }
        )

    (outdir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"[OK] Saved demo rollouts to: {outdir}")
    print("Index summary:")
    for row in index:
        print(row)


if __name__ == "__main__":
    main()
