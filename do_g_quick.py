#!/usr/bin/env python3
# do_g_quick.py
"""
Quick do(g) on regime test results.

Fixes from previous version:
  1. Uses final g from training trajectory (not freshly harvested)
  2. PE-based measurement: compares prediction error against ACTUAL next obs
     (this makes decoder measurement probe-dependent!)

Usage:
    python do_g_quick.py --regime_root outputs/regime_test_YYYYMMDD
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_THIS = Path(__file__).resolve()
for _p in (_THIS.parents[1], _THIS.parents[0]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cear_pilot.envs.nzone_phase2 import NZonePhase2Config, NZonePhase2Env
from cear_pilot.models.agent import CEARAgent, AgentConfig
from cear_pilot.models.decoder import ObsDecoder, DecoderConfig

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.2, "font.size": 11, "figure.dpi": 120,
})
SAVE_DPI = 180

ERR_FEATURE_NAMES = [
    "pred_err_now_ratio", "pred_err_ema_short_ratio",
    "pred_err_ema_long_log", "pred_err_rise_ratio",
    "recent_event_trace", "recent_consequence_trace",
    "current_event_flag", "c_state_signed",
    "supportive_flag", "misleading_flag",
]


def load_agent_decoder(ckpt_path, device):
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
        agent_cfg.world.update_mode = str(a.get("update_mode", "fixed"))
        agent_cfg.world.alpha_fixed = float(a.get("alpha_fixed", 0.10))
        agent_cfg.world.energy_mode = str(a.get("energy_mode", "prototype_wells"))
        agent_cfg.world.dyn_eta = float(a.get("dyn_eta", 0.10))
        agent_cfg.world.n_prototypes = int(a.get("n_prototypes", 2))

    agent_cfg.world.use_error_feedback = True
    agent_cfg.world.err_dim = len(ERR_FEATURE_NAMES)

    agent = CEARAgent(agent_cfg)
    agent.load_state_dict(ckpt["agent_state"], strict=False)

    dec_cfg = DecoderConfig(**base["decoder_cfg"])
    decoder = ObsDecoder(dec_cfg)
    decoder.load_state_dict(ckpt["decoder_state"], strict=False)
    return agent, decoder, meta


def extract_final_g(traj_path, g_cols):
    """Get the g vector from the last timestep of the last episode."""
    if traj_path.suffix == ".parquet":
        df = pd.read_parquet(traj_path)
    else:
        df = pd.read_csv(traj_path)
    last_ep = df["episode"].max()
    last_row = df[(df["episode"] == last_ep) & (df["t"] == df[df["episode"] == last_ep]["t"].max())]
    g_vec = last_row[g_cols].values[0]
    return g_vec


def make_env(valence, schedule, max_steps, meta):
    env_cfg_data = meta.get("env_cfg", meta.get("base_meta", {}).get("env_cfg", {}))
    env_cfg = NZonePhase2Config(**env_cfg_data) if env_cfg_data else NZonePhase2Config()
    env_cfg.valence_sequence = valence
    env_cfg.schedule_pattern = schedule
    env_cfg.max_steps = max_steps
    return NZonePhase2Env(config=env_cfg)


def run_do_g(agent, decoder, env, g_dict, n_episodes, seed, device):
    """
    Run episodes and at each step:
      1. Get actual next observation
      2. For each g in g_dict, compute decoder prediction
      3. Measure PE = ||prediction - actual_next||^2 for each g
      4. Also measure policy logits difference
    """
    n_actions = int(env.action_space.n)
    rows = []

    for ep in range(n_episodes):
        agent.reset(batch_size=1)
        obs, info = env.reset(seed=seed + ep)
        last_action = 4
        done = False
        t = 0

        while not done:
            x_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            p_t = F.one_hot(torch.tensor([last_action], device=device), num_classes=n_actions).float()
            err_t = torch.zeros((1, len(ERR_FEATURE_NAMES)), device=device)

            # Step agent normally to get z_t, p_emb
            with torch.no_grad():
                out = agent.forward_step(x_t, p_t, err_t=err_t)
                action = agent.policy.sample_action(out["logits"], greedy=True)
                z_t = out["z"].detach()
                p_emb = out["p_emb"].detach() if "p_emb" in out else p_t

            a_int = int(action.item())
            obs_next, _, terminated, truncated, info_next = env.step(a_int)

            # Actual next observation (ground truth)
            x_next = torch.tensor(obs_next, dtype=torch.float32, device=device).unsqueeze(0)

            # Classify probe
            event_now = int(info.get("event_now", 0))
            consequence_now = int(info.get("consequence_now", 0))
            c_state = float(info.get("c_state", 0.0))
            if event_now:
                probe_type = "event"
            elif consequence_now:
                probe_type = "consequence"
            elif abs(c_state) > 0.1:
                probe_type = "active_regime"
            else:
                probe_type = "quiet"

            row = {
                "episode": ep, "t": t, "probe_type": probe_type,
                "action": a_int, "c_state": c_state,
            }

            # ── For each g condition, measure PE and policy ──
            a_onehot = F.one_hot(torch.tensor([a_int], device=device), num_classes=n_actions).float()

            for g_label, g_vec in g_dict.items():
                g_tensor = torch.tensor(g_vec, dtype=torch.float32, device=device).unsqueeze(0)

                with torch.no_grad():
                    # Decoder: predict next obs given g and actual action taken
                    pred_next = decoder(g_tensor, a_onehot)  # (1, obs_dim)
                    pe = float(F.mse_loss(pred_next, x_next).item())

                    # Decoder: predict all actions
                    pred_all = decoder.predict_all_actions(g_tensor)  # (1, n_actions, obs_dim)
                    # PE for each action (against actual next obs)
                    pe_per_action = torch.mean((pred_all.squeeze(0) - x_next) ** 2, dim=-1)  # (n_actions,)

                    # Best-action PE (which action would have predicted best?)
                    pe_best_action = float(pe_per_action.min().item())
                    best_action = int(pe_per_action.argmin().item())

                    # Policy logits
                    s_t = agent.state(z_t, p_emb, g_tensor)
                    logits = agent.policy(s_t)
                    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

                row[f"pe_{g_label}"] = pe
                row[f"pe_best_{g_label}"] = pe_best_action
                row[f"best_action_{g_label}"] = best_action
                row[f"entropy_{g_label}"] = float(-np.sum(probs * np.log(probs + 1e-9)))
                for ai in range(n_actions):
                    row[f"prob_a{ai}_{g_label}"] = float(probs[ai])
                    row[f"pe_a{ai}_{g_label}"] = float(pe_per_action[ai].item())

            # ── Pairwise comparisons ──
            g_labels = list(g_dict.keys())
            for i in range(len(g_labels)):
                for j in range(i + 1, len(g_labels)):
                    la, lb = g_labels[i], g_labels[j]
                    pair = f"{la}_vs_{lb}"
                    row[f"pe_diff_{pair}"] = row[f"pe_{la}"] - row[f"pe_{lb}"]
                    row[f"pe_best_diff_{pair}"] = row[f"pe_best_{la}"] - row[f"pe_best_{lb}"]

                    # Policy divergence
                    pa = np.array([row[f"prob_a{ai}_{la}"] for ai in range(n_actions)])
                    pb = np.array([row[f"prob_a{ai}_{lb}"] for ai in range(n_actions)])
                    row[f"policy_tvd_{pair}"] = float(0.5 * np.sum(np.abs(pa - pb)))
                    row[f"policy_kl_{pair}"] = float(np.sum(pa * np.log((pa + 1e-9) / (pb + 1e-9))))

            rows.append(row)

            obs = obs_next
            info = info_next
            last_action = a_int
            t += 1
            done = bool(terminated or truncated)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime_root", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--probe_episodes", type=int, default=5)
    ap.add_argument("--probe_seed", type=int, default=7000)
    args = ap.parse_args()

    root = Path(args.regime_root) / "runs"
    device = args.device
    outdir = Path(args.regime_root) / "do_g_results"
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Find SSSS and MMMM runs ──
    ssss_dir = mmmm_dir = None
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if "_SSSS_" in d.name:
            ssss_dir = d
        elif "_MMMM_" in d.name:
            mmmm_dir = d

    if not ssss_dir or not mmmm_dir:
        print("ERROR: Need both SSSS and MMMM runs")
        return

    print(f"SSSS: {ssss_dir.name}")
    print(f"MMMM: {mmmm_dir.name}")

    # ── Detect g columns ──
    traj_ssss = ssss_dir / "traj.parquet"
    if not traj_ssss.exists():
        traj_ssss = ssss_dir / "traj.csv"
    sample_df = pd.read_parquet(traj_ssss) if traj_ssss.suffix == ".parquet" else pd.read_csv(traj_ssss)
    g_cols = sorted([c for c in sample_df.columns if re.match(r"g_\d+$", c)],
                    key=lambda x: int(x.split("_")[-1]))
    print(f"G dims: {len(g_cols)}")

    # ── Extract final g from each training run ──
    traj_mmmm = mmmm_dir / "traj.parquet"
    if not traj_mmmm.exists():
        traj_mmmm = mmmm_dir / "traj.csv"

    g_ssss = extract_final_g(traj_ssss, g_cols)
    g_mmmm = extract_final_g(traj_mmmm, g_cols)
    g_zero = np.zeros_like(g_ssss)

    print(f"\n||g_ssss||={np.linalg.norm(g_ssss):.4f}")
    print(f"||g_mmmm||={np.linalg.norm(g_mmmm):.4f}")
    print(f"||g_ssss - g_mmmm||={np.linalg.norm(g_ssss - g_mmmm):.4f}")
    print(f"cosine={np.dot(g_ssss, g_mmmm) / (np.linalg.norm(g_ssss) * np.linalg.norm(g_mmmm) + 1e-8):.4f}")

    g_dict = {"ssss": g_ssss, "mmmm": g_mmmm, "zero": g_zero}

    # ── Load agent (use SSSS checkpoint — decoder is frozen/same) ──
    ckpt_path = ssss_dir / "ckpt_final.pt"
    agent, decoder, meta = load_agent_decoder(str(ckpt_path), device)
    agent.to(device).eval()
    decoder.to(device).eval()

    # ── Run probes in BOTH environments ──
    for env_valence in ["SSSS", "MMMM"]:
        print(f"\n=== Probing in {env_valence} environment ===")
        env = make_env(env_valence, "1-1-1-1", 300, meta)
        results = run_do_g(agent, decoder, env, g_dict,
                           n_episodes=args.probe_episodes,
                           seed=args.probe_seed, device=device)

        results["probe_env"] = env_valence
        results.to_csv(outdir / f"do_g_probes_{env_valence}.csv", index=False)

        # ── Print summary ──
        print(f"\n  Probe types: {dict(results['probe_type'].value_counts())}")

        for pair in ["ssss_vs_mmmm", "ssss_vs_zero", "mmmm_vs_zero"]:
            pe_col = f"pe_diff_{pair}"
            tvd_col = f"policy_tvd_{pair}"
            if pe_col in results.columns:
                print(f"\n  {pair}:")
                print(f"    PE diff: mean={results[pe_col].mean():.6f} std={results[pe_col].std():.6f}")
                if tvd_col in results.columns:
                    print(f"    Policy TVD: mean={results[tvd_col].mean():.6f} std={results[tvd_col].std():.6f}")

                # By probe type
                for pt in results["probe_type"].unique():
                    sub = results[results["probe_type"] == pt]
                    print(f"    [{pt:>14s}] PE diff={sub[pe_col].mean():.6f}±{sub[pe_col].std():.6f}  "
                          f"TVD={sub[tvd_col].mean():.6f}" if tvd_col in sub.columns else "")

    # ── Combined analysis ──
    all_results = []
    for env_val in ["SSSS", "MMMM"]:
        p = outdir / f"do_g_probes_{env_val}.csv"
        if p.exists():
            all_results.append(pd.read_csv(p))

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(outdir / "do_g_combined.csv", index=False)

        # ── Figures ──
        print("\n=== Generating figures ===")

        # Fig 1: PE with g_ssss vs g_mmmm, by probe environment
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ei, env_val in enumerate(["SSSS", "MMMM"]):
            ax = axes[ei]
            sub = combined[combined["probe_env"] == env_val]
            for g_label, color in [("ssss", "#185FA5"), ("mmmm", "#D85A30"), ("zero", "#888")]:
                pe_col = f"pe_{g_label}"
                if pe_col not in sub.columns:
                    continue
                # Group by probe type
                for pi, pt in enumerate(["quiet", "active_regime", "event", "consequence"]):
                    pt_data = sub[sub["probe_type"] == pt][pe_col]
                    if len(pt_data) == 0:
                        continue
                    offset = {"ssss": -0.2, "mmmm": 0, "zero": 0.2}[g_label]
                    ax.bar(pi + offset, pt_data.mean(), 0.18, color=color, alpha=0.7,
                           label=f"g={g_label}" if pi == 0 else "")
            ax.set_xticks(range(4))
            ax.set_xticklabels(["quiet", "active\nregime", "event", "consequence"], fontsize=9)
            ax.set_ylabel("Prediction error")
            ax.set_title(f"Probing in {env_val} env")
            if ei == 0:
                ax.legend(fontsize=8)
        fig.suptitle("do(g): PE with different perspectives in different worlds", fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(outdir / "do_g_pe_by_perspective.png", dpi=SAVE_DPI, bbox_inches="tight")
        plt.close()
        print(f"  [fig] do_g_pe_by_perspective.png")

        # Fig 2: PE difference (g_ssss - g_mmmm) by probe type and env
        fig, ax = plt.subplots(figsize=(8, 5))
        pe_diff_col = "pe_diff_ssss_vs_mmmm"
        if pe_diff_col in combined.columns:
            for ei, (env_val, color) in enumerate([("SSSS", "#185FA5"), ("MMMM", "#D85A30")]):
                sub = combined[combined["probe_env"] == env_val]
                pts = ["quiet", "active_regime", "event", "consequence"]
                means, stds = [], []
                for pt in pts:
                    pt_data = sub[sub["probe_type"] == pt][pe_diff_col]
                    means.append(pt_data.mean() if len(pt_data) > 0 else 0)
                    stds.append(pt_data.std() if len(pt_data) > 0 else 0)
                offset = -0.15 if ei == 0 else 0.15
                ax.bar(np.arange(len(pts)) + offset, means, 0.28, yerr=stds,
                       color=color, alpha=0.7, capsize=3, label=f"In {env_val} env")
            ax.set_xticks(range(len(pts)))
            ax.set_xticklabels(["quiet", "active\nregime", "event", "consequence"], fontsize=9)
            ax.set_ylabel("PE(g_ssss) - PE(g_mmmm)")
            ax.axhline(0, color="#ccc", lw=0.5)
            ax.set_title("Interpretive dissociation: which perspective predicts better?")
            ax.legend()
            fig.tight_layout()
            fig.savefig(outdir / "do_g_pe_difference.png", dpi=SAVE_DPI, bbox_inches="tight")
            plt.close()
            print(f"  [fig] do_g_pe_difference.png")

        # Fig 3: Policy divergence
        fig, ax = plt.subplots(figsize=(8, 5))
        tvd_col = "policy_tvd_ssss_vs_mmmm"
        if tvd_col in combined.columns:
            for ei, (env_val, color) in enumerate([("SSSS", "#185FA5"), ("MMMM", "#D85A30")]):
                sub = combined[combined["probe_env"] == env_val]
                pts = ["quiet", "active_regime", "event", "consequence"]
                means = []
                for pt in pts:
                    pt_data = sub[sub["probe_type"] == pt][tvd_col]
                    means.append(pt_data.mean() if len(pt_data) > 0 else 0)
                offset = -0.15 if ei == 0 else 0.15
                ax.bar(np.arange(len(pts)) + offset, means, 0.28,
                       color=color, alpha=0.7, label=f"In {env_val} env")
            ax.set_xticks(range(len(pts)))
            ax.set_xticklabels(["quiet", "active\nregime", "event", "consequence"], fontsize=9)
            ax.set_ylabel("Policy TVD (g_ssss vs g_mmmm)")
            ax.set_title("Behavioral divergence under perspective swap")
            ax.legend()
            fig.tight_layout()
            fig.savefig(outdir / "do_g_policy_divergence.png", dpi=SAVE_DPI, bbox_inches="tight")
            plt.close()
            print(f"  [fig] do_g_policy_divergence.png")

    print(f"\nDone. Output: {outdir}")


if __name__ == "__main__":
    main()