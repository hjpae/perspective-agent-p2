# cear_pilot/experiments/do_g_experiment.py
# -*- coding: utf-8 -*-
"""
do(g) experiment: interpretive dissociation.

Core question: same observation, same encoder, same policy weights —
does swapping g change decoder prediction more than policy logits?

Protocol:
  1. Harvest g_ssss and g_mmmm from the SAME checkpoint by running
     carry_g episodes in SSSS vs MMMM environments
  2. Collect probe observations from a neutral episode
  3. For each probe, inject g_ssss / g_mmmm / g_zero and measure
     decoder prediction difference + policy logit difference

Usage:
    python -m cear_pilot.experiments.do_g_experiment \
        --sweep_root outputs/experiment_full_YYYYMMDD \
        --outdir outputs/experiment_full_YYYYMMDD/analysis_do_g
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_THIS_FILE = Path(__file__).resolve()
for _p in (_THIS_FILE.parents[2], _THIS_FILE.parents[1], _THIS_FILE.parent):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from cear_pilot.envs.nzone_phase2 import NZonePhase2Config, NZonePhase2Env
from cear_pilot.models.agent import CEARAgent, AgentConfig
from cear_pilot.models.decoder import ObsDecoder, DecoderConfig

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.2,
    "font.size": 11,
    "figure.dpi": 120,
})
SAVE_DPI = 180

ERR_FEATURE_NAMES = [
    "pred_err_now_ratio", "pred_err_ema_short_ratio",
    "pred_err_ema_long_log", "pred_err_rise_ratio",
    "recent_event_trace", "recent_consequence_trace",
    "current_event_flag", "c_state_signed",
    "supportive_flag", "misleading_flag",
]


def savefig(fig, path, title=""):
    fig.tight_layout()
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {path.name}" + (f" — {title}" if title else ""))


def parse_label(name):
    m = re.match(r"a([\d.]+)_(SSSS|MMMM)_s(\d+)", name)
    if not m:
        return None
    return {"alpha": float(m.group(1)), "valence": m.group(2), "seed": int(m.group(3))}


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
        agent_cfg.world.energy_mode = str(a.get("energy_mode", agent_cfg.world.energy_mode))
        agent_cfg.world.dyn_eta = float(a.get("dyn_eta", agent_cfg.world.dyn_eta))
        agent_cfg.world.confine_lambda = float(a.get("confine_lambda", agent_cfg.world.confine_lambda))
        agent_cfg.world.n_prototypes = int(a.get("n_prototypes", agent_cfg.world.n_prototypes))
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


def make_env(valence, schedule, max_steps, meta):
    env_cfg_data = meta.get("env_cfg", meta.get("base_meta", {}).get("env_cfg", {}))
    env_cfg = NZonePhase2Config(**env_cfg_data) if env_cfg_data else NZonePhase2Config()
    env_cfg.valence_sequence = valence
    env_cfg.schedule_pattern = schedule
    env_cfg.max_steps = max_steps
    return NZonePhase2Env(config=env_cfg)


def build_zero_err(device, err_dim=10):
    return torch.zeros((1, err_dim), device=device, dtype=torch.float32)


# ═══════════════════════════════════════════════════════
# Step 1: Harvest g from carry_g episodes
# ═══════════════════════════════════════════════════════

def harvest_g(agent, decoder, env, n_episodes, seed, device):
    """Run n_episodes with carry_g, return final g and per-episode g snapshots."""
    n_actions = int(env.action_space.n)
    agent.reset(batch_size=1)

    g_snapshots = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        last_action = 4
        done = False

        while not done:
            x_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            p_t = F.one_hot(torch.tensor([last_action], device=device), num_classes=n_actions).float()
            err_t = build_zero_err(device)

            with torch.no_grad():
                action, out = agent.step(x_t, p_t, greedy=True, err_t=err_t)

            a_int = int(action.item())
            obs, _, terminated, truncated, info = env.step(a_int)
            last_action = a_int
            done = bool(terminated or truncated)

        g_end = agent.get_latents()["g"].detach().clone()
        g_snapshots.append(g_end.cpu().numpy()[0].copy())

    final_g = agent.get_latents()["g"].detach().clone()
    return final_g, g_snapshots


# ═══════════════════════════════════════════════════════
# Step 2: Collect probe observations
# ═══════════════════════════════════════════════════════

def collect_probes(agent, decoder, env, n_episodes, seed, device):
    """Run episodes and collect (obs, z, p_emb, info) at each step as probes."""
    n_actions = int(env.action_space.n)
    probes = []

    for ep in range(n_episodes):
        agent.reset(batch_size=1)
        obs, info = env.reset(seed=seed + ep)
        last_action = 4
        done = False
        t = 0

        while not done:
            x_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            p_t = F.one_hot(torch.tensor([last_action], device=device), num_classes=n_actions).float()
            err_t = build_zero_err(device)

            with torch.no_grad():
                out = agent.forward_step(x_t, p_t, err_t=err_t)
                action = agent.policy.sample_action(out["logits"], greedy=True)

            # Classify probe type
            event_now = int(info.get("event_now", 0))
            consequence_now = int(info.get("consequence_now", 0))
            if event_now:
                probe_type = "event"
            elif consequence_now:
                probe_type = "consequence"
            else:
                probe_type = "quiet"

            probes.append({
                "episode": ep,
                "t": t,
                "probe_type": probe_type,
                "x_t": x_t.detach().clone(),
                "z_t": out["z"].detach().clone(),
                "p_emb": out["p_emb"].detach().clone() if "p_emb" in out else p_t.clone(),
                "obs_raw": obs.copy(),
                "info": dict(info),
            })

            a_int = int(action.item())
            obs_next, _, terminated, truncated, info = env.step(a_int)
            obs = obs_next
            last_action = a_int
            t += 1
            done = bool(terminated or truncated)

    return probes


# ═══════════════════════════════════════════════════════
# Step 3: Intervention — measure decoder + policy shift
# ═══════════════════════════════════════════════════════

def intervene(agent, decoder, probes, g_dict, device):
    """
    For each probe, set g to each value in g_dict and measure:
      - decoder predictions (all actions)
      - policy logits
      - state head output

    g_dict: {"ssss": tensor, "mmmm": tensor, "zero": tensor, ...}
    """
    results = []
    n_actions = agent.cfg.policy.n_actions

    for pi, probe in enumerate(probes):
        z_t = probe["z_t"].to(device)
        p_emb = probe["p_emb"].to(device)

        row = {
            "probe_idx": pi,
            "episode": probe["episode"],
            "t": probe["t"],
            "probe_type": probe["probe_type"],
        }

        predictions = {}
        logits_all = {}

        for g_label, g_val in g_dict.items():
            g = g_val.to(device)

            with torch.no_grad():
                # Decoder: predict next obs for all actions
                pred_all = decoder.predict_all_actions(g)  # (1, n_actions, obs_dim)

                # State head + policy
                s_t = agent.state(z_t, p_emb, g)
                logits = agent.policy(s_t)  # (1, n_actions)

            predictions[g_label] = pred_all.cpu().numpy()[0]  # (n_actions, obs_dim)
            logits_all[g_label] = logits.cpu().numpy()[0]      # (n_actions,)

        # Compute pairwise differences
        g_labels = list(g_dict.keys())
        for i in range(len(g_labels)):
            for j in range(i + 1, len(g_labels)):
                la, lb = g_labels[i], g_labels[j]
                pair = f"{la}_vs_{lb}"

                # Decoder prediction difference (mean L2 across actions)
                pred_diff = predictions[la] - predictions[lb]
                decoder_shift = float(np.sqrt(np.mean(pred_diff ** 2)))

                # Per-action decoder difference
                per_action_shift = np.sqrt(np.mean(pred_diff ** 2, axis=1))  # (n_actions,)

                # Policy logit difference
                logit_diff = logits_all[la] - logits_all[lb]
                policy_shift = float(np.sqrt(np.mean(logit_diff ** 2)))

                # Policy probability difference (KL-ish)
                probs_a = np.exp(logits_all[la]) / np.exp(logits_all[la]).sum()
                probs_b = np.exp(logits_all[lb]) / np.exp(logits_all[lb]).sum()
                policy_tvd = float(0.5 * np.sum(np.abs(probs_a - probs_b)))

                row[f"decoder_shift_{pair}"] = decoder_shift
                row[f"policy_shift_{pair}"] = policy_shift
                row[f"policy_tvd_{pair}"] = policy_tvd
                for ai in range(n_actions):
                    row[f"decoder_shift_{pair}_a{ai}"] = float(per_action_shift[ai])

        results.append(row)

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════

def fig_interpretive_dissociation(results_by_condition, outdir):
    """
    Panel A: decoder shift by probe type (cross-basin > within-basin)
    Panel B: policy shift by probe type
    Panel C: decoder vs policy scatter
    """
    # Collect all results
    all_rows = []
    for key, df in results_by_condition.items():
        df = df.copy()
        df["condition_key"] = key
        # Parse alpha from key
        m = re.match(r"a([\d.]+)", key)
        if m:
            df["alpha"] = float(m.group(1))
        all_rows.append(df)

    if not all_rows:
        print("  [skip] No results for interpretive dissociation figure")
        return

    all_df = pd.concat(all_rows, ignore_index=True)

    # Find the cross-basin comparison column
    cross_col_dec = None
    cross_col_pol = None
    cross_col_tvd = None
    for c in all_df.columns:
        if c.startswith("decoder_shift_ssss_vs_mmmm"):
            if c == "decoder_shift_ssss_vs_mmmm":
                cross_col_dec = c
        if c.startswith("policy_shift_ssss_vs_mmmm"):
            if c == "policy_shift_ssss_vs_mmmm":
                cross_col_pol = c
        if c.startswith("policy_tvd_ssss_vs_mmmm"):
            if c == "policy_tvd_ssss_vs_mmmm":
                cross_col_tvd = c

    if cross_col_dec is None:
        print("  [skip] No ssss_vs_mmmm comparison found")
        return

    # ── Panel A+B: decoder vs policy shift by probe type ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    probe_types = ["quiet", "event", "consequence"]
    probe_colors = {"quiet": "#888", "event": "#185FA5", "consequence": "#D85A30"}

    # Panel A: decoder shift
    ax = axes[0]
    for pi, pt in enumerate(probe_types):
        pt_data = all_df[all_df["probe_type"] == pt]
        if len(pt_data) == 0:
            continue
        vals = pt_data[cross_col_dec].values
        ax.bar(pi, np.median(vals),
               yerr=[[np.median(vals) - np.percentile(vals, 25)],
                     [np.percentile(vals, 75) - np.median(vals)]],
               color=probe_colors[pt], alpha=0.7, capsize=4, width=0.6)
    ax.set_xticks(range(len(probe_types)))
    ax.set_xticklabels(probe_types)
    ax.set_ylabel("Decoder prediction shift (RMSE)")
    ax.set_title("Interpretive shift (decoder)")

    # Panel B: policy shift
    ax = axes[1]
    for pi, pt in enumerate(probe_types):
        pt_data = all_df[all_df["probe_type"] == pt]
        if len(pt_data) == 0:
            continue
        vals = pt_data[cross_col_pol].values
        ax.bar(pi, np.median(vals),
               yerr=[[np.median(vals) - np.percentile(vals, 25)],
                     [np.percentile(vals, 75) - np.median(vals)]],
               color=probe_colors[pt], alpha=0.7, capsize=4, width=0.6)
    ax.set_xticks(range(len(probe_types)))
    ax.set_xticklabels(probe_types)
    ax.set_ylabel("Policy logit shift (RMSE)")
    ax.set_title("Behavioral shift (policy)")

    # Panel C: decoder vs policy scatter
    ax = axes[2]
    for pt, color in probe_colors.items():
        pt_data = all_df[all_df["probe_type"] == pt]
        if len(pt_data) == 0:
            continue
        ax.scatter(pt_data[cross_col_dec], pt_data[cross_col_pol],
                   c=color, alpha=0.3, s=10, label=pt, edgecolors="none")
    # Diagonal reference
    lims = [0, max(all_df[cross_col_dec].max(), all_df[cross_col_pol].max()) * 1.1]
    ax.plot(lims, lims, "--", color="#ccc", lw=0.8, zorder=0)
    ax.set_xlabel("Decoder shift")
    ax.set_ylabel("Policy shift")
    ax.set_title("Interpretation vs behavior")
    ax.legend(fontsize=8, framealpha=0.7)

    fig.suptitle("do(g): same observation, different perspective",
                 fontsize=14, y=1.02)
    savefig(fig, outdir / "do_g_interpretive_dissociation.png",
            "Interpretive dissociation")


def fig_cross_vs_within_basin(results_by_condition, outdir):
    """Cross-basin shift vs within-basin shift: signal vs noise."""

    all_rows = []
    for key, df in results_by_condition.items():
        df = df.copy()
        df["condition_key"] = key
        all_rows.append(df)
    if not all_rows:
        return

    all_df = pd.concat(all_rows, ignore_index=True)

    # Find columns
    cross_dec = "decoder_shift_ssss_vs_mmmm"
    within_candidates = [c for c in all_df.columns
                         if c.startswith("decoder_shift_ssss_vs_ssss")
                         or c.startswith("decoder_shift_mmmm_vs_mmmm")]

    if cross_dec not in all_df.columns:
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    cross_vals = all_df[cross_dec].dropna().values
    ax.hist(cross_vals, bins=40, alpha=0.6, color="#534AB7",
            label=f"Cross-basin (SSSS vs MMMM)\nmedian={np.median(cross_vals):.4f}", density=True)

    for wc in within_candidates:
        within_vals = all_df[wc].dropna().values
        if len(within_vals) > 0:
            ax.hist(within_vals, bins=40, alpha=0.4, color="#888",
                    label=f"Within-basin\nmedian={np.median(within_vals):.4f}", density=True)
            break

    ax.set_xlabel("Decoder prediction shift (RMSE)")
    ax.set_ylabel("Density")
    ax.set_title("Cross-basin vs within-basin decoder shift")
    ax.legend(framealpha=0.7)

    savefig(fig, outdir / "do_g_cross_vs_within.png", "Cross vs within basin")


def fig_per_alpha(results_by_condition, outdir):
    """Decoder and policy shift as function of alpha."""

    alpha_decoder = {}
    alpha_policy = {}

    for key, df in results_by_condition.items():
        m = re.match(r"a([\d.]+)", key)
        if not m:
            continue
        alpha = float(m.group(1))

        cross_dec = "decoder_shift_ssss_vs_mmmm"
        cross_pol = "policy_shift_ssss_vs_mmmm"

        if cross_dec in df.columns:
            if alpha not in alpha_decoder:
                alpha_decoder[alpha] = []
            alpha_decoder[alpha].extend(df[cross_dec].dropna().tolist())

        if cross_pol in df.columns:
            if alpha not in alpha_policy:
                alpha_policy[alpha] = []
            alpha_policy[alpha].extend(df[cross_pol].dropna().tolist())

    if not alpha_decoder:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    alphas = sorted(alpha_decoder.keys())
    dec_med = [np.median(alpha_decoder[a]) for a in alphas]
    dec_q1 = [np.percentile(alpha_decoder[a], 25) for a in alphas]
    dec_q3 = [np.percentile(alpha_decoder[a], 75) for a in alphas]

    pol_med = [np.median(alpha_policy.get(a, [0])) for a in alphas]
    pol_q1 = [np.percentile(alpha_policy.get(a, [0]), 25) for a in alphas]
    pol_q3 = [np.percentile(alpha_policy.get(a, [0]), 75) for a in alphas]

    ax.plot(alphas, dec_med, "o-", color="#534AB7", lw=1.5, markersize=6,
            label="Decoder shift (interpretation)")
    ax.fill_between(alphas, dec_q1, dec_q3, color="#534AB7", alpha=0.12)

    ax.plot(alphas, pol_med, "s--", color="#D4537E", lw=1.5, markersize=6,
            label="Policy shift (behavior)")
    ax.fill_between(alphas, pol_q1, pol_q3, color="#D4537E", alpha=0.12)

    ax.set_xlabel("α (update rate)")
    ax.set_ylabel("Shift magnitude (RMSE)")
    ax.set_title("Interpretive vs behavioral sensitivity to perspective swap")
    ax.legend(framealpha=0.7)

    savefig(fig, outdir / "do_g_alpha_profile.png", "Alpha profile")


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def run_do_g_for_checkpoint(
    ckpt_path, device, meta, outdir,
    harvest_episodes=5, probe_episodes=3,
    harvest_seed=5000, probe_seed=6000,
    schedule="1-1-1-1", max_steps=300,
):
    """Run the full do(g) protocol for one checkpoint."""

    agent, decoder, meta = load_agent_decoder(str(ckpt_path), device)
    agent.to(device).eval()
    decoder.to(device).eval()

    # ── Harvest g_ssss ──
    print(f"    Harvesting g_ssss ({harvest_episodes} episodes)...")
    env_s = make_env("SSSS", schedule, max_steps, meta)
    g_ssss, g_ssss_snapshots = harvest_g(
        agent, decoder, env_s, harvest_episodes, harvest_seed, device)

    # ── Harvest g_mmmm (SAME checkpoint, different env) ──
    print(f"    Harvesting g_mmmm ({harvest_episodes} episodes)...")
    env_m = make_env("MMMM", schedule, max_steps, meta)
    g_mmmm, g_mmmm_snapshots = harvest_g(
        agent, decoder, env_m, harvest_episodes, harvest_seed + 500, device)

    # ── g_zero baseline ──
    g_zero = torch.zeros_like(g_ssss)

    g_dict = {"ssss": g_ssss, "mmmm": g_mmmm, "zero": g_zero}

    print(f"    ||g_ssss||={g_ssss.norm().item():.4f}  "
          f"||g_mmmm||={g_mmmm.norm().item():.4f}  "
          f"||g_ssss - g_mmmm||={torch.norm(g_ssss - g_mmmm).item():.4f}")

    # ── Collect probes ──
    print(f"    Collecting probes ({probe_episodes} episodes)...")
    env_probe = make_env("SSSS", schedule, max_steps, meta)
    probes = collect_probes(agent, decoder, env_probe, probe_episodes, probe_seed, device)
    print(f"    {len(probes)} probes collected")

    # Count probe types
    types = {}
    for p in probes:
        pt = p["probe_type"]
        types[pt] = types.get(pt, 0) + 1
    print(f"    Probe types: {types}")

    # ── Intervention ──
    print(f"    Running interventions...")
    results = intervene(agent, decoder, probes, g_dict, device)

    # ── Within-basin control: harvest second g_ssss with different seed ──
    print(f"    Harvesting g_ssss_b (within-basin control)...")
    g_ssss_b, _ = harvest_g(
        agent, decoder, env_s, harvest_episodes, harvest_seed + 100, device)

    g_dict_within = {"ssss": g_ssss, "ssss_b": g_ssss_b}
    results_within = intervene(agent, decoder, probes, g_dict_within, device)

    # Merge within-basin columns
    within_cols = [c for c in results_within.columns if "ssss_vs_ssss_b" in c]
    for c in within_cols:
        results[c] = results_within[c]

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_root", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--harvest_episodes", type=int, default=5)
    ap.add_argument("--probe_episodes", type=int, default=3)
    ap.add_argument("--schedule", type=str, default="1-1-1-1")
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--max_ckpts", type=int, default=0,
                    help="Max checkpoints to process (0=all)")
    args = ap.parse_args()

    sweep_root = Path(args.sweep_root)
    outdir = Path(args.outdir) if args.outdir else sweep_root / "analysis_do_g"
    outdir.mkdir(parents=True, exist_ok=True)

    runs_dir = sweep_root / "runs"
    if not runs_dir.exists():
        print(f"ERROR: {runs_dir} not found")
        return

    # ── Discover checkpoints ──
    # For do(g), we only need ONE checkpoint per (alpha, seed) pair.
    # We use the SSSS-trained checkpoint (arbitrary choice — GRU weights
    # are what matter, and we harvest g in both envs from the same ckpt).
    ckpts = {}
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        parsed = parse_label(run_dir.name)
        if not parsed:
            continue
        # Use SSSS checkpoints as base (could also use MMMM, doesn't matter)
        if parsed["valence"] != "SSSS":
            continue
        ckpt_path = run_dir / "ckpt_final.pt"
        if not ckpt_path.exists():
            continue
        key = f"a{parsed['alpha']}_s{parsed['seed']}"
        ckpts[key] = {
            "path": ckpt_path,
            "alpha": parsed["alpha"],
            "seed": parsed["seed"],
            "dir_name": run_dir.name,
        }

    if not ckpts:
        print("No SSSS checkpoints found")
        return

    print(f"Found {len(ckpts)} checkpoints for do(g)")

    if args.max_ckpts > 0:
        keys = sorted(ckpts.keys())[:args.max_ckpts]
        ckpts = {k: ckpts[k] for k in keys}
        print(f"  (limited to {len(ckpts)})")

    # ── Run do(g) for each checkpoint ──
    results_by_condition = {}

    for key, info in sorted(ckpts.items()):
        print(f"\n[do(g)] {key} — {info['dir_name']}")

        try:
            meta_dummy = {}
            results = run_do_g_for_checkpoint(
                ckpt_path=info["path"],
                device=args.device,
                meta=meta_dummy,
                outdir=outdir,
                harvest_episodes=args.harvest_episodes,
                probe_episodes=args.probe_episodes,
                harvest_seed=5000 + info["seed"] * 100,
                probe_seed=6000 + info["seed"] * 100,
                schedule=args.schedule,
                max_steps=args.max_steps,
            )
            results["alpha"] = info["alpha"]
            results["seed"] = info["seed"]
            results_by_condition[key] = results

            # Save per-checkpoint results
            results.to_csv(outdir / f"do_g_{key}.csv", index=False)

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            continue

    if not results_by_condition:
        print("\nNo results generated")
        return

    # ── Save combined results ──
    combined = pd.concat(results_by_condition.values(), ignore_index=True)
    combined.to_csv(outdir / "do_g_combined.csv", index=False)
    print(f"\nCombined results: {len(combined)} rows")

    # ── Summary statistics ──
    cross_dec = "decoder_shift_ssss_vs_mmmm"
    cross_pol = "policy_shift_ssss_vs_mmmm"
    within_dec = "decoder_shift_ssss_vs_ssss_b"

    if cross_dec in combined.columns:
        print(f"\n=== do(g) Summary ===")
        print(f"  Cross-basin decoder shift:  "
              f"median={combined[cross_dec].median():.4f}  "
              f"IQR=[{combined[cross_dec].quantile(0.25):.4f}, "
              f"{combined[cross_dec].quantile(0.75):.4f}]")
        if cross_pol in combined.columns:
            print(f"  Cross-basin policy shift:   "
                  f"median={combined[cross_pol].median():.4f}  "
                  f"IQR=[{combined[cross_pol].quantile(0.25):.4f}, "
                  f"{combined[cross_pol].quantile(0.75):.4f}]")
            ratio = combined[cross_dec].median() / max(combined[cross_pol].median(), 1e-8)
            print(f"  Decoder/Policy ratio:       {ratio:.2f}x")
        if within_dec in combined.columns:
            print(f"  Within-basin decoder shift:  "
                  f"median={combined[within_dec].median():.4f}")
            snr = combined[cross_dec].median() / max(combined[within_dec].median(), 1e-8)
            print(f"  Cross/Within ratio (SNR):   {snr:.2f}x")

    # ── Generate figures ──
    print("\nGenerating figures...")
    fig_interpretive_dissociation(results_by_condition, outdir)
    fig_cross_vs_within_basin(results_by_condition, outdir)
    fig_per_alpha(results_by_condition, outdir)

    print(f"\nDone. Output in: {outdir}")


if __name__ == "__main__":
    main()