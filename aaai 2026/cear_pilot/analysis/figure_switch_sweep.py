# cear_pilot/analysis/figure_switch_sweep.py
# -*- coding: utf-8 -*-
"""
3-panel figure for regime-switch sweep runs.

Panel 1: d_g(t) = ||g(t) - mean_pre|| (pre = t < first switch)
Panel 2: policy snapshots (pi_max, entropy, margin) + argmax change markers
Panel 3: cumulative deviation (CUSUM-style) to visualize integration / hysteresis:
         cum_g(t)  = sum_{k>=t_pre_end} (d_g(k) - d_pre)
         cum_pi(t) = sum_{k>=t_pre_end} (entropy(k) - ent_pre)

This is designed to show that g behaves like a slow integrator / low-pass memory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np


def load_table(run_dir: Path):
    import pandas as pd
    p_parq = run_dir / "traj.parquet"
    p_csv = run_dir / "traj.csv"
    if p_parq.exists():
        return pd.read_parquet(p_parq)
    if p_csv.exists():
        return pd.read_csv(p_csv)
    raise FileNotFoundError(f"No traj.parquet or traj.csv under {run_dir}")


def find_first_switch_t(regime: np.ndarray) -> int:
    # Return the first t where regime changes; if never changes, return -1.
    if len(regime) < 2:
        return -1
    diffs = np.where(regime[1:] != regime[:-1])[0]
    return int(diffs[0] + 1) if len(diffs) > 0 else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--pre_window", type=int, default=-1,
                    help="If >0, use only last pre_window steps before first switch as baseline.")
    ap.add_argument("--save_name", type=str, default="fig_switch_sweep.png")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    df = load_table(run_dir)

    # Extract arrays
    t = df["t"].to_numpy().astype(int)
    regime = df["regime_id"].to_numpy().astype(int)
    zone = df["zone_id"].to_numpy().astype(int)

    # g vectors
    g_cols = [c for c in df.columns if c.startswith("g_")]
    g = df[g_cols].to_numpy(dtype=np.float32)

    # policy snapshots
    pi_max = df["pi_max"].to_numpy(dtype=np.float32) if "pi_max" in df.columns else None
    ent = df["pi_entropy"].to_numpy(dtype=np.float32) if "pi_entropy" in df.columns else None
    margin = df["pi_margin"].to_numpy(dtype=np.float32) if "pi_margin" in df.columns else None
    argmax = df["pi_argmax"].to_numpy().astype(int) if "pi_argmax" in df.columns else None

    t_sw = find_first_switch_t(regime)

    # Define "pre" baseline region
    if t_sw >= 0:
        pre_mask = (t < t_sw)
    else:
        pre_mask = np.ones_like(t, dtype=bool)

    pre_idx = np.where(pre_mask)[0]
    if len(pre_idx) < 5:
        # Fallback: use first 10 steps if no clear pre region
        pre_idx = np.arange(min(10, len(t)))

    if args.pre_window and args.pre_window > 0 and len(pre_idx) > args.pre_window:
        pre_idx = pre_idx[-args.pre_window:]

    g_pre_mean = g[pre_idx].mean(axis=0, keepdims=True)
    d_g = np.linalg.norm(g - g_pre_mean, axis=1)
    d_pre = float(d_g[pre_idx].mean())

    # Policy baselines (for cumulative)
    ent_pre = float(ent[pre_idx].mean()) if ent is not None else 0.0

    # CUSUM-like cumulative deviations (start at end of pre)
    start = int(pre_idx[-1] + 1) if len(pre_idx) > 0 else 0
    cum_g = np.zeros_like(d_g, dtype=np.float32)
    cum_e = np.zeros_like(d_g, dtype=np.float32)

    for i in range(start, len(t)):
        cum_g[i] = cum_g[i - 1] + float(d_g[i] - d_pre)
        if ent is not None:
            cum_e[i] = cum_e[i - 1] + float(ent[i] - ent_pre)

    # Argmax flip markers
    flip_ts = []
    if argmax is not None and len(argmax) > 1:
        flip_ts = list(t[1:][argmax[1:] != argmax[:-1]])

    # Plot
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(14, 7.5), sharex=True)

    # Shade B-regime regions
    if t_sw >= 0:
        # Shade any segment where regime==1 (B)
        in_B = (regime == 1)
        if np.any(in_B):
            # Convert to contiguous spans
            idx = np.where(in_B)[0]
            spans = []
            s = idx[0]
            for k in range(1, len(idx)):
                if idx[k] != idx[k - 1] + 1:
                    spans.append((t[s], t[idx[k - 1]]))
                    s = idx[k]
            spans.append((t[s], t[idx[-1]]))
            for (a, b) in spans:
                for ax in axes:
                    ax.axvspan(a, b, alpha=0.08)

    # Switch line
    if t_sw >= 0:
        for ax in axes:
            ax.axvline(t_sw, linestyle="--")

    # Panel 1: g distance
    axes[0].plot(t, d_g)
    axes[0].set_ylabel("d_g(t)=||g-mean_pre||")
    axes[0].set_title(f"Switch-sweep overview (t_switch={t_sw})")

    # Panel 2: policy stats
    if pi_max is not None:
        axes[1].plot(t, pi_max, label="pi_max")
    if ent is not None:
        axes[1].plot(t, ent, label="entropy")
    if margin is not None:
        axes[1].plot(t, margin, label="margin(top1-top2)")
    # Argmax flip markers
    for ft in flip_ts[:2000]:
        axes[1].axvline(ft, alpha=0.05)
    axes[1].legend()
    axes[1].set_ylabel("policy")

    # Panel 3: cumulative deviations
    axes[2].plot(t, cum_g, label="CUSUM_g")
    if ent is not None:
        axes[2].plot(t, cum_e, label="CUSUM_entropy")
    axes[2].legend()
    axes[2].set_ylabel("cumulative")
    axes[2].set_xlabel("t")

    out_path = run_dir / "figs" / args.save_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()
