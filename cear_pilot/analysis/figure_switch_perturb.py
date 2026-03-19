# cear_pilot/analysis/figure_switch_perturb.py
# -*- coding: utf-8 -*-
"""
Make two separate demo figures from a run_switch_perturb output.

Figure 1) switch_overview.png
  - g distance to pre-switch baseline
  - policy stats (pi_max, pi_entropy)
  - zone_id

Figure 2) perturb_recovery.png
  - recovery curve(s) around perturb time(s)
  - each curve uses its own local baseline mean g (window before perturb)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


def load_table(run_dir: Path):
    import pandas as pd
    p_parquet = run_dir / "traj.parquet"
    p_csv = run_dir / "traj.csv"
    if p_parquet.exists():
        return pd.read_parquet(p_parquet)
    if p_csv.exists():
        return pd.read_csv(p_csv)
    raise FileNotFoundError("traj.parquet or traj.csv not found in run_dir")


def get_g_matrix(df) -> np.ndarray:
    g_cols = [c for c in df.columns if c.startswith("g_")]
    g_cols = sorted(g_cols, key=lambda x: int(x.split("_")[1]))
    if len(g_cols) == 0:
        raise ValueError("No g_* columns found")
    G = df[g_cols].to_numpy(dtype=np.float32)
    return G


def mean_g_over_window(G: np.ndarray, t: int, w: int) -> np.ndarray:
    """
    Mean g over [t-w, t-1]. If window is out of range, it clamps safely.
    """
    a = max(0, t - w)
    b = max(0, t)  # exclusive
    if b <= a:
        return G[max(0, min(t, len(G)-1))].copy()
    return G[a:b].mean(axis=0)


def g_dist_to_ref(G: np.ndarray, ref: np.ndarray) -> np.ndarray:
    D = np.linalg.norm(G - ref[None, :], axis=1)
    return D


def recovery_curve(G: np.ndarray, t0: int, w_ref: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      t = [0..T-1]
      d(t)=||g(t)-mean(g[t0-w_ref:t0])||
    """
    ref = mean_g_over_window(G, t0, w_ref)
    d = g_dist_to_ref(G, ref)
    return np.arange(len(d), dtype=np.int32), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--w_ref", type=int, default=30, help="Window size for baseline mean g")
    ap.add_argument("--shade_after_switch", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    t_switch = int(meta.get("t_switch", 80))
    t_perturb = int(meta.get("t_perturb", -1))
    t_perturb2 = int(meta.get("t_perturb2", -1))

    df = load_table(run_dir)
    G = get_g_matrix(df)

    t = df["t"].to_numpy(dtype=np.int32)
    zone_id = df["zone_id"].to_numpy(dtype=np.int32)

    pi_max = df["pi_max"].to_numpy(dtype=np.float32) if "pi_max" in df.columns else None
    pi_entropy = df["pi_entropy"].to_numpy(dtype=np.float32) if "pi_entropy" in df.columns else None

    figs_dir = run_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Figure 1: switch overview
    # -----------------------------
    # baseline = mean g over first w_ref steps (or up to switch, whichever is smaller)
    w0 = min(args.w_ref, max(1, t_switch))
    ref_pre = G[:w0].mean(axis=0)
    d_pre = g_dist_to_ref(G, ref_pre)

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.0, 0.7], hspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, d_pre)
    ax1.axvline(t_switch, linestyle="--")
    ax1.set_ylabel("d_pre(t)=||g-mean_pre||")
    ax1.set_title(f"Switch overview (t_switch={t_switch})")

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    if pi_max is not None:
        ax2.plot(t, pi_max, label="pi_max")
    if pi_entropy is not None:
        ax2.plot(t, pi_entropy, label="pi_entropy")
    ax2.axvline(t_switch, linestyle="--")
    ax2.set_ylabel("policy")
    ax2.legend()

    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax3.plot(t, zone_id)
    ax3.axvline(t_switch, linestyle="--")
    ax3.set_ylabel("zone_id")
    ax3.set_xlabel("t")

    if args.shade_after_switch:
        for ax in (ax1, ax2, ax3):
            ax.axvspan(t_switch, t[-1], alpha=0.08)

    out1 = figs_dir / "switch_overview.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out1}")

    # -----------------------------
    # Figure 2: perturb recovery (separate)
    # -----------------------------
    curves = []
    labels = []

    if t_perturb >= 0:
        tt, dd = recovery_curve(G, t_perturb, args.w_ref)
        curves.append((tt, dd, t_perturb))
        labels.append(f"perturb@{t_perturb}")

    if t_perturb2 >= 0:
        tt, dd = recovery_curve(G, t_perturb2, args.w_ref)
        curves.append((tt, dd, t_perturb2))
        labels.append(f"perturb2@{t_perturb2}")

    if len(curves) > 0:
        fig2 = plt.figure(figsize=(12, 4))
        ax = fig2.add_subplot(1, 1, 1)

        for (tt, dd, t0), lab in zip(curves, labels):
            ax.plot(tt, dd, label=lab)
            ax.axvline(t0, linestyle="--")

        ax.axvline(t_switch, linestyle="--")
        ax.set_title("Perturb recovery (distance to local pre-perturb baseline)")
        ax.set_xlabel("t")
        ax.set_ylabel("d_local(t)=||g-mean_pre_perturb||")
        ax.legend()

        out2 = figs_dir / "perturb_recovery.png"
        fig2.savefig(out2, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"[OK] Saved: {out2}")
    else:
        print("[WARN] No perturb times provided; perturb_recovery.png not generated.")


if __name__ == "__main__":
    main()
