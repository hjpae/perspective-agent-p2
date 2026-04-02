#!/usr/bin/env python3
"""
Phase 2 analysis: rollout figures for salience gating verification.

Usage:
    python -m cear_pilot.analysis.analyze_phase2 --root outputs/nperturb4_s0
    python -m cear_pilot.analysis.analyze_phase2 --sweep_root outputs/sweep_20260401
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DPI = 150
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.2,
    "font.size": 11,
})


def load_traj(d: Path) -> pd.DataFrame | None:
    for name in ["traj.parquet", "traj.csv"]:
        p = d / name
        if p.exists():
            try:
                return pd.read_parquet(p) if name.endswith(".parquet") else pd.read_csv(p)
            except Exception as e:
                print(f"  [warn] Failed to load {p}: {e}")
    return None


def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {path.name}")


# ────────────────────────────────────────

def fig_pe(df, outdir):
    ep_pe = df.groupby("episode")["pred_err"].mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ep_pe.index, ep_pe.values, lw=0.6, color="#534AB7", alpha=0.5)
    if len(ep_pe) > 10:
        ax.plot(ep_pe.rolling(10, min_periods=1).mean(), lw=2, color="#534AB7")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean PE")
    ax.set_title("Prediction error over training")
    savefig(fig, outdir / "pe_trajectory.png")


def fig_g_alpha(df, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ep_g = df.groupby("episode")["g_norm"].mean()
    axes[0].plot(ep_g.index, ep_g.values, lw=1, color="#0F6E56")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("||g||")
    axes[0].set_title("Perspective magnitude")

    if "alpha" in df.columns:
        ep_a = df.groupby("episode")["alpha"].mean()
        axes[1].plot(ep_a.index, ep_a.values, lw=1, color="#D85A30")
        axes[1].set_ylabel("α")
        axes[1].set_title("Self-modulating plasticity")
    axes[1].set_xlabel("Episode")
    savefig(fig, outdir / "g_alpha_trajectory.png")


def fig_zone_dwell(df, outdir):
    counts = df["zone_id"].value_counts().sort_index()
    total = counts.sum()
    fracs = counts / total

    colors = ["#E24B4A", "#D85A30", "#888780", "#1D9E75", "#0F6E56"]
    labels = [f"Z{i}" for i in fracs.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(fracs)), fracs.values,
                  color=[colors[i % len(colors)] for i in range(len(fracs))], alpha=0.7)
    ax.set_xticks(range(len(fracs)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction of steps")
    ax.set_title("Zone dwell time")
    for bar, frac in zip(bars, fracs.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{frac:.1%}", ha="center", fontsize=9)
    savefig(fig, outdir / "zone_dwell.png")


def fig_perturbation_response(df, outdir):
    if "perturbation_active" not in df.columns:
        return
    if df["perturbation_active"].sum() == 0:
        print("  [skip] No perturbations")
        return

    ds = df.sort_values(["episode", "t"]).reset_index(drop=True)
    pa = ds["perturbation_active"].values

    # Onset detection
    onsets = []
    for i in range(1, len(pa)):
        if pa[i] == 1 and pa[i - 1] == 0:
            onsets.append(i)
    if not onsets:
        print("  [skip] No onsets detected")
        return

    W = 40

    def triggered_avg(vals):
        traces = []
        for loc in onsets:
            row = np.full(2 * W + 1, np.nan)
            for j in range(max(0, loc - W), min(len(vals), loc + W + 1)):
                row[j - loc + W] = vals[j]
            traces.append(row)
        arr = np.array(traces)
        return np.nanmedian(arr, axis=0), np.nanpercentile(arr, 25, axis=0), np.nanpercentile(arr, 75, axis=0)

    t_ax = np.arange(-W, W + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    med, q1, q3 = triggered_avg(ds["g_norm"].values)
    axes[0].plot(t_ax, med, color="#D85A30", lw=2)
    axes[0].fill_between(t_ax, q1, q3, color="#D85A30", alpha=0.15)
    axes[0].axvline(0, color="#ccc", lw=0.5, ls="--")
    axes[0].set_xlabel("Steps from onset")
    axes[0].set_ylabel("||g||")
    axes[0].set_title("g response to perturbation")

    if "alpha" in ds.columns:
        med_a, _, _ = triggered_avg(ds["alpha"].values)
        axes[1].plot(t_ax, med_a, color="#534AB7", lw=2)
        axes[1].axvline(0, color="#ccc", lw=0.5, ls="--")
    axes[1].set_xlabel("Steps from onset")
    axes[1].set_ylabel("α")
    axes[1].set_title("Plasticity response")

    savefig(fig, outdir / "perturbation_response.png")


def fig_zone_trajectory(df, outdir):
    max_ep = df["episode"].max()
    late = df[df["episode"] >= max(0, max_ep - 4)]

    fig, ax = plt.subplots(figsize=(12, 4))
    for ep in sorted(late["episode"].unique())[:5]:
        ed = late[late["episode"] == ep]
        ax.plot(ed["t"].values, ed["zone_id"].values, lw=0.6, alpha=0.6, label=f"ep {ep}")
        if "perturbation_active" in ed.columns:
            pm = ed[ed["perturbation_active"] == 1]
            if len(pm) > 0:
                ax.scatter(pm["t"].values, pm["zone_id"].values, c="red", s=8, zorder=5, alpha=0.5)

    ax.set_xlabel("Step")
    ax.set_ylabel("Zone ID")
    ax.set_title("Zone trajectory (last episodes, red=perturbation)")
    ax.set_yticks(range(5))
    savefig(fig, outdir / "zone_trajectory_late.png")


def fig_sweep(sweep_root, outdir):
    runs = {}
    for d in sorted(sweep_root.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        m = re.search(r"nperturb(\d+)", d.name)
        if not m:
            continue
        tdf = load_traj(d)
        if tdf is not None:
            runs[int(m.group(1))] = tdf

    if len(runs) < 2:
        print("  [skip] Need >=2 runs for sweep comparison")
        return

    ns = sorted(runs.keys())
    zone_colors = ["#E24B4A", "#D85A30", "#888780", "#1D9E75", "#0F6E56"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # ── (0,0) Zone dwell fractions ──
    ax = axes[0, 0]
    for zi in range(5):
        fracs = []
        for n in ns:
            total = len(runs[n])
            fracs.append(len(runs[n][runs[n]["zone_id"] == zi]) / max(total, 1))
        ax.plot(ns, fracs, "o-", color=zone_colors[zi], lw=1.5, markersize=5, label=f"Z{zi}")
    ax.set_xlabel("Perturbations / episode")
    ax.set_ylabel("Fraction of steps")
    ax.set_title("Zone dwell time")
    ax.legend(fontsize=8)

    # ── (0,1) Late PE ──
    ax = axes[0, 1]
    pes = []
    for n in ns:
        d = runs[n]
        mx = d["episode"].max()
        late = d[d["episode"] >= max(0, mx - 19)]
        pes.append(late["pred_err"].mean())
    ax.plot(ns, pes, "s-", color="#534AB7", lw=1.5, markersize=6)
    ax.set_xlabel("Perturbations / episode")
    ax.set_ylabel("Mean PE (last 20 eps)")
    ax.set_title("Prediction error")

    # ── (0,2) Late ||g|| ──
    ax = axes[0, 2]
    gnorms = []
    for n in ns:
        d = runs[n]
        mx = d["episode"].max()
        late = d[d["episode"] >= max(0, mx - 19)]
        gnorms.append(late["g_norm"].mean())
    ax.plot(ns, gnorms, "o-", color="#0F6E56", lw=1.5, markersize=6)
    ax.set_xlabel("Perturbations / episode")
    ax.set_ylabel("||g|| (last 20 eps)")
    ax.set_title("Perspective magnitude")

    # ── (1,0) Late alpha ──
    ax = axes[1, 0]
    if "alpha" in list(runs.values())[0].columns:
        alphas = []
        for n in ns:
            d = runs[n]
            mx = d["episode"].max()
            late = d[d["episode"] >= max(0, mx - 19)]
            alphas.append(late["alpha"].mean())
        ax.plot(ns, alphas, "D-", color="#D85A30", lw=1.5, markersize=6)
    ax.set_xlabel("Perturbations / episode")
    ax.set_ylabel("α (last 20 eps)")
    ax.set_title("Self-modulating plasticity")

    # ── (1,1) Zone dwell LATE only (last 20 eps) ──
    ax = axes[1, 1]
    for zi in range(5):
        fracs = []
        for n in ns:
            d = runs[n]
            mx = d["episode"].max()
            late = d[d["episode"] >= max(0, mx - 19)]
            total = len(late)
            fracs.append(len(late[late["zone_id"] == zi]) / max(total, 1))
        ax.plot(ns, fracs, "o-", color=zone_colors[zi], lw=1.5, markersize=5, label=f"Z{zi}")
    ax.set_xlabel("Perturbations / episode")
    ax.set_ylabel("Fraction (last 20 eps)")
    ax.set_title("Zone dwell (late training only)")
    ax.legend(fontsize=8)

    # ── (1,2) PE trajectory overlay ──
    ax = axes[1, 2]
    cmap = plt.cm.viridis
    for i, n in enumerate(ns):
        d = runs[n]
        ep_pe = d.groupby("episode")["pred_err"].mean()
        if len(ep_pe) > 10:
            ep_pe = ep_pe.rolling(10, min_periods=1).mean()
        color = cmap(i / max(len(ns) - 1, 1))
        ax.plot(ep_pe.index, ep_pe.values, lw=1, color=color, alpha=0.8, label=f"n={n}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("PE (smoothed)")
    ax.set_title("PE learning curves")
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle("Perturbation sweep comparison", fontsize=16, y=1.01)
    savefig(fig, outdir / "sweep_comparison.png")

    # ── Summary table ──
    rows = []
    for n in ns:
        d = runs[n]
        mx = d["episode"].max()
        late = d[d["episode"] >= max(0, mx - 19)]
        row = {"n_perturb": n, "pe_late": late["pred_err"].mean(), "g_norm_late": late["g_norm"].mean()}
        if "alpha" in late.columns:
            row["alpha_late"] = late["alpha"].mean()
        for zi in range(5):
            row[f"zone{zi}_frac"] = len(late[late["zone_id"] == zi]) / max(len(late), 1)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "sweep_summary.csv", index=False)
    print(f"  [csv] sweep_summary.csv")
    print(summary.to_string(index=False))


# ────────────────────────────────────────

def analyze_run(run_dir: Path):
    outdir = run_dir / "analysis"
    outdir.mkdir(exist_ok=True)

    df = load_traj(run_dir)
    if df is None:
        print(f"  [skip] No trajectory in {run_dir}")
        return

    print(f"\n  Analyzing {run_dir.name}: {len(df)} rows, {df['episode'].nunique()} eps")

    for fn in [fig_pe, fig_g_alpha, fig_zone_dwell, fig_perturbation_response, fig_zone_trajectory]:
        try:
            fn(df, outdir)
        except Exception as e:
            print(f"  [error] {fn.__name__}: {e}")

    # Summary
    mx = df["episode"].max()
    late = df[df["episode"] >= max(0, mx - 19)]
    print(f"  Late PE: {late['pred_err'].mean():.4f}")
    print(f"  Late ||g||: {late['g_norm'].mean():.4f}")
    if "alpha" in late.columns:
        print(f"  Late α: {late['alpha'].mean():.4f}")
    dwell = late["zone_id"].value_counts().sort_index()
    print(f"  Zone dwell: {dict(dwell)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="")
    ap.add_argument("--sweep_root", type=str, default="")
    args = ap.parse_args()

    if args.root:
        analyze_run(Path(args.root))

    if args.sweep_root:
        sr = Path(args.sweep_root)
        for d in sorted(sr.iterdir()):
            if d.is_dir() and not d.name.startswith("_"):
                analyze_run(d)

        sa = sr / "sweep_analysis"
        sa.mkdir(exist_ok=True)
        try:
            fig_sweep(sr, sa)
        except Exception as e:
            print(f"  [error] sweep: {e}")
        print(f"\n  Sweep output: {sa}")


if __name__ == "__main__":
    main()