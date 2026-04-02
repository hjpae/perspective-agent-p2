#!/usr/bin/env python3
"""
Phase 2 analysis.

Usage:
    python -m cear_pilot.analysis.analyze_phase2 --root outputs/run_dir
    python -m cear_pilot.analysis.analyze_phase2 --sweep_root outputs/sweep_dir
    python -m cear_pilot.analysis.analyze_phase2 --ablation_root outputs/ablation_dir
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DPI = 150
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                      "axes.grid": True, "grid.alpha": 0.2, "font.size": 11})
ZONE_COLORS = ["#E24B4A", "#D85A30", "#888780", "#1D9E75", "#0F6E56"]


def load_traj(d):
    for name in ["traj.parquet", "traj.csv"]:
        p = d / name
        if p.exists():
            try:
                return pd.read_parquet(p) if name.endswith(".parquet") else pd.read_csv(p)
            except Exception as e:
                print(f"  [warn] {p}: {e}")
    return None


def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {path.name}")


def g_cols(df):
    return sorted([c for c in df.columns if re.match(r"g_\d+$", c)],
                  key=lambda x: int(x.split("_")[-1]))


def gamma_cols(df):
    return sorted([c for c in df.columns if re.match(r"gamma_\d+$", c)],
                  key=lambda x: int(x.split("_")[-1]))


def beta_cols(df):
    return sorted([c for c in df.columns if re.match(r"beta_\d+$", c)],
                  key=lambda x: int(x.split("_")[-1]))


# ════════════════════════════════════════
# Per-run basics
# ════════════════════════════════════════

def fig_pe(df, outdir):
    ep_pe = df.groupby("episode")["pred_err"].mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ep_pe.index, ep_pe.values, lw=0.5, alpha=0.4, color="#534AB7")
    if len(ep_pe) > 10:
        ax.plot(ep_pe.rolling(10, min_periods=1).mean(), lw=2, color="#534AB7")
    ax.set_xlabel("Episode"); ax.set_ylabel("PE"); ax.set_title("Prediction error")
    savefig(fig, outdir / "pe_trajectory.png")


def fig_g_alpha(df, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ep_g = df.groupby("episode")["g_norm"].mean()
    axes[0].plot(ep_g.index, ep_g.values, lw=1, color="#0F6E56")
    axes[0].set_xlabel("Episode"); axes[0].set_ylabel("||g||"); axes[0].set_title("g magnitude")
    if "alpha" in df.columns:
        ep_a = df.groupby("episode")["alpha"].mean()
        axes[1].plot(ep_a.index, ep_a.values, lw=1, color="#D85A30")
    axes[1].set_xlabel("Episode"); axes[1].set_ylabel("α"); axes[1].set_title("Plasticity")
    savefig(fig, outdir / "g_alpha_trajectory.png")


def fig_zone_dwell(df, outdir):
    counts = df["zone_id"].value_counts().sort_index()
    fracs = counts / counts.sum()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(fracs)), fracs.values,
                  color=[ZONE_COLORS[i % 5] for i in range(len(fracs))], alpha=0.7)
    ax.set_xticks(range(len(fracs)))
    ax.set_xticklabels([f"Z{i}" for i in fracs.index])
    ax.set_ylabel("Fraction"); ax.set_title("Zone dwell")
    for b, f in zip(bars, fracs.values):
        ax.text(b.get_x() + b.get_width()/2, b.get_height()+0.005, f"{f:.1%}", ha="center", fontsize=9)
    savefig(fig, outdir / "zone_dwell.png")


def fig_perturbation_response(df, outdir):
    if "perturbation_active" not in df.columns or df["perturbation_active"].sum() == 0:
        return
    ds = df.sort_values(["episode", "t"]).reset_index(drop=True)
    pa = ds["perturbation_active"].values
    onsets = [i for i in range(1, len(pa)) if pa[i] == 1 and pa[i-1] == 0]
    if not onsets:
        return
    W = 40
    t_ax = np.arange(-W, W+1)

    def trig(vals):
        tr = []
        for loc in onsets:
            row = np.full(2*W+1, np.nan)
            for j in range(max(0, loc-W), min(len(vals), loc+W+1)):
                row[j-loc+W] = vals[j]
            tr.append(row)
        return np.array(tr)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    med = np.nanmedian(trig(ds["g_norm"].values), axis=0)
    axes[0].plot(t_ax, med, color="#D85A30", lw=2)
    axes[0].axvline(0, color="#ccc", lw=0.5, ls="--")
    axes[0].set_xlabel("Steps from onset"); axes[0].set_ylabel("||g||"); axes[0].set_title("g response")
    if "alpha" in ds.columns:
        axes[1].plot(t_ax, np.nanmedian(trig(ds["alpha"].values), axis=0), color="#534AB7", lw=2)
        axes[1].axvline(0, color="#ccc", lw=0.5, ls="--")
    axes[1].set_xlabel("Steps from onset"); axes[1].set_ylabel("α"); axes[1].set_title("α response")
    savefig(fig, outdir / "perturbation_response.png")


# ════════════════════════════════════════
# Mixed history: block-aligned analysis
# ════════════════════════════════════════

def fig_mixed_history(df, outdir):
    if "block_id" not in df.columns:
        return
    if df["block_id"].nunique() < 2:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ep_data = df.groupby("episode").agg({
        "pred_err": "mean", "g_norm": "mean",
        "alpha": "mean" if "alpha" in df.columns else "first",
        "block_id": "first", "n_perturb_setting": "first",
    }).reset_index()

    # Block boundaries
    block_changes = ep_data[ep_data["block_id"] != ep_data["block_id"].shift(1)]["episode"].values

    # Panel (0,0): PE with block boundaries
    ax = axes[0, 0]
    ax.plot(ep_data["episode"], ep_data["pred_err"], lw=0.5, alpha=0.4, color="#534AB7")
    if len(ep_data) > 10:
        ax.plot(ep_data["episode"], ep_data["pred_err"].rolling(10, min_periods=1).mean(),
                lw=2, color="#534AB7")
    for bc in block_changes:
        ax.axvline(bc, color="#D85A30", lw=1, ls="--", alpha=0.5)
    ax.set_xlabel("Episode"); ax.set_ylabel("PE"); ax.set_title("PE across blocks")

    # Panel (0,1): alpha with block boundaries
    ax = axes[0, 1]
    if "alpha" in ep_data.columns:
        ax.plot(ep_data["episode"], ep_data["alpha"], lw=1, color="#D85A30")
        for bc in block_changes:
            ax.axvline(bc, color="#D85A30", lw=1, ls="--", alpha=0.5)
    ax.set_xlabel("Episode"); ax.set_ylabel("α"); ax.set_title("Plasticity across blocks")

    # Panel (1,0): g_norm with block boundaries
    ax = axes[1, 0]
    ax.plot(ep_data["episode"], ep_data["g_norm"], lw=1, color="#0F6E56")
    for bc in block_changes:
        ax.axvline(bc, color="#D85A30", lw=1, ls="--", alpha=0.5)
    ax.set_xlabel("Episode"); ax.set_ylabel("||g||"); ax.set_title("g magnitude across blocks")

    # Panel (1,1): n_perturb setting (block structure visualization)
    ax = axes[1, 1]
    ax.plot(ep_data["episode"], ep_data["n_perturb_setting"], lw=2, color="#888780", drawstyle="steps-post")
    ax.set_xlabel("Episode"); ax.set_ylabel("n_perturbations"); ax.set_title("Block schedule")
    ax.set_ylim(-0.5, ep_data["n_perturb_setting"].max() + 1)

    # Add block labels
    for i, bc in enumerate(block_changes):
        blk = ep_data[ep_data["episode"] >= bc]["block_id"].iloc[0]
        nP = ep_data[ep_data["episode"] >= bc]["n_perturb_setting"].iloc[0]
        for a in axes.flat:
            a.annotate(f"blk{blk}\nnP={nP}", xy=(bc, a.get_ylim()[1]),
                       fontsize=7, color="#D85A30", alpha=0.7, va="top", ha="left")

    fig.suptitle("Mixed perturbation history (one agent)", fontsize=14)
    savefig(fig, outdir / "mixed_history.png")


# ════════════════════════════════════════
# FiLM gating analysis
# ════════════════════════════════════════

def fig_film_gating(df, outdir):
    gc = gamma_cols(df)
    bc = beta_cols(df)
    if not gc:
        print("  [skip] No gamma columns in trajectory")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    n_dims = len(gc)

    # Get block info if available
    has_blocks = "block_id" in df.columns and df["block_id"].nunique() > 1

    # Panel (0,0): gamma per episode (mean abs)
    ax = axes[0, 0]
    ep_gamma_abs = df.groupby("episode")[gc].apply(lambda x: np.abs(x.values).mean())
    ax.plot(ep_gamma_abs.index, ep_gamma_abs.values, lw=1, color="#D85A30")
    ax.set_xlabel("Episode"); ax.set_ylabel("Mean |γ|"); ax.set_title("FiLM modulation strength")

    # Panel (0,1): gamma per dim, late episodes
    ax = axes[0, 1]
    mx = df["episode"].max()
    late = df[df["episode"] >= max(0, mx - 19)]
    gamma_means = late[gc].mean().values
    beta_means = late[bc].mean().values if bc else np.zeros(n_dims)
    x_pos = np.arange(n_dims)
    ax.bar(x_pos - 0.15, gamma_means, 0.3, color="#D85A30", alpha=0.7, label="γ (scale)")
    ax.bar(x_pos + 0.15, beta_means, 0.3, color="#534AB7", alpha=0.7, label="β (shift)")
    ax.set_xticks(x_pos); ax.set_xticklabels([f"z{i}" for i in range(n_dims)], fontsize=8)
    ax.set_ylabel("Value"); ax.set_title("Late gating pattern (per z dim)")
    ax.legend(fontsize=8)
    ax.axhline(0, color="#ccc", lw=0.5)

    # Panel (1,0): gamma evolution per dim
    ax = axes[1, 0]
    ep_gamma = df.groupby("episode")[gc].mean()
    for i, col in enumerate(gc):
        ax.plot(ep_gamma.index, ep_gamma[col].values, lw=0.7, alpha=0.6, label=f"z{i}")
    ax.set_xlabel("Episode"); ax.set_ylabel("γ"); ax.set_title("Per-dim gamma over training")
    ax.legend(fontsize=6, ncol=4)

    # Panel (1,1): gating comparison between blocks (if mixed)
    ax = axes[1, 1]
    if has_blocks:
        blocks = sorted(df["block_id"].unique())
        width = 0.8 / len(blocks)
        for bi, blk in enumerate(blocks):
            blk_data = df[df["block_id"] == blk]
            # Use late portion of each block
            blk_max = blk_data["episode"].max()
            blk_late = blk_data[blk_data["episode"] >= max(blk_data["episode"].min(), blk_max - 9)]
            gvals = blk_late[gc].mean().values
            nP = int(blk_data["n_perturb_setting"].iloc[0])
            offset = (bi - len(blocks)/2 + 0.5) * width
            ax.bar(x_pos + offset, gvals, width, alpha=0.7, label=f"blk{blk} (nP={nP})")
        ax.set_xticks(x_pos); ax.set_xticklabels([f"z{i}" for i in range(n_dims)], fontsize=8)
        ax.set_ylabel("γ (late block mean)"); ax.set_title("Gating per block")
        ax.legend(fontsize=7)
        ax.axhline(0, color="#ccc", lw=0.5)
    else:
        ax.text(0.5, 0.5, "Single block (no comparison)", transform=ax.transAxes, ha="center")

    fig.suptitle("FiLM salience gating analysis", fontsize=14)
    savefig(fig, outdir / "film_gating.png")


# ════════════════════════════════════════
# Sweep comparison (6-panel)
# ════════════════════════════════════════

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
        print("  [skip] Need >=2 runs for sweep")
        return

    ns = sorted(runs.keys())
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (0,0) Zone dwell
    ax = axes[0, 0]
    for zi in range(5):
        fracs = [len(runs[n][runs[n]["zone_id"]==zi])/max(len(runs[n]),1) for n in ns]
        ax.plot(ns, fracs, "o-", color=ZONE_COLORS[zi], lw=1.5, markersize=5, label=f"Z{zi}")
    ax.set_xlabel("nP"); ax.set_ylabel("Frac"); ax.set_title("Zone dwell"); ax.legend(fontsize=8)

    # (0,1) Late PE
    ax = axes[0, 1]
    pes = []
    for n in ns:
        d = runs[n]; mx = d["episode"].max()
        pes.append(d[d["episode"]>=max(0,mx-19)]["pred_err"].mean())
    ax.plot(ns, pes, "s-", color="#534AB7", lw=1.5, markersize=6)
    ax.set_xlabel("nP"); ax.set_ylabel("PE"); ax.set_title("Late PE")

    # (0,2) Late ||g||
    ax = axes[0, 2]
    gnorms = []
    for n in ns:
        d = runs[n]; mx = d["episode"].max()
        gnorms.append(d[d["episode"]>=max(0,mx-19)]["g_norm"].mean())
    ax.plot(ns, gnorms, "o-", color="#0F6E56", lw=1.5, markersize=6)
    ax.set_xlabel("nP"); ax.set_ylabel("||g||"); ax.set_title("g magnitude")

    # (1,0) Late alpha
    ax = axes[1, 0]
    alphas = []
    for n in ns:
        d = runs[n]; mx = d["episode"].max()
        late = d[d["episode"]>=max(0,mx-19)]
        alphas.append(late["alpha"].mean() if "alpha" in late.columns else 0)
    ax.plot(ns, alphas, "D-", color="#D85A30", lw=1.5, markersize=6)
    ax.set_xlabel("nP"); ax.set_ylabel("α"); ax.set_title("Plasticity")

    # (1,1) Zone dwell late only
    ax = axes[1, 1]
    for zi in range(5):
        fracs = []
        for n in ns:
            d = runs[n]; mx = d["episode"].max()
            late = d[d["episode"]>=max(0,mx-19)]
            fracs.append(len(late[late["zone_id"]==zi])/max(len(late),1))
        ax.plot(ns, fracs, "o-", color=ZONE_COLORS[zi], lw=1.5, markersize=5, label=f"Z{zi}")
    ax.set_xlabel("nP"); ax.set_ylabel("Frac (late)"); ax.set_title("Zone dwell (late)"); ax.legend(fontsize=8)

    # (1,2) PE curves overlay
    ax = axes[1, 2]
    cmap = plt.cm.viridis
    for i, n in enumerate(ns):
        ep_pe = runs[n].groupby("episode")["pred_err"].mean()
        if len(ep_pe) > 10:
            ep_pe = ep_pe.rolling(10, min_periods=1).mean()
        ax.plot(ep_pe.index, ep_pe.values, lw=1, color=cmap(i/max(len(ns)-1,1)), alpha=0.8, label=f"n={n}")
    ax.set_xlabel("Episode"); ax.set_ylabel("PE"); ax.set_title("Learning curves"); ax.legend(fontsize=7, ncol=2)

    fig.suptitle("Perturbation sweep comparison", fontsize=16)
    savefig(fig, outdir / "sweep_comparison.png")

    # Summary CSV
    rows = []
    for n in ns:
        d = runs[n]; mx = d["episode"].max()
        late = d[d["episode"]>=max(0,mx-19)]
        row = {"n_perturb": n, "pe_late": late["pred_err"].mean(), "g_norm_late": late["g_norm"].mean()}
        if "alpha" in late.columns:
            row["alpha_late"] = late["alpha"].mean()
        for zi in range(5):
            row[f"zone{zi}_frac"] = len(late[late["zone_id"]==zi])/max(len(late),1)
        rows.append(row)
    pd.DataFrame(rows).to_csv(outdir / "sweep_summary.csv", index=False)
    print(f"  [csv] sweep_summary.csv")


# ════════════════════════════════════════
# Ablation comparison: adaptive vs fixed vs fast
# ════════════════════════════════════════

def fig_ablation(ablation_root, outdir):
    runs = {}
    for d in sorted(ablation_root.iterdir()):
        if not d.is_dir():
            continue
        tdf = load_traj(d)
        if tdf is not None:
            runs[d.name] = tdf
    if len(runs) < 2:
        print("  [skip] Need >=2 ablation runs")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    labels = sorted(runs.keys())
    cmap_list = ["#534AB7", "#D85A30", "#0F6E56", "#D4537E"]

    # (0,0) PE curves
    ax = axes[0, 0]
    for i, lbl in enumerate(labels):
        ep_pe = runs[lbl].groupby("episode")["pred_err"].mean()
        if len(ep_pe) > 10:
            ep_pe = ep_pe.rolling(10, min_periods=1).mean()
        ax.plot(ep_pe.index, ep_pe.values, lw=1.5, color=cmap_list[i%len(cmap_list)], label=lbl)
    ax.set_xlabel("Episode"); ax.set_ylabel("PE"); ax.set_title("PE learning curves")
    ax.legend(fontsize=7)

    # (0,1) Alpha over episodes
    ax = axes[0, 1]
    for i, lbl in enumerate(labels):
        d = runs[lbl]
        if "alpha" in d.columns:
            ep_a = d.groupby("episode")["alpha"].mean()
            ax.plot(ep_a.index, ep_a.values, lw=1.5, color=cmap_list[i%len(cmap_list)], label=lbl)
    ax.set_xlabel("Episode"); ax.set_ylabel("α"); ax.set_title("Plasticity dynamics")
    ax.legend(fontsize=7)

    # (1,0) g_norm over episodes
    ax = axes[1, 0]
    for i, lbl in enumerate(labels):
        ep_g = runs[lbl].groupby("episode")["g_norm"].mean()
        ax.plot(ep_g.index, ep_g.values, lw=1.5, color=cmap_list[i%len(cmap_list)], label=lbl)
    ax.set_xlabel("Episode"); ax.set_ylabel("||g||"); ax.set_title("g magnitude")
    ax.legend(fontsize=7)

    # (1,1) Late stats bar comparison
    ax = axes[1, 1]
    metrics = ["pe_late", "alpha_late", "g_norm_late"]
    metric_vals = {m: [] for m in metrics}
    for lbl in labels:
        d = runs[lbl]; mx = d["episode"].max()
        late = d[d["episode"]>=max(0,mx-19)]
        metric_vals["pe_late"].append(late["pred_err"].mean())
        metric_vals["alpha_late"].append(late["alpha"].mean() if "alpha" in late.columns else 0)
        metric_vals["g_norm_late"].append(late["g_norm"].mean())

    x = np.arange(len(labels))
    w = 0.25
    ax.bar(x - w, metric_vals["pe_late"], w, color="#534AB7", alpha=0.7, label="PE")
    ax.bar(x, metric_vals["alpha_late"], w, color="#D85A30", alpha=0.7, label="α")
    ax.bar(x + w, [v/10 for v in metric_vals["g_norm_late"]], w, color="#0F6E56", alpha=0.7, label="||g||/10")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=15)
    ax.set_title("Late metrics comparison"); ax.legend(fontsize=8)

    fig.suptitle("Ablation: adaptive vs fixed vs fast", fontsize=14)
    savefig(fig, outdir / "ablation_comparison.png")


# ════════════════════════════════════════
# Main
# ════════════════════════════════════════

def analyze_run(run_dir):
    outdir = run_dir / "analysis"
    outdir.mkdir(exist_ok=True)
    df = load_traj(run_dir)
    if df is None:
        print(f"  [skip] No traj in {run_dir}")
        return
    print(f"\n  {run_dir.name}: {len(df)} rows, {df['episode'].nunique()} eps")
    for fn in [fig_pe, fig_g_alpha, fig_zone_dwell, fig_perturbation_response,
               fig_mixed_history, fig_film_gating]:
        try:
            fn(df, outdir)
        except Exception as e:
            print(f"  [error] {fn.__name__}: {e}")
    mx = df["episode"].max()
    late = df[df["episode"]>=max(0,mx-19)]
    print(f"  PE={late['pred_err'].mean():.4f} ||g||={late['g_norm'].mean():.3f}", end="")
    if "alpha" in late.columns:
        print(f" α={late['alpha'].mean():.3f}", end="")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="")
    ap.add_argument("--sweep_root", type=str, default="")
    ap.add_argument("--ablation_root", type=str, default="")
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

    if args.ablation_root:
        ar = Path(args.ablation_root)
        for d in sorted(ar.iterdir()):
            if d.is_dir():
                analyze_run(d)
        ao = ar / "ablation_analysis"
        ao.mkdir(exist_ok=True)
        try:
            fig_ablation(ar, ao)
        except Exception as e:
            print(f"  [error] ablation: {e}")


if __name__ == "__main__":
    main()