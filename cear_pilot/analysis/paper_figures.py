# cear_pilot/analysis/paper_figures.py
# -*- coding: utf-8 -*-
"""
Paper figure generation for ALIFE 2026 submission.

Hierarchical aggregation:
    Phase 2 seeds: aggregated within each Phase 1 seed (median over p2 seeds)
    Phase 1 seeds: final paper summary uses median ± IQR across p1 seeds

Expected directory structure:
    <root>/
      from_p1_s0/
        sweep/
          seed0/
            nperturb0/
            nperturb1/
            ...
          seed1/
            ...
        mixed/
          seed0/
            mixed_0_4_0/
            mixed_0_6_0/
            mixed_ramp/
          ...
        ablation/
          seed0/
            adaptive/
            fixed_005/   (or fixed_010 if older naming exists)
            fast_080/
          ...
      from_p1_s1/
      ...
      from_p1_s4/

Generates:
  Fig 2: Sweep dose-response (4-panel)
  Fig 3: Mixed history hysteresis (alpha + g_norm with block boundaries)
  Fig 4: Probe representation (cross-g z_t distance + zone ratio + heatmap)
  Fig 5: Ablation dynamics (adaptive vs fixed vs fast)
  Fig S1: Perturbation event-triggered response
  Fig S2: FiLM gating per block

Usage:
    python -m cear_pilot.analysis.paper_figures \
        --sweep_root outputs/phase2_all_xxx \
        --mixed_root outputs/phase2_all_xxx \
        --ablation_root outputs/phase2_all_xxx \
        --outdir paper_figures
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

DPI = 300
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.4,
    "font.size": 9,
    "font.family": "sans-serif",
    "axes.linewidth": 0.6,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "figure.titlesize": 12,
})

C_AD = "#534AB7"
C_FX = "#0F6E56"
C_FA = "#D85A30"
C_GM = "#993C1D"
ZC = ["#E24B4A", "#D85A30", "#888780", "#1D9E75", "#0F6E56"]


# ---------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------

def savefig(fig, path: Path):
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  [fig] {path.name}")


def load_traj(d: Path):
    for n in ["traj.parquet", "traj.csv"]:
        p = d / n
        if p.exists():
            try:
                return pd.read_parquet(p) if n.endswith(".parquet") else pd.read_csv(p)
            except Exception:
                try:
                    return pd.read_csv(p)
                except Exception:
                    pass
    return None


def load_probe(d: Path):
    p = d / "analysis" / "probe_representation.csv"
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            return None
    return None


def zt_cols(df: pd.DataFrame):
    return sorted(
        [c for c in df.columns if re.match(r"z_t_\d+$", c)],
        key=lambda x: int(x.split("_")[-1]),
    )


def gm_cols(df: pd.DataFrame):
    return sorted(
        [c for c in df.columns if re.match(r"gamma_\d+$", c)],
        key=lambda x: int(x.split("_")[-1]),
    )


def late(df: pd.DataFrame, n: int = 20):
    if "episode" not in df.columns or len(df) == 0:
        return df
    mx = int(df["episode"].max())
    return df[df["episode"] >= max(0, mx - n + 1)]


def ensure_numeric_series(s: pd.Series):
    return pd.to_numeric(s, errors="coerce")


def hierarchical_scalar_summary(data_by_p1: dict[int, list], fn):
    """
    data_by_p1: {p1_seed: [df_for_p2_seed0, df_for_p2_seed1, ...]}
    fn(df) -> scalar

    Returns:
        med, q1, q3, p1_medians
    """
    p1_medians = []
    for p1, dfs in sorted(data_by_p1.items()):
        vals = []
        for df in dfs:
            try:
                v = fn(df)
                if v is not None and np.isfinite(v):
                    vals.append(float(v))
            except Exception:
                continue
        if vals:
            p1_medians.append(float(np.median(vals)))

    if not p1_medians:
        return np.nan, np.nan, np.nan, []

    arr = np.asarray(p1_medians, dtype=float)
    return float(np.median(arr)), float(np.percentile(arr, 25)), float(np.percentile(arr, 75)), p1_medians


def hierarchical_curve_summary(data_by_p1: dict[int, list], series_fn):
    """
    data_by_p1: {p1_seed: [df_for_p2_seed0, ...]}
    series_fn(df) -> pd.Series indexed by episode

    Steps:
      1) Within each p1, align p2 curves and take median across p2 at each episode
      2) Across p1, align those median curves and take median/IQR
    """
    p1_curves = []

    for p1, dfs in sorted(data_by_p1.items()):
        curves = []
        for df in dfs:
            try:
                s = series_fn(df)
                if s is None or len(s) == 0:
                    continue
                s = ensure_numeric_series(s).dropna()
                if len(s) == 0:
                    continue
                curves.append(s)
            except Exception:
                continue

        if not curves:
            continue

        p2_aligned = pd.concat(curves, axis=1)
        p1_curve = p2_aligned.median(axis=1, skipna=True)
        p1_curves.append(p1_curve)

    if not p1_curves:
        return None, None, None, []

    p1_aligned = pd.concat(p1_curves, axis=1)
    med = p1_aligned.median(axis=1, skipna=True)
    q1 = p1_aligned.quantile(0.25, axis=1, interpolation="linear")
    q3 = p1_aligned.quantile(0.75, axis=1, interpolation="linear")
    return med, q1, q3, p1_curves


def hierarchical_event_trigger(data_by_p1: dict[int, list], value_getter, window: int = 40):
    """
    Event-triggered median/IQR with hierarchical aggregation.
    value_getter(df_sorted) -> 1D np.array of metric values aligned with rows of df_sorted
    """
    t_ax = np.arange(-window, window + 1)
    p1_traces = []

    for p1, dfs in sorted(data_by_p1.items()):
        p2_traces = []

        for df in dfs:
            if "perturbation_active" not in df.columns:
                continue

            ds = df.sort_values(["episode", "t"]).reset_index(drop=True)
            pa = pd.to_numeric(ds["perturbation_active"], errors="coerce").fillna(0).values
            onsets = [i for i in range(1, len(pa)) if pa[i] == 1 and pa[i - 1] == 0]
            if not onsets:
                continue

            vals = value_getter(ds)
            if vals is None:
                continue

            vals = np.asarray(vals, dtype=float)
            if len(vals) != len(ds):
                continue

            tr = []
            for loc in onsets:
                row = np.full(2 * window + 1, np.nan)
                lo = max(0, loc - window)
                hi = min(len(vals), loc + window + 1)
                for j in range(lo, hi):
                    row[j - loc + window] = vals[j]
                tr.append(row)

            if not tr:
                continue

            tr = np.asarray(tr, dtype=float)
            p2_trace = np.nanmedian(tr, axis=0)
            p2_traces.append(p2_trace)

        if not p2_traces:
            continue

        p2_traces = np.asarray(p2_traces, dtype=float)
        p1_trace = np.nanmedian(p2_traces, axis=0)
        p1_traces.append(p1_trace)

    if not p1_traces:
        return t_ax, None, None, None

    p1_traces = np.asarray(p1_traces, dtype=float)
    med = np.nanmedian(p1_traces, axis=0)
    q1 = np.nanpercentile(p1_traces, 25, axis=0)
    q3 = np.nanpercentile(p1_traces, 75, axis=0)
    return t_ax, med, q1, q3


# ---------------------------------------------------------
# Directory collectors
# ---------------------------------------------------------

def iter_from_p1_dirs(root: Path):
    if not root.exists():
        return
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r"from_p1_s(\d+)$", d.name)
        if m:
            yield int(m.group(1)), d


def collect_sweep_runs(root: Path):
    """
    Returns:
      data[nperturb][p1_seed] = [df_for_p2_seed0, ...]
    """
    data = defaultdict(lambda: defaultdict(list))

    for p1_seed, p1_dir in iter_from_p1_dirs(root):
        sweep_dir = p1_dir / "sweep"
        if not sweep_dir.exists():
            continue

        for seed_dir in sorted(sweep_dir.iterdir()):
            if not seed_dir.is_dir() or not re.match(r"seed\d+$", seed_dir.name):
                continue

            for run_dir in sorted(seed_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                m = re.match(r"nperturb(\d+)$", run_dir.name)
                if not m:
                    continue
                npert = int(m.group(1))
                df = load_traj(run_dir)
                if df is not None:
                    data[npert][p1_seed].append(df)

    return data


def collect_mixed_runs(root: Path):
    """
    Returns:
      data[schedule_key][p1_seed] = [df_for_p2_seed0, ...]
    """
    data = defaultdict(lambda: defaultdict(list))
    valid_keys = {"mixed_0_4_0", "mixed_0_6_0", "mixed_ramp"}

    for p1_seed, p1_dir in iter_from_p1_dirs(root):
        mixed_dir = p1_dir / "mixed"
        if not mixed_dir.exists():
            continue

        for seed_dir in sorted(mixed_dir.iterdir()):
            if not seed_dir.is_dir() or not re.match(r"seed\d+$", seed_dir.name):
                continue

            for run_dir in sorted(seed_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                if run_dir.name not in valid_keys:
                    continue
                df = load_traj(run_dir)
                if df is not None:
                    data[run_dir.name][p1_seed].append(df)

    return data


def collect_ablation_runs(root: Path):
    """
    Returns:
      data[condition][p1_seed] = [df_for_p2_seed0, ...]
    condition canonicalized to:
      adaptive, fixed_005, fixed_010, fast_080
    """
    data = defaultdict(lambda: defaultdict(list))

    for p1_seed, p1_dir in iter_from_p1_dirs(root):
        abl_dir = p1_dir / "ablation"
        if not abl_dir.exists():
            continue

        for seed_dir in sorted(abl_dir.iterdir()):
            if not seed_dir.is_dir() or not re.match(r"seed\d+$", seed_dir.name):
                continue

            for run_dir in sorted(seed_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                cond = run_dir.name
                if cond not in {"adaptive", "fixed_005", "fixed_010", "fast_080"}:
                    continue
                df = load_traj(run_dir)
                if df is not None:
                    data[cond][p1_seed].append(df)

    return data


def collect_probe_runs(root: Path):
    """
    Returns:
      data[schedule_key][p1_seed] = [probe_df_for_p2_seed0, ...]
    """
    data = defaultdict(lambda: defaultdict(list))
    valid_keys = {"mixed_0_4_0", "mixed_0_6_0", "mixed_ramp"}

    for p1_seed, p1_dir in iter_from_p1_dirs(root):
        mixed_dir = p1_dir / "mixed"
        if not mixed_dir.exists():
            continue

        for seed_dir in sorted(mixed_dir.iterdir()):
            if not seed_dir.is_dir() or not re.match(r"seed\d+$", seed_dir.name):
                continue

            for run_dir in sorted(seed_dir.iterdir()):
                if not run_dir.is_dir() or run_dir.name not in valid_keys:
                    continue
                pdf = load_probe(run_dir)
                if pdf is not None:
                    data[run_dir.name][p1_seed].append(pdf)

    return data


# ---------------------------------------------------------
# Fig 2: Sweep dose-response
# ---------------------------------------------------------

def fig2_sweep(sweep_root: Path, outdir: Path):
    runs = collect_sweep_runs(sweep_root)
    if len(runs) < 2:
        print("  [skip] fig2: need >=2 sweep conditions")
        return

    ns = sorted(runs.keys())

    def agg_scalar(fn):
        meds, q1s, q3s = [], [], []
        for n in ns:
            med, q1, q3, _ = hierarchical_scalar_summary(runs[n], fn)
            meds.append(med)
            q1s.append(q1)
            q3s.append(q3)
        return np.asarray(meds), np.asarray(q1s), np.asarray(q3s)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))

    # (a) PE
    ax = axes[0]
    m, lo, hi = agg_scalar(lambda df: late(df)["pred_err"].mean() if "pred_err" in df.columns else np.nan)
    ax.fill_between(ns, lo, hi, color=C_AD, alpha=0.15)
    ax.plot(ns, m, "s-", color=C_AD, lw=1.5, markersize=5)
    ax.set_xlabel("Perturbations / ep")
    ax.set_ylabel("Late PE")
    ax.set_title("(a) Prediction error")

    # (b) Alpha
    ax = axes[1]
    m, lo, hi = agg_scalar(lambda df: late(df)["alpha"].mean() if "alpha" in df.columns else np.nan)
    ax.fill_between(ns, lo, hi, color=C_FA, alpha=0.15)
    ax.plot(ns, m, "D-", color=C_FA, lw=1.5, markersize=5)
    ax.set_xlabel("Perturbations / ep")
    ax.set_ylabel("Late α")
    ax.set_title("(b) Self-modulating plasticity")

    # (c) FiLM |γ|
    ax = axes[2]

    def get_gabs(df):
        gc = gm_cols(df)
        if not gc:
            return np.nan
        return late(df)[gc].abs().mean().mean()

    m, lo, hi = agg_scalar(get_gabs)
    ax.fill_between(ns, lo, hi, color=C_GM, alpha=0.15)
    ax.plot(ns, m, "o-", color=C_GM, lw=1.5, markersize=5)
    ax.set_xlabel("Perturbations / ep")
    ax.set_ylabel("Late |γ|")
    ax.set_title("(c) Salience modulation")

    # (d) Zone dwell
    ax = axes[3]
    for zi in range(5):
        m, _, _ = agg_scalar(
            lambda df, z=zi: (
                len(late(df)[late(df)["zone_id"] == z]) / max(len(late(df)), 1)
                if "zone_id" in df.columns else np.nan
            )
        )
        ax.plot(ns, m, "o-", color=ZC[zi], lw=1.2, markersize=4, label=f"Z{zi}")

    ax.set_xlabel("Perturbations / ep")
    ax.set_ylabel("Frac (late)")
    ax.set_title("(d) Zone dwell (stable)")
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle(
        "Fig 2: Dose-dependent internal reorganization under stable behavior",
        fontsize=11,
        y=1.04,
    )
    savefig(fig, outdir / "fig2_sweep_dose_response.png")


# ---------------------------------------------------------
# Fig 3: Mixed history hysteresis
# ---------------------------------------------------------

def fig3_mixed(mixed_root: Path, outdir: Path):
    data = collect_mixed_runs(mixed_root)
    schedule_meta = [
        ("mixed_0_4_0", "0→4→0"),
        ("mixed_0_6_0", "0→6→0"),
    ]

    found = [(key, label, data[key]) for key, label in schedule_meta if key in data and data[key]]
    if not found:
        print("  [skip] fig3: no mixed runs")
        return

    fig, axes = plt.subplots(len(found), 2, figsize=(12, 3.5 * len(found)), squeeze=False)

    for row, (sched_key, label, by_p1) in enumerate(found):
        # Reference block info from first available df
        ref = None
        for dfs in by_p1.values():
            if dfs:
                ref = dfs[0]
                break

        blk_info = []
        if ref is not None and "block_id" in ref.columns and "n_perturb_setting" in ref.columns:
            ep_blk = ref.groupby("episode")[["block_id", "n_perturb_setting"]].first()
            changes = ep_blk[ep_blk["block_id"] != ep_blk["block_id"].shift(1)]
            for ep_idx in changes.index:
                blk_info.append(
                    (
                        int(ep_idx),
                        int(changes.loc[ep_idx, "block_id"]),
                        int(changes.loc[ep_idx, "n_perturb_setting"]),
                    )
                )

        for col, (metric, ylabel, color) in enumerate([
            ("alpha", "α", C_FA),
            ("g_norm", "||g||", C_FX),
        ]):
            ax = axes[row, col]

            med, q1, q3, p1_curves = hierarchical_curve_summary(
                by_p1,
                lambda df, m=metric: df.groupby("episode")[m].mean() if m in df.columns else pd.Series(dtype=float),
            )

            # Thin lines = p1-level medians over p2 seeds
            for curve in p1_curves:
                ax.plot(curve.index, curve.values, lw=0.7, alpha=0.25, color=color)

            # Bold line + IQR across p1 seeds
            if med is not None:
                ax.fill_between(med.index, q1.values, q3.values, color=color, alpha=0.15)
                ax.plot(med.index, med.values, lw=2, color=color)

            for bc, blk, nP in blk_info:
                ax.axvline(bc, color="#999", lw=0.8, ls="--", alpha=0.5)
                y_top = ax.get_ylim()[1] if len(ax.lines) or len(ax.collections) else 1.0
                ax.annotate(
                    f"nP={nP}",
                    xy=(bc + 1, y_top * 0.95),
                    fontsize=7,
                    color="#666",
                    va="top",
                )

            ax.set_xlabel("Episode")
            ax.set_ylabel(ylabel)
            sub = chr(97 + row * 2 + col)
            title_word = "Plasticity" if metric == "alpha" else "Perspective magnitude"
            ax.set_title(f"({sub}) {title_word} — {label}")

    fig.suptitle("Fig 3: Perceptual hysteresis within one agent", fontsize=11, y=1.02)
    savefig(fig, outdir / "fig3_mixed_hysteresis.png")


# ---------------------------------------------------------
# Fig 4: Probe representation + zone ratio
# ---------------------------------------------------------

def fig4_probe(mixed_root: Path, outdir: Path):
    probe_runs = collect_probe_runs(mixed_root)
    # Prefer 0_4_0, then 0_6_0, then ramp
    chosen_key = None
    for key in ["mixed_0_4_0", "mixed_0_6_0", "mixed_ramp"]:
        if key in probe_runs and probe_runs[key]:
            chosen_key = key
            break

    if chosen_key is None:
        print("  [skip] fig4: no probe data found")
        return

    # Flatten probe dfs only after p2→p1 selection target is identified.
    # For fig4 panels, we summarize probe-level scalar quantities hierarchically.
    # For heatmap, we use the p1-median representative by taking the first p1's p2-median-compatible run.
    by_p1 = probe_runs[chosen_key]

    first_df = None
    for dfs in by_p1.values():
        if dfs:
            first_df = dfs[0]
            break
    if first_df is None:
        print("  [skip] fig4: empty probe data")
        return

    labels = sorted([str(l) for l in first_df["g_label"].dropna().unique()])
    g_labels = sorted([l for l in labels if l != "g_zero"])
    all_labels = g_labels + (["g_zero"] if "g_zero" in labels else [])

    if len(all_labels) < 2:
        print("  [skip] fig4: insufficient g_label diversity")
        return

    zc = zt_cols(first_df)
    if not zc:
        print("  [skip] fig4: no z_t columns")
        return

    zones = sorted(first_df["zone_id"].dropna().unique())
    if len(g_labels) < 2:
        print("  [skip] fig4: need at least two non-zero g labels")
        return

    gA, gB = g_labels[0], g_labels[-1]
    gl_main = g_labels[-1]

    fig = plt.figure(figsize=(14, 4.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.2])

    # (a) Cross-g z_t distance by zone
    ax = fig.add_subplot(gs[0])

    zone_meds = []
    zone_q1s = []
    zone_q3s = []

    for z in zones:
        def zone_dist_stat(pdf: pd.DataFrame, zone=z):
            dists = []
            for pi in sorted(pdf["probe_idx"].dropna().unique()):
                va = pdf[
                    (pdf["probe_idx"] == pi) &
                    (pdf["zone_id"] == zone) &
                    (pdf["g_label"].astype(str) == gA)
                ][zc].values
                vb = pdf[
                    (pdf["probe_idx"] == pi) &
                    (pdf["zone_id"] == zone) &
                    (pdf["g_label"].astype(str) == gB)
                ][zc].values
                if len(va) > 0 and len(vb) > 0:
                    dists.append(float(np.linalg.norm(va[0] - vb[0])))
            return np.median(dists) if dists else np.nan

        med, q1, q3, p1_vals = hierarchical_scalar_summary(by_p1, zone_dist_stat)
        zone_meds.append(med)
        zone_q1s.append(q1)
        zone_q3s.append(q3)

        if p1_vals:
            ax.scatter([z] * len(p1_vals), p1_vals, c="black", s=10, alpha=0.35, zorder=5)

    ax.bar(zones, zone_meds, color=[ZC[int(z) % 5] for z in zones], alpha=0.7, edgecolor="white", lw=0.5)
    for z, q1, q3 in zip(zones, zone_q1s, zone_q3s):
        if np.isfinite(q1) and np.isfinite(q3):
            ax.vlines(z, q1, q3, color="black", lw=1.0, alpha=0.8)
    ax.set_xticks(zones)
    ax.set_xticklabels([f"Z{int(z)}" for z in zones], fontsize=8)
    ax.set_ylabel(f"||z_t({gA}) − z_t({gB})||")
    ax.set_title("(a) Perceptual distance\n(same input, different history)")

    # (b) Zone-dependent ratio
    ax = fig.add_subplot(gs[1])
    ratio_vals, ratio_q1, ratio_q3 = [], [], []

    for z in zones:
        def ratio_stat(pdf: pd.DataFrame, zone=z):
            df_g = pdf[(pdf["zone_id"] == zone) & (pdf["g_label"].astype(str) == gl_main)]
            df_0 = pdf[(pdf["zone_id"] == zone) & (pdf["g_label"].astype(str) == "g_zero")]
            if len(df_g) == 0 or len(df_0) == 0 or "z_shift" not in pdf.columns:
                return np.nan
            zs_g = pd.to_numeric(df_g["z_shift"], errors="coerce").mean()
            zs_0 = pd.to_numeric(df_0["z_shift"], errors="coerce").mean()
            if not np.isfinite(zs_g) or not np.isfinite(zs_0):
                return np.nan
            return float(zs_g / max(zs_0, 1e-6))

        med, q1, q3, _ = hierarchical_scalar_summary(by_p1, ratio_stat)
        ratio_vals.append(med)
        ratio_q1.append(q1)
        ratio_q3.append(q3)

    bars = ax.bar(
        zones,
        ratio_vals,
        color=[ZC[int(z) % 5] for z in zones],
        alpha=0.7,
        edgecolor="white",
        lw=0.5,
    )
    for z, q1, q3 in zip(zones, ratio_q1, ratio_q3):
        if np.isfinite(q1) and np.isfinite(q3):
            ax.vlines(z, q1, q3, color="black", lw=1.0, alpha=0.8)
    for bar, rv in zip(bars, ratio_vals):
        if np.isfinite(rv):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f"{rv:.1f}×",
                ha="center",
                fontsize=8,
                color="#444",
            )
    ax.set_xticks(zones)
    ax.set_xticklabels([f"Z{int(z)}" for z in zones], fontsize=8)
    ax.set_ylabel(f"z_shift({gl_main}) / z_shift(g_zero)")
    ax.set_title("(b) Relative perspective\ninfluence by zone")

    # (c) z_t difference heatmap
    ax = fig.add_subplot(gs[2])

    # Use a representative p1-level median-compatible sample:
    # pick first available p1, then first pdf inside it.
    rep_pdf = None
    for _, pdfs in sorted(by_p1.items()):
        if pdfs:
            rep_pdf = pdfs[0]
            break

    if rep_pdf is not None:
        dfA = rep_pdf[rep_pdf["g_label"].astype(str) == gA].sort_values(["zone_id", "probe_idx"])[zc].values
        dfB = rep_pdf[rep_pdf["g_label"].astype(str) == gB].sort_values(["zone_id", "probe_idx"])[zc].values
        if len(dfA) == len(dfB) and len(dfA) > 0:
            diff = dfA - dfB
            vmax = float(np.abs(diff).max()) if np.abs(diff).max() > 0 else 1.0
            im = ax.imshow(
                diff,
                aspect="auto",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                interpolation="nearest",
            )
            ax.set_xlabel("z dimension")
            ax.set_ylabel("Probe (sorted by zone)")
            ax.set_title(f"(c) z_t difference\n({gA} vs {gB})")
            plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

            zone_counts = rep_pdf[rep_pdf["g_label"].astype(str) == gA].groupby("zone_id").size()
            cumsum = 0
            for z in sorted(zone_counts.index)[:-1]:
                cumsum += zone_counts[z]
                ax.axhline(cumsum - 0.5, color="black", lw=0.5, ls="--", alpha=0.5)

    fig.suptitle(
        "Fig 4: Same observation, different perspective → different perception",
        fontsize=11,
        y=1.04,
    )
    savefig(fig, outdir / "fig4_probe_representation.png")


# ---------------------------------------------------------
# Fig 5: Ablation dynamics
# ---------------------------------------------------------

def fig5_ablation(ablation_root: Path, outdir: Path):
    data = collect_ablation_runs(ablation_root)

    conds = [
        ("adaptive", C_AD, "Adaptive"),
        ("fixed_005", C_FX, "Fixed (α=0.05)"),
        ("fixed_010", C_FX, "Fixed (α=0.10)"),
        ("fast_080", C_FA, "Fast (α=0.80)"),
    ]

    found = [(cond, color, label, data[cond]) for cond, color, label in conds if cond in data and data[cond]]
    if len(found) < 2:
        print("  [skip] fig5: need >=2 ablation conditions")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))

    for i, (metric, ylabel, title) in enumerate([
        ("alpha", "α", "(a) Plasticity dynamics"),
        ("pred_err", "PE (smoothed)", "(b) Prediction error"),
        ("g_norm", "||g||", "(c) Perspective magnitude"),
    ]):
        ax = axes[i]

        for cond, color, label, by_p1 in found:
            def make_series(df, m=metric):
                if m not in df.columns:
                    return pd.Series(dtype=float)
                s = df.groupby("episode")[m].mean()
                if m == "pred_err" and len(s) > 1:
                    s = s.rolling(10, min_periods=1).mean()
                return s

            med, q1, q3, p1_curves = hierarchical_curve_summary(by_p1, make_series)

            for curve in p1_curves:
                ax.plot(curve.index, curve.values, lw=0.7, alpha=0.22, color=color)

            if med is not None:
                ax.fill_between(med.index, q1.values, q3.values, color=color, alpha=0.12)
                ax.plot(med.index, med.values, lw=2, color=color, label=label)

        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)

    fig.suptitle(
        "Fig 5: Self-modulating plasticity vs fixed vs fast update",
        fontsize=11,
        y=1.04,
    )
    savefig(fig, outdir / "fig5_ablation_dynamics.png")


# ---------------------------------------------------------
# Fig S1: Perturbation event-triggered response
# ---------------------------------------------------------

def figS1_event(sweep_root: Path, outdir: Path):
    runs = collect_sweep_runs(sweep_root)
    if 4 not in runs:
        print("  [skip] figS1: no nP=4 run")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    specs = [
        ("g_norm", "||g||", C_FX, "(a) Perspective response"),
        ("alpha", "α", C_FA, "(b) Plasticity response"),
    ]

    for i, (metric, ylabel, color, title) in enumerate(specs):
        ax = axes[i]

        t_ax, med, q1, q3 = hierarchical_event_trigger(
            runs[4],
            value_getter=lambda ds, m=metric: pd.to_numeric(ds[m], errors="coerce").values if m in ds.columns else None,
            window=40,
        )
        if med is not None:
            ax.fill_between(t_ax, q1, q3, color=color, alpha=0.15)
            ax.plot(t_ax, med, color=color, lw=2)
            ax.axvline(0, color="#999", lw=0.5, ls="--")
            ax.axvspan(0, 15, color=C_FA, alpha=0.04)

        ax.set_xlabel("Steps from onset")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    fig.suptitle("Fig S1: Event-triggered perturbation response (nP=4)", fontsize=11, y=1.04)
    savefig(fig, outdir / "figS1_perturbation_response.png")


# ---------------------------------------------------------
# Fig S2: FiLM gating per block
# ---------------------------------------------------------

def figS2_film(mixed_root: Path, outdir: Path):
    data = collect_mixed_runs(mixed_root)
    chosen_key = None
    for key in ["mixed_0_4_0", "mixed_0_6_0", "mixed_ramp"]:
        if key in data and data[key]:
            chosen_key = key
            break

    if chosen_key is None:
        print("  [skip] figS2: no mixed runs")
        return

    by_p1 = data[chosen_key]

    ref = None
    for dfs in by_p1.values():
        if dfs:
            ref = dfs[0]
            break

    if ref is None or "block_id" not in ref.columns:
        print("  [skip] figS2: no block_id")
        return

    gc = gm_cols(ref)
    if not gc:
        print("  [skip] figS2: no gamma columns")
        return

    blocks = sorted(ref["block_id"].dropna().unique())
    n_dims = len(gc)
    x_pos = np.arange(n_dims)
    width = 0.8 / max(len(blocks), 1)
    bc = [C_AD, C_FA, C_FX, "#D4537E", "#888780"]

    fig, ax = plt.subplots(figsize=(10, 3.5))

    for bi, blk in enumerate(blocks):
        def block_gamma_stat(df, block=blk):
            if "block_id" not in df.columns:
                return None
            bd = df[df["block_id"] == block]
            if len(bd) == 0:
                return None
            mx = int(bd["episode"].max())
            bl = bd[bd["episode"] >= max(int(bd["episode"].min()), mx - 9)]
            return bl[gc].mean().values

        # hierarchical vector aggregation
        p1_vecs = []
        for p1, dfs in sorted(by_p1.items()):
            p2_vecs = []
            for df in dfs:
                vec = block_gamma_stat(df, blk)
                if vec is not None and np.all(np.isfinite(vec)):
                    p2_vecs.append(vec)
            if p2_vecs:
                p1_vecs.append(np.median(np.vstack(p2_vecs), axis=0))

        if not p1_vecs:
            continue

        gvals = np.median(np.vstack(p1_vecs), axis=0)
        offset = (bi - len(blocks) / 2 + 0.5) * width
        nP = None
        if "n_perturb_setting" in ref.columns:
            try:
                nP = int(ref[ref["block_id"] == blk]["n_perturb_setting"].iloc[0])
            except Exception:
                nP = None

        label = f"Block {int(blk)}" if nP is None else f"Block {int(blk)} (nP={nP})"
        ax.bar(
            x_pos + offset,
            gvals,
            width,
            alpha=0.7,
            color=bc[bi % len(bc)],
            label=label,
            edgecolor="white",
            lw=0.3,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"z{i}" for i in range(n_dims)], fontsize=7)
    ax.axhline(0, color="#ccc", lw=0.5)
    ax.set_ylabel("γ (late block mean)")
    ax.set_title("Fig S2: Salience gating pattern per block")
    ax.legend(fontsize=7, ncol=3)
    savefig(fig, outdir / "figS2_film_per_block.png")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_root", type=str, default="")
    ap.add_argument("--mixed_root", type=str, default="")
    ap.add_argument("--ablation_root", type=str, default="")
    ap.add_argument("--outdir", type=str, default="paper_figures")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {outdir}")

    if args.sweep_root:
        sr = Path(args.sweep_root)
        print(f"\n=== Sweep root: {sr} ===")
        try:
            fig2_sweep(sr, outdir)
        except Exception as e:
            print(f"  [error] fig2: {e}")
        try:
            figS1_event(sr, outdir)
        except Exception as e:
            print(f"  [error] figS1: {e}")

    if args.mixed_root:
        mr = Path(args.mixed_root)
        print(f"\n=== Mixed root: {mr} ===")
        try:
            fig3_mixed(mr, outdir)
        except Exception as e:
            print(f"  [error] fig3: {e}")
        try:
            fig4_probe(mr, outdir)
        except Exception as e:
            print(f"  [error] fig4: {e}")
        try:
            figS2_film(mr, outdir)
        except Exception as e:
            print(f"  [error] figS2: {e}")

    if args.ablation_root:
        ar = Path(args.ablation_root)
        print(f"\n=== Ablation root: {ar} ===")
        try:
            fig5_ablation(ar, outdir)
        except Exception as e:
            print(f"  [error] fig5: {e}")

    print(f"\nDone. Figures in: {outdir}")


if __name__ == "__main__":
    main()