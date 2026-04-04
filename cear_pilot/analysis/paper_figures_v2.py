# cear_pilot/analysis/paper_figures_v2.py
# -*- coding: utf-8 -*-
"""
Paper figures for ALIFE 2026.

Hierarchical aggregation:
    Phase 2 seeds: aggregated within each Phase 1 seed (median over p2 seeds)
    Phase 1 seeds: final summary uses median ± IQR across p1 seeds

Expected directory structure:
    <root>/
      from_p1_s0/
        sweep/
          seed0/
            nperturb0/
            nperturb4/
            ...
        mixed/
          seed0/
            mixed_0_4_0/
            mixed_0_6_0/
            mixed_ramp/
        ablation/
          seed0/
            adaptive/
            fixed_005/ or fixed_010/
            fast_080/
      from_p1_s1/
      ...

Generates:
  Fig 2: Mixed 0→4→0 — schedule strip + alpha + ||g|| + |γ|
  Fig 3: Perspective shapes perception — (a) z_shift, (b) per-dim reshape, (c) perturbation response
  Fig 4: Ablation — alpha dynamics + PE + ||g||
  Fig S1: Sweep dose-response
  Fig S2: Per-block gating

Usage: 
python -m cear_pilot.analysis.paper_figures_v2 \
  --sweep_root outputs/phase2_all_20260403_011254 \
  --mixed_root outputs/phase2_all_20260403_011254 \
  --ablation_root outputs/phase2_all_20260403_011254 \
  --outdir outputs/phase2_all_20260403_011254/paper_figures_v2
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DPI = 300
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.12,
    "grid.linewidth": 0.3,
    "font.size": 9,
    "font.family": "sans-serif",
    "axes.linewidth": 0.5,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
})

C_ALPHA = "#D85A30"
C_GNORM = "#0F6E56"
C_GAMMA = "#993C1D"
C_AD = "#534AB7"
C_FX = "#0F6E56"
C_FA = "#D85A30"
C_PE = "#534AB7"
ZC = ["#E24B4A", "#D85A30", "#888780", "#1D9E75", "#0F6E56"]


# ---------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------

def savefig(fig, path: Path):
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.12)
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


def zraw_cols(df: pd.DataFrame):
    return sorted(
        [c for c in df.columns if re.match(r"z_raw_\d+$", c)],
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
    p1_medians = []
    for _, dfs in sorted(data_by_p1.items()):
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
    p1_curves = []
    for _, dfs in sorted(data_by_p1.items()):
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
    t_ax = np.arange(-window, window + 1)
    p1_traces = []

    for _, dfs in sorted(data_by_p1.items()):
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

            p2_traces.append(np.nanmedian(np.asarray(tr, dtype=float), axis=0))

        if p2_traces:
            p1_traces.append(np.nanmedian(np.asarray(p2_traces, dtype=float), axis=0))

    if not p1_traces:
        return t_ax, None, None, None

    p1_traces = np.asarray(p1_traces, dtype=float)
    med = np.nanmedian(p1_traces, axis=0)
    q1 = np.nanpercentile(p1_traces, 25, axis=0)
    q3 = np.nanpercentile(p1_traces, 75, axis=0)
    return t_ax, med, q1, q3


def hierarchical_vector_summary(data_by_p1: dict[int, list], fn):
    p1_vecs = []
    for _, dfs in sorted(data_by_p1.items()):
        p2_vecs = []
        for df in dfs:
            try:
                vec = fn(df)
                if vec is None:
                    continue
                arr = np.asarray(vec, dtype=float)
                if arr.ndim != 1 or not np.all(np.isfinite(arr)):
                    continue
                p2_vecs.append(arr)
            except Exception:
                continue
        if p2_vecs:
            p1_vecs.append(np.median(np.vstack(p2_vecs), axis=0))
    if not p1_vecs:
        return None, []
    arr = np.vstack(p1_vecs)
    return np.median(arr, axis=0), p1_vecs


# ---------------------------------------------------------
# Directory collectors
# ---------------------------------------------------------

def iter_from_p1_dirs(root: Path):
    if root is None or not root.exists():
        return
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r"from_p1_s(\d+)$", d.name)
        if m:
            yield int(m.group(1)), d


def collect_sweep_runs(root: Path):
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
                df = load_traj(run_dir)
                if df is not None:
                    data[run_dir.name][p1_seed].append(df)
    return data


def collect_ablation_runs(root: Path):
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
# Fig 2: Mixed 0→4→0 — stacked time series
# ---------------------------------------------------------

def fig2_mixed_history(mixed_root: Path, outdir: Path):
    data = collect_mixed_runs(mixed_root)
    if "mixed_0_4_0" not in data or not data["mixed_0_4_0"]:
        print("  [skip] fig2: no mixed_0_4_0 traj")
        return

    by_p1 = data["mixed_0_4_0"]
    ref = next((dfs[0] for _, dfs in sorted(by_p1.items()) if dfs), None)
    if ref is None:
        print("  [skip] fig2: empty mixed_0_4_0")
        return

    blk_info = []
    if "block_id" in ref.columns and "n_perturb_setting" in ref.columns:
        ep_blk = ref.groupby("episode")[["block_id", "n_perturb_setting"]].first()
        changes = ep_blk[ep_blk["block_id"] != ep_blk["block_id"].shift(1)]
        for ep_idx in changes.index:
            blk_info.append((int(ep_idx), int(changes.loc[ep_idx, "n_perturb_setting"])))

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(10, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [0.6, 1, 1, 1], "hspace": 0.08},
    )

    ax = axes[0]
    if "n_perturb_setting" in ref.columns:
        ep_nP = ref.groupby("episode")["n_perturb_setting"].first()
        ax.fill_between(ep_nP.index, 0, ep_nP.values, step="post", color="#ddd", alpha=0.6)
        ax.plot(ep_nP.index, ep_nP.values, drawstyle="steps-post", color="#888", lw=1.5)
    ax.set_ylabel("nP", fontsize=9)
    ymax = 5
    if "n_perturb_setting" in ref.columns:
        ymax = max(float(ref["n_perturb_setting"].max()) + 0.5, 5)
    ax.set_ylim(-0.3, ymax)
    ax.set_title("Block schedule", fontsize=10, pad=4)
    ax.set_yticks(sorted(set([0, 4] + [int(x) for _, x in blk_info])))
    for bc, _ in blk_info:
        ax.axvline(bc, color="#aaa", lw=0.6, ls="--", alpha=0.5)

    for i, (metric, ylabel, color) in enumerate([
        ("alpha", "α (plasticity)", C_ALPHA),
        ("g_norm", "||g||", C_GNORM),
    ]):
        ax = axes[i + 1]
        med, q1, q3, p1_curves = hierarchical_curve_summary(
            by_p1,
            lambda df, m=metric: df.groupby("episode")[m].mean() if m in df.columns else pd.Series(dtype=float),
        )
        for curve in p1_curves:
            ax.plot(curve.index, curve.values, lw=0.3, alpha=0.2, color=color)
        if med is not None:
            ax.fill_between(med.index, q1.values, q3.values, color=color, alpha=0.12)
            ax.plot(med.index, med.values, lw=2, color=color)
        ax.set_ylabel(ylabel, fontsize=9)
        for bc, _ in blk_info:
            ax.axvline(bc, color="#aaa", lw=0.6, ls="--", alpha=0.5)

    ax = axes[3]
    med, q1, q3, p1_curves = hierarchical_curve_summary(
        by_p1,
        lambda df: df.groupby("episode")[gm_cols(df)].apply(lambda x: np.abs(x.values).mean()) if gm_cols(df) else pd.Series(dtype=float),
    )
    for curve in p1_curves:
        ax.plot(curve.index, curve.values, lw=0.3, alpha=0.2, color=C_GAMMA)
    if med is not None:
        ax.fill_between(med.index, q1.values, q3.values, color=C_GAMMA, alpha=0.12)
        ax.plot(med.index, med.values, lw=2, color=C_GAMMA)
    ax.set_ylabel("Mean |γ|", fontsize=9)
    ax.set_xlabel("Episode")
    for bc, _ in blk_info:
        ax.axvline(bc, color="#aaa", lw=0.6, ls="--", alpha=0.5)

    savefig(fig, outdir / "fig2_mixed_history.png")


# ---------------------------------------------------------
# Fig 3: Perspective shapes perception
# ---------------------------------------------------------

def fig3_perspective_perception(mixed_root: Path | None, sweep_root: Path | None, outdir: Path):
    probe_runs = collect_probe_runs(mixed_root) if mixed_root else {}

    chosen_key = None
    for key in ["mixed_0_4_0", "mixed_0_6_0", "mixed_ramp"]:
        if key in probe_runs and probe_runs[key]:
            chosen_key = key
            break
    if chosen_key is None:
        print("  [skip] fig3: no probe data")
        return

    by_p1_probe = probe_runs[chosen_key]
    first_df = next((dfs[0] for _, dfs in sorted(by_p1_probe.items()) if dfs), None)
    if first_df is None:
        print("  [skip] fig3: empty probe data")
        return

    zc = zt_cols(first_df)
    zr = zraw_cols(first_df)
    n_z = len(zc)
    if n_z == 0:
        print("  [skip] fig3: no z_t columns")
        return

    labels = sorted([str(l) for l in first_df["g_label"].dropna().unique()])
    trained_candidates = [l for l in labels if l != "g_zero"]
    if not trained_candidates or "g_zero" not in labels:
        print("  [skip] fig3: need trained label and g_zero")
        return

    best_label = None
    best_shift = -np.inf
    for label in trained_candidates:
        med, _, _, _ = hierarchical_scalar_summary(
            by_p1_probe,
            lambda df, lab=label: pd.to_numeric(df[df["g_label"].astype(str) == lab]["z_shift"], errors="coerce").mean()
            if "z_shift" in df.columns else np.nan,
        )
        if np.isfinite(med) and med > best_shift:
            best_shift = med
            best_label = label

    if best_label is None:
        print("  [skip] fig3: could not identify trained g label")
        return

    sweep_runs = collect_sweep_runs(sweep_root) if sweep_root else {}
    has_panel_c = 4 in sweep_runs and bool(sweep_runs[4])
    n_cols = 3 if has_panel_c else 2
    width_ratios = [1, 1.2, 1.3] if has_panel_c else [1, 1.2]

    fig = plt.figure(figsize=(4.5 * n_cols, 4))
    gs = gridspec.GridSpec(1, n_cols, width_ratios=width_ratios, wspace=0.35)

    ax = fig.add_subplot(gs[0])
    tm, tq1, tq3, tp1 = hierarchical_scalar_summary(
        by_p1_probe,
        lambda df: pd.to_numeric(df[df["g_label"].astype(str) == best_label]["z_shift"], errors="coerce").mean()
        if "z_shift" in df.columns else np.nan,
    )
    zm, zq1, zq3, zp1 = hierarchical_scalar_summary(
        by_p1_probe,
        lambda df: pd.to_numeric(df[df["g_label"].astype(str) == "g_zero"]["z_shift"], errors="coerce").mean()
        if "z_shift" in df.columns else np.nan,
    )

    means = [tm, zm]
    errs = [max(tm - tq1, tq3 - tm) if np.isfinite(tm) else 0.0, max(zm - zq1, zq3 - zm) if np.isfinite(zm) else 0.0]
    ax.bar(
        [0, 1],
        means,
        yerr=errs,
        capsize=4,
        color=[C_AD, "#bbb"],
        alpha=0.75,
        edgecolor="white",
        lw=0.5,
        error_kw={"lw": 0.8},
    )
    ratio = tm / max(zm, 1e-6) if np.isfinite(tm) and np.isfinite(zm) else np.nan
    if np.isfinite(ratio):
        bracket_y = max(means[0] + errs[0], means[1] + errs[1]) + 0.15
        ax.plot([0, 0, 1, 1], [bracket_y - 0.08, bracket_y, bracket_y, bracket_y - 0.08], color="#444", lw=0.8)
        ax.text(0.5, bracket_y + 0.05, f"{ratio:.0f}×", ha="center", fontsize=11, fontweight="bold", color="#444")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["g (trained)", "g = 0"], fontsize=9)
    ax.set_ylabel("||z_t − z_raw||  (perception shift)")
    ax.set_title("(a) g reshapes perception", fontsize=10)

    ax = fig.add_subplot(gs[1])
    trained_vec, _ = hierarchical_vector_summary(
        by_p1_probe,
        lambda df: df[df["g_label"].astype(str) == best_label][zc].mean().values if set(zc).issubset(df.columns) else None,
    )
    zero_vec, _ = hierarchical_vector_summary(
        by_p1_probe,
        lambda df: df[df["g_label"].astype(str) == "g_zero"][zc].mean().values if set(zc).issubset(df.columns) else None,
    )
    if trained_vec is not None and zero_vec is not None:
        diff = trained_vec - zero_vec
        colors = [C_AD if v >= 0 else "#7799CC" for v in diff]
        x_pos = np.arange(n_z)
        ax.bar(x_pos, diff, color=colors, alpha=0.7, edgecolor="white", lw=0.3)
        ax.axhline(0, color="#ccc", lw=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"z{i}" for i in range(n_z)], fontsize=7)
        ax.set_xlabel("Latent dimension")
        ax.set_ylabel("Δz_t  (trained − zero)")
        ax.set_title("(b) Per-dimension perceptual reorganization", fontsize=10)

    if has_panel_c:
        ax = fig.add_subplot(gs[2])
        t_ax, med, q1, q3 = hierarchical_event_trigger(
            sweep_runs[4],
            lambda ds: pd.to_numeric(ds["g_norm"], errors="coerce").values if "g_norm" in ds.columns else None,
            window=40,
        )
        if med is not None:
            ax.fill_between(t_ax, q1, q3, color=C_GNORM, alpha=0.12)
            ax.plot(t_ax, med, color=C_GNORM, lw=2, label="||g||")
            ax.axvline(0, color="#aaa", lw=0.5, ls="--")
            ax.axvspan(0, 15, color=C_ALPHA, alpha=0.04)
            ax.set_xlabel("Steps from perturbation onset")
            ax.set_ylabel("||g||")
            ax.set_title("(c) Perturbation engages perspective", fontsize=10)

            _, med_a, _, _ = hierarchical_event_trigger(
                sweep_runs[4],
                lambda ds: pd.to_numeric(ds["alpha"], errors="coerce").values if "alpha" in ds.columns else None,
                window=40,
            )
            if med_a is not None:
                ax2 = ax.twinx()
                ax2.plot(t_ax, med_a, color=C_ALPHA, lw=1.5, ls="--", label="α")
                ax2.set_ylabel("α", color=C_ALPHA, fontsize=9)
                ax2.tick_params(axis="y", labelcolor=C_ALPHA)

    savefig(fig, outdir / "fig3_perspective_perception.png")


# ---------------------------------------------------------
# Fig 4: Ablation
# ---------------------------------------------------------

def fig4_ablation(ablation_root: Path, outdir: Path):
    data = collect_ablation_runs(ablation_root)
    conds = [
        ("adaptive", C_AD, "Adaptive"),
        ("fixed_010", C_FX, "Fixed (α=0.10)"),
        ("fixed_005", C_FX, "Fixed (α=0.05)"),
        ("fast_080", C_FA, "Fast (α=0.80)"),
    ]
    found = [(cond, color, label, data[cond]) for cond, color, label in conds if cond in data and data[cond]]
    if len(found) < 2:
        print("  [skip] fig4: need >=2 ablation conditions")
        return

    fig = plt.figure(figsize=(13, 3.8))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.5, 1, 1], wspace=0.3)

    panels = [
        (0, "alpha", "α (plasticity)", "(a) Plasticity dynamics"),
        (1, "pred_err", "PE (smoothed)", "(b) Prediction error"),
        (2, "g_norm", "||g||", "(c) Perspective magnitude"),
    ]

    for gi, metric, ylabel, title in panels:
        ax = fig.add_subplot(gs[gi])
        for _, color, label, by_p1 in found:
            def make_series(df, m=metric):
                if m not in df.columns:
                    return pd.Series(dtype=float)
                s = df.groupby("episode")[m].mean()
                if m == "pred_err" and len(s) > 1:
                    s = s.rolling(10, min_periods=1).mean()
                return s

            med, q1, q3, p1_curves = hierarchical_curve_summary(by_p1, make_series)
            for curve in p1_curves:
                ax.plot(curve.index, curve.values, lw=0.3, alpha=0.15, color=color)
            if med is not None:
                ax.fill_between(med.index, q1.values, q3.values, color=color, alpha=0.10)
                ax.plot(med.index, med.values, lw=2, color=color, label=label)

        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)

    savefig(fig, outdir / "fig4_ablation.png")


# ---------------------------------------------------------
# Supplementary
# ---------------------------------------------------------

def figS1_sweep(sweep_root: Path, outdir: Path):
    runs = collect_sweep_runs(sweep_root)
    if len(runs) < 2:
        print("  [skip] figS1: need >=2 sweep conditions")
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

    ax = axes[0]
    m, lo, hi = agg_scalar(lambda df: late(df)["pred_err"].mean() if "pred_err" in df.columns else np.nan)
    ax.fill_between(ns, lo, hi, color=C_PE, alpha=0.15)
    ax.plot(ns, m, "s-", color=C_PE, lw=1.5, markersize=5)
    ax.set_xlabel("nP")
    ax.set_ylabel("Late PE")
    ax.set_title("(a) Prediction error")

    ax = axes[1]
    m, lo, hi = agg_scalar(lambda df: late(df)["alpha"].mean() if "alpha" in df.columns else np.nan)
    ax.fill_between(ns, lo, hi, color=C_ALPHA, alpha=0.15)
    ax.plot(ns, m, "D-", color=C_ALPHA, lw=1.5, markersize=5)
    ax.set_xlabel("nP")
    ax.set_ylabel("Late α")
    ax.set_title("(b) Plasticity")

    ax = axes[2]
    def get_gabs(df):
        gc = gm_cols(df)
        if not gc:
            return np.nan
        return late(df)[gc].abs().mean().mean()
    m, lo, hi = agg_scalar(get_gabs)
    ax.fill_between(ns, lo, hi, color=C_GAMMA, alpha=0.15)
    ax.plot(ns, m, "o-", color=C_GAMMA, lw=1.5, markersize=5)
    ax.set_xlabel("nP")
    ax.set_ylabel("Late |γ|")
    ax.set_title("(c) Salience modulation")

    ax = axes[3]
    for zi in range(5):
        m, _, _ = agg_scalar(
            lambda df, z=zi: len(late(df)[late(df)["zone_id"] == z]) / max(len(late(df)), 1)
            if "zone_id" in df.columns else np.nan
        )
        ax.plot(ns, m, "o-", color=ZC[zi], lw=1.2, markersize=4, label=f"Z{zi}")
    ax.set_xlabel("nP")
    ax.set_ylabel("Frac")
    ax.set_title("(d) Zone dwell")
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle("Fig S1: Perturbation sweep dose-response", fontsize=11, y=1.03)
    savefig(fig, outdir / "figS1_sweep.png")


def figS2_film_blocks(mixed_root: Path, outdir: Path):
    data = collect_mixed_runs(mixed_root)
    if "mixed_0_4_0" not in data or not data["mixed_0_4_0"]:
        print("  [skip] figS2: no mixed_0_4_0 traj")
        return

    by_p1 = data["mixed_0_4_0"]
    ref = next((dfs[0] for _, dfs in sorted(by_p1.items()) if dfs), None)
    if ref is None:
        print("  [skip] figS2: empty mixed_0_4_0")
        return

    gc = gm_cols(ref)
    if not gc or "block_id" not in ref.columns:
        print("  [skip] figS2: missing columns")
        return

    blocks = sorted(ref["block_id"].dropna().unique())
    n_dims = len(gc)
    x_pos = np.arange(n_dims)
    width = 0.8 / max(len(blocks), 1)
    bc = [C_AD, C_FA, C_FX, "#D4537E", "#888"]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    for bi, blk in enumerate(blocks):
        def block_gamma_vec(df, block=blk):
            if "block_id" not in df.columns:
                return None
            bd = df[df["block_id"] == block]
            if len(bd) == 0:
                return None
            mx = int(bd["episode"].max())
            bl = bd[bd["episode"] >= max(int(bd["episode"].min()), mx - 9)]
            return bl[gm_cols(df)].mean().values if gm_cols(df) else None

        gvals, _ = hierarchical_vector_summary(by_p1, block_gamma_vec)
        if gvals is None:
            continue
        nP = None
        if "n_perturb_setting" in ref.columns:
            try:
                nP = int(ref[ref["block_id"] == blk]["n_perturb_setting"].iloc[0])
            except Exception:
                nP = None
        offset = (bi - len(blocks) / 2 + 0.5) * width
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
    ax.set_title("Fig S2: Salience gating pattern per block (0→4→0)")
    ax.legend(fontsize=7, ncol=3)
    savefig(fig, outdir / "figS2_film_blocks.png")


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
    sr = Path(args.sweep_root) if args.sweep_root else None
    mr = Path(args.mixed_root) if args.mixed_root else None
    ar = Path(args.ablation_root) if args.ablation_root else None

    print(f"Output: {outdir}\n")

    if mr:
        print("=== Fig 2: Mixed history ===")
        try:
            fig2_mixed_history(mr, outdir)
        except Exception as e:
            print(f"  [error] {e}")

    print("\n=== Fig 3: Perspective shapes perception ===")
    try:
        fig3_perspective_perception(mr, sr, outdir)
    except Exception as e:
        print(f"  [error] {e}")

    if ar:
        print("\n=== Fig 4: Ablation ===")
        try:
            fig4_ablation(ar, outdir)
        except Exception as e:
            print(f"  [error] {e}")

    if sr:
        print("\n=== Fig S1: Sweep ===")
        try:
            figS1_sweep(sr, outdir)
        except Exception as e:
            print(f"  [error] {e}")

    if mr:
        print("\n=== Fig S2: FiLM per block ===")
        try:
            figS2_film_blocks(mr, outdir)
        except Exception as e:
            print(f"  [error] {e}")

    print(f"\nDone. {outdir}")


if __name__ == "__main__":
    main()