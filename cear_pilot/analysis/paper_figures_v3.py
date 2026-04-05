# cear_pilot/analysis/paper_figures_v3.py
# -*- coding: utf-8 -*-
"""
Paper figures v3 for ALIFE 2026.

Main figures generated:
  Fig 3: mixed-history vs nP=0 baseline
  Fig 4: perspective shapes perception + blockwise gating residue
  Fig 5: update-law comparison + adaptive nP=0 baseline

Expected directory structure:
    <root>/
      from_p1_s0/
        sweep/
          seed0/
            nperturb0/
            nperturb1/
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

Usage:
python -m cear_pilot.analysis.paper_figures_v3 \
  --sweep_root outputs/phase2_all_20260403_011254 \
  --mixed_root outputs/phase2_all_20260403_011254 \
  --ablation_root outputs/phase2_all_20260403_011254 \
  --outdir outputs/phase2_all_20260403_011254/paper_figures_v3
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


# ---------------------------------------------------------
# Style
# ---------------------------------------------------------

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
C_BASE = "#444444"
C_BASE_FILL = "#777777"
ZC = ["#E24B4A", "#D85A30", "#888780", "#1D9E75", "#0F6E56"]


# ---------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------

def savefig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"  [fig] {path.name}")


def load_traj(d: Path):
    for n in ["traj.parquet", "traj.csv"]:
        p = d / n
        if p.exists():
            try:
                return pd.read_parquet(p) if n.endswith(".parquet") else pd.read_csv(p)
            except Exception as e:
                print(f"[load_traj error] {p}: {type(e).__name__}: {e}")
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


def ensure_numeric_series(s: pd.Series):
    return pd.to_numeric(s, errors="coerce")


def late(df: pd.DataFrame, n: int = 20):
    if "episode" not in df.columns or len(df) == 0:
        return df
    mx = int(df["episode"].max())
    return df[df["episode"] >= max(0, mx - n + 1)]


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
        return None, None, None, []

    arr = np.vstack(p1_vecs)
    med = np.median(arr, axis=0)
    q1 = np.percentile(arr, 25, axis=0)
    q3 = np.percentile(arr, 75, axis=0)
    return med, q1, q3, p1_vecs


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

            traces = []
            for loc in onsets:
                row = np.full(2 * window + 1, np.nan)
                lo = max(0, loc - window)
                hi = min(len(vals), loc + window + 1)
                for j in range(lo, hi):
                    row[j - loc + window] = vals[j]
                traces.append(row)

            if traces:
                p2_traces.append(np.nanmedian(np.asarray(traces, dtype=float), axis=0))

        if p2_traces:
            p1_traces.append(np.nanmedian(np.asarray(p2_traces, dtype=float), axis=0))

    if not p1_traces:
        return t_ax, None, None, None

    arr = np.asarray(p1_traces, dtype=float)
    med = np.nanmedian(arr, axis=0)
    q1 = np.nanpercentile(arr, 25, axis=0)
    q3 = np.nanpercentile(arr, 75, axis=0)
    return t_ax, med, q1, q3


def slice_series_into_blocks(s: pd.Series, block_len: int = 50, n_blocks: int = 3):
    s = ensure_numeric_series(s).dropna()
    if len(s) == 0:
        return None

    vals = []
    for bi in range(n_blocks):
        lo = bi * block_len
        hi = (bi + 1) * block_len - 1
        chunk = s[(s.index >= lo) & (s.index <= hi)]
        vals.append(float(chunk.mean()) if len(chunk) > 0 else np.nan)
    return np.asarray(vals, dtype=float)


def hierarchical_block_summary(data_by_p1: dict[int, list], series_fn, block_len: int = 50, n_blocks: int = 3):
    p1_block_vecs = []

    for _, dfs in sorted(data_by_p1.items()):
        p2_vecs = []
        for df in dfs:
            try:
                s = series_fn(df)
                if s is None or len(s) == 0:
                    continue
                vec = slice_series_into_blocks(s, block_len=block_len, n_blocks=n_blocks)
                if vec is None or np.all(~np.isfinite(vec)):
                    continue
                p2_vecs.append(vec)
            except Exception:
                continue

        if p2_vecs:
            p2_arr = np.vstack(p2_vecs)
            p1_block_vecs.append(np.nanmedian(p2_arr, axis=0))

    if not p1_block_vecs:
        return None, None, None

    arr = np.vstack(p1_block_vecs)
    med = np.nanmedian(arr, axis=0)
    q1 = np.nanpercentile(arr, 25, axis=0)
    q3 = np.nanpercentile(arr, 75, axis=0)
    return med, q1, q3


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
# Figure 3: mixed-history vs nP=0 baseline
# ---------------------------------------------------------

def fig3_mixed_history_vs_nP0(mixed_data, sweep_data, outdir: Path):
    if "mixed_0_4_0" not in mixed_data:
        print("  [skip] mixed_0_4_0 not found")
        return
    if 0 not in sweep_data:
        print("  [skip] sweep nperturb0 not found")
        return

    mixed = mixed_data["mixed_0_4_0"]
    base0 = sweep_data[0]

    def series_alpha(df):
        return df.groupby("episode")["alpha"].mean() if "alpha" in df.columns else None

    def series_g(df):
        return df.groupby("episode")["g_norm"].mean() if "g_norm" in df.columns else None

    def series_gamma(df):
        if "gamma_abs_mean" in df.columns:
            return df.groupby("episode")["gamma_abs_mean"].mean()
        gm = gm_cols(df)
        if gm:
            return df.groupby("episode")[gm].mean().abs().mean(axis=1)
        return None

    med_a_m, q1_a_m, q3_a_m, _ = hierarchical_curve_summary(mixed, series_alpha)
    med_g_m, q1_g_m, q3_g_m, _ = hierarchical_curve_summary(mixed, series_g)
    med_y_m, q1_y_m, q3_y_m, _ = hierarchical_curve_summary(mixed, series_gamma)

    med_a_b, q1_a_b, q3_a_b = hierarchical_block_summary(base0, series_alpha, block_len=50, n_blocks=3)
    med_g_b, q1_g_b, q3_g_b = hierarchical_block_summary(base0, series_g, block_len=50, n_blocks=3)
    med_y_b, q1_y_b, q3_y_b = hierarchical_block_summary(base0, series_gamma, block_len=50, n_blocks=3)

    if med_a_m is None or med_g_m is None or med_y_m is None:
        print("  [skip] insufficient mixed-history data")
        return

    fig = plt.figure(figsize=(11.8, 7.0))
    gs = gridspec.GridSpec(2, 2, height_ratios=[0.65, 3.0], hspace=0.38, wspace=0.28)

    # schedule strip
    ax0 = fig.add_subplot(gs[0, :])
    block_colors = ["#E8E8E8", "#D85A30", "#E8E8E8"]
    labels = ["0", "4", "0"]
    for i in range(3):
        ax0.axvspan(i * 50, (i + 1) * 50, color=block_colors[i], alpha=0.75)
        ax0.text(i * 50 + 25, 0.5, labels[i], ha="center", va="center", fontsize=10)
    for x in [50, 100]:
        ax0.axvline(x, color="k", lw=0.8, ls="--", alpha=0.45)

    ax0.set_xlim(0, 150)
    ax0.set_ylim(0, 1)
    ax0.set_yticks([])
    ax0.set_xlabel("Episode")
    ax0.set_title("Mixed-history schedule (nP per episode)")
    ax0.spines["left"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    # alpha
    ax1 = fig.add_subplot(gs[1, 0])
    x_m = med_a_m.index.values.astype(float)
    ax1.plot(x_m, med_a_m.values, color=C_ALPHA, lw=2.2, label="mixed 0→4→0")
    ax1.fill_between(x_m, q1_a_m.values, q3_a_m.values, color=C_ALPHA, alpha=0.18)

    if med_a_b is not None:
        xb = np.array([25, 75, 125], dtype=float)
        ax1.plot(xb, med_a_b, color=C_BASE, lw=2.0, marker="o", label="nP=0 pseudo-baseline")
        ax1.fill_between(xb, q1_a_b, q3_a_b, color=C_BASE_FILL, alpha=0.15)

    for x in [50, 100]:
        ax1.axvline(x, color="k", lw=0.8, ls="--", alpha=0.35)
    ax1.set_xlim(0, 150)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel(r"$\alpha$")
    ax1.set_title("Plasticity")
    ax1.legend(frameon=False, loc="best")

    # g norm
    ax2 = fig.add_subplot(gs[1, 1])
    x_m = med_g_m.index.values.astype(float)
    ax2.plot(x_m, med_g_m.values, color=C_GNORM, lw=2.2, label="mixed 0→4→0")
    ax2.fill_between(x_m, q1_g_m.values, q3_g_m.values, color=C_GNORM, alpha=0.18)

    if med_g_b is not None:
        xb = np.array([25, 75, 125], dtype=float)
        ax2.plot(xb, med_g_b, color=C_BASE, lw=2.0, marker="o", label="nP=0 pseudo-baseline")
        ax2.fill_between(xb, q1_g_b, q3_g_b, color=C_BASE_FILL, alpha=0.15)

    for x in [50, 100]:
        ax2.axvline(x, color="k", lw=0.8, ls="--", alpha=0.35)
    ax2.set_xlim(0, 150)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel(r"$\|g\|$")
    ax2.set_title("Perspective magnitude")
    ax2.legend(frameon=False, loc="best")

    savefig(fig, outdir / "fig3_mixed_history_vs_nP0.png")

    # separate gamma panel to keep readability
    fig_g, ax = plt.subplots(figsize=(6.2, 4.0))
    x_m = med_y_m.index.values.astype(float)
    ax.plot(x_m, med_y_m.values, color=C_GAMMA, lw=2.2, label="mixed 0→4→0")
    ax.fill_between(x_m, q1_y_m.values, q3_y_m.values, color=C_GAMMA, alpha=0.18)

    if med_y_b is not None:
        xb = np.array([25, 75, 125], dtype=float)
        ax.plot(xb, med_y_b, color=C_BASE, lw=2.0, marker="o", label="nP=0 pseudo-baseline")
        ax.fill_between(xb, q1_y_b, q3_y_b, color=C_BASE_FILL, alpha=0.15)

    for x in [50, 100]:
        ax.axvline(x, color="k", lw=0.8, ls="--", alpha=0.35)
    ax.set_xlim(0, 150)
    ax.set_xlabel("Episode")
    ax.set_ylabel(r"$|\gamma|$")
    ax.set_title("Salience modulation")
    ax.legend(frameon=False, loc="best")

    savefig(fig_g, outdir / "fig3b_gamma_vs_nP0.png")


# ---------------------------------------------------------
# Figure 4: perspective shapes perception + gating residue
# ---------------------------------------------------------

def _probe_distance_zero_vs_trained(df: pd.DataFrame):
    zone_col = None
    for c in ["zone_id", "zone", "probe_zone", "zone_idx"]:
        if c in df.columns:
            zone_col = c
            break
    if zone_col is None:
        return None, None, None

    # Try to infer comparison columns.
    candidates_trained = [c for c in df.columns if re.search(r"(trained|blk2|block2|hist)", c, re.I)]
    candidates_zero = [c for c in df.columns if re.search(r"(zero|g0|null)", c, re.I)]

    # If explicit scalar shift column exists, use it.
    for c in df.columns:
        if re.search(r"(shift|dist|delta)", c, re.I) and zone_col in df.columns:
            tmp = df.groupby(zone_col)[c].mean().sort_index()
            vals = tmp.values.astype(float)
            return vals, np.full_like(vals, np.nan), np.full_like(vals, np.nan)

    # Otherwise attempt latent-vector distance from prefixed columns.
    # Example expected patterns:
    #   trained_z_t_0 ... trained_z_t_15
    #   zero_z_t_0    ... zero_z_t_15
    def prefixed_cols(prefixes):
        for pref in prefixes:
            cols = sorted(
                [c for c in df.columns if re.match(rf"{pref}\d+$", c)],
                key=lambda x: int(re.findall(r"(\d+)$", x)[0]),
            )
            if cols:
                return cols
        return []

    zt_tr = prefixed_cols([r"trained_z_t_", r"blk2_z_t_", r"hist_z_t_"])
    zt_ze = prefixed_cols([r"zero_z_t_", r"g0_z_t_", r"null_z_t_"])

    if not zt_tr or not zt_ze or len(zt_tr) != len(zt_ze):
        return None, None, None

    grouped = []
    for z, g in df.groupby(zone_col):
        a = g[zt_tr].to_numpy(dtype=float)
        b = g[zt_ze].to_numpy(dtype=float)
        d = np.linalg.norm(a - b, axis=1)
        grouped.append((z, np.nanmean(d), np.nanpercentile(d, 25), np.nanpercentile(d, 75)))

    grouped = sorted(grouped, key=lambda x: x[0])
    med = np.array([x[1] for x in grouped], dtype=float)
    q1 = np.array([x[2] for x in grouped], dtype=float)
    q3 = np.array([x[3] for x in grouped], dtype=float)
    return med, q1, q3


def _probe_signed_dim_shift(df: pd.DataFrame):
    # Try explicit delta columns first.
    dcols = sorted(
        [c for c in df.columns if re.match(r"(delta_z_|dz_)\d+$", c)],
        key=lambda x: int(re.findall(r"(\d+)$", x)[0]),
    )
    if dcols:
        return df[dcols].mean(axis=0).to_numpy(dtype=float)

    # Fallback: trained - zero vector
    tr = sorted(
        [c for c in df.columns if re.match(r"(trained_z_t_|blk2_z_t_|hist_z_t_)\d+$", c)],
        key=lambda x: int(re.findall(r"(\d+)$", x)[0]),
    )
    ze = sorted(
        [c for c in df.columns if re.match(r"(zero_z_t_|g0_z_t_|null_z_t_)\d+$", c)],
        key=lambda x: int(re.findall(r"(\d+)$", x)[0]),
    )
    if tr and ze and len(tr) == len(ze):
        return (df[tr].to_numpy(dtype=float) - df[ze].to_numpy(dtype=float)).mean(axis=0)
    return None


def _blockwise_gamma_vector(df: pd.DataFrame, n_last_eps: int = 20):
    gm = gm_cols(df)
    if not gm or "episode" not in df.columns:
        return None

    # block 0: early baseline
    b0 = df[(df["episode"] >= 0) & (df["episode"] < 50)]
    # block 1: perturbation
    b1 = df[(df["episode"] >= 50) & (df["episode"] < 100)]
    # block 2: late recovery
    b2 = df[(df["episode"] >= max(100, int(df["episode"].max()) - n_last_eps + 1))]

    if len(b0) == 0 or len(b1) == 0 or len(b2) == 0:
        return None

    v0 = b0[gm].mean(axis=0).to_numpy(dtype=float)
    v1 = b1[gm].mean(axis=0).to_numpy(dtype=float)
    v2 = b2[gm].mean(axis=0).to_numpy(dtype=float)
    return np.vstack([v0, v1, v2])


def fig4_perception_and_gating(probe_data, mixed_data, outdir: Path):
    if "mixed_0_4_0" not in probe_data:
        print("  [skip] mixed_0_4_0 probe data not found")
        return
    if "mixed_0_4_0" not in mixed_data:
        print("  [skip] mixed_0_4_0 traj data not found")
        return

    probe = probe_data["mixed_0_4_0"]
    mixed = mixed_data["mixed_0_4_0"]

    # Panel A: same-input shift by zone
    zone_meds = []
    zone_q1s = []
    zone_q3s = []
    for _, dfs in sorted(probe.items()):
        p2_vals = []
        for df in dfs:
            med, q1, q3 = _probe_distance_zero_vs_trained(df)
            if med is not None:
                p2_vals.append(med)
        if p2_vals:
            arr = np.vstack(p2_vals)
            zone_meds.append(np.median(arr, axis=0))
    if zone_meds:
        arr = np.vstack(zone_meds)
        med_a = np.median(arr, axis=0)
        q1_a = np.percentile(arr, 25, axis=0)
        q3_a = np.percentile(arr, 75, axis=0)
    else:
        med_a = q1_a = q3_a = None

    # Panel B: signed per-dim latent shift
    dim_vecs = []
    for _, dfs in sorted(probe.items()):
        p2_vecs = []
        for df in dfs:
            vec = _probe_signed_dim_shift(df)
            if vec is not None and np.all(np.isfinite(vec)):
                p2_vecs.append(vec)
        if p2_vecs:
            dim_vecs.append(np.median(np.vstack(p2_vecs), axis=0))
    if dim_vecs:
        arr = np.vstack(dim_vecs)
        med_b = np.median(arr, axis=0)
        q1_b = np.percentile(arr, 25, axis=0)
        q3_b = np.percentile(arr, 75, axis=0)
    else:
        med_b = q1_b = q3_b = None

    # Panel C: blockwise gamma residue (B0 vs B1 vs B2)
    gamma_block_vecs = []
    for _, dfs in sorted(mixed.items()):
        p2_stack = []
        for df in dfs:
            vec = _blockwise_gamma_vector(df)
            if vec is not None:
                p2_stack.append(vec)
        if p2_stack:
            gamma_block_vecs.append(np.median(np.stack(p2_stack, axis=0), axis=0))
    if gamma_block_vecs:
        arr = np.stack(gamma_block_vecs, axis=0)  # [p1, 3, D]
        med_c = np.median(arr, axis=0)
        q1_c = np.percentile(arr, 25, axis=0)
        q3_c = np.percentile(arr, 75, axis=0)
    else:
        med_c = q1_c = q3_c = None

    # Event-triggered g response for compact support inset/side panel
    def g_getter(ds):
        if "g_norm" not in ds.columns:
            return None
        return pd.to_numeric(ds["g_norm"], errors="coerce").to_numpy(dtype=float)

    t_ax, med_evt, q1_evt, q3_evt = hierarchical_event_trigger(mixed, g_getter, window=35)

    fig = plt.figure(figsize=(12.6, 4.2))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1.0, 1.15, 1.25, 1.0], wspace=0.35)

    # (a) same-input shift
    ax = fig.add_subplot(gs[0, 0])
    if med_a is not None:
        x = np.arange(len(med_a))
        ax.bar(x, med_a, color=[ZC[i % len(ZC)] for i in range(len(med_a))], alpha=0.75)
        ax.errorbar(x, med_a, yerr=[med_a - q1_a, q3_a - med_a], fmt="none", ecolor="k", lw=0.8, capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Z{i}" for i in x])
    ax.set_ylabel(r"$\|z_t(g)-z_t(0)\|$")
    ax.set_title("(a) Same-input shift")

    # (b) per-dim reorganization
    ax = fig.add_subplot(gs[0, 1])
    if med_b is not None:
        x = np.arange(len(med_b))
        colors = [C_AD if v >= 0 else C_GNORM for v in med_b]
        ax.bar(x, med_b, color=colors, alpha=0.8)
        if q1_b is not None and q3_b is not None:
            ax.errorbar(x, med_b, yerr=[med_b - q1_b, q3_b - med_b], fmt="none", ecolor="k", lw=0.7, capsize=1.5)
        ax.axhline(0, color="k", lw=0.8, alpha=0.4)
        ax.set_xticks(x[::2] if len(x) > 10 else x)
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel(r"$\Delta z_t$")
    ax.set_title("(b) Per-dimension shift")

    # (c) blockwise gamma residue
    ax = fig.add_subplot(gs[0, 2])
    if med_c is not None:
        D = med_c.shape[1]
        x = np.arange(D)
        width = 0.26
        labels = ["Block 0\n(nP=0)", "Block 1\n(nP=4)", "Block 2\n(nP=0)"]
        cols = ["#BDBDBD", C_ALPHA, C_GAMMA]
        for bi in range(3):
            ax.bar(x + (bi - 1) * width, med_c[bi], width=width, color=cols[bi], alpha=0.85, label=labels[bi])
        ax.axhline(0, color="k", lw=0.8, alpha=0.4)
        ax.set_xticks(x[::2] if D > 10 else x)
        ax.set_xlabel("FiLM dimension")
        ax.set_ylabel(r"$\gamma$")
        ax.legend(frameon=False, loc="best")
    ax.set_title("(c) Blockwise gating residue")

    # (d) perturbation-triggered g response
    ax = fig.add_subplot(gs[0, 3])
    if med_evt is not None:
        ax.plot(t_ax, med_evt, color=C_GNORM, lw=2.1)
        ax.fill_between(t_ax, q1_evt, q3_evt, color=C_GNORM, alpha=0.18)
        ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.45)
    ax.set_xlabel("Steps from onset")
    ax.set_ylabel(r"$\|g\|$")
    ax.set_title("(d) Perturbation engagement")

    savefig(fig, outdir / "fig4_perception_gating_v3.png")


# ---------------------------------------------------------
# Figure 5: update-law comparison + adaptive baseline
# ---------------------------------------------------------

def fig5_update_law_with_adaptive_baseline(ablation_data, sweep_data, outdir: Path):
    if "adaptive" not in ablation_data:
        print("  [skip] ablation adaptive not found")
        return
    if 0 not in sweep_data:
        print("  [skip] sweep nperturb0 not found")
        return

    adaptive_np4 = ablation_data["adaptive"]
    adaptive_np0 = sweep_data[0]

    fixed_key = None
    if "fixed_005" in ablation_data:
        fixed_key = "fixed_005"
    elif "fixed_010" in ablation_data:
        fixed_key = "fixed_010"

    fast_key = "fast_080" if "fast_080" in ablation_data else None

    def series_alpha(df):
        return df.groupby("episode")["alpha"].mean() if "alpha" in df.columns else None

    def series_pe(df):
        return df.groupby("episode")["pred_err"].mean() if "pred_err" in df.columns else None

    def series_g(df):
        return df.groupby("episode")["g_norm"].mean() if "g_norm" in df.columns else None

    med_a_ad4, q1_a_ad4, q3_a_ad4, _ = hierarchical_curve_summary(adaptive_np4, series_alpha)
    med_a_ad0, q1_a_ad0, q3_a_ad0, _ = hierarchical_curve_summary(adaptive_np0, series_alpha)

    med_pe_ad4, q1_pe_ad4, q3_pe_ad4, _ = hierarchical_curve_summary(adaptive_np4, series_pe)
    med_pe_ad0, q1_pe_ad0, q3_pe_ad0, _ = hierarchical_curve_summary(adaptive_np0, series_pe)

    med_g_ad4, q1_g_ad4, q3_g_ad4, _ = hierarchical_curve_summary(adaptive_np4, series_g)
    med_g_ad0, q1_g_ad0, q3_g_ad0, _ = hierarchical_curve_summary(adaptive_np0, series_g)

    fixed_curves = {}
    if fixed_key is not None:
        fixed_curves["alpha"] = hierarchical_curve_summary(ablation_data[fixed_key], series_alpha)
        fixed_curves["pe"] = hierarchical_curve_summary(ablation_data[fixed_key], series_pe)
        fixed_curves["g"] = hierarchical_curve_summary(ablation_data[fixed_key], series_g)

    fast_curves = {}
    if fast_key is not None:
        fast_curves["alpha"] = hierarchical_curve_summary(ablation_data[fast_key], series_alpha)
        fast_curves["pe"] = hierarchical_curve_summary(ablation_data[fast_key], series_pe)
        fast_curves["g"] = hierarchical_curve_summary(ablation_data[fast_key], series_g)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.1))

    # alpha
    ax = axes[0]
    if med_a_ad4 is not None:
        x = med_a_ad4.index.values.astype(float)
        ax.plot(x, med_a_ad4.values, color=C_AD, lw=2.3, label="Adaptive (nP=4)")
        ax.fill_between(x, q1_a_ad4.values, q3_a_ad4.values, color=C_AD, alpha=0.18)
    if med_a_ad0 is not None:
        x = med_a_ad0.index.values.astype(float)
        ax.plot(x, med_a_ad0.values, color=C_BASE, lw=2.0, label="Adaptive baseline (nP=0)")
        ax.fill_between(x, q1_a_ad0.values, q3_a_ad0.values, color=C_BASE_FILL, alpha=0.15)

    if fixed_curves:
        med, q1, q3, _ = fixed_curves["alpha"]
        if med is not None:
            x = med.index.values.astype(float)
            ax.plot(x, med.values, color=C_FX, lw=2.0, label="Fixed")
            ax.fill_between(x, q1.values, q3.values, color=C_FX, alpha=0.14)
    if fast_curves:
        med, q1, q3, _ = fast_curves["alpha"]
        if med is not None:
            x = med.index.values.astype(float)
            ax.plot(x, med.values, color=C_FA, lw=2.0, label="Fast")
            ax.fill_between(x, q1.values, q3.values, color=C_FA, alpha=0.14)

    ax.set_xlabel("Episode")
    ax.set_ylabel(r"$\alpha$")
    ax.set_title("Plasticity")
    ax.legend(frameon=False, loc="best")

    # PE
    ax = axes[1]
    if med_pe_ad4 is not None:
        x = med_pe_ad4.index.values.astype(float)
        ax.plot(x, med_pe_ad4.values, color=C_AD, lw=2.3, label="Adaptive (nP=4)")
        ax.fill_between(x, q1_pe_ad4.values, q3_pe_ad4.values, color=C_AD, alpha=0.18)
    if med_pe_ad0 is not None:
        x = med_pe_ad0.index.values.astype(float)
        ax.plot(x, med_pe_ad0.values, color=C_BASE, lw=2.0, label="Adaptive baseline (nP=0)")
        ax.fill_between(x, q1_pe_ad0.values, q3_pe_ad0.values, color=C_BASE_FILL, alpha=0.15)

    if fixed_curves:
        med, q1, q3, _ = fixed_curves["pe"]
        if med is not None:
            x = med.index.values.astype(float)
            ax.plot(x, med.values, color=C_FX, lw=2.0, label="Fixed")
            ax.fill_between(x, q1.values, q3.values, color=C_FX, alpha=0.14)
    if fast_curves:
        med, q1, q3, _ = fast_curves["pe"]
        if med is not None:
            x = med.index.values.astype(float)
            ax.plot(x, med.values, color=C_FA, lw=2.0, label="Fast")
            ax.fill_between(x, q1.values, q3.values, color=C_FA, alpha=0.14)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Prediction error")
    ax.set_title("Prediction error")
    ax.legend(frameon=False, loc="best")

    # g norm
    ax = axes[2]
    if med_g_ad4 is not None:
        x = med_g_ad4.index.values.astype(float)
        ax.plot(x, med_g_ad4.values, color=C_AD, lw=2.3, label="Adaptive (nP=4)")
        ax.fill_between(x, q1_g_ad4.values, q3_g_ad4.values, color=C_AD, alpha=0.18)
    if med_g_ad0 is not None:
        x = med_g_ad0.index.values.astype(float)
        ax.plot(x, med_g_ad0.values, color=C_BASE, lw=2.0, label="Adaptive baseline (nP=0)")
        ax.fill_between(x, q1_g_ad0.values, q3_g_ad0.values, color=C_BASE_FILL, alpha=0.15)

    if fixed_curves:
        med, q1, q3, _ = fixed_curves["g"]
        if med is not None:
            x = med.index.values.astype(float)
            ax.plot(x, med.values, color=C_FX, lw=2.0, label="Fixed")
            ax.fill_between(x, q1.values, q3.values, color=C_FX, alpha=0.14)
    if fast_curves:
        med, q1, q3, _ = fast_curves["g"]
        if med is not None:
            x = med.index.values.astype(float)
            ax.plot(x, med.values, color=C_FA, lw=2.0, label="Fast")
            ax.fill_between(x, q1.values, q3.values, color=C_FA, alpha=0.14)

    ax.set_xlabel("Episode")
    ax.set_ylabel(r"$\|g\|$")
    ax.set_title("Perspective magnitude")
    ax.legend(frameon=False, loc="best")

    savefig(fig, outdir / "fig5_update_law_with_adaptive_baseline.png")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_root", type=str, required=True)
    ap.add_argument("--mixed_root", type=str, required=True)
    ap.add_argument("--ablation_root", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sweep_root = Path(args.sweep_root)
    mixed_root = Path(args.mixed_root)
    ablation_root = Path(args.ablation_root)

    print("[collect] sweep")
    sweep_data = collect_sweep_runs(sweep_root)

    print("[collect] mixed")
    mixed_data = collect_mixed_runs(mixed_root)

    print("[collect] ablation")
    ablation_data = collect_ablation_runs(ablation_root)

    print("[collect] probe")
    probe_data = collect_probe_runs(mixed_root)

    print("[fig] Fig 3")
    fig3_mixed_history_vs_nP0(mixed_data, sweep_data, outdir)

    print("[fig] Fig 4")
    fig4_perception_and_gating(probe_data, mixed_data, outdir)

    print("[fig] Fig 5")
    fig5_update_law_with_adaptive_baseline(ablation_data, sweep_data, outdir)

    print("[done]")

if __name__ == "__main__":
    main()
    

# #%% spyder run cell 
# import sys
# from cear_pilot.analysis.paper_figures_v3 import main

# sys.argv = [
#     "paper_figures_v3",
#     "--sweep_root", "outputs/phase2_all_20260403_011254",
#     "--mixed_root", "outputs/phase2_all_20260403_011254",
#     "--ablation_root", "outputs/phase2_all_20260403_011254",
#     "--outdir", "outputs/phase2_all_20260403_011254/paper_figures_v3",
# ]

# main()
# #%%
# from pathlib import Path
# from cear_pilot.analysis.paper_figures_v3 import (
#     collect_sweep_runs, collect_mixed_runs, collect_ablation_runs
# )

# root = Path("outputs/phase2_all_20260403_011254")
# print("sweep keys:", sorted(collect_sweep_runs(root).keys()))
# print("mixed keys:", sorted(collect_mixed_runs(root).keys()))
# print("ablation keys:", sorted(collect_ablation_runs(root).keys()))