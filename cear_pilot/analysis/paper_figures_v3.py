#!/usr/bin/env python3
# cear_pilot/analysis/paper_figures_v3.py
# -*- coding: utf-8 -*-
"""
Paper figures v3 for ALIFE 2026.

Generates:
  Fig 3: Mixed 0→4→0 with baseline (sweep nP=0) — schedule + alpha + ||g||
  Fig 4: Perspective shapes perception — (a) z_shift, (b) per-dim, (c) block 0 vs 2 gating
  Fig 5: Ablation — (a) adaptive vs baseline alpha, (b) perspective magnitude across regimes

- Hierarchical aggregation:
    Phase 1 seeds: aggregated within each Phase 2 seed (median over p1 seeds)
    Phase 2 seeds: final summary uses median ± IQR across p1 seeds

- Expected directory structure:
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

Usage:
    python -m cear_pilot.analysis.paper_figures_v3 \
        --sweep_root  outputs/phase2_all_YYYYMMDD \
        --mixed_root  outputs/phase2_all_YYYYMMDD \
        --ablation_root outputs/phase2_all_YYYYMMDD \
        --outdir paper_figures_v3
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

# ── Style ──────────────────────────────────────────────────
DPI = 300
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.12, "grid.linewidth": 0.3,
    "font.size": 9, "font.family": "sans-serif",
    "axes.linewidth": 0.5, "axes.labelsize": 10, "axes.titlesize": 10,
    "legend.fontsize": 8,
})

C_ALPHA = "#D85A30"
C_GNORM = "#0F6E56"
C_GAMMA = "#993C1D"
C_AD    = "#534AB7"
C_FX    = "#0F6E56"
C_FA    = "#D85A30"
C_BL    = "#888888"

# ── Helpers ────────────────────────────────────────────────

def savefig(fig, path: Path):
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"  [fig] {path.name}")

def load_traj(d: Path):
    for n in ["traj.parquet", "traj.csv"]:
        p = d / n
        if not p.exists():
            continue
        try:
            if n.endswith(".parquet"):
                return pd.read_parquet(p, engine="fastparquet")
            return pd.read_csv(p)
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

def zt_cols(df):
    return sorted(
        [c for c in df.columns if re.match(r"z_t_\d+$", c)],
        key=lambda x: int(x.split("_")[-1])
    )

def gm_cols(df):
    return sorted(
        [c for c in df.columns if re.match(r"gamma_\d+$", c)],
        key=lambda x: int(x.split("_")[-1])
    )

def ensure_numeric_series(s):
    return pd.to_numeric(s, errors="coerce")

# ── Revised hierarchical summaries ─────────────────────────
# Central tendency = across p1 medians
# Spread (IQR)     = typical p2 variability, aggregated across p1

def hierarchical_scalar_summary(data_by_p1, fn):
    p1_medians = []
    p1_q1s = []
    p1_q3s = []

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
            arr = np.asarray(vals, dtype=float)
            p1_medians.append(float(np.median(arr)))
            p1_q1s.append(float(np.percentile(arr, 25)))
            p1_q3s.append(float(np.percentile(arr, 75)))

    if not p1_medians:
        return np.nan, np.nan, np.nan, []

    med = float(np.median(np.asarray(p1_medians, dtype=float)))
    q1 = float(np.median(np.asarray(p1_q1s, dtype=float)))
    q3 = float(np.median(np.asarray(p1_q3s, dtype=float)))
    q1 = min(q1, med)
    q3 = max(q3, med)
    return med, q1, q3, p1_medians

def hierarchical_curve_summary(data_by_p1, series_fn):
    p1_med_curves = []
    p1_q1_curves = []
    p1_q3_curves = []

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

        aligned = pd.concat(curves, axis=1)
        p1_med_curves.append(aligned.median(axis=1, skipna=True))
        p1_q1_curves.append(aligned.quantile(0.25, axis=1, interpolation="linear"))
        p1_q3_curves.append(aligned.quantile(0.75, axis=1, interpolation="linear"))

    if not p1_med_curves:
        return None, None, None, []

    med_aligned = pd.concat(p1_med_curves, axis=1)
    q1_aligned = pd.concat(p1_q1_curves, axis=1)
    q3_aligned = pd.concat(p1_q3_curves, axis=1)

    med = med_aligned.median(axis=1, skipna=True)
    q1 = q1_aligned.median(axis=1, skipna=True)
    q3 = q3_aligned.median(axis=1, skipna=True)

    q1 = np.minimum(q1.values, med.values)
    q3 = np.maximum(q3.values, med.values)
    q1 = pd.Series(q1, index=med.index)
    q3 = pd.Series(q3, index=med.index)

    return med, q1, q3, p1_med_curves

def hierarchical_vector_summary(data_by_p1, fn):
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

# ── Directory collectors ───────────────────────────────────

def iter_from_p1_dirs(root):
    if root is None or not root.exists():
        return
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r"from_p1_s(\d+)$", d.name)
        if m:
            yield int(m.group(1)), d

def collect_sweep_runs(root):
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
                df = load_traj(run_dir)
                if df is not None:
                    data[int(m.group(1))][p1_seed].append(df)
    return data

def collect_mixed_runs(root):
    data = defaultdict(lambda: defaultdict(list))
    for p1_seed, p1_dir in iter_from_p1_dirs(root):
        mixed_dir = p1_dir / "mixed"
        if not mixed_dir.exists():
            continue
        for seed_dir in sorted(mixed_dir.iterdir()):
            if not seed_dir.is_dir() or not re.match(r"seed\d+$", seed_dir.name):
                continue
            for run_dir in sorted(seed_dir.iterdir()):
                if not run_dir.is_dir() or run_dir.name not in {"mixed_0_4_0", "mixed_0_6_0", "mixed_ramp"}:
                    continue
                df = load_traj(run_dir)
                if df is not None:
                    data[run_dir.name][p1_seed].append(df)
    return data

def collect_ablation_runs(root):
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

def collect_probe_runs(root):
    data = defaultdict(lambda: defaultdict(list))
    for p1_seed, p1_dir in iter_from_p1_dirs(root):
        mixed_dir = p1_dir / "mixed"
        if not mixed_dir.exists():
            continue
        for seed_dir in sorted(mixed_dir.iterdir()):
            if not seed_dir.is_dir() or not re.match(r"seed\d+$", seed_dir.name):
                continue
            for run_dir in sorted(seed_dir.iterdir()):
                if not run_dir.is_dir() or run_dir.name not in {"mixed_0_4_0", "mixed_0_6_0", "mixed_ramp"}:
                    continue
                pdf = load_probe(run_dir)
                if pdf is not None:
                    data[run_dir.name][p1_seed].append(pdf)
    return data

# ═══════════════════════════════════════════════════════════
# Fig 3
# ═══════════════════════════════════════════════════════════

def fig3_mixed_with_baseline(mixed_root, sweep_root, outdir):
    mixed_data = collect_mixed_runs(mixed_root)
    if "mixed_0_4_0" not in mixed_data or not mixed_data["mixed_0_4_0"]:
        print("  [skip] fig3: no mixed_0_4_0 traj")
        return

    by_p1_mixed = mixed_data["mixed_0_4_0"]
    sweep_data = collect_sweep_runs(sweep_root) if sweep_root else {}
    by_p1_baseline = sweep_data.get(0, {})
    has_baseline = bool(by_p1_baseline)

    ref = next((dfs[0] for _, dfs in sorted(by_p1_mixed.items()) if dfs), None)
    if ref is None:
        print("  [skip] fig3: empty")
        return

    blk_info = []
    if "block_id" in ref.columns and "n_perturb_setting" in ref.columns:
        ep_blk = ref.groupby("episode")[["block_id", "n_perturb_setting"]].first()
        changes = ep_blk[ep_blk["block_id"] != ep_blk["block_id"].shift(1)]
        for ep_idx in changes.index:
            blk_info.append((int(ep_idx), int(changes.loc[ep_idx, "n_perturb_setting"])))

    fig, axes = plt.subplots(
        3, 1, figsize=(7.0, 4.5), sharex=True,
        gridspec_kw={"height_ratios": [0.45, 1, 1], "hspace": 0.08}
    )

    ax = axes[0]
    if "n_perturb_setting" in ref.columns:
        ep_nP = ref.groupby("episode")["n_perturb_setting"].first()
        ax.fill_between(ep_nP.index, 0, ep_nP.values, step="post", color="#ddd", alpha=0.6)
        ax.plot(ep_nP.index, ep_nP.values, drawstyle="steps-post", color="#888", lw=1.5)
    ax.set_ylabel("nP", fontsize=9)
    ax.set_ylim(-0.3, max(float(ref.get("n_perturb_setting", pd.Series([4])).max()) + 0.5, 5))
    ax.set_title("Block schedule (nP = 0 → 4 → 0)", fontsize=10, pad=4)
    ax.set_yticks([0, 4])
    for bc, _ in blk_info:
        ax.axvline(bc, color="#aaa", lw=0.6, ls="--", alpha=0.5)

    metrics = [("alpha", "α (plasticity)", C_ALPHA), ("g_norm", "||g||", C_GNORM)]

    for i, (metric, ylabel, color) in enumerate(metrics):
        ax = axes[i + 1]

        med, q1, q3, p1_curves = hierarchical_curve_summary(
            by_p1_mixed,
            lambda df, m=metric: df.groupby("episode")[m].mean() if m in df.columns else pd.Series(dtype=float),
        )
        for curve in p1_curves:
            ax.plot(curve.index, curve.values, lw=0.3, alpha=0.15, color=color)
        if med is not None:
            ax.fill_between(med.index, q1.values, q3.values, color=color, alpha=0.12)
            ax.plot(med.index, med.values, lw=2, color=color, label="Mixed (0→4→0)")

        if has_baseline:
            bl_med, bl_q1, bl_q3, _ = hierarchical_curve_summary(
                by_p1_baseline,
                lambda df, m=metric: df.groupby("episode")[m].mean() if m in df.columns else pd.Series(dtype=float),
            )
            if bl_med is not None:
                ax.fill_between(bl_med.index, bl_q1.values, bl_q3.values, color=C_BL, alpha=0.08)
                ax.plot(bl_med.index, bl_med.values, lw=1.5, color=C_BL, ls="--", label="Baseline (0→0→0)")

        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=7, loc="best")
        for bc, _ in blk_info:
            ax.axvline(bc, color="#aaa", lw=0.6, ls="--", alpha=0.5)

    axes[-1].set_xlabel("Episode")
    savefig(fig, outdir / "fig3_mixed_history.png")

# ═══════════════════════════════════════════════════════════
# Fig 4
# ═══════════════════════════════════════════════════════════

def fig4_perspective_perception(mixed_root, outdir):
    probe_runs = collect_probe_runs(mixed_root) if mixed_root else {}
    mixed_traj = collect_mixed_runs(mixed_root) if mixed_root else {}

    chosen_key = None
    for key in ["mixed_0_4_0", "mixed_0_6_0", "mixed_ramp"]:
        if key in probe_runs and probe_runs[key]:
            chosen_key = key
            break
    if chosen_key is None:
        print("  [skip] fig4: no probe data")
        return

    by_p1_probe = probe_runs[chosen_key]
    first_df = next((dfs[0] for _, dfs in sorted(by_p1_probe.items()) if dfs), None)
    if first_df is None:
        print("  [skip] fig4: empty probe")
        return

    zc = zt_cols(first_df)
    n_z = len(zc)
    if n_z == 0:
        print("  [skip] fig4: no z_t cols")
        return

    required_cols = {"probe_idx", "zone_id", "g_label", *zc}
    if not required_cols.issubset(first_df.columns):
        missing = sorted(required_cols - set(first_df.columns))
        print(f"  [skip] fig4: missing required columns: {missing}")
        return

    labels = sorted([str(l) for l in first_df["g_label"].dropna().unique()])
    target_label = "blk2_nP0"
    zero_label = "g_zero"

    if target_label not in labels:
        print(f"  [skip] fig4: missing required target label '{target_label}'")
        print(f"            available labels: {labels}")
        return
    if zero_label not in labels:
        print(f"  [skip] fig4: missing required zero label '{zero_label}'")
        print(f"            available labels: {labels}")
        return

    by_p1_traj = mixed_traj.get(chosen_key, {})
    has_panel_c = bool(by_p1_traj)

    n_cols = 3 if has_panel_c else 2
    fig = plt.figure(figsize=(12.5, 3.8) if has_panel_c else (8.4, 3.8))
    gs = gridspec.GridSpec(1, n_cols, width_ratios=[1.0] * n_cols, wspace=0.30)

    def to_numeric_block(df, cols):
        out = df.copy()
        for c in cols:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        return out.dropna(subset=cols)

    def make_paired_df(df, trained_label, zero_label):
        cols = ["probe_idx", "zone_id", "g_label"] + zc
        sub = df[cols].copy()

        trained = sub[sub["g_label"].astype(str) == trained_label].copy()
        zero = sub[sub["g_label"].astype(str) == zero_label].copy()
        if len(trained) == 0 or len(zero) == 0:
            return None

        trained = to_numeric_block(trained, zc)
        zero = to_numeric_block(zero, zc)
        if len(trained) == 0 or len(zero) == 0:
            return None

        trained = trained.rename(columns={c: f"{c}_tr" for c in zc})
        zero = zero.rename(columns={c: f"{c}_z0" for c in zc})

        merged = pd.merge(
            trained[["probe_idx", "zone_id"] + [f"{c}_tr" for c in zc]],
            zero[["probe_idx", "zone_id"] + [f"{c}_z0" for c in zc]],
            on=["probe_idx", "zone_id"],
            how="inner",
        )
        if len(merged) == 0:
            return None

        for c in zc:
            merged[f"d_{c}"] = merged[f"{c}_tr"] - merged[f"{c}_z0"]
        return merged

    def pca_project_shared(Xt, Xz):
        Xt = np.asarray(Xt, dtype=float)
        Xz = np.asarray(Xz, dtype=float)
        if Xt.ndim != 2 or Xz.ndim != 2 or len(Xt) < 2 or len(Xz) < 2:
            return None, None

        Xall = np.vstack([Xt, Xz])
        mu = Xall.mean(axis=0, keepdims=True)
        Xc = Xall - mu
        try:
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        except np.linalg.LinAlgError:
            return None, None

        k = min(2, Vt.shape[0])
        comps = Vt[:k].T
        Zall = Xc @ comps
        if k == 1:
            Zall = np.hstack([Zall, np.zeros((Zall.shape[0], 1))])

        return Zall[:len(Xt)], Zall[len(Xt):]

    def add_cov_ellipse(ax, X2, color, n_std=2.0, lw=1.4, alpha=0.18):
        from matplotlib.patches import Ellipse

        X2 = np.asarray(X2, dtype=float)
        if X2.ndim != 2 or X2.shape[1] != 2 or len(X2) < 3:
            return

        mean = X2.mean(axis=0)
        cov = np.cov(X2, rowvar=False)
        if not np.all(np.isfinite(cov)):
            return

        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        vals = np.clip(vals, 0, None)

        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        width, height = 2 * n_std * np.sqrt(vals)

        ell = Ellipse(
            xy=mean, width=width, height=height, angle=angle,
            facecolor=color, edgecolor=color, lw=lw, alpha=alpha, zorder=1
        )
        ax.add_patch(ell)

    # (a) PCA scatter + covariance ellipses
    ax = fig.add_subplot(gs[0])

    trained_blocks = []
    zero_blocks = []

    for _, dfs in sorted(by_p1_probe.items()):
        p2_trained = []
        p2_zero = []

        for df in dfs:
            sub = df[["g_label"] + zc].copy()

            tr = sub[sub["g_label"].astype(str) == target_label].copy()
            z0 = sub[sub["g_label"].astype(str) == zero_label].copy()

            tr = to_numeric_block(tr, zc)
            z0 = to_numeric_block(z0, zc)
            if len(tr) == 0 or len(z0) == 0:
                continue

            p2_trained.append(tr[zc].values)
            p2_zero.append(z0[zc].values)

        if not p2_trained or not p2_zero:
            continue

        min_len_tr = min(arr.shape[0] for arr in p2_trained)
        min_len_z0 = min(arr.shape[0] for arr in p2_zero)

        tr_stack = np.stack([arr[:min_len_tr] for arr in p2_trained], axis=0)
        z0_stack = np.stack([arr[:min_len_z0] for arr in p2_zero], axis=0)

        trained_blocks.append(np.median(tr_stack, axis=0))
        zero_blocks.append(np.median(z0_stack, axis=0))

    if not trained_blocks or not zero_blocks:
        print("  [skip] fig4(a): insufficient trained/zero probe data")
        plt.close(fig)
        return

    Xt = np.vstack(trained_blocks)
    Xz = np.vstack(zero_blocks)

    Zt, Zz = pca_project_shared(Xt, Xz)
    if Zt is None or Zz is None:
        print("  [skip] fig4(a): PCA failed")
        plt.close(fig)
        return

    ax.scatter(
        Zz[:, 0], Zz[:, 1],
        s=28, alpha=0.35, color="#999999",
        edgecolors="white", linewidths=0.3, label="g = 0", zorder=2
    )
    ax.scatter(
        Zt[:, 0], Zt[:, 1],
        s=28, alpha=0.60, color=C_AD,
        edgecolors="white", linewidths=0.3, label="g (trained)", zorder=3
    )

    add_cov_ellipse(ax, Zz, color="#777777", n_std=2.0, alpha=0.14, lw=1.2)
    add_cov_ellipse(ax, Zt, color=C_AD, n_std=2.0, alpha=0.16, lw=1.2)

    cz = Zz.mean(axis=0)
    ct = Zt.mean(axis=0)
    ax.scatter([cz[0]], [cz[1]], marker="x", s=95, lw=2.0, color="#555555", zorder=5)
    ax.scatter([ct[0]], [ct[1]], marker="x", s=95, lw=2.0, color=C_AD, zorder=6)
    ax.annotate("", xy=(ct[0], ct[1]), xytext=(cz[0], cz[1]),
                arrowprops=dict(arrowstyle="->", lw=1.1, color=C_AD, alpha=0.8))

    ax.axhline(0, color="#cccccc", lw=0.5)
    ax.axvline(0, color="#cccccc", lw=0.5)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("(a) PCA latent cloud geometry: trained g vs g = 0", fontsize=10)
    ax.legend(fontsize=7, loc="best")

    # (b)
    ax = fig.add_subplot(gs[1])
    p1_delta_vecs = []
    for _, dfs in sorted(by_p1_probe.items()):
        p2_vecs = []
        for df in dfs:
            pair_df = make_paired_df(df, target_label, zero_label)
            if pair_df is None or len(pair_df) == 0:
                continue
            vec = np.array([pair_df[f"d_{c}"].mean() for c in zc], dtype=float)
            if np.all(np.isfinite(vec)):
                p2_vecs.append(vec)
        if p2_vecs:
            p1_delta_vecs.append(np.median(np.vstack(p2_vecs), axis=0))

    if p1_delta_vecs:
        arr = np.vstack(p1_delta_vecs)
        diff = np.median(arr, axis=0)
        x_pos = np.arange(n_z)
        colors = [C_AD if v >= 0 else "#7799CC" for v in diff]
        ax.bar(x_pos, diff, color=colors, alpha=0.75, edgecolor="white", lw=0.3)
        ax.axhline(0, color="#ccc", lw=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(i) for i in range(n_z)], fontsize=7)
        ax.set_xlabel("Latent dimension z_i")
        ax.set_ylabel("Δz_t (trained − zero)")
        ax.set_title("(b) Per-dimension perceptual reorganization", fontsize=10)

    # (c)
    if has_panel_c:
        ax = fig.add_subplot(gs[2])
        ref_traj = next((dfs[0] for _, dfs in sorted(by_p1_traj.items()) if dfs), None)
        gc = gm_cols(ref_traj) if ref_traj is not None else []

        if gc and ref_traj is not None and "block_id" in ref_traj.columns:
            n_dims = len(gc)
            x_pos = np.arange(n_dims)
            width = 0.35

            for bi, (blk, color, label) in enumerate([
                (0, C_AD, "Block 0 (nP=0, pre)"),
                (2, C_GNORM, "Block 2 (nP=0, post)"),
            ]):
                def block_gamma_vec(df, block=blk):
                    if "block_id" not in df.columns:
                        return None
                    bd = df[df["block_id"] == block]
                    if len(bd) == 0:
                        return None
                    mx = int(bd["episode"].max())
                    bl = bd[bd["episode"] >= max(int(bd["episode"].min()), mx - 9)]
                    gc_df = gm_cols(df)
                    return bl[gc_df].mean().values if gc_df else None

                gvals, _ = hierarchical_vector_summary(by_p1_traj, block_gamma_vec)
                if gvals is not None:
                    offset = (bi - 0.5) * width
                    ax.bar(x_pos + offset, gvals, width,
                           alpha=0.7, color=color,
                           label=label, edgecolor="white", lw=0.3)

            ax.set_xticks(x_pos)
            ax.set_xticklabels([str(i) for i in range(n_dims)], fontsize=7)
            ax.axhline(0, color="#ccc", lw=0.5)
            ax.set_xlabel("Latent dimension z_i")
            ax.set_ylabel("γ (late block mean)")
            ax.set_title("(c) Gating residue: same env, different history", fontsize=10)
            ax.legend(fontsize=7)

    savefig(fig, outdir / "fig4_perspective_perception.png")

# ═══════════════════════════════════════════════════════════
# Fig 5
# ═══════════════════════════════════════════════════════════

def fig5_ablation(ablation_root, sweep_root, outdir):
    abl_data = collect_ablation_runs(ablation_root)
    sweep_data = collect_sweep_runs(sweep_root) if sweep_root else {}
    by_p1_baseline = sweep_data.get(0, {})
    has_baseline = bool(by_p1_baseline)

    conds_all = [
        ("adaptive",  C_AD, "Adaptive (nP=4)"),
        ("fixed_005", C_FX, "Fixed (α=0.05)"),
        ("fixed_010", C_FX, "Fixed (α=0.10)"),
        ("fast_080",  C_FA, "Fast (α=0.80)"),
    ]
    found_all = [
        (c, col, lab, abl_data[c])
        for c, col, lab in conds_all
        if c in abl_data and abl_data[c]
    ]

    adaptive_data = None
    for c, _, _, d in found_all:
        if c == "adaptive":
            adaptive_data = d
            break
    if adaptive_data is None:
        print("  [skip] fig5: no adaptive condition")
        return

    fig = plt.figure(figsize=(6.8, 7.0))
    gs = gridspec.GridSpec(2, 1, hspace=0.28)

    # (a) alpha
    ax = fig.add_subplot(gs[0])
    plot_specs = []
    if has_baseline:
        plot_specs.append((by_p1_baseline, C_BL, "Adaptive (nP=0)", "--"))
    plot_specs.append((adaptive_data, C_AD, "Adaptive (nP=4)", "-"))

    for by_p1, color, label, ls in plot_specs:
        med, q1, q3, p1c = hierarchical_curve_summary(
            by_p1,
            lambda df: df.groupby("episode")["alpha"].mean()
            if "alpha" in df.columns else pd.Series(dtype=float)
        )
        if med is not None:
            for curve in p1c:
                ax.plot(curve.index, curve.values, lw=0.3, alpha=0.12, color=color)
            ax.fill_between(med.index, q1.values, q3.values, color=color, alpha=0.10)
            ax.plot(med.index, med.values, lw=2, color=color, ls=ls, label=label)

    ax.set_xlabel("Episode")
    ax.set_ylabel("α (plasticity)")
    ax.set_title("(a) Adaptive plasticity dynamics", fontsize=10)
    ax.legend(fontsize=7, loc="lower center")
    ax.set_ylim(bottom=0.2)

    # (b) g_norm
    ax = fig.add_subplot(gs[1])
    for _, color, label, by_p1 in found_all:
        med, q1, q3, p1c = hierarchical_curve_summary(
            by_p1,
            lambda df: df.groupby("episode")["g_norm"].mean()
            if "g_norm" in df.columns else pd.Series(dtype=float)
        )
        if med is not None:
            for curve in p1c:
                ax.plot(curve.index, curve.values, lw=0.3, alpha=0.12, color=color)
            ax.fill_between(med.index, q1.values, q3.values, color=color, alpha=0.08)
            ax.plot(med.index, med.values, lw=2, color=color, label=label)

    if has_baseline:
        bl_med, bl_q1, bl_q3, bl_p1c = hierarchical_curve_summary(
            by_p1_baseline,
            lambda df: df.groupby("episode")["g_norm"].mean()
            if "g_norm" in df.columns else pd.Series(dtype=float)
        )
        if bl_med is not None:
            for curve in bl_p1c:
                ax.plot(curve.index, curve.values, lw=0.3, alpha=0.10, color=C_BL)
            ax.fill_between(bl_med.index, bl_q1.values, bl_q3.values, color=C_BL, alpha=0.06)
            ax.plot(bl_med.index, bl_med.values, lw=1.5, color=C_BL, ls="--",
                    label="Adaptive (nP=0)")

    ax.set_xlabel("Episode")
    ax.set_ylabel("||g||")
    ax.set_title("(b) Perspective magnitude across plasticity", fontsize=10)

    handles, labels = ax.get_legend_handles_labels()
    desired_order = {
        "Adaptive (nP=0)": 0,
        "Adaptive (nP=4)": 1,
        "Fixed (α=0.05)": 2,
        "Fixed (α=0.10)": 2,
        "Fast (α=0.80)": 3,
    }
    pairs = sorted(zip(handles, labels), key=lambda x: desired_order.get(x[1], 999))
    ax.legend([h for h, _ in pairs], [l for _, l in pairs], fontsize=7, loc="upper left")
    ax.set_ylim(top=4.5)

    savefig(fig, outdir / "fig5_ablation.png")

# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_root", type=str, default="")
    ap.add_argument("--mixed_root", type=str, default="")
    ap.add_argument("--ablation_root", type=str, default="")
    ap.add_argument("--outdir", type=str, default="paper_figures_v3")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sr = Path(args.sweep_root) if args.sweep_root else None
    mr = Path(args.mixed_root) if args.mixed_root else None
    ar = Path(args.ablation_root) if args.ablation_root else None

    print(f"Output: {outdir}\n")

    if mr:
        print("=== Fig 3: Mixed + baseline ===")
        try:
            fig3_mixed_with_baseline(mr, sr, outdir)
        except Exception as e:
            print(f"  [error] {e}")

    if mr:
        print("\n=== Fig 4: Perspective perception ===")
        try:
            fig4_perspective_perception(mr, outdir)
        except Exception as e:
            print(f"  [error] {e}")

    if ar:
        print("\n=== Fig 5: Ablation ===")
        try:
            fig5_ablation(ar, sr, outdir)
        except Exception as e:
            print(f"  [error] {e}")

    print(f"\nDone. {outdir}")

if __name__ == "__main__":
    main()

#%% spyder run cell 
import sys
from cear_pilot.analysis.paper_figures_v3 import main

sys.argv = [
    "paper_figures_v3",
    "--sweep_root", "outputs/phase2_all_20260403_011254",
    "--mixed_root", "outputs/phase2_all_20260403_011254",
    "--ablation_root", "outputs/phase2_all_20260403_011254",
    "--outdir", "outputs/phase2_all_20260403_011254/paper_figures_v3",
]

main()