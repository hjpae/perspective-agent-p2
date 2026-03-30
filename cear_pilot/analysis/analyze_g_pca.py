# cear_pilot/analysis/analyze_g_pca.py
# -*- coding: utf-8 -*-
"""
PCA analysis of g-space: visualize perspective basins.

Revised aggregation:
  1) aggregate across phase1 seeds within each phase2 seed
  2) show p2-level representatives
  3) for curves/bands, use median ± IQR across p2 seeds

Produces:
  1. PCA scatter: final g positions colored by valence (per alpha)
  2. PCA trajectories: g evolution within episodes (per alpha)
  3. PCA reversal: formation g -> reversal g migration (if reversal data exists)
  4. Basin separation over time

Usage:
    python cear_pilot/analysis/analyze_g_pca.py --sweep_root outputs/alpha_sweep_v2_YYYYMMDD
    python cear_pilot/analysis/analyze_g_pca.py --sweep_root outputs/alpha_sweep_v2_YYYYMMDD \
                                                --reversal_root outputs/reversal_YYYYMMDD
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.2,
    "font.size": 11,
    "axes.titlesize": 13,
    "figure.dpi": 120,
})
SAVE_DPI = 180

VALENCE_COLORS = {"SSSS": "#185FA5", "MMMM": "#D85A30"}


def savefig(fig, path, title=""):
    fig.tight_layout()
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {path.name}" + (f" — {title}" if title else ""))


def load_table(path):
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def find_file(d, stem):
    for ext in (".parquet", ".csv"):
        p = d / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def parse_label(name):
    m = re.match(r"a([\d.]+)_(SSSS|MMMM)_s(\d+)", name)
    if m:
        p1m = re.search(r"p1s(\d+)", name)
        result = {
            "alpha": float(m.group(1)),
            "valence": m.group(2),
            "seed": int(m.group(3)),                    # p2 seed
            "p1_seed": int(p1m.group(1)) if p1m else 0 # p1 seed
        }
        rev = re.search(r"_rev_(SSSS|MMMM)", name)
        if rev:
            result["formation_valence"] = result["valence"]
            result["test_valence"] = rev.group(1)
            result["is_reversal"] = True
        elif "_baseline" in name:
            result["formation_valence"] = result["valence"]
            result["test_valence"] = result["valence"]
            result["is_reversal"] = False
        else:
            result["formation_valence"] = result["valence"]
            result["test_valence"] = result["valence"]
            result["is_reversal"] = False
        return result
    return None


def detect_g_cols(df):
    cols = [c for c in df.columns if re.match(r"g_\d+$", c)]
    cols.sort(key=lambda x: int(x.split("_")[-1]))
    return cols


def load_all_traj(root, source="runs"):
    base = root / source
    if not base.exists():
        return pd.DataFrame()

    frames = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        parsed = parse_label(d.name)
        if not parsed:
            continue
        tp = find_file(d, "traj")
        if not tp:
            continue
        tr = load_table(tp)
        tr["alpha_fixed"] = parsed["alpha"]
        tr["valence"] = parsed["valence"]
        tr["seed"] = parsed["seed"]
        tr["p1_seed"] = parsed["p1_seed"]
        tr["formation_valence"] = parsed.get("formation_valence", parsed["valence"])
        tr["test_valence"] = parsed.get("test_valence", parsed["valence"])
        tr["is_reversal"] = parsed.get("is_reversal", False)
        frames.append(tr)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_p2_aggregated_traj(traj: pd.DataFrame, g_cols: List[str]) -> pd.DataFrame:
    """
    Merge p1 seeds inside each p2 seed.
    Returns one p2-level representative row per
    (alpha, valence, seed, episode, t, formation_valence, test_valence, is_reversal).
    """
    if len(traj) == 0:
        return traj.copy()

    group_cols = [
        "alpha_fixed", "valence", "seed", "episode", "t",
        "formation_valence", "test_valence", "is_reversal"
    ]

    frames = []
    for keys, sub in traj.groupby(group_cols, dropna=False):
        row = {col: val for col, val in zip(group_cols, keys)}
        for g in g_cols:
            row[g] = float(sub[g].mean())
        for c in ["x", "y", "event_now"]:
            if c in sub.columns:
                row[c] = float(sub[c].mean())
        frames.append(row)

    return pd.DataFrame(frames)


# ── Figure 1: PCA scatter of late-episode g ────────────

def fig_pca_scatter(traj, outdir):
    """PCA of late-episode g using p2-level representatives."""
    g_cols = detect_g_cols(traj)
    if len(g_cols) == 0:
        return None

    traj_p2 = build_p2_aggregated_traj(traj, g_cols)

    alphas = sorted(traj_p2["alpha_fixed"].unique())
    n_alpha = len(alphas)

    late = traj_p2[traj_p2["t"] > 200]
    if len(late) == 0:
        return None

    pca = PCA(n_components=2)
    pca.fit(late[g_cols].values)
    var_explained = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, n_alpha, figsize=(5 * n_alpha, 5))
    if n_alpha == 1:
        axes = [axes]

    for ai, alpha in enumerate(alphas):
        ax = axes[ai]
        a_late = late[late["alpha_fixed"] == alpha]

        for val in ["SSSS", "MMMM"]:
            v_data = a_late[a_late["valence"] == val]
            if len(v_data) == 0:
                continue

            g_proj = pca.transform(v_data[g_cols].values)
            ax.scatter(
                g_proj[:, 0], g_proj[:, 1],
                c=VALENCE_COLORS[val], alpha=0.35, s=20,
                label=val, edgecolors="none"
            )

            ax.scatter(
                np.median(g_proj[:, 0]), np.median(g_proj[:, 1]),
                c=VALENCE_COLORS[val], s=180, marker="X",
                edgecolors="black", linewidth=1.0, zorder=10
            )

        ax.set_xlabel(f"PC1 ({var_explained[0]:.0%})")
        if ai == 0:
            ax.set_ylabel(f"PC2 ({var_explained[1]:.0%})")
        ax.set_title(f"α={alpha}")
        ax.legend(markerscale=2.5, framealpha=0.7)

    fig.suptitle("PCA of late-episode g (p1 merged inside each p2 seed)", fontsize=14, y=1.02)
    savefig(fig, outdir / "pca_scatter_late_g.png", "PCA scatter")
    return pca


def fig_pca_trajectories(traj, pca, outdir):
    """PCA trajectories using p2-level representatives."""
    g_cols = detect_g_cols(traj)
    if len(g_cols) == 0 or pca is None:
        return

    traj_p2 = build_p2_aggregated_traj(traj, g_cols)

    alphas = sorted(traj_p2["alpha_fixed"].unique())
    n_alpha = len(alphas)

    fig, axes = plt.subplots(1, n_alpha, figsize=(5 * n_alpha, 5))
    if n_alpha == 1:
        axes = [axes]

    target_eps = [0, 50, 100]

    for ai, alpha in enumerate(alphas):
        ax = axes[ai]
        a_data = traj_p2[traj_p2["alpha_fixed"] == alpha]

        for val in ["SSSS", "MMMM"]:
            v_data = a_data[a_data["valence"] == val]
            color = VALENCE_COLORS[val]

            for seed in sorted(v_data["seed"].unique()):
                s_data = v_data[v_data["seed"] == seed]
                for ep in target_eps:
                    ep_data = s_data[s_data["episode"] == ep].sort_values("t")
                    if len(ep_data) < 10:
                        continue
                    g_proj = pca.transform(ep_data[g_cols].values)
                    ax.plot(g_proj[:, 0], g_proj[:, 1], color=color, alpha=0.18, lw=0.7)

            v_late = v_data[v_data["t"] > 250]
            if len(v_late) > 0:
                g_late_proj = pca.transform(v_late[g_cols].values)
                ax.scatter(
                    np.median(g_late_proj[:, 0]),
                    np.median(g_late_proj[:, 1]),
                    c=color, s=200, marker="X",
                    edgecolors="black", linewidth=1,
                    zorder=10, label=f"{val} p2-median"
                )

        ax.set_xlabel("PC1")
        if ai == 0:
            ax.set_ylabel("PC2")
        ax.set_title(f"α={alpha}")
        ax.legend(framealpha=0.7, fontsize=8)

    fig.suptitle("G trajectories in PCA space (p1 merged → p2 representatives)", fontsize=14, y=1.02)
    savefig(fig, outdir / "pca_trajectories.png", "PCA trajectories")


def fig_pca_basin_separation(traj, pca, outdir):
    """Basin separation with p1 merged inside p2, then median±IQR across p2."""
    g_cols = detect_g_cols(traj)
    if len(g_cols) == 0 or pca is None:
        return

    alphas = sorted(traj["alpha_fixed"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    t_points = list(range(0, 301, 10))

    for alpha in alphas:
        a_data = traj[traj["alpha_fixed"] == alpha]
        p2_curves = []

        for seed in sorted(a_data["seed"].unique()):
            s2_data = a_data[a_data["seed"] == seed]
            distances = []

            for t in t_points:
                p1_dists = []
                for p1_seed in sorted(s2_data["p1_seed"].unique()):
                    p1_data = s2_data[s2_data["p1_seed"] == p1_seed]
                    window = p1_data[(p1_data["t"] >= t) & (p1_data["t"] < t + 10)]
                    ssss_g = window[window["valence"] == "SSSS"][g_cols].values
                    mmmm_g = window[window["valence"] == "MMMM"][g_cols].values
                    if len(ssss_g) >= 3 and len(mmmm_g) >= 3:
                        p1_dists.append(np.linalg.norm(ssss_g.mean(axis=0) - mmmm_g.mean(axis=0)))

                if len(p1_dists) > 0:
                    distances.append(float(np.mean(p1_dists)))
                else:
                    distances.append(np.nan)

            p2_curves.append(distances)

        curves = np.array(p2_curves, dtype=float)
        med = np.nanmedian(curves, axis=0)
        q1 = np.nanpercentile(curves, 25, axis=0)
        q3 = np.nanpercentile(curves, 75, axis=0)

        color = {
            "0.05": "#185FA5", "0.1": "#3B8BD4", "0.2": "#1D9E75",
            "0.25": "#1D9E75", "0.3": "#D85A30", "0.5": "#E24B4A"
        }.get(str(alpha), "#888")

        ax.plot(t_points, med, color=color, lw=1.5, label=f"α={alpha}")
        ax.fill_between(t_points, q1, q3, color=color, alpha=0.12)

    for evt_t in [45, 105, 165, 225]:
        ax.axvline(evt_t, color="#E24B4A", ls="--", lw=0.8, alpha=0.4)

    ax.set_xlabel("Step within episode")
    ax.set_ylabel("||centroid_SSSS - centroid_MMMM|| in g-space")
    ax.set_title("Basin separation over time")
    ax.legend(framealpha=0.7)

    savefig(fig, outdir / "pca_basin_separation_over_time.png",
            "Basin separation dynamics")


# ── Reversal analysis ──────────────────────────────────

def fig_pca_reversal(formation_traj, reversal_traj, pca, outdir):
    """Show g migration in PCA space during reversal using p2-level representatives."""
    g_cols = detect_g_cols(formation_traj)
    if len(g_cols) == 0 or pca is None or len(reversal_traj) == 0:
        print("  [skip] reversal PCA — missing data")
        return

    formation_p2 = build_p2_aggregated_traj(formation_traj, g_cols)
    reversal_p2 = build_p2_aggregated_traj(reversal_traj, g_cols)

    alphas = sorted(reversal_p2["alpha_fixed"].unique())
    n_alpha = len(alphas)

    fig, axes = plt.subplots(1, n_alpha, figsize=(5 * n_alpha, 5))
    if n_alpha == 1:
        axes = [axes]

    form_late = formation_p2[formation_p2["t"] > 200]

    for ai, alpha in enumerate(alphas):
        ax = axes[ai]

        for val in ["SSSS", "MMMM"]:
            f_data = form_late[
                (form_late["alpha_fixed"] == alpha) &
                (form_late["valence"] == val)
            ]
            if len(f_data) > 0:
                g_proj = pca.transform(f_data[g_cols].values)
                ax.scatter(g_proj[:, 0], g_proj[:, 1],
                           c=VALENCE_COLORS[val], alpha=0.08, s=8,
                           edgecolors="none")
                cx, cy = np.median(g_proj[:, 0]), np.median(g_proj[:, 1])
                ax.annotate(f"{val}\nbasin", (cx, cy), fontsize=8,
                            ha="center", color=VALENCE_COLORS[val], alpha=0.7)

        r_data = reversal_p2[
            (reversal_p2["alpha_fixed"] == alpha) &
            (reversal_p2["is_reversal"] == True)
        ]
        for form_val in ["SSSS", "MMMM"]:
            v_data = r_data[r_data["formation_valence"] == form_val]
            if len(v_data) == 0:
                continue

            color = VALENCE_COLORS[form_val]

            for seed in sorted(v_data["seed"].unique()):
                s_data = v_data[v_data["seed"] == seed]
                for ep in [0, 5]:
                    ep_data = s_data[s_data["episode"] == ep].sort_values("t")
                    if len(ep_data) < 10:
                        continue
                    g_proj = pca.transform(ep_data[g_cols].values)
                    ax.plot(g_proj[:, 0], g_proj[:, 1],
                            color=color, alpha=0.45, lw=1.0, ls="--")
                    ax.scatter(g_proj[0, 0], g_proj[0, 1],
                               c=color, s=30, marker="o", edgecolors="black",
                               linewidth=0.7, zorder=5)
                    ax.scatter(g_proj[-1, 0], g_proj[-1, 1],
                               c=color, s=45, marker="*", edgecolors="black",
                               linewidth=0.7, zorder=5)

        ax.set_xlabel("PC1")
        if ai == 0:
            ax.set_ylabel("PC2")
        ax.set_title(f"α={alpha}")

    fig.suptitle("Reversal: g migration from formation basin toward opposite evidence",
                 fontsize=13, y=1.02)
    savefig(fig, outdir / "pca_reversal_migration.png", "Reversal PCA")


# ── Main ──────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_root", type=str, required=True)
    ap.add_argument("--reversal_root", type=str, default="")
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--source", type=str, default="runs")
    args = ap.parse_args()

    sweep_root = Path(args.sweep_root)
    outdir = Path(args.outdir) if args.outdir else sweep_root / "pca_analysis"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading formation data from {sweep_root}...")
    traj = load_all_traj(sweep_root, source=args.source)
    if len(traj) == 0:
        print("No trajectory data found.")
        return
    print(f"  {len(traj)} rows")

    g_cols = detect_g_cols(traj)
    print(f"  {len(g_cols)} g dimensions")

    pca = fig_pca_scatter(traj, outdir)
    if pca is not None:
        fig_pca_trajectories(traj, pca, outdir)
        fig_pca_basin_separation(traj, pca, outdir)

    if args.reversal_root:
        rev_root = Path(args.reversal_root)
        print(f"\nLoading reversal data from {rev_root}...")
        rev_traj = load_all_traj(rev_root, source=args.source)
        if len(rev_traj) > 0 and pca is not None:
            fig_pca_reversal(traj, rev_traj, pca, outdir)
        else:
            print("  No reversal data or PCA unavailable")

    print(f"\nDone. Figures in: {outdir}")


if __name__ == "__main__":
    main()