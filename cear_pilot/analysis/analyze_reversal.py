# cear_pilot/analysis/analyze_reversal.py
# -*- coding: utf-8 -*-
"""
Reversal analysis: "same world, different perspective."

Core question: agents with identical architecture and alpha,
but different formation histories (SSSS vs MMMM),
encounter the OPPOSITE evidence. How does g respond?

Produces:
  1. Reversal g-score trajectory: formed-SSSS in MMMM world vs formed-MMMM in SSSS world
  2. Recalibration speed: how fast does g cross the midpoint?
  3. Formation residue: does formation history leave a permanent trace?
  4. PCA migration: reversal trajectories overlaid on formation basins
  5. PE during reversal: prediction error trajectory (perspective-world mismatch)

Usage:
    python cear_pilot/analysis/analyze_reversal.py \
        --formation_root outputs/alpha_sweep_p1s0_a05_a25_a50 \
        --reversal_root outputs/alpha_sweep_p1s0_a05_a25_a50/reversal_YYYYMMDD
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 120,
})
SAVE_DPI = 180

COLORS = {
    "SSSS_in_SSSS": "#185FA5",  # blue — native supportive
    "SSSS_in_MMMM": "#85B7EB",  # light blue — supportive-formed, now in misleading
    "MMMM_in_MMMM": "#D85A30",  # coral — native misleading
    "MMMM_in_SSSS": "#F0997B",  # light coral — misleading-formed, now in supportive
}


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
    if not m:
        return None
    result = {
        "alpha": float(m.group(1)),
        "formation_valence": m.group(2),
        "seed": int(m.group(3)),
    }
    rev = re.search(r"_rev_(SSSS|MMMM)", name)
    if rev:
        result["test_valence"] = rev.group(1)
        result["condition"] = "reversal"
    elif "_baseline" in name:
        result["test_valence"] = result["formation_valence"]
        result["condition"] = "baseline"
    else:
        result["test_valence"] = result["formation_valence"]
        result["condition"] = "formation"
    return result


def detect_g_cols(df):
    cols = [c for c in df.columns if re.match(r"g_\d+$", c)]
    cols.sort(key=lambda x: int(x.split("_")[-1]))
    return cols


def load_traj_from(root, source="runs"):
    base = root / source
    if not base.exists():
        # Also try loading directly from root if no runs/ subdir
        base = root
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

        # If data already has phase/formation_valence columns (from collect_reversal),
        # use those. Otherwise use parsed directory name info.
        if "formation_valence" not in tr.columns:
            for k, v in parsed.items():
                tr[k] = v
        else:
            # Data has its own columns — just add alpha/seed from parsed
            tr["alpha"] = parsed.get("alpha", tr.get("alpha", 0))
            tr["seed"] = parsed.get("seed", tr.get("seed", 0))
            # Map phase column to condition column for compatibility
            if "phase" in tr.columns and "condition" not in tr.columns:
                tr["condition"] = tr["phase"].map(
                    lambda p: "reversal" if p == "reversal" else "baseline"
                )

        frames.append(tr)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def compute_discriminant(formation_traj, g_cols, alpha):
    """Compute SSSS-MMMM discriminant direction from formation data."""
    a_data = formation_traj[formation_traj["alpha"] == alpha]
    late = a_data[a_data["t"] > 200]
    g_ssss = late[late["formation_valence"] == "SSSS"][g_cols].values
    g_mmmm = late[late["formation_valence"] == "MMMM"][g_cols].values
    if len(g_ssss) < 10 or len(g_mmmm) < 10:
        return None
    u = g_ssss.mean(axis=0) - g_mmmm.mean(axis=0)
    norm = np.linalg.norm(u)
    if norm < 1e-8:
        return None
    return u / norm


# ── Figure 1: Reversal g-score trajectories ────────────

def fig_reversal_gscore(formation, reversal, outdir):
    """The skull-water figure: same stimulus, different history → different g trajectory."""

    g_cols = detect_g_cols(formation)
    if len(g_cols) == 0:
        print("  [skip] No g columns")
        return

    alphas = sorted(formation["alpha"].unique())
    n_alpha = len(alphas)

    fig, axes = plt.subplots(1, n_alpha, figsize=(5.5 * n_alpha, 5))
    if n_alpha == 1:
        axes = [axes]

    for ai, alpha in enumerate(alphas):
        ax = axes[ai]
        u_hat = compute_discriminant(formation, g_cols, alpha)
        if u_hat is None:
            continue

        # Four conditions to plot:
        # 1. SSSS-formed in SSSS (baseline) — native
        # 2. SSSS-formed in MMMM (reversal) — skull-water moment
        # 3. MMMM-formed in MMMM (baseline) — native
        # 4. MMMM-formed in SSSS (reversal) — skull-water moment

        conditions = [
            ("SSSS", "SSSS", "baseline", COLORS["SSSS_in_SSSS"], "-",  "SSSS native"),
            ("SSSS", "MMMM", "reversal", COLORS["SSSS_in_MMMM"], "--", "SSSS → MMMM"),
            ("MMMM", "MMMM", "baseline", COLORS["MMMM_in_MMMM"], "-",  "MMMM native"),
            ("MMMM", "SSSS", "reversal", COLORS["MMMM_in_SSSS"], "--", "MMMM → SSSS"),
        ]

        for form_val, test_val, cond, color, ls, label in conditions:
            # Get data from appropriate source
            if cond == "baseline":
                source = reversal  # baselines are collected alongside reversals
            else:
                source = reversal

            c_data = source[
                (source["alpha"] == alpha) &
                (source["formation_valence"] == form_val) &
                (source["test_valence"] == test_val) &
                (source["condition"] == cond)
            ]

            if len(c_data) == 0:
                # Try formation data for baselines
                if cond == "baseline":
                    c_data = formation[
                        (formation["alpha"] == alpha) &
                        (formation["formation_valence"] == form_val)
                    ]
                if len(c_data) == 0:
                    continue

            # Project onto discriminant
            g_mat = c_data[g_cols].values
            c_data = c_data.copy()
            c_data["gscore"] = g_mat @ u_hat

            # Aggregate: per-seed median by timestep, then cross-seed median ± IQR
            seed_curves = {}
            for seed in c_data["seed"].unique():
                s = c_data[c_data["seed"] == seed]
                curve = s.groupby("t")["gscore"].median()
                seed_curves[seed] = curve

            if len(seed_curves) == 0:
                continue

            all_t = sorted(set().union(*[set(c.index) for c in seed_curves.values()]))
            med, q1, q3 = [], [], []
            for t in all_t:
                vals = [seed_curves[s][t] for s in seed_curves if t in seed_curves[s].index]
                if len(vals) >= 2:
                    med.append(np.median(vals))
                    q1.append(np.percentile(vals, 25))
                    q3.append(np.percentile(vals, 75))
                else:
                    med.append(np.nan)
                    q1.append(np.nan)
                    q3.append(np.nan)

            ax.plot(all_t, med, color=color, ls=ls, lw=1.5, label=label)
            ax.fill_between(all_t, q1, q3, color=color, alpha=0.1)

        # Event markers
        for evt_t in [45, 105, 165, 225]:
            ax.axvline(evt_t, color="#ccc", ls=":", lw=0.8, alpha=0.5)
        ax.axhline(0, color="#999", ls="-", lw=0.5)

        ax.set_xlabel("Step within episode")
        if ai == 0:
            ax.set_ylabel("g-score (SSSS-MMMM axis)")
        ax.set_title(f"α={alpha}")
        ax.legend(fontsize=8, framealpha=0.7, loc="best")

    fig.suptitle("Reversal: same world, different formation history",
                 fontsize=14, y=1.02)
    savefig(fig, outdir / "reversal_gscore_trajectory.png",
            "Reversal g-score trajectories")


# ── Figure 2: Formation residue ────────────────────────

def fig_formation_residue(formation, reversal, outdir):
    """After 300 steps of opposite evidence, how much of formation history remains?"""

    g_cols = detect_g_cols(formation)
    if len(g_cols) == 0:
        return

    alphas = sorted(formation["alpha"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))

    results = []

    for alpha in alphas:
        u_hat = compute_discriminant(formation, g_cols, alpha)
        if u_hat is None:
            continue

        # Late-episode g-scores from baselines (reference)
        for form_val in ["SSSS", "MMMM"]:
            # Native endpoint
            native = formation[
                (formation["alpha"] == alpha) &
                (formation["formation_valence"] == form_val)
            ]
            native_late = native[native["t"] > 250]
            if len(native_late) == 0:
                continue
            native_gscore = (native_late[g_cols].values @ u_hat).mean()

            # Reversed endpoint
            rev = reversal[
                (reversal["alpha"] == alpha) &
                (reversal["formation_valence"] == form_val) &
                (reversal["condition"] == "reversal")
            ]
            rev_late = rev[rev["t"] > 250]
            if len(rev_late) == 0:
                continue

            # Per-seed residue
            for seed in rev_late["seed"].unique():
                s_rev = rev_late[rev_late["seed"] == seed]
                rev_gscore = (s_rev[g_cols].values @ u_hat).mean()

                # Target: where native opposite-valence agents end up
                opposite = "MMMM" if form_val == "SSSS" else "SSSS"
                opp_native = formation[
                    (formation["alpha"] == alpha) &
                    (formation["formation_valence"] == opposite)
                ]
                opp_late = opp_native[opp_native["t"] > 250]
                if len(opp_late) == 0:
                    continue
                target_gscore = (opp_late[g_cols].values @ u_hat).mean()

                # Residue: how far from target, as fraction of total distance
                total_dist = abs(native_gscore - target_gscore)
                if total_dist < 1e-6:
                    continue
                residue = abs(rev_gscore - target_gscore) / total_dist

                results.append({
                    "alpha": alpha,
                    "formation": form_val,
                    "seed": seed,
                    "residue": residue,
                    "native_gscore": native_gscore,
                    "reversed_gscore": rev_gscore,
                    "target_gscore": target_gscore,
                })

    if len(results) == 0:
        print("  [skip] No residue data")
        return

    res_df = pd.DataFrame(results)

    # Plot: residue by alpha, colored by formation
    for form_val, color in [("SSSS", "#185FA5"), ("MMMM", "#D85A30")]:
        f_data = res_df[res_df["formation"] == form_val]
        if len(f_data) == 0:
            continue
        agg = f_data.groupby("alpha")["residue"].agg(
            median="median",
            q1=lambda x: x.quantile(0.25),
            q3=lambda x: x.quantile(0.75),
        )
        ax.plot(agg.index, agg["median"], color=color, marker="o",
                lw=1.5, label=f"Formed in {form_val}", markersize=6)
        ax.fill_between(agg.index, agg["q1"], agg["q3"], color=color, alpha=0.12)

    ax.set_xlabel("α (update rate)")
    ax.set_ylabel("Formation residue (1 = no recalibration, 0 = fully recalibrated)")
    ax.set_title("How much formation history persists after reversal?")
    ax.axhline(0, color="#ccc", lw=0.5)
    ax.axhline(1, color="#ccc", lw=0.5)
    ax.set_ylim(-0.1, 1.3)
    ax.legend(framealpha=0.7)

    savefig(fig, outdir / "formation_residue.png", "Formation residue")


# ── Figure 3: PCA reversal migration ──────────────────

def fig_pca_reversal(formation, reversal, outdir):
    """PCA scatter with reversal trajectories overlaid."""

    g_cols = detect_g_cols(formation)
    if len(g_cols) == 0:
        return

    # Shared PCA from formation data
    late = formation[formation["t"] > 200]
    g_all = late[g_cols].values
    if len(g_all) < 50:
        return
    pca = PCA(n_components=2)
    pca.fit(g_all)
    var_exp = pca.explained_variance_ratio_

    alphas = sorted(formation["alpha"].unique())
    n_alpha = len(alphas)

    fig, axes = plt.subplots(1, n_alpha, figsize=(5.5 * n_alpha, 5))
    if n_alpha == 1:
        axes = [axes]

    for ai, alpha in enumerate(alphas):
        ax = axes[ai]

        # Formation basins as background
        a_late = late[late["alpha"] == alpha]
        for val, color in [("SSSS", "#185FA5"), ("MMMM", "#D85A30")]:
            v_data = a_late[a_late["formation_valence"] == val]
            if len(v_data) > 1500:
                v_data = v_data.sample(1500, random_state=42)
            if len(v_data) == 0:
                continue
            g_proj = pca.transform(v_data[g_cols].values)
            ax.scatter(g_proj[:, 0], g_proj[:, 1],
                       c=color, alpha=0.08, s=5, edgecolors="none")

            # Centroid label
            cx, cy = g_proj[:, 0].mean(), g_proj[:, 1].mean()
            ax.annotate(f"{val}", (cx, cy), fontsize=9, ha="center",
                        color=color, fontweight="bold", alpha=0.5)

        # Reversal trajectories
        a_rev = reversal[
            (reversal["alpha"] == alpha) &
            (reversal["condition"] == "reversal")
        ]

        for form_val in ["SSSS", "MMMM"]:
            fv_data = a_rev[a_rev["formation_valence"] == form_val]
            color = COLORS[f"{form_val}_in_{'MMMM' if form_val == 'SSSS' else 'SSSS'}"]

            # Show 2-3 representative trajectories per seed
            seeds_to_show = sorted(fv_data["seed"].unique())[:3]
            for seed in seeds_to_show:
                s_data = fv_data[fv_data["seed"] == seed]
                # Take first episode
                eps = sorted(s_data["episode"].unique())[:1]
                for ep in eps:
                    ep_data = s_data[s_data["episode"] == ep].sort_values("t")
                    if len(ep_data) < 20:
                        continue
                    g_proj = pca.transform(ep_data[g_cols].values)

                    # Draw trajectory
                    ax.plot(g_proj[:, 0], g_proj[:, 1],
                            color=color, alpha=0.5, lw=0.8)
                    # Start marker
                    ax.scatter(g_proj[0, 0], g_proj[0, 1],
                               c=color, s=40, marker="o",
                               edgecolors="black", linewidth=0.5, zorder=5)
                    # End marker
                    ax.scatter(g_proj[-1, 0], g_proj[-1, 1],
                               c=color, s=60, marker="*",
                               edgecolors="black", linewidth=0.5, zorder=5)

        ax.set_xlabel(f"PC1 ({var_exp[0]:.0%})")
        if ai == 0:
            ax.set_ylabel(f"PC2 ({var_exp[1]:.0%})")
        ax.set_title(f"α={alpha}")

    fig.suptitle("PCA: reversal trajectories (o=start in formation basin, *=end)",
                 fontsize=13, y=1.02)
    savefig(fig, outdir / "pca_reversal_migration.png", "PCA reversal migration")


# ── Figure 4: Recalibration speed ─────────────────────

def fig_recalibration_speed(formation, reversal, outdir):
    """How many steps until g-score crosses the midpoint during reversal?"""

    g_cols = detect_g_cols(formation)
    if len(g_cols) == 0:
        return

    alphas = sorted(formation["alpha"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))

    for form_val, color in [("SSSS", "#185FA5"), ("MMMM", "#D85A30")]:
        cross_times_by_alpha = {}

        for alpha in alphas:
            u_hat = compute_discriminant(formation, g_cols, alpha)
            if u_hat is None:
                continue

            rev = reversal[
                (reversal["alpha"] == alpha) &
                (reversal["formation_valence"] == form_val) &
                (reversal["condition"] == "reversal")
            ]

            cross_times = []
            for seed in rev["seed"].unique():
                s_data = rev[rev["seed"] == seed]
                for ep in s_data["episode"].unique():
                    ep_data = s_data[s_data["episode"] == ep].sort_values("t")
                    if len(ep_data) < 20:
                        continue
                    gscores = ep_data[g_cols].values @ u_hat

                    # Find first crossing of zero (midpoint)
                    initial_sign = np.sign(gscores[0]) if abs(gscores[0]) > 0.01 else 0
                    if initial_sign == 0:
                        continue
                    crossed = False
                    for ti, gs in enumerate(gscores):
                        if np.sign(gs) != initial_sign and np.sign(gs) != 0:
                            cross_times.append(ep_data["t"].values[ti])
                            crossed = True
                            break
                    if not crossed:
                        cross_times.append(300)  # never crossed

            if len(cross_times) >= 2:
                cross_times_by_alpha[alpha] = cross_times

        if len(cross_times_by_alpha) == 0:
            continue

        a_list = sorted(cross_times_by_alpha.keys())
        medians = [np.median(cross_times_by_alpha[a]) for a in a_list]
        q1s = [np.percentile(cross_times_by_alpha[a], 25) for a in a_list]
        q3s = [np.percentile(cross_times_by_alpha[a], 75) for a in a_list]

        ax.plot(a_list, medians, color=color, marker="o", lw=1.5,
                label=f"Formed in {form_val}", markersize=6)
        ax.fill_between(a_list, q1s, q3s, color=color, alpha=0.12)

    ax.set_xlabel("α (update rate)")
    ax.set_ylabel("Steps to cross midpoint (lower = faster recalibration)")
    ax.set_title("Recalibration speed: how quickly does perspective shift?")
    ax.legend(framealpha=0.7)

    savefig(fig, outdir / "recalibration_speed.png", "Recalibration speed")


# ── Figure 5: PE during reversal ──────────────────────

def fig_reversal_pe(formation, reversal, outdir):
    """Prediction error during reversal — should spike then decrease."""

    pe_col = None
    for c in ["pred_err_min", "pred_err", "loss_pred"]:
        if c in reversal.columns:
            pe_col = c
            break
    if pe_col is None:
        print("  [skip] No prediction error column found")
        return

    alphas = sorted(reversal["alpha"].unique())
    n_alpha = len(alphas)

    fig, axes = plt.subplots(1, n_alpha, figsize=(5.5 * n_alpha, 4.5))
    if n_alpha == 1:
        axes = [axes]

    for ai, alpha in enumerate(alphas):
        ax = axes[ai]

        conditions = [
            ("SSSS", "baseline", COLORS["SSSS_in_SSSS"], "-",  "SSSS native"),
            ("SSSS", "reversal", COLORS["SSSS_in_MMMM"], "--", "SSSS → MMMM"),
            ("MMMM", "baseline", COLORS["MMMM_in_MMMM"], "-",  "MMMM native"),
            ("MMMM", "reversal", COLORS["MMMM_in_SSSS"], "--", "MMMM → SSSS"),
        ]

        for form_val, cond, color, ls, label in conditions:
            c_data = reversal[
                (reversal["alpha"] == alpha) &
                (reversal["formation_valence"] == form_val) &
                (reversal["condition"] == cond)
            ]
            if len(c_data) == 0:
                continue

            # Aggregate by t
            seed_curves = {}
            for seed in c_data["seed"].unique():
                s = c_data[c_data["seed"] == seed]
                curve = s.groupby("t")[pe_col].median()
                # Smooth for readability
                if len(curve) > 10:
                    curve = curve.rolling(window=10, min_periods=3, center=True).median()
                seed_curves[seed] = curve

            if len(seed_curves) == 0:
                continue

            all_t = sorted(set().union(*[set(c.index) for c in seed_curves.values()]))
            med, q1, q3 = [], [], []
            for t in all_t:
                vals = [seed_curves[s][t] for s in seed_curves
                        if t in seed_curves[s].index and not np.isnan(seed_curves[s][t])]
                if len(vals) >= 2:
                    med.append(np.median(vals))
                    q1.append(np.percentile(vals, 25))
                    q3.append(np.percentile(vals, 75))
                else:
                    med.append(np.nan)
                    q1.append(np.nan)
                    q3.append(np.nan)

            ax.plot(all_t, med, color=color, ls=ls, lw=1.2, label=label)
            ax.fill_between(all_t, q1, q3, color=color, alpha=0.08)

        for evt_t in [45, 105, 165, 225]:
            ax.axvline(evt_t, color="#ccc", ls=":", lw=0.6)
        ax.set_xlabel("Step")
        if ai == 0:
            ax.set_ylabel("Prediction error")
        ax.set_title(f"α={alpha}")
        ax.legend(fontsize=7, framealpha=0.7)

    fig.suptitle("Prediction error: reversal vs native", fontsize=14, y=1.02)
    savefig(fig, outdir / "reversal_prediction_error.png", "Reversal PE")


# ── Main ──────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formation_root", type=str, required=True)
    ap.add_argument("--reversal_root", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="")
    args = ap.parse_args()

    formation_root = Path(args.formation_root)
    reversal_root = Path(args.reversal_root)
    outdir = Path(args.outdir) if args.outdir else reversal_root / "reversal_analysis"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Formation root: {formation_root}")
    print(f"Reversal root:  {reversal_root}")
    print(f"Output:         {outdir}")

    # Load formation data
    print("\nLoading formation trajectories...")
    formation = load_traj_from(formation_root, source="runs")
    # Tag formation data
    if "condition" not in formation.columns:
        formation["condition"] = "formation"
    if "test_valence" not in formation.columns:
        formation["test_valence"] = formation.get("formation_valence",
                                                    formation.get("valence", ""))
    if "formation_valence" not in formation.columns and "valence" in formation.columns:
        formation["formation_valence"] = formation["valence"]
    if "alpha" not in formation.columns and "alpha_fixed" in formation.columns:
        formation["alpha"] = formation["alpha_fixed"]
    print(f"  {len(formation)} rows")

    # Load reversal data
    print("Loading reversal trajectories...")
    reversal = load_traj_from(reversal_root, source="runs")
    if "alpha" not in reversal.columns and "alpha_fixed" in reversal.columns:
        reversal["alpha"] = reversal["alpha_fixed"]
    print(f"  {len(reversal)} rows")

    if len(reversal) == 0:
        print("No reversal data found!")
        return

    # Print condition summary
    print("\nConditions found in reversal data:")
    for cond in reversal["condition"].unique():
        c_data = reversal[reversal["condition"] == cond]
        for fv in c_data["formation_valence"].unique():
            for tv in c_data["test_valence"].unique():
                n = len(c_data[(c_data["formation_valence"] == fv) &
                               (c_data["test_valence"] == tv)])
                if n > 0:
                    print(f"  {cond}: {fv} → {tv}: {n} rows")

    # Generate figures
    print("\nGenerating reversal analysis figures...")

    fig_reversal_gscore(formation, reversal, outdir)
    fig_formation_residue(formation, reversal, outdir)
    fig_pca_reversal(formation, reversal, outdir)
    fig_recalibration_speed(formation, reversal, outdir)
    fig_reversal_pe(formation, reversal, outdir)

    print(f"\nDone. Figures in: {outdir}")


if __name__ == "__main__":
    main()