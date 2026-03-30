# cear_pilot/analysis/reanalyze_gscore.py
# -*- coding: utf-8 -*-
"""
Reanalysis of existing alpha sweep data using g-score (discriminant projection)
instead of Δ||g|| norm.

Revised aggregation:
  1) aggregate across phase1 seeds within each phase2 seed
  2) aggregate across phase2 seeds using median ± IQR

Produces:
  1. Event-triggered g-score response
  2. Per-event SSSS-MMMM separation increment
  3. Prediction loss / responsiveness / separation sweet-spot
  4. G-score trajectory overlay

Usage:
  python cear_pilot/analysis/reanalyze_gscore.py --sweep_root outputs/alpha_sweep_YYYYMMDD
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 120,
})

SAVE_DPI = 180

ALPHA_COLORS = {
    0.05: "#185FA5",
    0.10: "#3B8BD4",
    0.20: "#1D9E75",
    0.25: "#1D9E75",
    0.30: "#D85A30",
    0.35: "#D85A30",
    0.50: "#E24B4A",
}
ALPHA_LABELS = {
    0.05: "α=0.05",
    0.10: "α=0.10",
    0.20: "α=0.20",
    0.25: "α=0.25",
    0.30: "α=0.30",
    0.35: "α=0.35",
    0.50: "α=0.50",
}


def savefig(fig, path, title=""):
    fig.tight_layout()
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {path.name}" + (f" — {title}" if title else ""))


def load_table(path):
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def find_file(directory, stem):
    for ext in (".parquet", ".csv"):
        p = directory / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def parse_run_label(name):
    m = re.match(r"a([\d.]+)_(SSSS|MMMM)_s(\d+)", name)
    if m:
        p1m = re.search(r"p1s(\d+)", name)
        p1_seed = int(p1m.group(1)) if p1m else 0
        return {
            "alpha": float(m.group(1)),
            "valence": m.group(2),
            "seed": int(m.group(3)),       # p2 seed
            "p1_seed": p1_seed,            # p1 seed
        }
    return None


def detect_g_cols(df):
    cols = [c for c in df.columns if re.match(r"g_\d+$", c)]
    cols.sort(key=lambda x: int(x.split("_")[-1]))
    return cols


def discover_and_load_traj(sweep_root, source="runs"):
    """Load all trajectory data from sweep."""
    base = sweep_root / source
    if not base.exists():
        raise FileNotFoundError(f"No {source}/ directory in {sweep_root}")

    frames = []
    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir():
            continue
        parsed = parse_run_label(run_dir.name)
        if parsed is None:
            continue
        traj_path = find_file(run_dir, "traj")
        if traj_path is None:
            continue
        tr = load_table(traj_path)
        tr["alpha_fixed"] = parsed["alpha"]
        tr["valence"] = parsed["valence"]
        tr["seed"] = parsed["seed"]
        tr["p1_seed"] = parsed["p1_seed"]
        frames.append(tr)

    if len(frames) == 0:
        raise FileNotFoundError("No trajectory files found")

    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(frames)} runs")
    print(f"  Alphas: {sorted(df['alpha_fixed'].unique())}")
    print(f"  Valences: {sorted(df['valence'].unique())}")
    print(f"  p2 Seeds: {sorted(df['seed'].unique())}")
    print(f"  p1 Seeds: {sorted(df['p1_seed'].unique())}")
    return df


def hierarchical_scalar_agg(
    df: pd.DataFrame,
    group_cols: List[str],
    value_col: str,
) -> pd.DataFrame:
    req = set(group_cols + ["p1_seed", "seed", value_col])
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    lvl0 = (
        df.groupby(group_cols + ["seed", "p1_seed"])[value_col]
          .mean()
          .reset_index()
    )

    lvl1 = (
        lvl0.groupby(group_cols + ["seed"])[value_col]
            .mean()
            .reset_index()
            .rename(columns={value_col: "p2_mean"})
    )

    out = (
        lvl1.groupby(group_cols)["p2_mean"]
            .agg(
                median="median",
                q1=lambda x: x.quantile(0.25),
                q3=lambda x: x.quantile(0.75),
                mean="mean",
                n="count",
            )
            .reset_index()
    )

    out.rename(columns={
        "median": f"{value_col}_median",
        "q1": f"{value_col}_q1",
        "q3": f"{value_col}_q3",
        "mean": f"{value_col}_mean",
        "n": "n_p2_seeds",
    }, inplace=True)
    return out


def hierarchical_pair_curves(curve_dicts):
    if len(curve_dicts) == 0:
        return {}

    xs = sorted(set().union(*[d.keys() for d in curve_dicts]))
    out = {}
    for x in xs:
        vals = [d[x] for d in curve_dicts if x in d]
        if len(vals) == 0:
            continue
        out[x] = {
            "median": float(np.median(vals)),
            "q1": float(np.percentile(vals, 25)),
            "q3": float(np.percentile(vals, 75)),
            "mean": float(np.mean(vals)),
            "n": int(len(vals)),
        }
    return out


def compute_discriminant_direction(traj, g_cols, alpha, min_t=200):
    """Compute SSSS-MMMM discriminant direction for a given alpha, using late-episode g."""
    a_data = traj[traj["alpha_fixed"] == alpha]
    late = a_data[a_data["t"] > min_t]

    g_ssss = late[late["valence"] == "SSSS"][g_cols].values
    g_mmmm = late[late["valence"] == "MMMM"][g_cols].values

    if len(g_ssss) < 10 or len(g_mmmm) < 10:
        return None

    u = g_ssss.mean(axis=0) - g_mmmm.mean(axis=0)
    norm = np.linalg.norm(u)
    if norm < 1e-8:
        return None
    return u / norm


def project_g_score(g_values, u_hat):
    return g_values @ u_hat


# ── Figure 1: Event-triggered g-score response ────────

def fig_event_triggered_gscore(traj, outdir):
    """Event-triggered response using g-score projection with p1->p2 hierarchy."""
    g_cols = detect_g_cols(traj)
    if len(g_cols) == 0 or "event_now" not in traj.columns:
        print("  [skip] Missing g columns or event_now")
        return

    window = 30
    tau_range = range(-window, window + 1)
    alphas = sorted(traj["alpha_fixed"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for vi, val in enumerate(["SSSS", "MMMM"]):
        ax = axes[vi]
        val_data = traj[traj["valence"] == val]

        for alpha in alphas:
            u_hat = compute_discriminant_direction(traj, g_cols, alpha)
            if u_hat is None:
                continue

            a_data = val_data[val_data["alpha_fixed"] == alpha]
            p2_curves = []

            for seed in sorted(a_data["seed"].unique()):
                s2_data = a_data[a_data["seed"] == seed]
                p1_curves = []

                for p1_seed in sorted(s2_data["p1_seed"].unique()):
                    s_data = s2_data[s2_data["p1_seed"] == p1_seed].copy()
                    s_data = s_data.sort_values(["episode", "t"])
                    if len(s_data) == 0:
                        continue

                    g_mat = s_data[g_cols].values
                    s_data["g_score"] = project_g_score(g_mat, u_hat)

                    snippets = []
                    for ep in s_data["episode"].unique():
                        ep_data = s_data[s_data["episode"] == ep]
                        ep_events = ep_data[ep_data["event_now"] == 1]
                        if len(ep_events) == 0:
                            continue

                        for evt_t in ep_events["t"].values:
                            snippet = {}
                            for tau in tau_range:
                                t_abs = evt_t + tau
                                row = ep_data[ep_data["t"] == t_abs]
                                if len(row) > 0:
                                    snippet[tau] = row["g_score"].values[0]

                            if len(snippet) > window:
                                pre_vals = [snippet[t] for t in range(-window, -5) if t in snippet]
                                if len(pre_vals) > 0:
                                    baseline = np.median(pre_vals)
                                    snippets.append({t: v - baseline for t, v in snippet.items()})

                    if len(snippets) == 0:
                        continue

                    all_taus = sorted(set().union(*[s.keys() for s in snippets]))
                    p1_curve = {}
                    for tau in all_taus:
                        vals = [s[tau] for s in snippets if tau in s]
                        if len(vals) >= 2:
                            p1_curve[tau] = float(np.median(vals))

                    if len(p1_curve) > 0:
                        p1_curves.append(p1_curve)

                if len(p1_curves) == 0:
                    continue

                p2_curve = {}
                all_taus = sorted(set().union(*[d.keys() for d in p1_curves]))
                for tau in all_taus:
                    vals = [d[tau] for d in p1_curves if tau in d]
                    if len(vals) >= 1:
                        p2_curve[tau] = float(np.mean(vals))

                if len(p2_curve) > 0:
                    p2_curves.append(p2_curve)

            if len(p2_curves) == 0:
                continue

            agg = hierarchical_pair_curves(p2_curves)
            xs = sorted(agg.keys())
            med = [agg[x]["median"] for x in xs]
            q1 = [agg[x]["q1"] for x in xs]
            q3 = [agg[x]["q3"] for x in xs]

            color = ALPHA_COLORS.get(alpha, "#888")
            label = ALPHA_LABELS.get(alpha, f"α={alpha}")
            ax.plot(xs, med, color=color, label=label, lw=1.5)
            ax.fill_between(xs, q1, q3, color=color, alpha=0.12)

        ax.axvline(0, color="#999", ls=":", lw=0.8)
        ax.axhline(0, color="#ccc", ls="-", lw=0.5)
        ax.set_title(f"Valence: {val}")
        ax.set_xlabel("Steps relative to event")
        if vi == 0:
            ax.set_ylabel("Δ g-score (baseline-subtracted)")
        ax.legend(loc="best", framealpha=0.7)

    fig.suptitle("Event-triggered g-score response", fontsize=14, y=1.02)
    savefig(fig, outdir / "event_triggered_gscore.png",
            "Event-triggered g-score (discriminant projection)")


# ── Figure 2: Per-event separation increment ──────────

def fig_per_event_separation(traj, outdir):
    """SSSS-MMMM g-score gap measured after each of the 4 events."""
    g_cols = detect_g_cols(traj)
    if len(g_cols) == 0 or "event_now" not in traj.columns:
        print("  [skip] Missing g columns or event_now")
        return

    alphas = sorted(traj["alpha_fixed"].unique())
    post_window = 15

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_data = {}

    for alpha in alphas:
        u_hat = compute_discriminant_direction(traj, g_cols, alpha)
        if u_hat is None:
            continue

        a_data = traj[traj["alpha_fixed"] == alpha].copy()
        a_data["g_score"] = project_g_score(a_data[g_cols].values, u_hat)

        p2_gaps = []

        for seed in sorted(a_data["seed"].unique()):
            s2_data = a_data[a_data["seed"] == seed]
            p1_gap_rows = []

            for p1_seed in sorted(s2_data["p1_seed"].unique()):
                s_data = s2_data[s2_data["p1_seed"] == p1_seed]
                event_gaps = []

                for event_idx in range(4):
                    ssss_scores = []
                    mmmm_scores = []

                    for val in ["SSSS", "MMMM"]:
                        v_data = s_data[s_data["valence"] == val]
                        for ep in v_data["episode"].unique():
                            ep_data = v_data[v_data["episode"] == ep]
                            events = ep_data[ep_data["event_now"] == 1]
                            if len(events) <= event_idx:
                                continue
                            evt_t = events["t"].values[event_idx]

                            post = ep_data[
                                (ep_data["t"] >= evt_t + 1) &
                                (ep_data["t"] <= evt_t + post_window)
                            ]
                            if len(post) >= 3:
                                score = post["g_score"].mean()
                                if val == "SSSS":
                                    ssss_scores.append(score)
                                else:
                                    mmmm_scores.append(score)

                    if len(ssss_scores) >= 3 and len(mmmm_scores) >= 3:
                        gap = np.median(ssss_scores) - np.median(mmmm_scores)
                        event_gaps.append(gap)
                    else:
                        event_gaps.append(np.nan)

                p1_gap_rows.append(event_gaps)

            if len(p1_gap_rows) == 0:
                continue

            p1_gap_arr = np.array(p1_gap_rows, dtype=float)
            p2_gap = np.nanmean(p1_gap_arr, axis=0)
            p2_gaps.append(p2_gap)

        if len(p2_gaps) == 0:
            continue

        gaps_arr = np.array(p2_gaps, dtype=float)
        med = np.nanmedian(gaps_arr, axis=0)
        q1 = np.nanpercentile(gaps_arr, 25, axis=0)
        q3 = np.nanpercentile(gaps_arr, 75, axis=0)

        bar_data[alpha] = (med, q1, q3)

    event_labels = ["After event 1", "After event 2", "After event 3", "After event 4"]
    x = np.arange(4)
    n_alpha = len(bar_data)
    width = 0.8 / max(n_alpha, 1)

    for i, (alpha, (med, q1, q3)) in enumerate(sorted(bar_data.items())):
        offset = (i - (n_alpha - 1) / 2) * width
        color = ALPHA_COLORS.get(alpha, "#888")
        label = ALPHA_LABELS.get(alpha, f"α={alpha}")
        yerr = [med - q1, q3 - med]
        ax.bar(x + offset, med, width, yerr=yerr, color=color, alpha=0.7,
               label=label, capsize=3, error_kw={"lw": 1})

    ax.set_xticks(x)
    ax.set_xticklabels(event_labels)
    ax.set_ylabel("SSSS − MMMM g-score gap")
    ax.set_title("Per-event evidence separation increment")
    ax.legend(framealpha=0.7)
    ax.axhline(0, color="#ccc", lw=0.5)

    savefig(fig, outdir / "per_event_separation.png",
            "Per-event SSSS-MMMM separation")


# ── Figure 3: Maturity sweet spot ─────────────────────

def fig_maturity_sweet_spot(traj, ep_data, outdir):
    """Show that moderate alpha achieves best prediction while maintaining responsive g."""
    if len(ep_data) == 0:
        return

    alphas = sorted(ep_data["alpha_fixed"].unique())
    g_cols = detect_g_cols(traj)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel A: Prediction loss
    ax = axes[0]
    for val in ["SSSS", "MMMM"]:
        meds, q1s, q3s = [], [], []
        for alpha in alphas:
            subset = ep_data[
                (ep_data["alpha_fixed"] == alpha) &
                (ep_data["valence"] == val)
            ]
            agg = hierarchical_scalar_agg(
                subset,
                group_cols=["alpha_fixed", "valence"],
                value_col="mean_pred_loss",
            )
            if len(agg) == 0:
                meds.append(np.nan); q1s.append(np.nan); q3s.append(np.nan)
            else:
                rr = agg.iloc[0]
                meds.append(rr["mean_pred_loss_median"])
                q1s.append(rr["mean_pred_loss_q1"])
                q3s.append(rr["mean_pred_loss_q3"])

        color = "#185FA5" if val == "SSSS" else "#D85A30"
        ls = "-" if val == "SSSS" else "--"
        ax.plot(alphas, meds, color=color, ls=ls, marker="o", lw=1.5, label=val, markersize=6)
        ax.fill_between(alphas, q1s, q3s, color=color, alpha=0.1)

    ax.set_xlabel("α (update rate)")
    ax.set_ylabel("Prediction loss")
    ax.set_title("Prediction accuracy")
    ax.legend(framealpha=0.7)

    # Panel B: Responsiveness
    ax = axes[1]
    for val in ["SSSS", "MMMM"]:
        meds, q1s, q3s = [], [], []
        for alpha in alphas:
            subset = ep_data[
                (ep_data["alpha_fixed"] == alpha) &
                (ep_data["valence"] == val)
            ]
            agg = hierarchical_scalar_agg(
                subset,
                group_cols=["alpha_fixed", "valence"],
                value_col="mean_delta_g",
            )
            if len(agg) == 0:
                meds.append(np.nan); q1s.append(np.nan); q3s.append(np.nan)
            else:
                rr = agg.iloc[0]
                meds.append(rr["mean_delta_g_median"])
                q1s.append(rr["mean_delta_g_q1"])
                q3s.append(rr["mean_delta_g_q3"])

        color = "#185FA5" if val == "SSSS" else "#D85A30"
        ls = "-" if val == "SSSS" else "--"
        ax.plot(alphas, meds, color=color, ls=ls, marker="o", lw=1.5, label=val, markersize=6)
        ax.fill_between(alphas, q1s, q3s, color=color, alpha=0.1)

    ax.set_xlabel("α (update rate)")
    ax.set_ylabel("Mean Δg per step")
    ax.set_title("Evidence responsiveness")
    ax.legend(framealpha=0.7)

    # Panel C: Late separation in g-score
    ax = axes[2]
    if len(g_cols) > 0:
        sep_vals_ssss = []
        sep_q1_ssss = []
        sep_q3_ssss = []
        sep_vals_mmmm = []
        sep_q1_mmmm = []
        sep_q3_mmmm = []

        for val in ["SSSS", "MMMM"]:
            meds, q1s, q3s = [], [], []
            for alpha in alphas:
                u_hat = compute_discriminant_direction(traj, g_cols, alpha)
                if u_hat is None:
                    meds.append(np.nan); q1s.append(np.nan); q3s.append(np.nan)
                    continue

                a_data = traj[
                    (traj["alpha_fixed"] == alpha) &
                    (traj["valence"] == val) &
                    (traj["t"] > 200)
                ].copy()
                if len(a_data) == 0:
                    meds.append(np.nan); q1s.append(np.nan); q3s.append(np.nan)
                    continue

                a_data["g_score"] = project_g_score(a_data[g_cols].values, u_hat)
                agg = hierarchical_scalar_agg(
                    a_data,
                    group_cols=["alpha_fixed", "valence"],
                    value_col="g_score",
                )
                if len(agg) == 0:
                    meds.append(np.nan); q1s.append(np.nan); q3s.append(np.nan)
                else:
                    rr = agg.iloc[0]
                    meds.append(rr["g_score_median"])
                    q1s.append(rr["g_score_q1"])
                    q3s.append(rr["g_score_q3"])

            color = "#185FA5" if val == "SSSS" else "#D85A30"
            ls = "-" if val == "SSSS" else "--"
            ax.plot(alphas, meds, color=color, ls=ls, marker="o", lw=1.5, label=val, markersize=6)
            ax.fill_between(alphas, q1s, q3s, color=color, alpha=0.1)

    ax.set_xlabel("α (update rate)")
    ax.set_ylabel("Late g-score")
    ax.set_title("Late perspective separation")
    ax.legend(framealpha=0.7)

    fig.suptitle("Maturity profile: prediction vs responsiveness vs separation",
                 fontsize=14, y=1.02)
    savefig(fig, outdir / "maturity_sweet_spot.png", "Maturity sweet spot")


# ── Figure 4: G-score trajectory overlay ──────────────

def fig_gscore_trajectory_overlay(traj, outdir):
    """G-score trajectories with event markers overlaid, using p1->p2 hierarchy."""
    g_cols = detect_g_cols(traj)
    if len(g_cols) == 0:
        return

    alphas = sorted(traj["alpha_fixed"].unique())
    n_alpha = len(alphas)

    fig, axes = plt.subplots(1, n_alpha, figsize=(5 * n_alpha, 5), sharey=True)
    if n_alpha == 1:
        axes = [axes]

    for ai, alpha in enumerate(alphas):
        ax = axes[ai]
        u_hat = compute_discriminant_direction(traj, g_cols, alpha)
        if u_hat is None:
            continue

        a_data = traj[traj["alpha_fixed"] == alpha]

        for val in ["SSSS", "MMMM"]:
            v_data = a_data[a_data["valence"] == val].copy()
            if len(v_data) == 0:
                continue

            v_data["proj"] = project_g_score(v_data[g_cols].values, u_hat)

            p2_curves = {}
            for seed in sorted(v_data["seed"].unique()):
                s2 = v_data[v_data["seed"] == seed]
                p1_curves = []

                for p1_seed in sorted(s2["p1_seed"].unique()):
                    s = s2[s2["p1_seed"] == p1_seed]
                    curve = s.groupby("t")["proj"].median()
                    p1_curves.append(curve)

                if len(p1_curves) == 0:
                    continue

                all_t = sorted(set().union(*[set(c.index) for c in p1_curves]))
                vals = []
                for t in all_t:
                    cur_vals = [c.loc[t] for c in p1_curves if t in c.index]
                    vals.append(np.mean(cur_vals) if len(cur_vals) > 0 else np.nan)
                p2_curves[seed] = pd.Series(vals, index=all_t)

            if len(p2_curves) == 0:
                continue

            all_t = sorted(set().union(*[set(c.index) for c in p2_curves.values()]))
            med, q1, q3 = [], [], []
            for t in all_t:
                vals = [p2_curves[s][t] for s in p2_curves if t in p2_curves[s].index]
                if len(vals) >= 2:
                    med.append(np.median(vals))
                    q1.append(np.percentile(vals, 25))
                    q3.append(np.percentile(vals, 75))
                else:
                    med.append(np.nan)
                    q1.append(np.nan)
                    q3.append(np.nan)

            color = "#185FA5" if val == "SSSS" else "#D85A30"
            ls = "-" if val == "SSSS" else "--"
            ax.plot(all_t, med, color=color, ls=ls, lw=1.5, label=val)
            ax.fill_between(all_t, q1, q3, color=color, alpha=0.1)

        for evt_t in [45, 105, 165, 225]:
            ax.axvline(evt_t, color="#ccc", ls=":", lw=0.6, alpha=0.6)

        ax.axhline(0, color="#ccc", ls="-", lw=0.5)
        ax.set_title(ALPHA_LABELS.get(alpha, f"α={alpha}"))
        ax.set_xlabel("Step within episode")
        if ai == 0:
            ax.set_ylabel("g-score")
        ax.legend(loc="best", framealpha=0.7)

    fig.suptitle("G-score trajectories with event markers", fontsize=14, y=1.02)
    savefig(fig, outdir / "gscore_trajectory_overlay.png", "G-score + events")


# ── Main ──────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_root", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="")
    ap.add_argument("--source", type=str, default="runs", choices=["runs", "collect"])
    args = ap.parse_args()

    sweep_root = Path(args.sweep_root)
    outdir = Path(args.outdir) if args.outdir else sweep_root / "reanalysis"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Sweep root: {sweep_root}")
    print(f"Output: {outdir}")

    traj = discover_and_load_traj(sweep_root, source=args.source)

    ep_frames = []
    base = sweep_root / args.source
    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir():
            continue
        parsed = parse_run_label(run_dir.name)
        if parsed is None:
            continue
        ep_path = find_file(run_dir, "episode_summary")
        if ep_path is None:
            continue
        ep = load_table(ep_path)
        ep["alpha_fixed"] = parsed["alpha"]
        ep["valence"] = parsed["valence"]
        ep["seed"] = parsed["seed"]
        ep["p1_seed"] = parsed["p1_seed"]
        ep_frames.append(ep)

    ep_data = pd.concat(ep_frames, ignore_index=True) if ep_frames else pd.DataFrame()

    print("\nGenerating reanalysis figures...")
    fig_event_triggered_gscore(traj, outdir)
    fig_per_event_separation(traj, outdir)
    fig_gscore_trajectory_overlay(traj, outdir)

    if len(ep_data) > 0:
        fig_maturity_sweet_spot(traj, ep_data, outdir)

    print(f"\nDone. Figures in: {outdir}")


if __name__ == "__main__":
    main()