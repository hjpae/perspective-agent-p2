# compare_alpha_sweep.py
# -*- coding: utf-8 -*-
"""
Alpha sweep analysis: aggregate across seeds, compare alpha × valence conditions.
Produces publication-ready figures with median ± IQR across seeds.

Usage:
    python compare_alpha_sweep.py --sweep_root outputs/alpha_sweep_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Style ───────────────────────────────────────────────
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
    0.05: "#185FA5",   # blue – rigid
    0.25: "#1D9E75",   # teal – moderate
    0.50: "#D85A30",   # coral – reactive
}
ALPHA_LABELS = {
    0.05: "α=0.05 (rigid)",
    0.25: "α=0.25 (moderate)",
    0.50: "α=0.50 (reactive)",
}
VALENCE_STYLES = {
    "SSSS": {"ls": "-",  "marker": "o"},
    "MMMM": {"ls": "--", "marker": "s"},
}


def timestamp_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


# ── Data loading ────────────────────────────────────────

def load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def find_file(directory: Path, stem: str) -> Optional[Path]:
    for ext in (".parquet", ".csv"):
        p = directory / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def parse_run_label(name: str) -> Optional[Dict[str, Any]]:
    """Parse 'a0.05_SSSS_s0' or similar patterns."""
    m = re.match(r"a([\d.]+)_(SSSS|MMMM)_s(\d+)", name)
    if m:
        return {
            "alpha": float(m.group(1)),
            "valence": m.group(2),
            "seed": int(m.group(3)),
        }
    return None


def discover_runs(sweep_root: Path) -> pd.DataFrame:
    """Find all runs under sweep_root/runs/ and sweep_root/collect/."""
    records = []

    # Check both runs/ (training) and collect/ (greedy eval)
    for subdir_name in ("runs", "collect"):
        base = sweep_root / subdir_name
        if not base.exists():
            continue
        for run_dir in sorted(base.iterdir()):
            if not run_dir.is_dir():
                continue

            parsed = parse_run_label(run_dir.name)
            if parsed is None:
                # Try subdirectories (train_phase2 appends timestamp)
                for sub in sorted(run_dir.iterdir()):
                    if sub.is_dir():
                        parsed = parse_run_label(run_dir.name)
                        if parsed:
                            run_dir = sub
                            break
                if parsed is None:
                    continue

            ep_path = find_file(run_dir, "episode_summary")
            traj_path = find_file(run_dir, "traj")

            records.append({
                "alpha": parsed["alpha"],
                "valence": parsed["valence"],
                "seed": parsed["seed"],
                "source": subdir_name,
                "run_dir": str(run_dir),
                "ep_path": str(ep_path) if ep_path else None,
                "traj_path": str(traj_path) if traj_path else None,
            })

    df = pd.DataFrame(records)
    if len(df) == 0:
        raise FileNotFoundError(f"No valid runs found under {sweep_root}")
    return df


def load_all_episode_data(run_index: pd.DataFrame) -> pd.DataFrame:
    """Load and concatenate all episode summaries."""
    frames = []
    for _, row in run_index.iterrows():
        if row["ep_path"] is None:
            continue
        ep = load_table(Path(row["ep_path"]))
        ep["alpha_fixed"] = row["alpha"]
        ep["valence"] = row["valence"]
        ep["seed"] = row["seed"]
        ep["source"] = row["source"]
        frames.append(ep)
    return pd.concat(frames, ignore_index=True)


def load_all_traj_data(run_index: pd.DataFrame, source: str = "runs") -> pd.DataFrame:
    """Load and concatenate all trajectory data from a given source."""
    frames = []
    subset = run_index[run_index["source"] == source]
    for _, row in subset.iterrows():
        if row["traj_path"] is None:
            continue
        tr = load_table(Path(row["traj_path"]))
        tr["alpha_fixed"] = row["alpha"]
        tr["valence"] = row["valence"]
        tr["seed"] = row["seed"]
        frames.append(tr)
    if len(frames) == 0:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def detect_g_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if re.match(r"g_\d+$", c)]
    cols.sort(key=lambda x: int(x.split("_")[-1]))
    return cols


# ── Aggregation helpers ─────────────────────────────────

def agg_median_iqr(
    df: pd.DataFrame,
    group_cols: List[str],
    value_col: str,
) -> pd.DataFrame:
    """Compute median, Q1, Q3 across seeds within each group."""
    grouped = df.groupby(group_cols)[value_col]
    out = grouped.agg(
        median="median",
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75),
        mean="mean",
        n="count",
    ).reset_index()
    out.rename(columns={
        "median": f"{value_col}_median",
        "q1": f"{value_col}_q1",
        "q3": f"{value_col}_q3",
        "mean": f"{value_col}_mean",
        "n": f"{value_col}_n",
    }, inplace=True)
    return out


def savefig(fig: plt.Figure, path: Path, title: str = "") -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {path.name}" + (f" — {title}" if title else ""))


# ── Figure functions ────────────────────────────────────

def fig_episode_training_curves(ep_data: pd.DataFrame, outdir: Path) -> None:
    """Training curves (episode-level metrics) by alpha, panel per valence."""

    metrics = [
        ("mean_pred_loss", "Prediction loss"),
        ("mean_delta_g",   "Mean Δg per step"),
        ("mean_energy",    "Energy"),
        ("mean_entropy",   "Policy entropy"),
    ]

    for metric_col, metric_label in metrics:
        if metric_col not in ep_data.columns:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for vi, val in enumerate(["SSSS", "MMMM"]):
            ax = axes[vi]
            subset = ep_data[ep_data["valence"] == val]
            if len(subset) == 0:
                continue

            for alpha in sorted(subset["alpha_fixed"].unique()):
                a_sub = subset[subset["alpha_fixed"] == alpha]

                agg = a_sub.groupby("episode")[metric_col].agg(
                    median="median",
                    q1=lambda x: x.quantile(0.25),
                    q3=lambda x: x.quantile(0.75),
                )

                color = ALPHA_COLORS.get(alpha, "#888")
                label = ALPHA_LABELS.get(alpha, f"α={alpha}")

                ax.plot(agg.index, agg["median"], color=color, label=label, lw=1.5)
                ax.fill_between(agg.index, agg["q1"], agg["q3"],
                                color=color, alpha=0.15)

            ax.set_title(f"Valence: {val}")
            ax.set_xlabel("Episode")
            if vi == 0:
                ax.set_ylabel(metric_label)
            ax.legend(loc="best", framealpha=0.7)

        fig.suptitle(metric_label, fontsize=14, y=1.02)
        savefig(fig, outdir / f"train_{metric_col}.png", metric_label)


def fig_event_triggered_g_response(traj_data: pd.DataFrame, outdir: Path) -> None:
    """Event-triggered g displacement: Δ||g|| around event timesteps."""

    if "event_now" not in traj_data.columns or len(traj_data) == 0:
        print("  [skip] event_triggered_g — missing event_now column")
        return

    g_cols = detect_g_cols(traj_data)
    if len(g_cols) == 0:
        print("  [skip] event_triggered_g — no g columns")
        return

    window = 30  # steps before/after event
    tau_range = range(-window, window + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for vi, val in enumerate(["SSSS", "MMMM"]):
        ax = axes[vi]
        val_data = traj_data[traj_data["valence"] == val]
        if len(val_data) == 0:
            continue

        for alpha in sorted(val_data["alpha_fixed"].unique()):
            a_data = val_data[val_data["alpha_fixed"] == alpha]

            # Collect per-seed median trajectories
            seed_medians = []
            for seed in a_data["seed"].unique():
                s_data = a_data[a_data["seed"] == seed]

                # Find event timesteps
                event_mask = s_data["event_now"] == 1
                if event_mask.sum() == 0:
                    continue

                # Compute g norm for each row
                g_mat = s_data[g_cols].values
                g_norm = np.linalg.norm(g_mat, axis=1)

                # For each episode, compute delta_g relative to event time
                snippets = []
                for ep in s_data["episode"].unique():
                    ep_data_local = s_data[s_data["episode"] == ep]
                    ep_events = ep_data_local[ep_data_local["event_now"] == 1]
                    if len(ep_events) == 0:
                        continue

                    ep_idx = ep_data_local.index
                    ep_g_norm = g_norm[np.isin(s_data.index, ep_idx)]

                    for evt_pos in ep_events["t"].values:
                        snippet = {}
                        for tau in tau_range:
                            t_abs = evt_pos + tau
                            mask = ep_data_local["t"] == t_abs
                            if mask.any():
                                idx_in_ep = np.where(mask.values)[0][0]
                                if idx_in_ep < len(ep_g_norm):
                                    snippet[tau] = ep_g_norm[idx_in_ep]
                        if len(snippet) > window:
                            # Normalize to baseline (median of pre-event window)
                            pre_vals = [snippet[t] for t in range(-window, -2) if t in snippet]
                            if len(pre_vals) > 0:
                                baseline = np.median(pre_vals)
                                snippets.append({t: v - baseline for t, v in snippet.items()})

                if len(snippets) > 0:
                    # Median across snippets within this seed
                    all_taus = sorted(set().union(*[s.keys() for s in snippets]))
                    seed_curve = {}
                    for tau in all_taus:
                        vals = [s[tau] for s in snippets if tau in s]
                        if len(vals) >= 2:
                            seed_curve[tau] = np.median(vals)
                    seed_medians.append(seed_curve)

            if len(seed_medians) == 0:
                continue

            # Aggregate across seeds: median ± IQR
            all_taus = sorted(set().union(*[s.keys() for s in seed_medians]))
            med_line, q1_line, q3_line = [], [], []
            valid_taus = []
            for tau in all_taus:
                vals = [s[tau] for s in seed_medians if tau in s]
                if len(vals) >= 2:
                    valid_taus.append(tau)
                    med_line.append(np.median(vals))
                    q1_line.append(np.percentile(vals, 25))
                    q3_line.append(np.percentile(vals, 75))

            color = ALPHA_COLORS.get(alpha, "#888")
            label = ALPHA_LABELS.get(alpha, f"α={alpha}")

            ax.plot(valid_taus, med_line, color=color, label=label, lw=1.5)
            ax.fill_between(valid_taus, q1_line, q3_line, color=color, alpha=0.15)

        ax.axvline(0, color="#999", ls=":", lw=0.8, label="Event onset")
        ax.axhline(0, color="#ccc", ls="-", lw=0.5)
        ax.set_title(f"Valence: {val}")
        ax.set_xlabel("Steps relative to event")
        if vi == 0:
            ax.set_ylabel("Δ||g|| (baseline-subtracted)")
        ax.legend(loc="best", framealpha=0.7)

    fig.suptitle("Event-triggered g response", fontsize=14, y=1.02)
    savefig(fig, outdir / "event_triggered_g_response.png",
            "Event-triggered g displacement")


def fig_c_state_trajectory(traj_data: pd.DataFrame, outdir: Path) -> None:
    """c_t trajectory within a typical episode, showing accumulation."""

    if "c_state" not in traj_data.columns or len(traj_data) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for vi, val in enumerate(["SSSS", "MMMM"]):
        ax = axes[vi]
        val_data = traj_data[traj_data["valence"] == val]
        if len(val_data) == 0:
            continue

        # Just pick alpha=0.25 (moderate) and show c_t across seeds
        alpha = 0.25
        a_data = val_data[val_data["alpha_fixed"] == alpha]

        # Average c_state by timestep across episodes and seeds
        for seed in sorted(a_data["seed"].unique()):
            s_data = a_data[a_data["seed"] == seed]
            c_by_t = s_data.groupby("t")["c_state"].median()
            ax.plot(c_by_t.index, c_by_t.values, alpha=0.3, color="#888", lw=0.5)

        # Median across all
        c_agg = a_data.groupby("t")["c_state"].agg(
            median="median",
            q1=lambda x: x.quantile(0.25),
            q3=lambda x: x.quantile(0.75),
        )
        ax.plot(c_agg.index, c_agg["median"], color="#185FA5", lw=2, label="Median c(t)")
        ax.fill_between(c_agg.index, c_agg["q1"], c_agg["q3"],
                        color="#185FA5", alpha=0.15)

        ax.axhline(0, color="#ccc", ls="-", lw=0.5)
        ax.set_title(f"Valence: {val}")
        ax.set_xlabel("Step within episode")
        if vi == 0:
            ax.set_ylabel("c(t) hidden state")
        ax.legend(loc="best", framealpha=0.7)

    fig.suptitle(f"c(t) accumulation (α={alpha}, c_decay=0.985)", fontsize=14, y=1.02)
    savefig(fig, outdir / "c_state_trajectory.png", "c(t) accumulation")


def fig_g_projection_ssss_vs_mmmm(traj_data: pd.DataFrame, outdir: Path) -> None:
    """Project g onto the SSSS-MMMM discriminant axis (like AAAI g-score)."""

    g_cols = detect_g_cols(traj_data)
    if len(g_cols) == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ai, alpha in enumerate(sorted(traj_data["alpha_fixed"].unique())):
        ax = axes[ai] if len(axes) > ai else axes[-1]
        a_data = traj_data[traj_data["alpha_fixed"] == alpha]

        # Compute discriminant direction from late-episode g means
        late_data = a_data[a_data["t"] > 200]
        g_ssss = late_data[late_data["valence"] == "SSSS"][g_cols].values
        g_mmmm = late_data[late_data["valence"] == "MMMM"][g_cols].values

        if len(g_ssss) < 10 or len(g_mmmm) < 10:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center")
            continue

        u = g_ssss.mean(axis=0) - g_mmmm.mean(axis=0)
        u_norm = np.linalg.norm(u)
        if u_norm < 1e-8:
            ax.text(0.5, 0.5, "No discriminant direction", transform=ax.transAxes, ha="center")
            continue
        u_hat = u / u_norm

        # Project all g onto u_hat, aggregate by t
        for val in ["SSSS", "MMMM"]:
            v_data = a_data[a_data["valence"] == val]
            g_all = v_data[g_cols].values
            proj = g_all @ u_hat

            v_data_copy = v_data[["t", "seed"]].copy()
            v_data_copy["proj"] = proj

            # Per-seed median, then across seeds
            seed_curves = {}
            for seed in v_data_copy["seed"].unique():
                s_proj = v_data_copy[v_data_copy["seed"] == seed]
                curve = s_proj.groupby("t")["proj"].median()
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

            style = VALENCE_STYLES[val]
            color = "#185FA5" if val == "SSSS" else "#D85A30"
            ax.plot(all_t, med, color=color, ls=style["ls"], lw=1.5, label=val)
            ax.fill_between(all_t, q1, q3, color=color, alpha=0.12)

        ax.axhline(0, color="#ccc", ls="-", lw=0.5)
        ax.set_title(ALPHA_LABELS.get(alpha, f"α={alpha}"))
        ax.set_xlabel("Step within episode")
        if ai == 0:
            ax.set_ylabel("g-score (SSSS-MMMM projection)")
        ax.legend(loc="best", framealpha=0.7)

    fig.suptitle("G-score: SSSS vs MMMM discriminant projection", fontsize=14, y=1.02)
    savefig(fig, outdir / "g_projection_ssss_vs_mmmm.png", "G-score projection")


def fig_summary_bars(ep_data: pd.DataFrame, outdir: Path) -> None:
    """Summary bar chart: key metrics by condition (median ± IQR across seeds)."""

    metrics = [
        ("mean_pred_loss", "Prediction loss"),
        ("mean_delta_g",   "Mean Δg"),
        ("mean_energy",    "Energy"),
        ("g_shift",        "G shift (start→end)"),
    ]

    # Aggregate: per (alpha, valence), take median across episodes within each seed,
    # then aggregate across seeds.
    available_metrics = [m for m in metrics if m[0] in ep_data.columns]
    if len(available_metrics) == 0:
        return

    fig, axes = plt.subplots(1, len(available_metrics),
                              figsize=(4 * len(available_metrics), 5))
    if len(available_metrics) == 1:
        axes = [axes]

    alphas = sorted(ep_data["alpha_fixed"].unique())
    valences = ["SSSS", "MMMM"]
    n_alpha = len(alphas)
    bar_width = 0.35
    x = np.arange(n_alpha)

    for mi, (metric_col, metric_label) in enumerate(available_metrics):
        ax = axes[mi]

        for vi, val in enumerate(valences):
            medians, errs_lo, errs_hi = [], [], []
            for alpha in alphas:
                subset = ep_data[
                    (ep_data["alpha_fixed"] == alpha) &
                    (ep_data["valence"] == val)
                ]
                # Per-seed mean, then cross-seed median±IQR
                seed_means = subset.groupby("seed")[metric_col].mean()
                med = seed_means.median()
                q1 = seed_means.quantile(0.25)
                q3 = seed_means.quantile(0.75)
                medians.append(med)
                errs_lo.append(med - q1)
                errs_hi.append(q3 - med)

            offset = (vi - 0.5) * bar_width
            color = "#185FA5" if val == "SSSS" else "#D85A30"
            ax.bar(x + offset, medians, bar_width,
                   yerr=[errs_lo, errs_hi],
                   color=color, alpha=0.7, label=val,
                   capsize=3, error_kw={"lw": 1})

        ax.set_xticks(x)
        ax.set_xticklabels([f"α={a}" for a in alphas])
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        if mi == 0:
            ax.legend(framealpha=0.7)

    fig.suptitle("Summary: median ± IQR across seeds", fontsize=14, y=1.02)
    savefig(fig, outdir / "summary_bars.png", "Summary bars")


def fig_basin_occupancy(traj_data: pd.DataFrame, outdir: Path) -> None:
    """Basin occupancy (fraction of time in each basin) by condition."""

    if "basin_id" not in traj_data.columns:
        return

    n_basins = traj_data["basin_id"].max() + 1
    alphas = sorted(traj_data["alpha_fixed"].unique())
    valences = ["SSSS", "MMMM"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for vi, val in enumerate(valences):
        ax = axes[vi]
        val_data = traj_data[traj_data["valence"] == val]

        for alpha in alphas:
            a_data = val_data[val_data["alpha_fixed"] == alpha]

            # Per-seed basin fraction, then median across seeds
            fracs = []
            for seed in a_data["seed"].unique():
                s_data = a_data[a_data["seed"] == seed]
                total = len(s_data)
                if total == 0:
                    continue
                frac = [(s_data["basin_id"] == b).sum() / total for b in range(n_basins)]
                fracs.append(frac)

            if len(fracs) == 0:
                continue

            fracs = np.array(fracs)
            med = np.median(fracs, axis=0)
            color = ALPHA_COLORS.get(alpha, "#888")
            label = ALPHA_LABELS.get(alpha, f"α={alpha}")
            ax.bar(
                np.arange(n_basins) + list(alphas).index(alpha) * 0.25,
                med, 0.22, color=color, alpha=0.7, label=label,
            )

        ax.set_xticks(np.arange(n_basins) + 0.25)
        ax.set_xticklabels([f"Basin {b}" for b in range(n_basins)])
        ax.set_ylabel("Occupancy fraction")
        ax.set_title(f"Valence: {val}")
        ax.legend(framealpha=0.7)

    fig.suptitle("Basin occupancy by condition", fontsize=14, y=1.02)
    savefig(fig, outdir / "basin_occupancy.png", "Basin occupancy")


def fig_delta_g_distribution(traj_data: pd.DataFrame, outdir: Path) -> None:
    """Distribution of per-step Δg by condition."""

    if "delta_g" not in traj_data.columns:
        g_cols = detect_g_cols(traj_data)
        if len(g_cols) == 0:
            return
        g_arr = traj_data[g_cols].values
        dg = np.linalg.norm(np.diff(g_arr, axis=0), axis=1)
        # This doesn't handle episode boundaries correctly, so skip
        print("  [skip] delta_g distribution — compute delta_g in collect")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for vi, val in enumerate(["SSSS", "MMMM"]):
        ax = axes[vi]
        val_data = traj_data[traj_data["valence"] == val]

        for alpha in sorted(val_data["alpha_fixed"].unique()):
            a_data = val_data[val_data["alpha_fixed"] == alpha]
            dg = a_data["delta_g"].dropna().values
            dg = dg[dg > 1e-8]  # skip zero (first step)
            if len(dg) == 0:
                continue
            color = ALPHA_COLORS.get(alpha, "#888")
            label = ALPHA_LABELS.get(alpha, f"α={alpha}")
            ax.hist(dg, bins=60, color=color, alpha=0.5, label=label, density=True)

        ax.set_title(f"Valence: {val}")
        ax.set_xlabel("Δg per step")
        if vi == 0:
            ax.set_ylabel("Density")
        ax.legend(framealpha=0.7)

    fig.suptitle("Distribution of per-step g displacement", fontsize=14, y=1.02)
    savefig(fig, outdir / "delta_g_distribution.png", "Δg distribution")


def write_summary_table(ep_data: pd.DataFrame, outdir: Path) -> None:
    """Write a summary CSV with per-condition aggregated stats."""

    metrics = [
        "mean_pred_loss", "mean_delta_g", "mean_energy",
        "mean_entropy", "mean_alpha", "g_shift", "switches",
    ]
    available = [m for m in metrics if m in ep_data.columns]

    rows = []
    for alpha in sorted(ep_data["alpha_fixed"].unique()):
        for val in ["SSSS", "MMMM"]:
            subset = ep_data[
                (ep_data["alpha_fixed"] == alpha) &
                (ep_data["valence"] == val)
            ]
            seed_means = subset.groupby("seed")[available].mean()

            row = {"alpha": alpha, "valence": val, "n_seeds": len(seed_means)}
            for m in available:
                row[f"{m}_median"] = seed_means[m].median()
                row[f"{m}_q1"] = seed_means[m].quantile(0.25)
                row[f"{m}_q3"] = seed_means[m].quantile(0.75)
            rows.append(row)

    df = pd.DataFrame(rows)
    path = outdir / "summary_table.csv"
    df.to_csv(path, index=False)
    print(f"  [table] {path}")

    # Print to console
    print("\n  === Summary (median across seeds) ===")
    for _, r in df.iterrows():
        line = f"  α={r['alpha']:.2f} {r['valence']}: "
        for m in available[:4]:
            line += f"{m.replace('mean_','')}={r[f'{m}_median']:.4f}  "
        print(line)


# ── Main ────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Alpha sweep comparison analysis")
    ap.add_argument("--sweep_root", type=str, required=True,
                    help="Root directory of the alpha sweep experiment")
    ap.add_argument("--outdir", type=str, default="",
                    help="Output directory for figures")
    ap.add_argument("--source", type=str, default="runs",
                    choices=["runs", "collect"],
                    help="Use training or collection trajectories")
    args = ap.parse_args()

    sweep_root = Path(args.sweep_root)
    outdir = Path(args.outdir) if args.outdir else sweep_root / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Sweep root: {sweep_root}")
    print(f"Output dir: {outdir}")

    # Discover runs
    run_index = discover_runs(sweep_root)
    print(f"\nDiscovered {len(run_index)} run entries:")
    for (a, v), grp in run_index.groupby(["alpha", "valence"]):
        seeds = sorted(grp["seed"].unique())
        sources = sorted(grp["source"].unique())
        print(f"  α={a:.2f} {v}: seeds={seeds} sources={sources}")

    # Load episode data
    print("\nLoading episode summaries...")
    ep_data = load_all_episode_data(run_index)
    print(f"  {len(ep_data)} episode rows loaded")

    # Load trajectory data
    print(f"\nLoading trajectory data (source={args.source})...")
    traj_data = load_all_traj_data(run_index, source=args.source)
    print(f"  {len(traj_data)} trajectory rows loaded")

    # Generate figures
    print(f"\nGenerating figures in {outdir}...")

    fig_episode_training_curves(ep_data, outdir)
    fig_summary_bars(ep_data, outdir)
    write_summary_table(ep_data, outdir)

    if len(traj_data) > 0:
        fig_event_triggered_g_response(traj_data, outdir)
        fig_g_projection_ssss_vs_mmmm(traj_data, outdir)
        fig_c_state_trajectory(traj_data, outdir)
        fig_basin_occupancy(traj_data, outdir)
        fig_delta_g_distribution(traj_data, outdir)

    print(f"\nDone. All outputs in: {outdir}")


if __name__ == "__main__":
    main()
