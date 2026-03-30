# cear_pilot/analysis/compare_alpha_sweep.py
# -*- coding: utf-8 -*-
"""
Alpha sweep analysis with hierarchical aggregation.

Revised aggregation:
  1) aggregate across phase1 seeds within each phase2 seed
  2) aggregate across phase2 seeds using median ± IQR

This is meant for runs named like:
  a0.05_SSSS_s0_p1s2
or timestamped variants like:
  a0.05_SSSS_s0_p1s2_20260330_123456

Usage:
    python -m cear_pilot.analysis.compare_alpha_sweep \
        --sweep_root outputs/alpha_sweep_v2_full_YYYYMMDD_HHMMSS \
        --outdir outputs/alpha_sweep_v2_full_YYYYMMDD_HHMMSS/analysis \
        --source runs
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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

VALENCE_STYLES = {
    "SSSS": {"ls": "-", "marker": "o"},
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
    """
    Parse run labels like:
      a0.05_SSSS_s0
      a0.05_SSSS_s0_p1s2
      a0.05_SSSS_s0_p1s2_20260330_123456
    """
    m = re.match(r"a([\d.]+)_(SSSS|MMMM)_s(\d+)", name)
    if not m:
        return None

    p1m = re.search(r"p1s(\d+)", name)
    return {
        "alpha": float(m.group(1)),
        "valence": m.group(2),
        "seed": int(m.group(3)),                     # phase2 seed
        "p1_seed": int(p1m.group(1)) if p1m else 0  # phase1 seed
    }


def discover_runs(sweep_root: Path) -> pd.DataFrame:
    """Find all runs under sweep_root/runs/ and sweep_root/collect/."""
    records: List[Dict[str, Any]] = []

    for subdir_name in ("runs", "collect"):
        base = sweep_root / subdir_name
        if not base.exists():
            continue

        for run_dir in sorted(base.iterdir()):
            if not run_dir.is_dir():
                continue

            parsed = parse_run_label(run_dir.name)
            resolved_dir = run_dir

            if parsed is None:
                for sub in sorted(run_dir.iterdir()):
                    if sub.is_dir():
                        parsed = parse_run_label(sub.name) or parse_run_label(run_dir.name)
                        if parsed is not None:
                            resolved_dir = sub
                            break
                if parsed is None:
                    continue

            ep_path = find_file(resolved_dir, "episode_summary")
            traj_path = find_file(resolved_dir, "traj")

            records.append({
                "alpha": parsed["alpha"],
                "valence": parsed["valence"],
                "seed": parsed["seed"],
                "p1_seed": parsed["p1_seed"],
                "source": subdir_name,
                "run_dir": str(resolved_dir),
                "ep_path": str(ep_path) if ep_path else None,
                "traj_path": str(traj_path) if traj_path else None,
            })

    df = pd.DataFrame(records)
    if len(df) == 0:
        raise FileNotFoundError(f"No valid runs found under {sweep_root}")
    return df


def load_all_episode_data(run_index: pd.DataFrame) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for _, row in run_index.iterrows():
        if row["ep_path"] is None:
            continue
        ep = load_table(Path(row["ep_path"]))
        ep["alpha_fixed"] = row["alpha"]
        ep["valence"] = row["valence"]
        ep["seed"] = row["seed"]          # p2 seed
        ep["p1_seed"] = row["p1_seed"]    # p1 seed
        ep["source"] = row["source"]
        frames.append(ep)

    if len(frames) == 0:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_all_traj_data(run_index: pd.DataFrame, source: str = "runs") -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    subset = run_index[run_index["source"] == source]

    for _, row in subset.iterrows():
        if row["traj_path"] is None:
            continue
        tr = load_table(Path(row["traj_path"]))
        tr["alpha_fixed"] = row["alpha"]
        tr["valence"] = row["valence"]
        tr["seed"] = row["seed"]          # p2 seed
        tr["p1_seed"] = row["p1_seed"]    # p1 seed
        frames.append(tr)

    if len(frames) == 0:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def detect_g_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if re.match(r"g_\d+$", c)]
    cols.sort(key=lambda x: int(x.split("_")[-1]))
    return cols


def savefig(fig: plt.Figure, path: Path, title: str = "") -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {path.name}" + (f" — {title}" if title else ""))


# ── Hierarchical aggregation helpers ────────────────────

def hierarchical_scalar_agg(
    df: pd.DataFrame,
    group_cols: List[str],
    value_col: str,
) -> pd.DataFrame:
    """
    Revised hierarchy:
      1) average within each (group_cols + p1_seed + seed)
      2) average across p1 seeds within each p2 seed
      3) median ± IQR across p2 seeds
    """
    req = set(group_cols + ["p1_seed", "seed", value_col])
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for hierarchical scalar aggregation: {missing}")

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


def hierarchical_curve_agg(
    df: pd.DataFrame,
    curve_group_cols: List[str],
    x_col: str,
    value_col: str,
) -> pd.DataFrame:
    """
    Curve version of revised hierarchy:
      1) mean within each (curve_group_cols + x_col + p1_seed + seed)
      2) mean across p1 seeds within each p2 seed
      3) median ± IQR across p2 seeds
    """
    req = set(curve_group_cols + [x_col, "p1_seed", "seed", value_col])
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for hierarchical curve aggregation: {missing}")

    lvl0 = (
        df.groupby(curve_group_cols + [x_col, "seed", "p1_seed"])[value_col]
          .mean()
          .reset_index()
    )

    lvl1 = (
        lvl0.groupby(curve_group_cols + [x_col, "seed"])[value_col]
            .mean()
            .reset_index()
            .rename(columns={value_col: "p2_mean"})
    )

    out = (
        lvl1.groupby(curve_group_cols + [x_col])["p2_mean"]
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


def hierarchical_pair_curves(
    curve_dicts: List[Dict[int, float]]
) -> Dict[int, Dict[str, float]]:
    """
    Aggregate a list of p2-level curves where p1 seeds have already been merged inside p2.
    """
    if len(curve_dicts) == 0:
        return {}

    xs = sorted(set().union(*[d.keys() for d in curve_dicts]))
    out: Dict[int, Dict[str, float]] = {}
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


# ── Figure functions ────────────────────────────────────

def fig_episode_training_curves(ep_data: pd.DataFrame, outdir: Path) -> None:
    """Training curves by alpha, panel per valence."""
    if len(ep_data) == 0:
        return

    metrics = [
        ("mean_pred_loss", "Prediction loss"),
        ("mean_delta_g", "Mean Δg per step"),
        ("mean_energy", "Energy"),
        ("mean_entropy", "Policy entropy"),
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
                agg = hierarchical_curve_agg(
                    a_sub,
                    curve_group_cols=[],
                    x_col="episode",
                    value_col=metric_col,
                ).sort_values("episode")

                if len(agg) == 0:
                    continue

                color = ALPHA_COLORS.get(alpha, "#888")
                label = ALPHA_LABELS.get(alpha, f"α={alpha}")

                ax.plot(
                    agg["episode"],
                    agg[f"{metric_col}_median"],
                    color=color,
                    label=label,
                    lw=1.5,
                )
                ax.fill_between(
                    agg["episode"],
                    agg[f"{metric_col}_q1"],
                    agg[f"{metric_col}_q3"],
                    color=color,
                    alpha=0.15,
                )

            ax.set_title(f"Valence: {val}")
            ax.set_xlabel("Episode")
            if vi == 0:
                ax.set_ylabel(metric_label)
            ax.legend(loc="best", framealpha=0.7)

        fig.suptitle(metric_label, fontsize=14, y=1.02)
        savefig(fig, outdir / f"train_{metric_col}.png", metric_label)


def fig_event_triggered_g_response(traj_data: pd.DataFrame, outdir: Path) -> None:
    """Event-triggered Δ||g|| around event timesteps with p1->p2 hierarchy."""
    if "event_now" not in traj_data.columns or len(traj_data) == 0:
        print("  [skip] event_triggered_g — missing event_now column")
        return

    g_cols = detect_g_cols(traj_data)
    if len(g_cols) == 0:
        print("  [skip] event_triggered_g — no g columns")
        return

    window = 30
    tau_range = range(-window, window + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for vi, val in enumerate(["SSSS", "MMMM"]):
        ax = axes[vi]
        val_data = traj_data[traj_data["valence"] == val]
        if len(val_data) == 0:
            continue

        for alpha in sorted(val_data["alpha_fixed"].unique()):
            a_data = val_data[val_data["alpha_fixed"] == alpha]

            p2_curves: List[Dict[int, float]] = []

            for seed in sorted(a_data["seed"].unique()):
                s2_data = a_data[a_data["seed"] == seed]
                p1_seed_curves: List[Dict[int, float]] = []

                for p1_seed in sorted(s2_data["p1_seed"].unique()):
                    s_data = s2_data[s2_data["p1_seed"] == p1_seed].copy()
                    if len(s_data) == 0 or (s_data["event_now"] == 1).sum() == 0:
                        continue

                    snippets = []
                    for ep in sorted(s_data["episode"].unique()):
                        ep_data_local = s_data[s_data["episode"] == ep].copy().reset_index(drop=True)
                        ep_events = ep_data_local[ep_data_local["event_now"] == 1]
                        if len(ep_events) == 0:
                            continue

                        ep_g_norm = np.linalg.norm(ep_data_local[g_cols].values, axis=1)

                        for evt_pos in ep_events["t"].values:
                            snippet: Dict[int, float] = {}
                            for tau in tau_range:
                                t_abs = evt_pos + tau
                                mask = ep_data_local["t"] == t_abs
                                if mask.any():
                                    idx_in_ep = np.where(mask.values)[0][0]
                                    if idx_in_ep < len(ep_g_norm):
                                        snippet[tau] = float(ep_g_norm[idx_in_ep])

                            if len(snippet) > window:
                                pre_vals = [snippet[t] for t in range(-window, -2) if t in snippet]
                                if len(pre_vals) > 0:
                                    baseline = float(np.median(pre_vals))
                                    snippets.append({t: v - baseline for t, v in snippet.items()})

                    if len(snippets) == 0:
                        continue

                    all_taus = sorted(set().union(*[s.keys() for s in snippets]))
                    p1_curve: Dict[int, float] = {}
                    for tau in all_taus:
                        vals = [s[tau] for s in snippets if tau in s]
                        if len(vals) >= 2:
                            p1_curve[tau] = float(np.median(vals))

                    if len(p1_curve) > 0:
                        p1_seed_curves.append(p1_curve)

                if len(p1_seed_curves) == 0:
                    continue

                p2_curve: Dict[int, float] = {}
                all_taus = sorted(set().union(*[d.keys() for d in p1_seed_curves]))
                for tau in all_taus:
                    vals = [d[tau] for d in p1_seed_curves if tau in d]
                    if len(vals) >= 1:
                        p2_curve[tau] = float(np.mean(vals))

                if len(p2_curve) > 0:
                    p2_curves.append(p2_curve)

            if len(p2_curves) == 0:
                continue

            agg = hierarchical_pair_curves(p2_curves)
            if len(agg) == 0:
                continue

            xs = sorted(agg.keys())
            med = [agg[x]["median"] for x in xs]
            q1 = [agg[x]["q1"] for x in xs]
            q3 = [agg[x]["q3"] for x in xs]

            color = ALPHA_COLORS.get(alpha, "#888")
            label = ALPHA_LABELS.get(alpha, f"α={alpha}")

            ax.plot(xs, med, color=color, label=label, lw=1.5)
            ax.fill_between(xs, q1, q3, color=color, alpha=0.15)

        ax.axvline(0, color="#999", ls=":", lw=0.8, label="Event onset")
        ax.axhline(0, color="#ccc", ls="-", lw=0.5)
        ax.set_title(f"Valence: {val}")
        ax.set_xlabel("Steps relative to event")
        if vi == 0:
            ax.set_ylabel("Δ||g|| (baseline-subtracted)")
        ax.legend(loc="best", framealpha=0.7)

    fig.suptitle("Event-triggered Δ||g||", fontsize=14, y=1.02)
    savefig(fig, outdir / "event_triggered_g.png", "Event-triggered g response")


def fig_summary_bars(ep_data: pd.DataFrame, outdir: Path) -> None:
    """Bar summary over alpha for key metrics."""
    if len(ep_data) == 0:
        return

    metrics = [
        ("mean_pred_loss", "Prediction loss"),
        ("mean_delta_g", "Mean Δg"),
        ("mean_energy", "Energy"),
        ("mean_entropy", "Entropy"),
    ]
    metrics = [(m, lab) for m, lab in metrics if m in ep_data.columns]
    if len(metrics) == 0:
        return

    alphas = sorted(ep_data["alpha_fixed"].unique())
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    x = np.arange(len(alphas))
    width = 0.36

    for mi, (metric_col, metric_label) in enumerate(metrics):
        ax = axes[mi]

        for vi, val in enumerate(["SSSS", "MMMM"]):
            meds, q1s, q3s = [], [], []

            for alpha in alphas:
                subset = ep_data[
                    (ep_data["alpha_fixed"] == alpha) &
                    (ep_data["valence"] == val)
                ]
                agg = hierarchical_scalar_agg(
                    subset,
                    group_cols=["alpha_fixed", "valence"],
                    value_col=metric_col,
                )
                if len(agg) == 0:
                    meds.append(np.nan)
                    q1s.append(np.nan)
                    q3s.append(np.nan)
                else:
                    rr = agg.iloc[0]
                    meds.append(rr[f"{metric_col}_median"])
                    q1s.append(rr[f"{metric_col}_q1"])
                    q3s.append(rr[f"{metric_col}_q3"])

            offset = (-0.5 + vi) * width
            yerr = [np.array(meds) - np.array(q1s), np.array(q3s) - np.array(meds)]
            style = VALENCE_STYLES[val]
            color = "#185FA5" if val == "SSSS" else "#D85A30"

            ax.bar(
                x + offset,
                meds,
                width=width,
                yerr=yerr,
                color=color,
                alpha=0.75,
                label=val,
                capsize=3,
                error_kw={"lw": 1},
            )

        ax.set_xticks(x)
        ax.set_xticklabels([f"{a:.2f}" for a in alphas])
        ax.set_title(metric_label)
        ax.set_xlabel("Alpha")
        if mi == 0:
            ax.legend(framealpha=0.7)

    fig.suptitle("Summary bars", fontsize=14, y=1.02)
    savefig(fig, outdir / "summary_bars.png", "Summary bars")


def write_summary_table(ep_data: pd.DataFrame, outdir: Path) -> None:
    """Write summary CSV with revised hierarchy."""
    if len(ep_data) == 0:
        return

    metrics = [
        "mean_pred_loss",
        "mean_delta_g",
        "mean_energy",
        "mean_entropy",
        "mean_alpha",
        "g_shift",
        "switches",
    ]
    available = [m for m in metrics if m in ep_data.columns]

    rows: List[Dict[str, Any]] = []

    for alpha in sorted(ep_data["alpha_fixed"].unique()):
        for val in ["SSSS", "MMMM"]:
            subset = ep_data[
                (ep_data["alpha_fixed"] == alpha) &
                (ep_data["valence"] == val)
            ]
            if len(subset) == 0:
                continue

            row: Dict[str, Any] = {
                "alpha": alpha,
                "valence": val,
                "n_p1_seeds": int(subset["p1_seed"].nunique()),
                "n_p2_seeds": int(subset["seed"].nunique()),
            }

            for m in available:
                agg = hierarchical_scalar_agg(
                    subset,
                    group_cols=["alpha_fixed", "valence"],
                    value_col=m,
                )
                if len(agg) == 0:
                    continue
                rr = agg.iloc[0]
                row[f"{m}_median"] = float(rr[f"{m}_median"])
                row[f"{m}_q1"] = float(rr[f"{m}_q1"])
                row[f"{m}_q3"] = float(rr[f"{m}_q3"])

            rows.append(row)

    df = pd.DataFrame(rows)
    path = outdir / "summary_table.csv"
    df.to_csv(path, index=False)
    print(f"  [table] {path}")

    print("\n  === Summary (median across p2 means) ===")
    for _, r in df.iterrows():
        line = f"  α={r['alpha']:.2f} {r['valence']}: "
        for m in available[:4]:
            key = f"{m}_median"
            if key in r and pd.notna(r[key]):
                line += f"{m.replace('mean_','')}={r[key]:.4f}  "
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

    run_index = discover_runs(sweep_root)

    print(f"\nDiscovered {len(run_index)} run entries:")
    for (a, v), grp in run_index.groupby(["alpha", "valence"]):
        p2s = sorted(grp["seed"].unique())
        p1s = sorted(grp["p1_seed"].unique())
        sources = sorted(grp["source"].unique())
        print(f"  α={a:.2f} {v}: p2_seeds={p2s} p1_seeds={p1s} sources={sources}")

    print("\nLoading episode summaries...")
    ep_data = load_all_episode_data(run_index)
    print(f"  {len(ep_data)} episode rows loaded")

    print(f"\nLoading trajectory data (source={args.source})...")
    traj_data = load_all_traj_data(run_index, source=args.source)
    print(f"  {len(traj_data)} trajectory rows loaded")

    meta_summary = {
        "timestamp": timestamp_id(),
        "sweep_root": str(sweep_root),
        "outdir": str(outdir),
        "source": args.source,
        "n_run_entries": int(len(run_index)),
        "n_episode_rows": int(len(ep_data)),
        "n_traj_rows": int(len(traj_data)),
        "alphas": sorted([float(x) for x in run_index["alpha"].unique()]) if len(run_index) else [],
        "valences": sorted([str(x) for x in run_index["valence"].unique()]) if len(run_index) else [],
        "p1_seeds": sorted([int(x) for x in run_index["p1_seed"].unique()]) if len(run_index) else [],
        "p2_seeds": sorted([int(x) for x in run_index["seed"].unique()]) if len(run_index) else [],
        "aggregation": "Option 2 revised: p1 within p2, then median±IQR across p2",
    }
    (outdir / "analysis_meta.json").write_text(json.dumps(meta_summary, indent=2))

    print("\nGenerating figures...")
    fig_episode_training_curves(ep_data, outdir)
    fig_event_triggered_g_response(traj_data, outdir)
    fig_summary_bars(ep_data, outdir)
    write_summary_table(ep_data, outdir)

    print(f"\nDone. Analysis outputs in: {outdir}")


if __name__ == "__main__":
    main()