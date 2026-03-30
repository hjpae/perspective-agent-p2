# cear_pilot/analysis/compare_phase2_runs.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SAVE_DPI = 160


def timestamp_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def find_existing(run_dir: Path, stem: str) -> Optional[Path]:
    for p in [run_dir / f"{stem}.parquet", run_dir / f"{stem}.csv"]:
        if p.exists():
            return p
    return None


def savefig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def detect_g_cols(traj: pd.DataFrame) -> List[str]:
    cols = []
    for c in traj.columns:
        if c.startswith("g_"):
            try:
                int(c.split("_")[-1])
                cols.append(c)
            except Exception:
                pass
    cols.sort(key=lambda x: int(x.split("_")[-1]))
    return cols


def add_delta_g(traj: pd.DataFrame) -> pd.DataFrame:
    if "delta_g" in traj.columns:
        return traj

    g_cols = detect_g_cols(traj)
    traj = traj.copy().sort_values(["episode", "t"]).reset_index(drop=True)
    if not g_cols:
        traj["delta_g"] = np.nan
        return traj

    deltas = np.zeros(len(traj), dtype=np.float32)
    for _, idx in traj.groupby("episode").groups.items():
        ix = np.array(list(idx))
        g = traj.loc[ix, g_cols].astype(float).to_numpy()
        d = np.zeros((len(ix),), dtype=np.float32)
        if len(ix) >= 2:
            d[1:] = np.linalg.norm(g[1:] - g[:-1], axis=1)
        deltas[ix] = d

    traj["delta_g"] = deltas
    return traj


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    w = max(int(w), 1)
    if len(x) == 0 or w == 1:
        return x
    return pd.Series(x).rolling(window=w, min_periods=1).mean().to_numpy()


def event_aligned_average(
    traj: pd.DataFrame,
    trigger_col: str,
    value_col: str,
    window_before: int,
    window_after: int,
    baseline_subtract: bool = False,
) -> Optional[pd.DataFrame]:
    if trigger_col not in traj.columns or value_col not in traj.columns:
        return None

    df = traj[["episode", "t", trigger_col, value_col]].copy()
    df["episode"] = df["episode"].astype(int)
    df["t"] = df["t"].astype(int)
    df[trigger_col] = df[trigger_col].astype(float)
    df[value_col] = df[value_col].astype(float)

    chunks = []
    for _, g in df.groupby("episode"):
        g = g.sort_values("t").reset_index(drop=True)
        trigger_idx = g.index[g[trigger_col] > 0].tolist()
        for idx in trigger_idx:
            t0 = int(g.loc[idx, "t"])
            w = g[(g["t"] >= t0 - window_before) & (g["t"] <= t0 + window_after)].copy()
            if len(w) == 0:
                continue
            w["rel_t"] = w["t"] - t0

            if baseline_subtract:
                base = w.loc[w["rel_t"] < 0, value_col]
                baseline = float(base.mean()) if len(base) > 0 else 0.0
                w[value_col] = w[value_col] - baseline

            chunks.append(w[["rel_t", value_col]])

    if not chunks:
        return None

    cat = pd.concat(chunks, axis=0, ignore_index=True)
    out = (
        cat.groupby("rel_t")[value_col]
        .mean()
        .reset_index()
        .sort_values("rel_t")
        .reset_index(drop=True)
    )
    return out


def compute_summary(ep: pd.DataFrame, traj: pd.DataFrame, label: str, meta: dict) -> Dict[str, float]:
    out: Dict[str, float] = {
        "label": label,
        "episodes": int(len(ep)),
        "schedule_pattern": meta.get("env_cfg", {}).get("schedule_pattern"),
        "valence_sequence": meta.get("env_cfg", {}).get("valence_sequence"),
    }

    for c in [
        "mean_alpha",
        "mean_pred_loss",
        "mean_energy",
        "g_shift",
        "mean_delta_g",
        "switches",
        "mean_err_norm",
        "supportive_event_count",
        "misleading_event_count",
        "supportive_consequence_count",
        "misleading_consequence_count",
    ]:
        if c in ep.columns and len(ep) > 0:
            out[f"{c}_overall_mean"] = float(ep[c].astype(float).mean())

    if "alpha" in traj.columns and len(traj) > 0:
        a = traj["alpha"].astype(float)
        out["alpha_min"] = float(a.min())
        out["alpha_max"] = float(a.max())
        out["alpha_mean"] = float(a.mean())
        out["alpha_std"] = float(a.std(ddof=0))
        out["alpha_var_all"] = float(a.var(ddof=0))

    for c in [
        "event_now",
        "consequence_now",
        "is_supportive_event",
        "is_misleading_event",
        "is_supportive_consequence",
        "is_misleading_consequence",
    ]:
        if c in traj.columns:
            out[f"{c}_mean"] = float(traj[c].astype(float).mean())

    return out


def load_run(run_dir: Path, label: str) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    ep_path = find_existing(run_dir, "episode_summary")
    traj_path = find_existing(run_dir, "traj")
    if ep_path is None:
        raise FileNotFoundError(f"Could not find episode_summary in {run_dir}")
    if traj_path is None:
        raise FileNotFoundError(f"Could not find traj in {run_dir}")

    ep = load_table(ep_path).sort_values("episode").reset_index(drop=True)
    traj = load_table(traj_path).sort_values(["episode", "t"]).reset_index(drop=True)
    traj = add_delta_g(traj)

    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["label"] = label
    return ep, traj, meta


def plot_episode_overlay(
    ep_a: pd.DataFrame,
    ep_b: pd.DataFrame,
    label_a: str,
    label_b: str,
    outdir: Path,
) -> None:
    specs = [
        ("mean_alpha", "Episode mean alpha", "fig1_mean_alpha_overlay.png"),
        ("mean_pred_loss", "Episode mean pred loss", "fig2_mean_pred_loss_overlay.png"),
        ("mean_energy", "Episode mean energy", "fig3_mean_energy_overlay.png"),
        ("g_shift", "Episode g_shift", "fig4_g_shift_overlay.png"),
        ("mean_delta_g", "Episode mean delta_g", "fig5_mean_delta_g_overlay.png"),
        ("switches", "Episode basin switches", "fig6_switches_overlay.png"),
    ]

    for col, title, fname in specs:
        if col not in ep_a.columns and col not in ep_b.columns:
            continue
        fig = plt.figure(figsize=(8, 5))
        ax = plt.gca()
        if col in ep_a.columns:
            ax.plot(ep_a["episode"], ep_a[col], marker="o", label=label_a)
        if col in ep_b.columns:
            ax.plot(ep_b["episode"], ep_b[col], marker="s", label=label_b)
        ax.set_xlabel("episode")
        ax.set_ylabel(col)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        savefig(fig, outdir / fname)


def plot_alpha_hist_overlay(
    traj_a: pd.DataFrame,
    traj_b: pd.DataFrame,
    label_a: str,
    label_b: str,
    outdir: Path,
) -> None:
    if "alpha" not in traj_a.columns or "alpha" not in traj_b.columns:
        return
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.hist(traj_a["alpha"].astype(float).to_numpy(), bins=40, alpha=0.55, label=label_a)
    ax.hist(traj_b["alpha"].astype(float).to_numpy(), bins=40, alpha=0.55, label=label_b)
    ax.set_xlabel("alpha")
    ax.set_ylabel("count")
    ax.set_title("Alpha histogram overlay")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, outdir / "fig7_alpha_hist_overlay.png")


def plot_alpha_context_overlay(
    traj_a: pd.DataFrame,
    traj_b: pd.DataFrame,
    label_a: str,
    label_b: str,
    outdir: Path,
    rolling_window: int,
) -> None:
    if "alpha" not in traj_a.columns or "alpha" not in traj_b.columns:
        return

    fig = plt.figure(figsize=(9, 5))
    ax = plt.gca()

    xa = np.arange(len(traj_a))
    xb = np.arange(len(traj_b))
    ya = rolling_mean(traj_a["alpha"].astype(float).to_numpy(), rolling_window)
    yb = rolling_mean(traj_b["alpha"].astype(float).to_numpy(), rolling_window)

    ax.plot(xa, ya, label=f"{label_a} alpha")
    ax.plot(xb, yb, label=f"{label_b} alpha")

    ax.set_xlabel("trajectory index")
    ax.set_ylabel("alpha")
    ax.set_title("Alpha rolling overlay")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, outdir / "fig8_alpha_rolling_overlay.png")


def plot_aligned_overlay(
    traj_a: pd.DataFrame,
    traj_b: pd.DataFrame,
    label_a: str,
    label_b: str,
    outdir: Path,
    window_before: int,
    window_after: int,
    baseline_subtract: bool,
) -> None:
    specs = [
        ("event_now", "alpha", "fig9_event_aligned_alpha_overlay.png", "Event-aligned alpha"),
        ("event_now", "delta_g", "fig10_event_aligned_delta_g_overlay.png", "Event-aligned delta_g"),
        ("consequence_now", "alpha", "fig11_consequence_aligned_alpha_overlay.png", "Consequence-aligned alpha"),
        ("consequence_now", "delta_g", "fig12_consequence_aligned_delta_g_overlay.png", "Consequence-aligned delta_g"),
    ]

    for trig, val, fname, title in specs:
        aa = event_aligned_average(traj_a, trig, val, window_before, window_after, baseline_subtract)
        bb = event_aligned_average(traj_b, trig, val, window_before, window_after, baseline_subtract)
        if aa is None and bb is None:
            continue

        fig = plt.figure(figsize=(8, 5))
        ax = plt.gca()

        if aa is not None:
            ax.plot(aa["rel_t"], aa[val], marker="o", label=label_a)
        if bb is not None:
            ax.plot(bb["rel_t"], bb[val], marker="s", label=label_b)

        ax.axvline(0.0)
        ax.set_xlabel("relative t")
        ax.set_ylabel(val)
        suffix = " (baseline-subtracted)" if baseline_subtract else ""
        ax.set_title(title + suffix)
        ax.legend()
        ax.grid(True, alpha=0.3)
        savefig(fig, outdir / fname)


def _collect_masked_values(traj: pd.DataFrame, value_col: str, mask_col: str) -> np.ndarray:
    if value_col not in traj.columns or mask_col not in traj.columns:
        return np.array([])
    mask = traj[mask_col].astype(int) > 0
    return traj.loc[mask, value_col].astype(float).to_numpy()


def plot_valence_boxplots(
    traj_a: pd.DataFrame,
    traj_b: pd.DataFrame,
    label_a: str,
    label_b: str,
    outdir: Path,
) -> None:
    specs = [
        (
            "alpha",
            [
                ("is_supportive_event", f"{label_a}:alpha@sup_event", traj_a),
                ("is_misleading_event", f"{label_a}:alpha@mis_event", traj_a),
                ("is_supportive_consequence", f"{label_a}:alpha@sup_cons", traj_a),
                ("is_misleading_consequence", f"{label_a}:alpha@mis_cons", traj_a),
                ("is_supportive_event", f"{label_b}:alpha@sup_event", traj_b),
                ("is_misleading_event", f"{label_b}:alpha@mis_event", traj_b),
                ("is_supportive_consequence", f"{label_b}:alpha@sup_cons", traj_b),
                ("is_misleading_consequence", f"{label_b}:alpha@mis_cons", traj_b),
            ],
            "fig13_alpha_valence_boxplots.png",
            "Alpha by valence-specific context",
        ),
        (
            "delta_g",
            [
                ("is_supportive_event", f"{label_a}:dg@sup_event", traj_a),
                ("is_misleading_event", f"{label_a}:dg@mis_event", traj_a),
                ("is_supportive_consequence", f"{label_a}:dg@sup_cons", traj_a),
                ("is_misleading_consequence", f"{label_a}:dg@mis_cons", traj_a),
                ("is_supportive_event", f"{label_b}:dg@sup_event", traj_b),
                ("is_misleading_event", f"{label_b}:dg@mis_event", traj_b),
                ("is_supportive_consequence", f"{label_b}:dg@sup_cons", traj_b),
                ("is_misleading_consequence", f"{label_b}:dg@mis_cons", traj_b),
            ],
            "fig14_delta_g_valence_boxplots.png",
            "delta_g by valence-specific context",
        ),
    ]

    for value_col, mask_specs, fname, title in specs:
        groups = []
        labels = []
        for mask_col, label, traj in mask_specs:
            arr = _collect_masked_values(traj, value_col, mask_col)
            if len(arr) > 0:
                groups.append(arr)
                labels.append(label)
        if not groups:
            continue

        fig = plt.figure(figsize=(12, 5))
        ax = plt.gca()
        ax.boxplot(groups, tick_labels=labels, showfliers=False)
        ax.set_ylabel(value_col)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        savefig(fig, outdir / fname)


def plot_variance_compare(
    traj_a: pd.DataFrame,
    traj_b: pd.DataFrame,
    label_a: str,
    label_b: str,
    outdir: Path,
) -> Dict[str, float]:
    def stats_for(traj: pd.DataFrame, prefix: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if "alpha" not in traj.columns:
            return out
        alpha = traj["alpha"].astype(float)
        out[f"{prefix}_alpha_var_all"] = float(alpha.var(ddof=0))
        for c in ["event_now", "consequence_now"]:
            if c in traj.columns:
                mask = traj[c].astype(float) > 0
                if mask.any():
                    key = c.replace("_now", "")
                    out[f"{prefix}_alpha_var_{key}"] = float(alpha[mask].var(ddof=0))
                    out[f"{prefix}_alpha_var_{key}_ratio_vs_all"] = float(
                        out[f"{prefix}_alpha_var_{key}"] / max(out[f"{prefix}_alpha_var_all"], 1e-12)
                    )
        return out

    stats = {}
    stats.update(stats_for(traj_a, "run_a"))
    stats.update(stats_for(traj_b, "run_b"))

    labels = []
    vals_a = []
    vals_b = []

    for k in ["all", "event", "consequence"]:
        ka = f"run_a_alpha_var_{k}"
        kb = f"run_b_alpha_var_{k}"
        if ka in stats and kb in stats:
            labels.append(k)
            vals_a.append(stats[ka])
            vals_b.append(stats[kb])

    if labels:
        x = np.arange(len(labels))
        width = 0.35

        fig = plt.figure(figsize=(8, 5))
        ax = plt.gca()
        ax.bar(x - width / 2, vals_a, width=width, label=label_a)
        ax.bar(x + width / 2, vals_b, width=width, label=label_b)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("alpha variance")
        ax.set_title("Alpha variance: all vs event/consequence")
        ax.legend()
        ax.grid(True, alpha=0.3)
        savefig(fig, outdir / "fig15_alpha_variance_compare.png")

    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_a", type=str, required=True)
    ap.add_argument("--run_b", type=str, required=True)
    ap.add_argument("--label_a", type=str, default="run_a")
    ap.add_argument("--label_b", type=str, default="run_b")
    ap.add_argument("--window_before", type=int, default=10)
    ap.add_argument("--window_after", type=int, default=16)
    ap.add_argument("--rolling_window", type=int, default=25)
    ap.add_argument("--baseline_subtract", action="store_true")
    ap.add_argument("--outdir", type=str, default="")
    args = ap.parse_args()

    run_a = Path(args.run_a)
    run_b = Path(args.run_b)

    ep_a, traj_a, meta_a = load_run(run_a, args.label_a)
    ep_b, traj_b, meta_b = load_run(run_b, args.label_b)

    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = Path("outputs") / f"compare_phase2_{args.label_a}_vs_{args.label_b}_{timestamp_id()}"
    outdir.mkdir(parents=True, exist_ok=True)

    plot_episode_overlay(ep_a, ep_b, args.label_a, args.label_b, outdir)
    plot_alpha_hist_overlay(traj_a, traj_b, args.label_a, args.label_b, outdir)
    plot_alpha_context_overlay(traj_a, traj_b, args.label_a, args.label_b, outdir, args.rolling_window)
    plot_aligned_overlay(
        traj_a,
        traj_b,
        args.label_a,
        args.label_b,
        outdir,
        args.window_before,
        args.window_after,
        args.baseline_subtract,
    )
    plot_valence_boxplots(traj_a, traj_b, args.label_a, args.label_b, outdir)
    var_stats = plot_variance_compare(traj_a, traj_b, args.label_a, args.label_b, outdir)

    summary = {
        "run_a": str(run_a),
        "run_b": str(run_b),
        "label_a": args.label_a,
        "label_b": args.label_b,
        "meta_a_valence_sequence": meta_a.get("env_cfg", {}).get("valence_sequence"),
        "meta_b_valence_sequence": meta_b.get("env_cfg", {}).get("valence_sequence"),
        "summary_a": compute_summary(ep_a, traj_a, args.label_a, meta_a),
        "summary_b": compute_summary(ep_b, traj_b, args.label_b, meta_b),
    }
    summary.update(var_stats)

    (outdir / "compare_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved merged comparison figures to: {outdir}")


if __name__ == "__main__":
    main()