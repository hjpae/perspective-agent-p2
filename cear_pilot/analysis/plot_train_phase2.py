# cear_pilot/analysis/plot_train_phase2.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SAVE_DPI = 160


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
    for ep, idx in traj.groupby("episode").groups.items():
        ix = np.array(list(idx))
        g = traj.loc[ix, g_cols].astype(float).to_numpy()
        d = np.zeros((len(ix),), dtype=np.float32)
        if len(ix) >= 2:
            d[1:] = np.linalg.norm(g[1:] - g[:-1], axis=1)
        deltas[ix] = d

    traj["delta_g"] = deltas
    return traj


def event_aligned_average(
    traj: pd.DataFrame,
    trigger_col: str,
    value_col: str,
    window_before: int,
    window_after: int,
) -> Optional[pd.DataFrame]:
    if trigger_col not in traj.columns or value_col not in traj.columns:
        return None

    df = traj[["episode", "t", trigger_col, value_col]].copy()
    df["episode"] = df["episode"].astype(int)
    df["t"] = df["t"].astype(int)
    df[trigger_col] = df[trigger_col].astype(float)
    df[value_col] = df[value_col].astype(float)

    chunks = []
    for ep, g in df.groupby("episode"):
        g = g.sort_values("t").reset_index(drop=True)
        trigger_idx = g.index[g[trigger_col] > 0].tolist()
        for idx in trigger_idx:
            t0 = int(g.loc[idx, "t"])
            w = g[(g["t"] >= t0 - window_before) & (g["t"] <= t0 + window_after)].copy()
            if len(w) == 0:
                continue
            w["rel_t"] = w["t"] - t0
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


def safe_plot(ax, df: pd.DataFrame, xcol: str, ycol: str, label: str, marker: str) -> bool:
    if xcol in df.columns and ycol in df.columns:
        ax.plot(df[xcol], df[ycol], marker=marker, label=label)
        return True
    return False


def plot_episode_overview(ep: pd.DataFrame, outdir: Path) -> None:
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()
    used = False
    used |= safe_plot(ax, ep, "episode", "mean_pred_loss", "mean_pred_loss", "o")
    used |= safe_plot(ax, ep, "episode", "mean_energy", "mean_energy", "s")
    used |= safe_plot(ax, ep, "episode", "mean_alpha", "mean_alpha", "^")
    ax.set_xlabel("episode")
    ax.set_ylabel("value")
    ax.set_title("Phase 2: prediction / energy / alpha")
    if used:
        ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, outdir / "fig1_episode_overview.png")


def plot_latent_overview(ep: pd.DataFrame, outdir: Path) -> None:
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()
    used = False
    used |= safe_plot(ax, ep, "episode", "g_shift", "g_shift", "o")
    used |= safe_plot(ax, ep, "episode", "mean_delta_g", "mean_delta_g", "s")
    used |= safe_plot(ax, ep, "episode", "switches", "switches", "^")
    ax.set_xlabel("episode")
    ax.set_ylabel("value")
    ax.set_title("Phase 2: latent movement / basin switching")
    if used:
        ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, outdir / "fig2_latent_movement.png")


def plot_valence_counts(ep: pd.DataFrame, outdir: Path) -> None:
    wanted = [
        "supportive_event_count",
        "misleading_event_count",
        "supportive_consequence_count",
        "misleading_consequence_count",
    ]
    cols = [c for c in wanted if c in ep.columns]
    if not cols:
        return

    fig = plt.figure(figsize=(9, 5))
    ax = plt.gca()
    for c in cols:
        ax.plot(ep["episode"], ep[c], marker="o", label=c)
    ax.set_xlabel("episode")
    ax.set_ylabel("count")
    ax.set_title("Valence counts per episode")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, outdir / "fig3_valence_counts.png")


def plot_alpha_hist(traj: pd.DataFrame, outdir: Path) -> None:
    if "alpha" not in traj.columns:
        return
    fig = plt.figure(figsize=(7, 5))
    ax = plt.gca()
    ax.hist(traj["alpha"].astype(float).to_numpy(), bins=40)
    ax.set_xlabel("alpha")
    ax.set_ylabel("count")
    ax.set_title("Trajectory-level alpha histogram")
    ax.grid(True, alpha=0.3)
    savefig(fig, outdir / "fig4_alpha_histogram.png")


def plot_alpha_by_basic_context(traj: pd.DataFrame, outdir: Path) -> None:
    if "alpha" not in traj.columns:
        return
    alpha = traj["alpha"].astype(float)

    labels = ["all"]
    values = [alpha.to_numpy()]

    for name in ["event_now", "consequence_now"]:
        if name in traj.columns:
            mask = traj[name].astype(float) > 0
            arr = alpha[mask].to_numpy()
            if len(arr) > 0:
                labels.append(name)
                values.append(arr)

    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.boxplot(values, tick_labels=labels, showfliers=False)
    ax.set_xlabel("context")
    ax.set_ylabel("alpha")
    ax.set_title("Alpha by basic context")
    ax.grid(True, alpha=0.3)
    savefig(fig, outdir / "fig5_alpha_by_context.png")


def plot_alpha_by_valence(traj: pd.DataFrame, outdir: Path) -> None:
    if "alpha" not in traj.columns:
        return
    alpha = traj["alpha"].astype(float)

    groups = []
    labels = []

    def add_mask(label: str, mask: pd.Series):
        arr = alpha[mask].to_numpy()
        if len(arr) > 0:
            labels.append(label)
            groups.append(arr)

    if "is_supportive_event" in traj.columns:
        add_mask("alpha@supportive_event", traj["is_supportive_event"].astype(int) > 0)
    if "is_misleading_event" in traj.columns:
        add_mask("alpha@misleading_event", traj["is_misleading_event"].astype(int) > 0)
    if "is_supportive_consequence" in traj.columns:
        add_mask("alpha@supportive_cons", traj["is_supportive_consequence"].astype(int) > 0)
    if "is_misleading_consequence" in traj.columns:
        add_mask("alpha@misleading_cons", traj["is_misleading_consequence"].astype(int) > 0)

    if not groups:
        return

    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.boxplot(groups, tick_labels=labels, showfliers=False)
    ax.set_ylabel("alpha")
    ax.set_title("Alpha by valence-specific context")
    ax.grid(True, alpha=0.3)
    savefig(fig, outdir / "fig6_alpha_by_valence.png")


def plot_delta_g_by_valence(traj: pd.DataFrame, outdir: Path) -> None:
    if "delta_g" not in traj.columns:
        return
    vals = traj["delta_g"].astype(float)

    groups = []
    labels = []

    def add_mask(label: str, mask: pd.Series):
        arr = vals[mask].to_numpy()
        if len(arr) > 0:
            labels.append(label)
            groups.append(arr)

    if "is_supportive_event" in traj.columns:
        add_mask("delta_g@supportive_event", traj["is_supportive_event"].astype(int) > 0)
    if "is_misleading_event" in traj.columns:
        add_mask("delta_g@misleading_event", traj["is_misleading_event"].astype(int) > 0)
    if "is_supportive_consequence" in traj.columns:
        add_mask("delta_g@supportive_cons", traj["is_supportive_consequence"].astype(int) > 0)
    if "is_misleading_consequence" in traj.columns:
        add_mask("delta_g@misleading_cons", traj["is_misleading_consequence"].astype(int) > 0)

    if not groups:
        return

    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.boxplot(groups, tick_labels=labels, showfliers=False)
    ax.set_ylabel("delta_g")
    ax.set_title("delta_g by valence-specific context")
    ax.grid(True, alpha=0.3)
    savefig(fig, outdir / "fig7_delta_g_by_valence.png")


def plot_aligned(traj: pd.DataFrame, outdir: Path, window_before: int, window_after: int) -> None:
    specs = [
        ("event_now", "alpha", "fig8_event_aligned_alpha.png", "Event-aligned alpha"),
        ("event_now", "delta_g", "fig9_event_aligned_delta_g.png", "Event-aligned delta_g"),
        ("consequence_now", "alpha", "fig10_consequence_aligned_alpha.png", "Consequence-aligned alpha"),
        ("consequence_now", "delta_g", "fig11_consequence_aligned_delta_g.png", "Consequence-aligned delta_g"),
    ]
    for trig, val, fname, title in specs:
        aligned = event_aligned_average(traj, trig, val, window_before, window_after)
        if aligned is None or len(aligned) == 0:
            continue
        fig = plt.figure(figsize=(8, 5))
        ax = plt.gca()
        ax.plot(aligned["rel_t"], aligned[val], marker="o")
        ax.axvline(0.0)
        ax.set_xlabel("relative t")
        ax.set_ylabel(val)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        savefig(fig, outdir / fname)


def compute_variance_summary(traj: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if "alpha" not in traj.columns:
        return out

    alpha = traj["alpha"].astype(float)
    out["alpha_var_all"] = float(alpha.var(ddof=0))

    for c in ["event_now", "consequence_now"]:
        if c in traj.columns:
            mask = traj[c].astype(float) > 0
            if mask.any():
                key = c.replace("_now", "")
                out[f"alpha_var_{key}"] = float(alpha[mask].var(ddof=0))
                out[f"alpha_var_{key}_ratio_vs_all"] = float(out[f"alpha_var_{key}"] / max(out["alpha_var_all"], 1e-12))

    return out


def plot_variance_summary(traj: pd.DataFrame, outdir: Path) -> Dict[str, float]:
    stats = compute_variance_summary(traj)
    if not stats:
        return stats

    labels = []
    vals = []
    for k in ["alpha_var_all", "alpha_var_event", "alpha_var_consequence"]:
        if k in stats:
            labels.append(k.replace("alpha_var_", ""))
            vals.append(stats[k])

    if vals:
        fig = plt.figure(figsize=(7, 5))
        ax = plt.gca()
        ax.bar(labels, vals)
        ax.set_ylabel("variance")
        ax.set_title("Alpha variance: all vs event/consequence windows")
        ax.grid(True, alpha=0.3)
        savefig(fig, outdir / "fig12_alpha_variance.png")

    return stats


def plot_episode_end(ep: pd.DataFrame, outdir: Path) -> None:
    if "episode" not in ep.columns:
        return
    if "final_x" not in ep.columns and "final_c" not in ep.columns:
        return
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()
    used = False
    used |= safe_plot(ax, ep, "episode", "final_x", "final_x", "o")
    used |= safe_plot(ax, ep, "episode", "final_c", "final_c", "s")
    ax.set_xlabel("episode")
    ax.set_ylabel("value")
    ax.set_title("Episode-end state traces")
    if used:
        ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, outdir / "fig13_episode_end_state.png")


def build_summary(ep: pd.DataFrame, traj: pd.DataFrame, meta: dict, var_stats: Dict[str, float]) -> dict:
    out = {
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

    if "alpha" in traj.columns:
        alpha = traj["alpha"].astype(float)
        out["alpha_min"] = float(alpha.min())
        out["alpha_max"] = float(alpha.max())
        out["alpha_mean"] = float(alpha.mean())
        out["alpha_std"] = float(alpha.std(ddof=0))

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

    out.update(var_stats)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--window_before", type=int, default=10)
    ap.add_argument("--window_after", type=int, default=16)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    outdir = run_dir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    ep_path = find_existing(run_dir, "episode_summary")
    traj_path = find_existing(run_dir, "traj")
    if ep_path is None:
        raise FileNotFoundError(f"Could not find episode_summary under {run_dir}")
    if traj_path is None:
        raise FileNotFoundError(f"Could not find traj under {run_dir}")

    ep = load_table(ep_path).sort_values("episode").reset_index(drop=True)
    traj = load_table(traj_path).sort_values(["episode", "t"]).reset_index(drop=True)
    traj = add_delta_g(traj)

    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    plot_episode_overview(ep, outdir)
    plot_latent_overview(ep, outdir)
    plot_valence_counts(ep, outdir)
    plot_alpha_hist(traj, outdir)
    plot_alpha_by_basic_context(traj, outdir)
    plot_alpha_by_valence(traj, outdir)
    plot_delta_g_by_valence(traj, outdir)
    plot_aligned(traj, outdir, int(args.window_before), int(args.window_after))
    var_stats = plot_variance_summary(traj, outdir)
    plot_episode_end(ep, outdir)

    summary = build_summary(ep, traj, meta, var_stats)
    (outdir / "plot_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Saved figures to: {outdir}")


if __name__ == "__main__":
    main()