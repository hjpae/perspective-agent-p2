# cear_pilot/analysis/plot_phase2_landscape.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_table(stem: Path) -> pd.DataFrame:
    pq = stem.with_suffix(".parquet")
    csv = stem.with_suffix(".csv")
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing table for {stem}")


def pca_2d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def save_fig(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    traj = load_table(run_dir / "traj")
    ep = load_table(run_dir / "episode_summary") if (run_dir / "episode_summary.csv").exists() or (run_dir / "episode_summary.parquet").exists() else None

    g_cols = sorted([c for c in traj.columns if c.startswith("g_")], key=lambda s: int(s.split("_")[1]))
    g = traj[g_cols].to_numpy()
    proj = pca_2d(g) if g.shape[1] >= 2 else np.concatenate([g, np.zeros((g.shape[0], 1))], axis=1)

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(traj["alpha"].to_numpy(), label="alpha")
    ax1.plot(traj["energy"].to_numpy(), label="energy")
    ax1.plot(traj["grad_norm"].to_numpy(), label="grad_norm")
    ax1.set_title("Alpha / energy / grad norm")
    ax1.set_xlabel("step")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(proj[:, 0], proj[:, 1], c=traj["basin_id"].to_numpy(), s=8)
    ax2.set_title("Latent PCA colored by basin")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    save_fig(fig, run_dir / "fig1_pca_and_energy.png")

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    basin_counts = traj.groupby("episode")["basin_id"].nunique()
    ax1.plot(basin_counts.index.to_numpy(), basin_counts.to_numpy())
    ax1.set_title("Unique basins per episode")
    ax1.set_xlabel("episode")
    ax1.set_ylabel("count")

    ax2 = fig.add_subplot(1, 2, 2)
    if ep is not None and "switches" in ep.columns:
        ax2.plot(ep["episode"].to_numpy(), ep["switches"].to_numpy())
        ax2.set_title("Basin switches per episode")
        ax2.set_xlabel("episode")
        ax2.set_ylabel("switches")
    else:
        ax2.text(0.5, 0.5, "episode summary unavailable", ha="center", va="center")
        ax2.set_axis_off()
    save_fig(fig, run_dir / "fig2_basin_switches.png")

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    if "recovery_dist" in traj.columns and np.isfinite(traj["recovery_dist"].to_numpy()).any():
        rec = traj[np.isfinite(traj["recovery_dist"]) & (traj["time_since_perturb"] >= 0)].copy()
        grp = rec.groupby("time_since_perturb")["recovery_dist"].mean()
        ax1.plot(grp.index.to_numpy(), grp.to_numpy())
        ax1.set_title("Recovery after perturbation")
        ax1.set_xlabel("steps since perturbation")
        ax1.set_ylabel("distance from anchor")
    else:
        ax1.text(0.5, 0.5, "no perturbation trace", ha="center", va="center")
        ax1.set_axis_off()

    ax2 = fig.add_subplot(1, 2, 2)
    grp = traj.groupby("encounter_outcome")["alpha"].mean()
    ax2.bar([str(k) for k in grp.index.tolist()], grp.to_numpy())
    ax2.set_title("Mean alpha by encounter outcome")
    ax2.set_xlabel("encounter_outcome")
    ax2.set_ylabel("mean alpha")
    save_fig(fig, run_dir / "fig3_recovery_and_alpha.png")

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    for col in g_cols[: min(4, len(g_cols))]:
        ax1.plot(traj[col].to_numpy(), label=col)
    ax1.set_title("Selected g dimensions")
    ax1.set_xlabel("step")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(traj["c_state"].to_numpy(), label="c_state")
    ax2.plot(traj["entropy"].to_numpy(), label="entropy")
    ax2.set_title("Context and policy trace")
    ax2.set_xlabel("step")
    ax2.legend()
    save_fig(fig, run_dir / "fig4_traces.png")

    print(f"[plot_phase2_landscape] saved figures to {run_dir}")


if __name__ == "__main__":
    main()
