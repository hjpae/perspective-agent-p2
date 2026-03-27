# cear_pilot/analysis/plot_train_phase2.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SAVE_DPI = 160


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def find_episode_summary(run_dir: Path) -> Path:
    candidates = [run_dir / "episode_summary.parquet", run_dir / "episode_summary.csv"]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find episode_summary in {run_dir}")


def pick_dims(ep_df: pd.DataFrame, n_dims: int) -> List[int]:
    cols = [c for c in ep_df.columns if c.startswith("g_end_")]
    if not cols:
        return []
    var_pairs = []
    for c in cols:
        try:
            dim = int(c.split("_")[-1])
        except Exception:
            continue
        v = float(ep_df[c].var(ddof=0))
        var_pairs.append((v, dim))
    var_pairs.sort(reverse=True)
    return [dim for _, dim in var_pairs[:n_dims]]


def savefig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot persistent-g Phase 2 training diagnostics.")
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--dims", type=int, default=4, help="How many g_end dimensions to plot.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    outdir = run_dir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    ep_path = find_episode_summary(run_dir)
    ep = load_table(ep_path).sort_values("episode").reset_index(drop=True)

    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    # Fig 1: episode-end norms and cumulative displacement.
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.plot(ep["episode"], ep["g_end_norm"], marker="o", label="||g_end||")
    ax.plot(ep["episode"], ep["g_delta_from_global_init"], marker="s", label="||g_end - g_init||")
    ax.set_xlabel("episode")
    ax.set_ylabel("magnitude")
    ax.set_title("Persistent-g formation: norm and cumulative displacement")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, outdir / "fig1_g_norm_and_cumulative_displacement.png")

    # Fig 2: episode-to-episode change and within-episode change.
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.plot(ep["episode"], ep["g_delta_from_prev_ep_end"], marker="o", label="||g_end(ep)-g_end(ep-1)||")
    ax.plot(ep["episode"], ep["g_delta_within_ep"], marker="s", label="||g_end-g_start||")
    ax.plot(ep["episode"], ep["mean_delta_g"], marker="^", label="mean stepwise delta_g")
    ax.set_xlabel("episode")
    ax.set_ylabel("change")
    ax.set_title("Persistent-g formation: stability vs. within-episode movement")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, outdir / "fig2_episode_to_episode_change.png")

    # Fig 3: selected g_end dimensions.
    dims = pick_dims(ep, int(args.dims))
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()
    for dim in dims:
        ax.plot(ep["episode"], ep[f"g_end_{dim}"], marker="o", label=f"g_end[{dim}]")
    ax.set_xlabel("episode")
    ax.set_ylabel("value")
    ax.set_title("Persistent-g formation: selected episode-end latent dimensions")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, outdir / "fig3_selected_g_end_dims.png")

    # Fig 4: behavior/context summary.
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.plot(ep["episode"], ep["final_x"], marker="o", label="final_x")
    ax.plot(ep["episode"], ep["mean_abs_c"], marker="s", label="mean |c|")
    ax.plot(ep["episode"], ep["mean_entropy"], marker="^", label="mean entropy")
    ax.set_xlabel("episode")
    ax.set_ylabel("value")
    ax.set_title("Persistent-g formation: task/context traces")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, outdir / "fig4_context_and_behavior_traces.png")

    summary = {
        "run_dir": str(run_dir),
        "alpha_fixed": meta.get("alpha_fixed"),
        "episodes": int(len(ep)),
        "schedule_pattern": meta.get("env_cfg", {}).get("schedule_pattern"),
        "valence_sequence": meta.get("env_cfg", {}).get("valence_sequence"),
        "carry_g_between_episodes": not bool(meta.get("reset_g_every_episode", False)),
        "g_end_norm_first": float(ep.iloc[0]["g_end_norm"]) if len(ep) else None,
        "g_end_norm_last": float(ep.iloc[-1]["g_end_norm"]) if len(ep) else None,
        "g_delta_from_global_init_last": float(ep.iloc[-1]["g_delta_from_global_init"]) if len(ep) else None,
        "g_delta_from_prev_ep_end_last": float(ep.iloc[-1]["g_delta_from_prev_ep_end"]) if len(ep) else None,
        "mean_within_ep_delta": float(ep["g_delta_within_ep"].mean()) if len(ep) else None,
        "mean_stepwise_delta": float(ep["mean_delta_g"].mean()) if len(ep) else None,
        "selected_dims": dims,
    }
    (outdir / "plot_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved figures to: {outdir}")


if __name__ == "__main__":
    main()
