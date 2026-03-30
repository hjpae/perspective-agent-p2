# cear_pilot/analysis/phase2_analysis.py
# -*- coding: utf-8 -*-
"""
Phase 2 analysis for encounter / rupture ecology.

Inputs:
- run_collect output directory containing traj.parquet/csv and meta.json

Outputs:
- phase2_summary.json
- figs/phase2_overview.png
- figs/phase2_encounter_panels.png
- figs/phase2_g_pca.png (if sklearn available)

This script is designed to run locally (e.g. Spyder) after training/probing on Vast.ai.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from cear_pilot.analysis.metrics import (
    g_columns,
    drift_norm,
    encounter_indices,
    rupture_indices,
    encounter_aligned_mean,
    encounter_induced_g_shift,
    false_alarm_reactivity,
    recovery_half_life_from_rows,
    context_sensitive_engagement,
)


def load_table(run_dir: Path):
    import pandas as pd
    p_parq = run_dir / "traj.parquet"
    p_csv = run_dir / "traj.csv"
    if p_parq.exists():
        return pd.read_parquet(p_parq)
    if p_csv.exists():
        return pd.read_csv(p_csv)
    raise FileNotFoundError(f"No traj.parquet or traj.csv under {run_dir}")


def load_meta(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def to_jsonable(x):
    if x is None:
        return None
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_jsonable(v) for v in x]
    if isinstance(x, tuple):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def get_time_array(df) -> np.ndarray:
    if "t_global" in df.columns:
        return df["t_global"].to_numpy(dtype=np.int32)
    if "t" in df.columns:
        return df["t"].to_numpy(dtype=np.int32)
    return np.arange(len(df), dtype=np.int32)


def get_G(df) -> np.ndarray:
    cols = g_columns(df)
    if len(cols) == 0:
        raise ValueError("No g_* columns found in trajectory table")
    cols = sorted(cols, key=lambda x: int(x.split("_")[1]))
    return df[cols].to_numpy(dtype=np.float32)


def get_entropy(df) -> Optional[np.ndarray]:
    if "pi_entropy" in df.columns:
        return df["pi_entropy"].to_numpy(dtype=np.float32)
    if "entropy" in df.columns:
        return df["entropy"].to_numpy(dtype=np.float32)
    return None


def get_zone(df) -> Optional[np.ndarray]:
    if "zone_id" not in df.columns:
        return None
    return df["zone_id"].to_numpy(dtype=np.int32)


def get_fragility(df) -> Optional[np.ndarray]:
    if "fragility" not in df.columns:
        return None
    return df["fragility"].to_numpy(dtype=np.float32)


def event_triggered_segments(arr: np.ndarray, idx: np.ndarray, pre: int, post: int) -> Optional[np.ndarray]:
    segs = []
    for t0 in idx:
        a = int(t0) - int(pre)
        b = int(t0) + int(post) + 1
        if a < 0 or b > len(arr):
            continue
        segs.append(arr[a:b])
    if len(segs) == 0:
        return None
    return np.stack(segs, axis=0)


def event_triggered_gnorm(G: np.ndarray, idx: np.ndarray, pre: int, post: int) -> Optional[np.ndarray]:
    segs = []
    for t0 in idx:
        a = int(t0) - int(pre)
        b = int(t0) + int(post) + 1
        if a < 0 or b > len(G):
            continue
        ref = G[t0]
        d = np.linalg.norm(G[a:b] - ref[None, :], axis=1)
        segs.append(d.astype(np.float32))
    if len(segs) == 0:
        return None
    return np.stack(segs, axis=0)


def baseline_recovery_curves(G: np.ndarray, rupt_idx: np.ndarray, pre_window: int, post_window: int) -> Optional[np.ndarray]:
    segs = []
    for t0 in rupt_idx:
        a = int(t0) - int(pre_window)
        b = int(t0) + int(post_window) + 1
        if a < 0 or b > len(G):
            continue
        ref = G[a:t0].mean(axis=0)
        d = np.linalg.norm(G[a:b] - ref[None, :], axis=1)
        segs.append(d.astype(np.float32))
    if len(segs) == 0:
        return None
    return np.stack(segs, axis=0)


def summarize_run(df, fragility_threshold: float, encounter_horizon: int, recovery_pre_window: int) -> Dict[str, Any]:
    G = get_G(df)
    dG = drift_norm(G)
    ent = get_entropy(df)
    enc_idx = encounter_indices(df)
    rup_idx = rupture_indices(df)

    out: Dict[str, Any] = {
        "n_steps": int(len(df)),
        "n_encounters": int(len(enc_idx)),
        "n_ruptures": int(len(rup_idx)),
        "encounter_rate": float(len(enc_idx) / max(1, len(df))),
        "rupture_rate": float(len(rup_idx) / max(1, len(df))),
        "g_drift_mean": float(np.mean(dG)),
        "g_drift_median": float(np.median(dG)),
        "g_norm_mean": float(np.mean(np.linalg.norm(G, axis=1))),
    }

    if ent is not None:
        out["entropy_mean"] = float(np.mean(ent))
        out["entropy_median"] = float(np.median(ent))

    out["encounter_g_shift"] = encounter_induced_g_shift(df, horizon=1)
    out["false_alarm"] = false_alarm_reactivity(df, horizon=encounter_horizon)
    out["recovery_half_life"] = recovery_half_life_from_rows(
        df, pre_window=recovery_pre_window, threshold=0.15
    )
    out["context_sensitive_engagement"] = context_sensitive_engagement(
        df, fragility_threshold=fragility_threshold
    )

    if "rupture" in df.columns and "encounter_event" in df.columns:
        enc = df["encounter_event"].to_numpy().astype(int)
        rup = df["rupture"].to_numpy().astype(int)
        realized = []
        for t0 in np.where(enc == 1)[0]:
            t1 = min(len(rup), t0 + encounter_horizon + 1)
            realized.append(int(np.any(rup[t0:t1] == 1)))
        out["rupture_after_encounter_prob"] = None if len(realized) == 0 else float(np.mean(realized))

    return out


def maybe_make_pca_plot(G: np.ndarray, zone: Optional[np.ndarray], out_png: Path) -> bool:
    try:
        from sklearn.decomposition import PCA
    except Exception:
        return False

    pca = PCA(n_components=2)
    Y = pca.fit_transform(G)

    plt.figure(figsize=(6.5, 5.5))
    if zone is None:
        plt.scatter(Y[:, 0], Y[:, 1], s=8, alpha=0.5)
    else:
        for zid in sorted(np.unique(zone)):
            m = zone == zid
            plt.scatter(Y[m, 0], Y[m, 1], s=8, alpha=0.5, label=f"zone {int(zid)}")
        plt.legend(frameon=False)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Latent g trajectory (PCA)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()
    return True


def make_overview_figure(df, out_png: Path) -> None:
    t = get_time_array(df)
    G = get_G(df)
    ent = get_entropy(df)
    zone = get_zone(df)
    frag = get_fragility(df)
    dG = drift_norm(G)
    enc_idx = encounter_indices(df)
    rup_idx = rupture_indices(df)

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(t, np.linalg.norm(G, axis=1), label="||g||")
    axes[0].plot(t, dG, label="Δg", alpha=0.8)
    for i in enc_idx:
        axes[0].axvline(t[i], color="gold", alpha=0.10, linewidth=1.0)
    for i in rup_idx:
        axes[0].axvline(t[i], color="red", alpha=0.18, linewidth=1.0)
    axes[0].set_ylabel("latent")
    axes[0].legend(frameon=False)

    if ent is not None:
        axes[1].plot(t, ent, label="entropy")
        for i in enc_idx:
            axes[1].axvline(t[i], color="gold", alpha=0.10, linewidth=1.0)
        for i in rup_idx:
            axes[1].axvline(t[i], color="red", alpha=0.18, linewidth=1.0)
        axes[1].set_ylabel("policy")
        axes[1].legend(frameon=False)
    else:
        axes[1].text(0.5, 0.5, "No entropy column", ha="center", va="center", transform=axes[1].transAxes)

    if frag is not None:
        axes[2].plot(t, frag, label="fragility")
    if "rupture_memory" in df.columns:
        axes[2].plot(t, df["rupture_memory"].to_numpy(dtype=np.float32), label="rupture_memory")
    for i in enc_idx:
        axes[2].axvline(t[i], color="gold", alpha=0.10, linewidth=1.0)
    for i in rup_idx:
        axes[2].axvline(t[i], color="red", alpha=0.18, linewidth=1.0)
    axes[2].set_ylabel("hidden ctx")
    axes[2].legend(frameon=False)

    if zone is not None:
        axes[3].plot(t, zone, label="zone_id")
        if "on_encounter" in df.columns:
            y_enc = np.where(df["on_encounter"].to_numpy().astype(int) == 1, 1.0, np.nan)
            axes[3].scatter(t, y_enc, s=8, alpha=0.35, label="on_encounter")
        axes[3].set_ylabel("zone")
        axes[3].set_xlabel("t")
        axes[3].legend(frameon=False)
    else:
        axes[3].text(0.5, 0.5, "No zone column", ha="center", va="center", transform=axes[3].transAxes)
        axes[3].set_xlabel("t")

    fig.suptitle("Phase 2 overview: latent, entropy, hidden context, encounters/ruptures")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_encounter_panels(df, out_png: Path, encounter_pre: int, encounter_post: int, recovery_pre_window: int, recovery_post_window: int) -> None:
    G = get_G(df)
    ent = get_entropy(df)
    enc_idx = encounter_indices(df)
    rup_idx = rupture_indices(df)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    if ent is not None:
        enc_mean = encounter_aligned_mean(
            df, event_col="encounter_event", value_col="pi_entropy", L_pre=encounter_pre, L_post=encounter_post
        )
        if enc_mean["mean"] is not None:
            tau = np.arange(-encounter_pre, encounter_post + 1)
            X = enc_mean["segments"]
            mu = X.mean(axis=0)
            se = X.std(axis=0) / max(1.0, np.sqrt(X.shape[0]))
            axes[0, 0].plot(tau, mu)
            axes[0, 0].fill_between(tau, mu - se, mu + se, alpha=0.25)
            axes[0, 0].axvline(0, linestyle="--", linewidth=1)
            axes[0, 0].set_title(f"Encounter-aligned entropy (n={enc_mean['n']})")
            axes[0, 0].set_xlabel("τ from encounter")
        else:
            axes[0, 0].text(0.5, 0.5, "No valid encounter-aligned entropy segments", ha="center", va="center")
    else:
        axes[0, 0].text(0.5, 0.5, "No entropy column", ha="center", va="center")

    Xg = event_triggered_gnorm(G, enc_idx, pre=encounter_pre, post=encounter_post)
    if Xg is not None:
        tau = np.arange(-encounter_pre, encounter_post + 1)
        mu = Xg.mean(axis=0)
        se = Xg.std(axis=0) / max(1.0, np.sqrt(Xg.shape[0]))
        axes[0, 1].plot(tau, mu)
        axes[0, 1].fill_between(tau, mu - se, mu + se, alpha=0.25)
        axes[0, 1].axvline(0, linestyle="--", linewidth=1)
        axes[0, 1].set_title(f"Encounter-aligned ||g(τ)-g(0)|| (n={Xg.shape[0]})")
        axes[0, 1].set_xlabel("τ from encounter")
    else:
        axes[0, 1].text(0.5, 0.5, "No valid encounter-aligned g segments", ha="center", va="center")

    Xr = baseline_recovery_curves(G, rup_idx, pre_window=recovery_pre_window, post_window=recovery_post_window)
    if Xr is not None:
        tau = np.arange(-recovery_pre_window, recovery_post_window + 1)
        mu = Xr.mean(axis=0)
        se = Xr.std(axis=0) / max(1.0, np.sqrt(Xr.shape[0]))
        axes[1, 0].plot(tau, mu)
        axes[1, 0].fill_between(tau, mu - se, mu + se, alpha=0.25)
        axes[1, 0].axvline(0, linestyle="--", linewidth=1)
        axes[1, 0].set_title(f"Rupture recovery to pre-rupture baseline (n={Xr.shape[0]})")
        axes[1, 0].set_xlabel("τ from rupture")
    else:
        axes[1, 0].text(0.5, 0.5, "No valid rupture recovery segments", ha="center", va="center")

    if "fragility" in df.columns and "rupture" in df.columns:
        F = df["fragility"].to_numpy(dtype=np.float32)
        R = df["rupture"].to_numpy(dtype=np.int32)
        axes[1, 1].hist(F[R == 0], bins=20, alpha=0.5, label="non-rupture steps")
        if np.any(R == 1):
            axes[1, 1].hist(F[R == 1], bins=20, alpha=0.5, label="rupture steps")
        axes[1, 1].set_title("Fragility distribution")
        axes[1, 1].set_xlabel("fragility")
        axes[1, 1].legend(frameon=False)
    else:
        axes[1, 1].text(0.5, 0.5, "No fragility/rupture columns", ha="center", va="center")

    fig.suptitle("Phase 2 encounter / rupture panels")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--fragility_threshold", type=float, default=0.5)
    ap.add_argument("--encounter_horizon", type=int, default=5)
    ap.add_argument("--encounter_pre", type=int, default=5)
    ap.add_argument("--encounter_post", type=int, default=15)
    ap.add_argument("--recovery_pre_window", type=int, default=20)
    ap.add_argument("--recovery_post_window", type=int, default=30)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    fig_dir = run_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_table(run_dir)
    meta = load_meta(run_dir)

    summary = summarize_run(
        df,
        fragility_threshold=float(args.fragility_threshold),
        encounter_horizon=int(args.encounter_horizon),
        recovery_pre_window=int(args.recovery_pre_window),
    )
    summary["meta"] = meta
    summary["args"] = vars(args)

    make_overview_figure(df, fig_dir / "phase2_overview.png")
    make_encounter_panels(
        df,
        fig_dir / "phase2_encounter_panels.png",
        encounter_pre=int(args.encounter_pre),
        encounter_post=int(args.encounter_post),
        recovery_pre_window=int(args.recovery_pre_window),
        recovery_post_window=int(args.recovery_post_window),
    )

    G = get_G(df)
    zone = get_zone(df)
    made_pca = maybe_make_pca_plot(G, zone, fig_dir / "phase2_g_pca.png")
    summary["made_pca"] = bool(made_pca)

    out_json = run_dir / "phase2_summary.json"
    out_json.write_text(json.dumps(to_jsonable(summary), indent=2))

    print(json.dumps(to_jsonable(summary), indent=2))
    print(f"[OK] Saved summary: {out_json}")
    print(f"[OK] Saved figures under: {fig_dir}")


if __name__ == "__main__":
    main()
