#!/usr/bin/env python3
"""
Stance-coloring diagnostic v2 — sanity checks.

Adds:
  - Episode-level coherence (excludes within-episode step-wander)
  - Drift direction analysis (do all seeds drift the SAME direction
    in g-space, or different directions?)
  - Sensitivity to n_late_eps choice

Usage (Spyder-style):
    sys.argv = ["stance_diagnostic_v2.py",
                "--runs_root", "outputs/phase2_all_20260403_011254",
                "--schedule", "mixed_0_4_0"]
    import stance_diagnostic_v2; stance_diagnostic_v2.main()
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def find_seed_runs(runs_root: Path, schedule: str):
    runs = []
    p1_pat = re.compile(r"from_p1_s(\d+)")
    p2_pat = re.compile(r"seed(\d+)")
    for p1_dir in sorted(runs_root.glob("from_p1_s*")):
        m1 = p1_pat.search(p1_dir.name)
        if not m1:
            continue
        p1_seed = int(m1.group(1))
        mixed_root = p1_dir / "mixed"
        if not mixed_root.exists():
            continue
        for seed_dir in sorted(mixed_root.glob("seed*")):
            m2 = p2_pat.search(seed_dir.name)
            if not m2:
                continue
            p2_seed = int(m2.group(1))
            run_dir = seed_dir / schedule
            traj = run_dir / "traj.parquet"
            if traj.exists():
                runs.append({"p1_seed": p1_seed, "p2_seed": p2_seed,
                             "run_dir": run_dir, "traj_path": traj})
    return runs


def g_cols(df):
    return sorted([c for c in df.columns if re.match(r"g_\d+$", c)],
                  key=lambda x: int(x.split("_")[-1]))


# ────────────────────────────────────────────────────────────
# Episode-level extraction (NOT step-level)
# ────────────────────────────────────────────────────────────

def episode_centroids(df, gc, block_id, n_late_eps):
    """Return (n_episodes, g_dim) — one g per episode (mean of all steps in that episode)."""
    sub = df[df["block_id"] == block_id]
    if len(sub) == 0:
        return None
    eps = sorted(sub["episode"].unique())
    late_eps = eps[-n_late_eps:] if len(eps) >= n_late_eps else eps
    out = []
    for e in late_eps:
        ep_df = sub[sub["episode"] == e]
        out.append(ep_df[gc].mean().values.astype(np.float32))
    return np.array(out)


def block_centroid(ep_centroids):
    """Mean across episode centroids."""
    return ep_centroids.mean(axis=0)


# ────────────────────────────────────────────────────────────
# Sanity check 1: episode-level coherence
# ────────────────────────────────────────────────────────────

def episode_level_coherence(runs_data):
    """RMS computed over EPISODE centroids (not step-level g).
    This excludes within-episode wander."""
    out = []
    for r in runs_data:
        c0 = r["ep_blk0"]
        c2 = r["ep_blk2"]
        if c0 is None or c2 is None:
            continue
        cen0 = c0.mean(axis=0)
        cen2 = c2.mean(axis=0)

        # Episode-to-centroid RMS (slow within-block variability)
        rms0 = float(np.sqrt(((c0 - cen0) ** 2).sum(axis=1).mean()))
        rms2 = float(np.sqrt(((c2 - cen2) ** 2).sum(axis=1).mean()))
        between = float(np.linalg.norm(cen0 - cen2))
        sil = between / max(0.5 * (rms0 + rms2), 1e-10)
        out.append({"p1": r["p1_seed"], "p2": r["p2_seed"],
                    "rms0_ep": rms0, "rms2_ep": rms2,
                    "between": between, "sil_ep": sil})
    return out


# ────────────────────────────────────────────────────────────
# Sanity check 2: drift direction analysis
# ────────────────────────────────────────────────────────────

def drift_direction_analysis(runs_data):
    """For each seed, compute drift vector v_i = centroid(blk2) - centroid(blk0).

    Then ask: do all v_i point the same direction (uniform drift),
    or do they spread (idiosyncratic stance formation)?

    Measures:
      - Mean cosine similarity between drift vectors (should be HIGH if uniform)
      - Norm of mean drift vector / mean of drift norms
        (= 1.0 if perfectly aligned, → 0 if random directions)
    """
    drifts = []
    for r in runs_data:
        c0 = r["ep_blk0"]
        c2 = r["ep_blk2"]
        if c0 is None or c2 is None:
            continue
        v = c2.mean(axis=0) - c0.mean(axis=0)
        drifts.append(v)
    drifts = np.array(drifts)
    if len(drifts) < 2:
        return None

    # Pairwise cosine similarity
    norms = np.linalg.norm(drifts, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-10, None)
    drifts_unit = drifts / norms
    cos_mat = drifts_unit @ drifts_unit.T
    triu = np.triu_indices(len(drifts), k=1)
    cos_pairs = cos_mat[triu]

    # Aligned-fraction = ||mean(v)|| / mean(||v||)
    mean_v = drifts.mean(axis=0)
    mean_v_norm = float(np.linalg.norm(mean_v))
    mean_norm = float(np.mean(np.linalg.norm(drifts, axis=1)))
    alignment = mean_v_norm / max(mean_norm, 1e-10)

    return {
        "n_seeds": len(drifts),
        "drift_norms": np.linalg.norm(drifts, axis=1),
        "cos_pairs": cos_pairs,
        "alignment": alignment,
        "mean_drift": mean_v,
    }


# ────────────────────────────────────────────────────────────
# Sanity check 3: n_late_eps sensitivity
# ────────────────────────────────────────────────────────────

def late_eps_sensitivity(df_list, gc, n_options=(5, 10, 20, 30)):
    """Run (a)-style centroid distance with varying n_late_eps."""
    results = {}
    for n in n_options:
        within = []
        for df in df_list:
            c0 = episode_centroids(df, gc, 0, n)
            c2 = episode_centroids(df, gc, 2, n)
            if c0 is None or c2 is None:
                continue
            within.append(float(np.linalg.norm(c0.mean(0) - c2.mean(0))))
        within = np.array(within)
        results[n] = {
            "median": float(np.median(within)),
            "iqr": (float(np.quantile(within, 0.25)), float(np.quantile(within, 0.75))),
            "n": len(within),
        }
    return results


# ────────────────────────────────────────────────────────────
# Reporting
# ────────────────────────────────────────────────────────────

def fmt(arr, label=""):
    arr = np.asarray(arr)
    if len(arr) == 0:
        return f"{label} n=0"
    return (f"{label} n={len(arr)} median={np.median(arr):.4f} "
            f"IQR=[{np.quantile(arr, 0.25):.4f}, {np.quantile(arr, 0.75):.4f}] "
            f"mean={arr.mean():.4f}±{arr.std():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, required=True)
    ap.add_argument("--schedule", type=str, default="mixed_0_4_0")
    ap.add_argument("--n_late_eps", type=int, default=10)
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    runs = find_seed_runs(runs_root, args.schedule)
    if not runs:
        print(f"ERROR: no runs found under {runs_root}")
        return
    print(f"Found {len(runs)} runs")

    runs_data = []
    df_list = []
    for r in runs:
        df = pd.read_parquet(r["traj_path"])
        gc_ = g_cols(df)
        if not gc_:
            continue
        ep_blk0 = episode_centroids(df, gc_, 0, args.n_late_eps)
        ep_blk2 = episode_centroids(df, gc_, 2, args.n_late_eps)
        runs_data.append({**r, "ep_blk0": ep_blk0, "ep_blk2": ep_blk2})
        df_list.append(df)
    gc = g_cols(df_list[0])

    print(f"Loaded {len(runs_data)} runs, g_dim={len(gc)}\n")

    # ──── Sanity 1: episode-level coherence ────
    print("=" * 70)
    print("SANITY 1: EPISODE-LEVEL COHERENCE")
    print("  (excludes within-episode step wander)")
    print("=" * 70)

    coh = episode_level_coherence(runs_data)
    if coh:
        rms0 = [d["rms0_ep"] for d in coh]
        rms2 = [d["rms2_ep"] for d in coh]
        between = [d["between"] for d in coh]
        sil = [d["sil_ep"] for d in coh]
        print(fmt(rms0, "  episode-RMS blk0:"))
        print(fmt(rms2, "  episode-RMS blk2:"))
        print(fmt(between, "  between-centroid:"))
        print(fmt(sil, "  silhouette (episode-level):"))
        med_sil = float(np.median(sil))
        print()
        print(f"  Compare to step-level silhouette ≈ 0.65 (original).")
        print(f"  Episode-level silhouette = {med_sil:.3f}")
        if med_sil > 1.5:
            print("  → episode centroids ARE well-separated; step-wander was inflating within-cloud RMS")
            print("    (d) verdict was actually misleading; coherence is OK")
        elif med_sil > 0.8:
            print("  → episode centroids partially separated")
            print("    (d) WEAK is real but not as bad as step-level suggested")
        else:
            print("  → episode centroids still overlap")
            print("    (d) WEAK confirmed independent of step wander")

    # ──── Sanity 2: drift direction ────
    print("\n" + "=" * 70)
    print("SANITY 2: DRIFT DIRECTION ANALYSIS")
    print("  (do all seeds drift the same way, or different ways?)")
    print("=" * 70)

    drift = drift_direction_analysis(runs_data)
    if drift:
        print(fmt(drift["drift_norms"], "  drift norms ||v_i||:"))
        print(fmt(drift["cos_pairs"], "  pairwise cos(v_i, v_j):"))
        print(f"  alignment ratio (||mean(v)|| / mean(||v||)) = {drift['alignment']:.3f}")
        print()
        print("  Interpretation:")
        if drift["alignment"] > 0.85:
            print(f"  → alignment {drift['alignment']:.2f}: ALL seeds drift in nearly same direction")
            print("    (a) WEAK reflects real lack of stance differentiation")
            print("    Formation history applies a UNIVERSAL push, not seed-specific stance")
        elif drift["alignment"] > 0.5:
            print(f"  → alignment {drift['alignment']:.2f}: dominant direction with some spread")
            print("    Partial stance differentiation atop a common drift")
        else:
            print(f"  → alignment {drift['alignment']:.2f}: drift directions SPREAD")
            print("    Each seed forms its own stance direction")
            print("    (a) low ratio was misleading: seeds DO differentiate, just not from blk0")

    # ──── Sanity 3: n_late_eps sensitivity ────
    print("\n" + "=" * 70)
    print("SANITY 3: SENSITIVITY TO N_LATE_EPS")
    print("=" * 70)

    sens = late_eps_sensitivity(df_list, gc, n_options=(5, 10, 20, 30))
    print(f"\n  within-seed centroid distance vs n_late_eps:")
    for n, info in sens.items():
        print(f"    n_late={n:3d}:  median={info['median']:.4f}  "
              f"IQR=[{info['iqr'][0]:.4f}, {info['iqr'][1]:.4f}]  n_runs={info['n']}")
    medians = [sens[n]["median"] for n in (5, 10, 20, 30)]
    if max(medians) - min(medians) < 0.1 * np.median(medians):
        print("\n  → result robust to n_late_eps choice (good)")
    else:
        print("\n  → result depends on n_late_eps; settle may not be complete")


if __name__ == "__main__":
    main()