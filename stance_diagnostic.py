#!/usr/bin/env python3
"""
Stance-coloring diagnostic for ALIFE Phase 2 runs.

Tests four criteria for whether perspective latent g carries
intentional-quality-like stance-coloring:

  (a) Persistence under common probe
      block 0 late g vs block 2 late g (same env condition, different history)
      Within-seed distance vs between-seed null distribution

  (b) Local structural distinctness
      Each block's g cloud has its own local geometric character
      (intrinsic dim, effective rank) — not just centroid offset

  (c) Non-reducibility to representational content
      z_t(g_blk0, x) - z_t(g_blk2, x) pattern across probes:
      probe-independent (= just translation, reducible) vs
      probe-dependent (= g induces non-trivial transform on z)

  (d) Within-region coherence
      Each block's g cloud is a coherent region (compact, unimodal),
      not random scatter

Usage:
    python stance_diagnostic.py \\
        --runs_root outputs/phase2_all_20260403_011254 \\
        --schedule mixed_0_4_0
"""

from __future__ import annotations

import argparse
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────
# Path discovery
# ──────────────────────────────────────────────────────────

def find_seed_runs(runs_root: Path, schedule: str) -> List[Dict]:
    """Walk runs_root and find all (p1_seed, p2_seed) runs for given schedule.

    Expected structure:
        runs_root/from_p1_s<P1>/mixed/seed<P2>/<schedule>/
            ├── traj.parquet
            └── ckpt_final.pt
    """
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
            ckpt = run_dir / "ckpt_final.pt"
            if traj.exists() and ckpt.exists():
                runs.append({
                    "p1_seed": p1_seed,
                    "p2_seed": p2_seed,
                    "run_dir": run_dir,
                    "traj_path": traj,
                    "ckpt_path": ckpt,
                })
    return runs


def g_cols(df: pd.DataFrame) -> List[str]:
    return sorted([c for c in df.columns if re.match(r"g_\d+$", c)],
                  key=lambda x: int(x.split("_")[-1]))


# ──────────────────────────────────────────────────────────
# Block-level extraction
# ──────────────────────────────────────────────────────────

def extract_block_clouds(df: pd.DataFrame, gc: List[str], n_late_eps: int = 10
                          ) -> Dict[int, np.ndarray]:
    """For each block_id, return a (n_steps, g_dim) array of g states
    from the late portion of that block."""
    clouds = {}
    if "block_id" not in df.columns:
        return clouds

    for blk in sorted(df["block_id"].unique()):
        sub = df[df["block_id"] == blk]
        if len(sub) == 0:
            continue
        eps = sorted(sub["episode"].unique())
        late_eps = eps[-n_late_eps:] if len(eps) >= n_late_eps else eps
        late = sub[sub["episode"].isin(late_eps)]
        cloud = late[gc].values.astype(np.float32)
        clouds[int(blk)] = cloud
    return clouds


def cloud_centroid(cloud: np.ndarray) -> np.ndarray:
    return cloud.mean(axis=0)


# ──────────────────────────────────────────────────────────
# (a) Persistence
# ──────────────────────────────────────────────────────────

def compute_persistence(runs_data: List[Dict]) -> Dict:
    """For each seed: distance between block 0 late centroid and block 2 late centroid.
    Compare to between-seed null (centroid_blk0 of seed_i vs centroid_blk2 of seed_j).

    Strong stance differentiation: within-seed distance comparable to between-seed.
    Weak: within-seed << between-seed (all agents drift same way).
    """
    within = []
    blk0_centroids = []
    blk2_centroids = []
    seed_labels = []

    for r in runs_data:
        clouds = r["clouds"]
        if 0 not in clouds or 2 not in clouds:
            continue
        c0 = cloud_centroid(clouds[0])
        c2 = cloud_centroid(clouds[2])
        within.append(np.linalg.norm(c0 - c2))
        blk0_centroids.append(c0)
        blk2_centroids.append(c2)
        seed_labels.append((r["p1_seed"], r["p2_seed"]))

    within = np.array(within)
    blk0_centroids = np.array(blk0_centroids)
    blk2_centroids = np.array(blk2_centroids)

    # Between-seed null: pair seed_i's blk0 with seed_j's blk2 (i != j)
    n = len(within)
    between = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            between.append(np.linalg.norm(blk0_centroids[i] - blk2_centroids[j]))
    between = np.array(between)

    # Also: between-seed within-block (blk0_i vs blk0_j) — should be small if no condition spread
    same_blk0 = []
    same_blk2 = []
    for i in range(n):
        for j in range(i+1, n):
            same_blk0.append(np.linalg.norm(blk0_centroids[i] - blk0_centroids[j]))
            same_blk2.append(np.linalg.norm(blk2_centroids[i] - blk2_centroids[j]))
    same_blk0 = np.array(same_blk0) if same_blk0 else np.array([0.0])
    same_blk2 = np.array(same_blk2) if same_blk2 else np.array([0.0])

    return {
        "within_seed_dist": within,
        "between_seed_null": between,
        "between_seed_blk0": same_blk0,
        "between_seed_blk2": same_blk2,
        "n_seeds": n,
    }


# ──────────────────────────────────────────────────────────
# (b) Local structural distinctness
# ──────────────────────────────────────────────────────────

def effective_rank(cloud: np.ndarray) -> float:
    """Shannon-entropy-based effective rank of the eigenvalue spectrum.
    er = exp(H(p)) where p_i = lambda_i / sum(lambda)."""
    if len(cloud) < 2:
        return 0.0
    centered = cloud - cloud.mean(axis=0, keepdims=True)
    cov = (centered.T @ centered) / max(1, len(cloud) - 1)
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 1e-12, None)
    p = eig / eig.sum()
    H = -np.sum(p * np.log(p))
    return float(np.exp(H))


def twonn_intrinsic_dim(cloud: np.ndarray, n_max: int = 500) -> float:
    """Two-NN estimator for intrinsic dimension (Facco et al. 2017).
    Robust for small samples; uses ratio of distances to 2nd vs 1st nearest neighbor."""
    if len(cloud) < 10:
        return 0.0

    # Subsample if huge
    if len(cloud) > n_max:
        idx = np.random.RandomState(0).choice(len(cloud), n_max, replace=False)
        X = cloud[idx]
    else:
        X = cloud

    # Pairwise distances
    n = len(X)
    diff = X[:, None, :] - X[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, np.inf)

    # First and second nearest
    d_sorted = np.sort(d, axis=1)
    r1 = d_sorted[:, 0]
    r2 = d_sorted[:, 1]

    # Filter out coincident points
    mask = (r1 > 1e-10) & (r2 > r1)
    if mask.sum() < 5:
        return 0.0
    mu = r2[mask] / r1[mask]

    # MLE estimate: d = N / sum(log mu)
    log_mu = np.log(mu)
    if log_mu.sum() <= 0:
        return 0.0
    d_est = len(log_mu) / log_mu.sum()
    return float(d_est)


def compute_local_structure(runs_data: List[Dict]) -> Dict:
    """For each block, compute effective rank and intrinsic dim.
    A genuine structural difference: blocks differ on these features,
    not just centroid distance."""
    results = {0: [], 2: []}
    for r in runs_data:
        clouds = r["clouds"]
        for blk in (0, 2):
            if blk not in clouds:
                continue
            cloud = clouds[blk]
            er = effective_rank(cloud)
            id_ = twonn_intrinsic_dim(cloud)
            results[blk].append({"er": er, "id": id_,
                                 "p1": r["p1_seed"], "p2": r["p2_seed"]})
    return results


# ──────────────────────────────────────────────────────────
# (c) Probe-dependence of z shift  (uses probe_representation.csv)
# ──────────────────────────────────────────────────────────

def compute_probe_dependence(runs_data: List[Dict]) -> Dict:
    """Load probe_representation.csv (if exists) and compute SVD of
    [z_t(g_blk0, x_i) - z_t(g_blk2, x_i)] across probes x_i.

    Top-1 singular value ratio → 1: reducible (just translation in z)
    Top-1 ratio < 1: g induces non-trivial transform on z (non-reducible)
    """
    out = []
    for r in runs_data:
        probe_csv = r["run_dir"] / "analysis" / "probe_representation.csv"
        if not probe_csv.exists():
            continue
        df = pd.read_csv(probe_csv)

        # Find blk0_nP0 and blk2_nP0 labels
        labels = sorted(df["g_label"].unique())
        blk0_label = next((l for l in labels if l.startswith("blk0")), None)
        blk2_label = next((l for l in labels if l.startswith("blk2")), None)
        if blk0_label is None or blk2_label is None:
            continue

        zt_cols = sorted([c for c in df.columns if re.match(r"z_t_\d+$", c)],
                          key=lambda x: int(x.split("_")[-1]))
        if not zt_cols:
            continue

        df0 = df[df["g_label"] == blk0_label].sort_values("probe_idx")
        df2 = df[df["g_label"] == blk2_label].sort_values("probe_idx")
        common = set(df0["probe_idx"]) & set(df2["probe_idx"])
        df0 = df0[df0["probe_idx"].isin(common)].sort_values("probe_idx")
        df2 = df2[df2["probe_idx"].isin(common)].sort_values("probe_idx")
        if len(df0) < 3:
            continue

        Z0 = df0[zt_cols].values  # (n_probes, z_dim)
        Z2 = df2[zt_cols].values
        diff = Z0 - Z2  # (n_probes, z_dim)

        # SVD: singular values reveal rank structure
        # If pure translation: rank 1 (all rows equal) → top sv dominates
        # If probe-dependent: rank > 1 → multiple sv's contribute
        u, s, vt = np.linalg.svd(diff, full_matrices=False)
        s_total = s.sum()
        if s_total < 1e-10:
            continue
        top1_ratio = float(s[0] / s_total)
        # Effective rank of the diff matrix
        p = s / s_total
        p = np.clip(p, 1e-12, None)
        H = -np.sum(p * np.log(p))
        diff_er = float(np.exp(H))

        out.append({
            "p1": r["p1_seed"], "p2": r["p2_seed"],
            "top1_ratio": top1_ratio,
            "diff_er": diff_er,
            "n_probes": len(df0),
            "diff_norm_mean": float(np.linalg.norm(diff, axis=1).mean()),
        })
    return {"per_seed": out}


# ──────────────────────────────────────────────────────────
# (d) Within-region coherence
# ──────────────────────────────────────────────────────────

def compute_coherence(runs_data: List[Dict]) -> Dict:
    """For each block in each seed:
      - within-cloud variance (RMS distance from centroid)
      - silhouette-like score: (between-block dist) / (within-block scatter)

    High silhouette → coherent regions
    Low silhouette → blocks are not really separated regions, just scatter
    """
    out = []
    for r in runs_data:
        clouds = r["clouds"]
        if 0 not in clouds or 2 not in clouds:
            continue
        c0, c2 = clouds[0], clouds[2]
        cen0, cen2 = c0.mean(axis=0), c2.mean(axis=0)

        # Within-cloud RMS
        rms0 = float(np.sqrt(((c0 - cen0) ** 2).sum(axis=1).mean()))
        rms2 = float(np.sqrt(((c2 - cen2) ** 2).sum(axis=1).mean()))

        # Between-centroid distance
        between = float(np.linalg.norm(cen0 - cen2))

        # Silhouette-like: ratio of between to mean within
        within_mean = 0.5 * (rms0 + rms2)
        sil = between / max(within_mean, 1e-10)

        out.append({
            "p1": r["p1_seed"], "p2": r["p2_seed"],
            "rms_blk0": rms0, "rms_blk2": rms2,
            "between": between,
            "silhouette_like": sil,
        })
    return {"per_seed": out}


# ──────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────

def summarize(arr) -> str:
    arr = np.asarray(arr)
    if len(arr) == 0:
        return "n=0"
    return f"n={len(arr)}  median={np.median(arr):.4f}  IQR=[{np.quantile(arr, 0.25):.4f}, {np.quantile(arr, 0.75):.4f}]  mean={arr.mean():.4f}±{arr.std():.4f}"


def report(persistence, structure, probe_dep, coherence, schedule: str, runs_data: List[Dict]):
    print("\n" + "=" * 70)
    print(f"STANCE-COLORING DIAGNOSTIC  ({schedule})")
    print(f"  {len(runs_data)} runs found")
    print("=" * 70)

    # ── (a) Persistence ──
    print("\n(a) PERSISTENCE  — block 0 vs block 2 centroid distance")
    print("    Question: do agents of different formation history persistently")
    print("              differ when in same environment condition?")
    w = persistence["within_seed_dist"]
    b = persistence["between_seed_null"]
    print(f"\n    within-seed (blk0 vs blk2, same agent):    {summarize(w)}")
    print(f"    between-seed null (blk0_i vs blk2_j, i≠j):  {summarize(b)}")
    print(f"    between-seed same-block blk0 vs blk0:       {summarize(persistence['between_seed_blk0'])}")
    print(f"    between-seed same-block blk2 vs blk2:       {summarize(persistence['between_seed_blk2'])}")
    if len(w) > 0 and len(b) > 0:
        ratio = float(np.median(w) / np.median(b)) if np.median(b) > 0 else float('inf')
        print(f"\n    within/null ratio (median):    {ratio:.3f}")
        print("    Interpretation:")
        if ratio > 0.7:
            print("      → within ≈ between: stance differentiation IS systematic across seeds")
        elif ratio > 0.4:
            print("      → within < between but non-trivial: partial differentiation")
        else:
            print("      → within << between: blk0/blk2 differ similarly in all seeds")
            print("        (= history shifts everyone same direction; weak stance-coloring)")

    # ── (b) Local structure ──
    print("\n(b) LOCAL STRUCTURAL DISTINCTNESS")
    print("    Question: do block 0 and block 2 clouds differ in geometric character,")
    print("              not just location?")
    er0 = [s["er"] for s in structure[0]]
    er2 = [s["er"] for s in structure[2]]
    id0 = [s["id"] for s in structure[0]]
    id2 = [s["id"] for s in structure[2]]
    print(f"\n    effective rank  blk0:  {summarize(er0)}")
    print(f"    effective rank  blk2:  {summarize(er2)}")
    print(f"    intrinsic dim   blk0:  {summarize(id0)}")
    print(f"    intrinsic dim   blk2:  {summarize(id2)}")
    if len(er0) > 0 and len(er2) > 0:
        from scipy.stats import wilcoxon
        try:
            stat_er, p_er = wilcoxon(er0, er2)
            stat_id, p_id = wilcoxon(id0, id2)
            print(f"\n    paired Wilcoxon  effective_rank:  W={stat_er:.2f}  p={p_er:.3f}")
            print(f"    paired Wilcoxon  intrinsic_dim:   W={stat_id:.2f}  p={p_id:.3f}")
            if p_er < 0.05 or p_id < 0.05:
                print("      → significant structural difference between blocks")
            else:
                print("      → no significant structural difference (likely just centroid offset)")
        except Exception as e:
            print(f"    [wilcoxon skipped: {e}]")

    # ── (c) Probe-dependence ──
    print("\n(c) NON-REDUCIBILITY TO REPRESENTATIONAL CONTENT")
    print("    Question: is z-level g-effect a simple translation, or probe-dependent?")
    pd_data = probe_dep["per_seed"]
    if not pd_data:
        print("    [no probe_representation.csv found in run_dirs — skipping]")
    else:
        top1 = [d["top1_ratio"] for d in pd_data]
        er_diff = [d["diff_er"] for d in pd_data]
        diff_norm = [d["diff_norm_mean"] for d in pd_data]
        print(f"\n    top-1 SV ratio of [z(g_blk0,x) - z(g_blk2,x)]:  {summarize(top1)}")
        print(f"    effective rank of diff matrix:                  {summarize(er_diff)}")
        print(f"    mean ||z_diff|| per probe:                      {summarize(diff_norm)}")
        print("    Interpretation:")
        med_top1 = float(np.median(top1))
        if med_top1 > 0.85:
            print(f"      → top1≈{med_top1:.2f}: g effect on z is ~rank-1 (REDUCIBLE to translation)")
            print("        Stance-coloring at z-level is mostly representational shift")
        elif med_top1 > 0.6:
            print(f"      → top1≈{med_top1:.2f}: dominant direction but not pure translation")
            print("        Partial non-reducibility")
        else:
            print(f"      → top1≈{med_top1:.2f}: probe-dependent transform (NON-REDUCIBLE)")
            print("        g induces structural transform on z, not just shift")

    # ── (d) Coherence ──
    print("\n(d) WITHIN-REGION COHERENCE")
    print("    Question: are block clouds compact regions, or just scatter?")
    coh = coherence["per_seed"]
    if coh:
        sils = [d["silhouette_like"] for d in coh]
        rms0 = [d["rms_blk0"] for d in coh]
        rms2 = [d["rms_blk2"] for d in coh]
        between = [d["between"] for d in coh]
        print(f"\n    within-cloud RMS  blk0:  {summarize(rms0)}")
        print(f"    within-cloud RMS  blk2:  {summarize(rms2)}")
        print(f"    between-centroid:        {summarize(between)}")
        print(f"    silhouette-like ratio:   {summarize(sils)}")
        print("    Interpretation:")
        med_sil = float(np.median(sils))
        if med_sil > 1.5:
            print(f"      → sil≈{med_sil:.2f}: clouds are well-separated regions")
        elif med_sil > 0.8:
            print(f"      → sil≈{med_sil:.2f}: clouds partially separated (scatter > separation)")
        else:
            print(f"      → sil≈{med_sil:.2f}: clouds heavily overlap (just scatter, not regions)")

    # ── Verdict ──
    print("\n" + "─" * 70)
    print("VERDICT (preliminary):")
    flags = []
    # (a)
    if len(persistence["within_seed_dist"]) > 0 and len(persistence["between_seed_null"]) > 0:
        ratio = float(np.median(persistence["within_seed_dist"]) /
                      max(np.median(persistence["between_seed_null"]), 1e-10))
        flags.append(("(a) persistence", "STRONG" if ratio > 0.7 else "PARTIAL" if ratio > 0.4 else "WEAK"))
    # (b)
    if len(er0) > 0 and len(er2) > 0:
        from scipy.stats import wilcoxon
        try:
            _, p_er = wilcoxon(er0, er2)
            _, p_id = wilcoxon(id0, id2)
            min_p = min(p_er, p_id)
            flags.append(("(b) local structure", "STRONG" if min_p < 0.01 else "PARTIAL" if min_p < 0.05 else "WEAK"))
        except Exception:
            pass
    # (c)
    if pd_data:
        med_top1 = float(np.median([d["top1_ratio"] for d in pd_data]))
        flags.append(("(c) non-reducibility", "STRONG" if med_top1 < 0.6 else "PARTIAL" if med_top1 < 0.85 else "WEAK"))
    # (d)
    if coh:
        med_sil = float(np.median([d["silhouette_like"] for d in coh]))
        flags.append(("(d) coherence", "STRONG" if med_sil > 1.5 else "PARTIAL" if med_sil > 0.8 else "WEAK"))

    for name, status in flags:
        marker = "✓" if status == "STRONG" else "~" if status == "PARTIAL" else "✗"
        print(f"    {marker} {name:30s} {status}")
    print("=" * 70)


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, required=True,
                    help="e.g. outputs/phase2_all_20260403_011254")
    ap.add_argument("--schedule", type=str, default="mixed_0_4_0",
                    choices=["mixed_0_4_0", "mixed_0_6_0", "mixed_ramp"])
    ap.add_argument("--n_late_eps", type=int, default=10,
                    help="late episodes per block to compute centroid")
    ap.add_argument("--save_json", type=str, default="",
                    help="optional path to dump raw numbers for further inspection")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        print(f"ERROR: {runs_root} does not exist")
        return

    runs = find_seed_runs(runs_root, args.schedule)
    if not runs:
        print(f"ERROR: no runs found under {runs_root} with schedule {args.schedule}")
        return
    print(f"Found {len(runs)} runs (P1 × P2 seeds)")

    # Load and pre-extract clouds
    runs_data = []
    for r in runs:
        try:
            df = pd.read_parquet(r["traj_path"])
        except Exception as e:
            print(f"  [skip] {r['run_dir']}: {e}")
            continue
        gc = g_cols(df)
        if not gc:
            continue
        clouds = extract_block_clouds(df, gc, n_late_eps=args.n_late_eps)
        runs_data.append({**r, "df": df, "gc": gc, "clouds": clouds})

    if not runs_data:
        print("ERROR: no runs with usable trajectory data")
        return

    print(f"Loaded {len(runs_data)} runs with valid trajectories")
    print(f"g_dim = {len(runs_data[0]['gc'])}")
    blk_counts = {0: 0, 1: 0, 2: 0}
    for r in runs_data:
        for blk in r["clouds"]:
            blk_counts[blk] = blk_counts.get(blk, 0) + 1
    print(f"Block presence: {blk_counts}")

    # Run diagnostics
    persistence = compute_persistence(runs_data)
    structure = compute_local_structure(runs_data)
    probe_dep = compute_probe_dependence(runs_data)
    coherence = compute_coherence(runs_data)

    # Report
    report(persistence, structure, probe_dep, coherence, args.schedule, runs_data)

    # Optional dump
    if args.save_json:
        out = {
            "schedule": args.schedule,
            "n_runs": len(runs_data),
            "persistence": {
                "within_seed_dist": persistence["within_seed_dist"].tolist(),
                "between_seed_null": persistence["between_seed_null"].tolist(),
                "between_seed_blk0": persistence["between_seed_blk0"].tolist(),
                "between_seed_blk2": persistence["between_seed_blk2"].tolist(),
            },
            "structure": {str(k): v for k, v in structure.items()},
            "probe_dep": probe_dep,
            "coherence": coherence,
        }
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(out, f, indent=2, default=float)
        print(f"\n[json] saved to {args.save_json}")


if __name__ == "__main__":
    main()