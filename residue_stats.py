# -*- coding: utf-8 -*-
"""
residue_stats.py — analysis-only statistics for the journal revision.
Spyder-friendly version: edit the CONFIG block below and press F5.

Computes, from EXISTING phase-2 run outputs (no new training):

  1. RESIDUE (Fig. 3 support): per-run block means of alpha
     (blocks: episodes 0-49 / 50-99 / 100-149), the residue statistic
         R = dAlpha_mixed - dAlpha_baseline,  dAlpha = mean_a(B0) - mean_a(B2)
     with a hierarchical bootstrap 95% CI (resample p1 seeds, then p2
     seeds within p1) and a cluster-respecting permutation test
     (condition labels shuffled within each p1 seed).

  2. GAMMA REORGANIZATION (Fig. 4c support): per-dimension mean gamma
     over the final 10 episodes of Block 0 vs Block 2 in mixed runs;
     reports across-run sign-consistency of the change, jointly with
     effect magnitude, so that near-zero dimensions cannot masquerade
     as "consistent".

  3. BEHAVIORAL STABILITY (R3's "why doesn't gross behavior change"):
     zone-occupancy total-variation distance between Block 0 and
     Block 2, and mean column position per block.

Expected layout (as produced by run_phase2.sh):
  ROOT/from_p1_s{P}/mixed/seed{S}/mixed_0_4_0/traj.parquet
  ROOT/from_p1_s{P}/sweep/seed{S}/nperturb0/traj.parquet   (Baseline)
  ROOT/from_p1_s{P}/sweep/seed{S}/nperturb4/traj.parquet   (Persistent)

After running in Spyder, these stay in the namespace for the Variable
Explorer:  A (per-run alpha blocks), G (per-dim gamma), B (behavior),
           C (gamma consistency table), RESULTS (summary dict)

Requires: numpy, pandas, and pyarrow (or fastparquet) for .parquet input.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================================
# CONFIG — edit these, then run (F5 in Spyder)
# =====================================================================

# Root of a phase-2 output tree, i.e. the directory that contains the
# from_p1_s0 / from_p1_s1 / ... folders.
#   Windows example: r"D:\cearlab\outputs\phase2_all_20260401_120000"
#   macOS / Linux  : "/home/me/cearlab/outputs/phase2_all_20260401_120000"
ROOT = r"outputs/phase2_all_20260403_011254"

# Where to write the CSVs. Empty string -> ROOT/revision_stats
OUTDIR = ""

BLOCK_LEN = 50        # episodes per block (150 episodes / 3 blocks)
LATE_WINDOW = 10      # trailing episodes per block used for gamma
N_BOOT = 10000        # hierarchical bootstrap resamples
N_PERM = 10000        # permutation iterations
RANDOM_SEED = 0

# Gamma dimensions are reported as "reorganized" only if they pass BOTH:
CONSISTENCY_MIN = 0.80    # fraction of runs agreeing on direction
MAGNITUDE_MIN = 0.02      # |median delta gamma| floor

SAVE_CSV = True

# =====================================================================


RNG = np.random.default_rng(RANDOM_SEED)


# ----------------------------------------------------------------- I/O

def load_traj(d: Path):
    """Load a run's trajectory table, parquet preferred."""
    for name in ("traj.parquet", "traj.csv"):
        p = d / name
        if p.exists():
            try:
                return pd.read_parquet(p) if name.endswith("parquet") else pd.read_csv(p)
            except ImportError:
                print(f"  [warn] cannot read {p.name}: install pyarrow "
                      f"(conda install pyarrow) or export runs as CSV")
                return None
            except Exception as e:
                print(f"  [warn] {p}: {e}")
    return None


def discover_runs(root: Path):
    """Return dict: condition -> list of (p1_seed, p2_seed, run_dir)."""
    out = {"mixed": [], "baseline": [], "persistent": []}
    for p1_dir in sorted(root.glob("from_p1_s*")):
        m = re.match(r"from_p1_s(\d+)", p1_dir.name)
        if not m:
            continue
        p1 = int(m.group(1))
        for pattern, key in (("mixed/seed*/mixed_0_4_0", "mixed"),
                             ("sweep/seed*/nperturb0", "baseline"),
                             ("sweep/seed*/nperturb4", "persistent")):
            for sd in sorted(p1_dir.glob(pattern)):
                m2 = re.search(r"seed(\d+)", str(sd))
                if m2:
                    out[key].append((p1, int(m2.group(1)), sd))
    return out


# ------------------------------------------------------- per-run measures

def block_of(ep, block_len: int):
    return (np.asarray(ep) // block_len).astype(int)


def run_alpha_blocks(df: pd.DataFrame, block_len: int) -> dict:
    if "alpha" not in df.columns:
        return {}
    b = block_of(df["episode"].to_numpy(), block_len)
    s = pd.Series(df["alpha"].to_numpy()).groupby(b).mean()
    return {int(k): float(v) for k, v in s.items()}


def run_gamma_late(df: pd.DataFrame, block_len: int, late: int):
    gcols = sorted([c for c in df.columns if re.match(r"gamma_\d+$", c)],
                   key=lambda c: int(c.split("_")[1]))
    if not gcols:
        return None, None
    ep = df["episode"].to_numpy()
    w0 = (ep >= block_len - late) & (ep < block_len)
    w2 = (ep >= 3 * block_len - late) & (ep < 3 * block_len)
    if w0.sum() == 0 or w2.sum() == 0:
        return None, None
    return (df.loc[w0, gcols].mean().to_numpy(),
            df.loc[w2, gcols].mean().to_numpy())


def run_behavior_blocks(df: pd.DataFrame, block_len: int) -> dict:
    if "zone_id" not in df.columns:
        return {}
    b = block_of(df["episode"].to_numpy(), block_len)
    res = {}
    for blk in np.unique(b):
        sel = df.loc[b == blk]
        vc = sel["zone_id"].value_counts(normalize=True)
        occ = np.array([float(vc.get(z, 0.0)) for z in range(5)])
        res[int(blk)] = {"occ": occ,
                         "mean_x": float(sel["x"].mean()) if "x" in sel else np.nan}
    return res


# --------------------------------------------- hierarchical stats (numpy)

def to_clusters(frame: pd.DataFrame, col: str):
    """Split a condition's per-run values into one array per p1 seed."""
    return {int(p1): sub[col].to_numpy(dtype=float)
            for p1, sub in frame.groupby("p1")}


def hier_median(clusters: dict) -> float:
    """Median across p2 within each p1, then median across p1."""
    inner = [np.median(v) for v in clusters.values() if len(v)]
    return float(np.median(inner)) if inner else np.nan


def hier_bootstrap_diff(cl_a: dict, cl_b: dict, n_boot: int):
    """Bootstrap CI for hier(a) - hier(b): resample p1, then runs within p1."""
    keys_a, keys_b = list(cl_a), list(cl_b)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        pa = RNG.choice(keys_a, size=len(keys_a), replace=True)
        pb = RNG.choice(keys_b, size=len(keys_b), replace=True)
        ma = [np.median(RNG.choice(cl_a[k], size=cl_a[k].size, replace=True)) for k in pa]
        mb = [np.median(RNG.choice(cl_b[k], size=cl_b[k].size, replace=True)) for k in pb]
        boots[i] = np.median(ma) - np.median(mb)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi), boots


def cluster_permutation_p(cl_a: dict, cl_b: dict, n_perm: int):
    """Permutation test for hier(a) - hier(b) != 0, shuffling condition
    labels within each p1 seed (runs are exchangeable within p1 under H0).
    Only p1 seeds present in BOTH conditions contribute."""
    shared = sorted(set(cl_a) & set(cl_b))
    if not shared:
        return np.nan, 0
    pooled, n_a = {}, {}
    for k in shared:
        pooled[k] = np.concatenate([cl_a[k], cl_b[k]])
        n_a[k] = cl_a[k].size
    obs = (np.median([np.median(cl_a[k]) for k in shared])
           - np.median([np.median(cl_b[k]) for k in shared]))
    count = 0
    for _ in range(n_perm):
        ma, mb = [], []
        for k in shared:
            perm = RNG.permutation(pooled[k])
            ma.append(np.median(perm[:n_a[k]]))
            mb.append(np.median(perm[n_a[k]:]))
        if abs(np.median(ma) - np.median(mb)) >= abs(obs) - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1), len(shared)


# --------------------------------------------------------------- main

def run_analysis(root, outdir="", block_len=BLOCK_LEN, late_window=LATE_WINDOW,
                 n_boot=N_BOOT, n_perm=N_PERM, save_csv=SAVE_CSV):
    """Run the full analysis. Returns (A, G, B, C, RESULTS)."""
    root = Path(root).expanduser()
    if not root.exists():
        raise FileNotFoundError(
            f"ROOT does not exist: {root}\n"
            f"Edit the CONFIG block at the top of this file. It should point at "
            f"the directory containing from_p1_s0, from_p1_s1, ..."
        )
    outdir = Path(outdir).expanduser() if outdir else root / "revision_stats"
    outdir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(root)
    for k, v in runs.items():
        print(f"[discover] {k}: {len(v)} runs")
    if not any(runs.values()):
        raise RuntimeError(
            f"No runs found under {root}. Expected e.g. "
            f"{root / 'from_p1_s0' / 'mixed' / 'seed0' / 'mixed_0_4_0'}"
        )

    rows, gamma_rows, behav_rows = [], [], []
    for cond, lst in runs.items():
        for p1, p2, d in lst:
            df = load_traj(d)
            if df is None:
                print(f"  [skip] no traj in {d}")
                continue
            ab = run_alpha_blocks(df, block_len)
            if {0, 2} <= set(ab):
                rows.append({"cond": cond, "p1": p1, "p2": p2,
                             "a_b0": ab[0], "a_b1": ab.get(1, np.nan),
                             "a_b2": ab[2], "d_alpha": ab[0] - ab[2]})
            if cond == "mixed":
                g0, g2 = run_gamma_late(df, block_len, late_window)
                if g0 is not None:
                    for i, (v0, v2) in enumerate(zip(g0, g2)):
                        gamma_rows.append({"p1": p1, "p2": p2, "dim": i,
                                           "gamma_b0": v0, "gamma_b2": v2,
                                           "d_gamma": v2 - v0})
                bb = run_behavior_blocks(df, block_len)
                if {0, 2} <= set(bb):
                    tv = 0.5 * float(np.abs(bb[0]["occ"] - bb[2]["occ"]).sum())
                    behav_rows.append({"p1": p1, "p2": p2, "tv_occ_b0_b2": tv,
                                       "mean_x_b0": bb[0]["mean_x"],
                                       "mean_x_b2": bb[2]["mean_x"]})

    A = pd.DataFrame(rows)
    G = pd.DataFrame(gamma_rows)
    B = pd.DataFrame(behav_rows)
    C = pd.DataFrame()
    RESULTS = {}

    if A.empty:
        raise RuntimeError("No usable alpha data — check that runs were saved "
                           "with --save_traj (traj.parquet must contain 'alpha').")
    if save_csv:
        A.to_csv(outdir / "alpha_blocks_per_run.csv", index=False)

    # ---- 1. alpha blocks + residue
    print("\n=== Hierarchical block means of alpha (median-of-medians) ===")
    summary = []
    clusters = {}
    for cond in ("baseline", "mixed", "persistent"):
        sub = A[A["cond"] == cond]
        if sub.empty:
            continue
        clusters[cond] = {c: to_clusters(sub, c) for c in ("a_b0", "a_b2", "d_alpha")}
        s = {c: hier_median(clusters[cond][c]) for c in ("a_b0", "a_b2", "d_alpha")}
        print(f"  {cond:11s} aB0={s['a_b0']:.3f}  aB2={s['a_b2']:.3f}  "
              f"dAlpha={s['d_alpha']:.3f}  (n={len(sub)})")
        summary.append({"cond": cond, **s, "n_runs": len(sub)})

    for ref in ("baseline", "persistent"):
        if "mixed" in clusters and ref in clusters:
            ca = clusters["mixed"]["d_alpha"]
            cb = clusters[ref]["d_alpha"]
            R = hier_median(ca) - hier_median(cb)
            print(f"\n=== RESIDUE: dAlpha(mixed) - dAlpha({ref}) ===")
            print(f"  bootstrapping ({n_boot})...", end="", flush=True)
            lo, hi, _ = hier_bootstrap_diff(ca, cb, n_boot)
            print(f" permuting ({n_perm})...", end="", flush=True)
            p, n_shared = cluster_permutation_p(ca, cb, n_perm)
            print(" done")
            excl = "excludes" if (lo > 0 or hi < 0) else "includes"
            print(f"  R = {R:.4f}   95% CI [{lo:.4f}, {hi:.4f}] ({excl} zero)"
                  f"   perm p = {p:.4f}  (on {n_shared} shared p1 clusters)")
            summary.append({"cond": f"RESIDUE_mixed_vs_{ref}", "d_alpha": R,
                            "ci_lo": lo, "ci_hi": hi, "perm_p": p})
            RESULTS[f"residue_vs_{ref}"] = {"R": R, "ci": (lo, hi), "p": p}

    if save_csv:
        pd.DataFrame(summary).to_csv(outdir / "residue_stats_summary.csv", index=False)

    # ---- 2. gamma sign-consistency (joint with magnitude)
    if not G.empty:
        print("\n=== Gamma reorganization (B2 - B0) across runs ===")
        cons = []
        for dim, sub in G.groupby("dim"):
            d = sub["d_gamma"].to_numpy()
            frac_pos = float((d > 0).mean())
            cons.append({"dim": int(dim),
                         "median_d_gamma": float(np.median(d)),
                         "abs_median": abs(float(np.median(d))),
                         "sign_consistency": max(frac_pos, 1.0 - frac_pos),
                         "n_runs": int(d.size)})
        C = pd.DataFrame(cons).sort_values(
            ["sign_consistency", "abs_median"], ascending=False).reset_index(drop=True)
        C["passes"] = ((C["sign_consistency"] >= CONSISTENCY_MIN)
                       & (C["abs_median"] >= MAGNITUDE_MIN))
        if save_csv:
            C.to_csv(outdir / "gamma_consistency.csv", index=False)
        for _, r in C.head(6).iterrows():
            flag = "*" if r["passes"] else " "
            print(f" {flag} dim {int(r['dim']):2d}: median dGamma={r['median_d_gamma']:+.4f}"
                  f"  consistency={r['sign_consistency']:.2f}")
        n_pass = int(C["passes"].sum())
        print(f"  dimensions passing BOTH consistency >= {CONSISTENCY_MIN:.2f} "
              f"and |median| >= {MAGNITUDE_MIN:.3f}: {n_pass}/{len(C)}")
        print("  -> report this count in the Fig. 4(c) robustness sentence.")
        RESULTS["gamma_n_pass"] = n_pass
        RESULTS["gamma_n_dims"] = len(C)

    # ---- 3. behavioral stability
    if not B.empty:
        if save_csv:
            B.to_csv(outdir / "behavior_stability.csv", index=False)
        tv_med = hier_median(to_clusters(B, "tv_occ_b0_b2"))
        dx = B["mean_x_b2"] - B["mean_x_b0"]
        print("\n=== Behavioral stability (mixed runs, Block 0 vs Block 2) ===")
        print(f"  zone-occupancy total-variation distance: median = {tv_med:.3f} "
              f"(0 = identical, 1 = disjoint)")
        print(f"  mean column shift: median = {float(dx.median()):+.2f} cells "
              f"(IQR {float(dx.quantile(.25)):+.2f} .. {float(dx.quantile(.75)):+.2f})")
        RESULTS["tv_occupancy"] = tv_med
        RESULTS["mean_x_shift"] = float(dx.median())

    print(f"\n[done] CSVs in {outdir}" if save_csv else "\n[done]")
    return A, G, B, C, RESULTS


# Run on F5 in Spyder; also works as `python residue_stats.py [ROOT]`
if len(sys.argv) > 1 and not sys.argv[0].endswith(("spyder-script.py", "ipykernel_launcher.py")):
    ROOT = sys.argv[1]

A, G, B, C, RESULTS = run_analysis(
    ROOT, OUTDIR, BLOCK_LEN, LATE_WINDOW, N_BOOT, N_PERM, SAVE_CSV)
