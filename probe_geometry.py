# -*- coding: utf-8 -*-
"""
probe_geometry.py — cross-run statistics for the probe panel (Fig. probe(a)).
Spyder-friendly: edit CONFIG, press F5.

WHY THIS EXISTS
---------------
The paper currently claims, from one illustrative run, that the encoding
under g2 shows "visibly larger" covariance than under g=0, and reads this
as reorganization rather than a uniform shift. Two problems:

  (1) It is a single-run eyeball claim, whereas the gating result is now
      backed by all 30 runs. This script makes panel (a) match.

  (2) Total variance alone cannot distinguish reorganization from gain.
      FiLM is diagonal: z = (1+gamma) * z_raw + beta, so across probes
          cov(z) = D C D,   D = diag(1+gamma),  C = cov(z_raw).
      If every gamma_i were equal (uniform amplification), total variance
      would grow with NO reorganization whatsoever. A larger variance
      ratio is therefore consistent with the null the sentence means to
      reject.

  So this script reports the gain metric AND two scale-invariant shape
  metrics that a uniform gain cannot move:

    total_var_ratio : trace(cov(z_g2)) / trace(cov(z_zero))   [gain]
    participation ratio  PR = (sum lambda)^2 / sum(lambda^2)  [shape]
        = effective number of dimensions the encoding spreads over.
        Invariant to overall scaling; changes only if variance is
        redistributed ACROSS dimensions.
    pc1_angle : angle (deg) between the leading principal axis under
        g2 and under g=0.                                     [shape]
        Exactly 0 under uniform gain; nonzero means the dominant
        direction of the encoding has rotated.

Reads the probe_representation.csv files that run_phase2.sh already
produced for every run — no retraining, no checkpoints needed.

Expected layout:
  ROOT/from_p1_s{P}/mixed/seed{S}/mixed_0_4_0/analysis/probe_representation.csv

Requires: numpy, pandas.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================================
# CONFIG
# =====================================================================

# Same ROOT as residue_stats.py (the folder containing from_p1_s0, ...)
ROOT = r"outputs/phase2_all_20260403_011254"

OUTDIR = ""            # "" -> ROOT/revision_stats

# Which g condition is the "post-perturbation" state. probe_representation
# labels blocks as f"blk{block}_nP{nP}", so Block 2 of a 0->4->0 run is
# "blk2_nP0". Left as a prefix match so it works if the naming differs.
G2_PREFIX = "blk2"
G_NULL_LABEL = "g_zero"

SAVE_CSV = True

# =====================================================================


def find_probe_csvs(root: Path):
    """Return list of (p1, p2, csv_path) for mixed-history runs."""
    out = []
    for p in sorted(root.glob("from_p1_s*/mixed/seed*/mixed_0_4_0/analysis/probe_representation.csv")):
        m1 = re.search(r"from_p1_s(\d+)", str(p))
        m2 = re.search(r"seed(\d+)", str(p))
        if m1 and m2:
            out.append((int(m1.group(1)), int(m2.group(1)), p))
    return out


def z_matrix(df: pd.DataFrame, label_sel) -> np.ndarray | None:
    """Probes x z_dim matrix for one g condition."""
    zcols = sorted([c for c in df.columns if re.match(r"z_t_\d+$", c)],
                   key=lambda c: int(c.split("_")[-1]))
    if not zcols:
        return None
    sub = df[label_sel]
    if len(sub) < 3:
        return None
    return sub[zcols].to_numpy(dtype=float)


def geometry(Z: np.ndarray) -> dict:
    """Total variance, participation ratio, leading principal axis."""
    C = np.cov(Z, rowvar=False)
    lam = np.linalg.eigvalsh(C)
    lam = np.clip(lam, 0, None)
    tot = float(lam.sum())
    pr = float(tot**2 / np.sum(lam**2)) if np.sum(lam**2) > 0 else np.nan
    w, v = np.linalg.eigh(C)
    pc1 = v[:, int(np.argmax(w))]
    return {"total_var": tot, "pr": pr, "pc1": pc1}


def angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    c = abs(float(np.dot(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
    return float(np.degrees(np.arccos(np.clip(c, 0.0, 1.0))))


def hier_median(df: pd.DataFrame, col: str) -> float:
    return float(df.groupby("p1")[col].median().median())


def run_analysis(root, outdir="", save_csv=SAVE_CSV):
    root = Path(root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"ROOT does not exist: {root}")
    outdir = Path(outdir).expanduser() if outdir else root / "revision_stats"
    outdir.mkdir(parents=True, exist_ok=True)

    csvs = find_probe_csvs(root)
    print(f"[discover] probe_representation.csv files: {len(csvs)}")
    if not csvs:
        raise RuntimeError(
            "None found. Expected e.g.\n  "
            f"{root / 'from_p1_s0/mixed/seed0/mixed_0_4_0/analysis/probe_representation.csv'}"
        )

    rows = []
    labels_seen = set()
    for p1, p2, path in csvs:
        df = pd.read_csv(path)
        if "g_label" not in df.columns:
            print(f"  [skip] no g_label column: {path}")
            continue
        labels_seen.update(df["g_label"].unique())

        sel_null = df["g_label"] == G_NULL_LABEL
        sel_g2 = df["g_label"].astype(str).str.startswith(G2_PREFIX)
        Zn, Z2 = z_matrix(df, sel_null), z_matrix(df, sel_g2)
        if Zn is None or Z2 is None:
            print(f"  [skip] missing condition in {path.parent.parent.name}")
            continue

        gn, g2 = geometry(Zn), geometry(Z2)
        rows.append({
            "p1": p1, "p2": p2,
            "total_var_null": gn["total_var"], "total_var_g2": g2["total_var"],
            "total_var_ratio": g2["total_var"] / gn["total_var"] if gn["total_var"] > 0 else np.nan,
            "pr_null": gn["pr"], "pr_g2": g2["pr"],
            "pr_ratio": g2["pr"] / gn["pr"] if gn["pr"] > 0 else np.nan,
            "pc1_angle_deg": angle_deg(gn["pc1"], g2["pc1"]),
            "n_probes": Zn.shape[0],
        })

    P = pd.DataFrame(rows)
    if P.empty:
        print(f"  g_labels present in files: {sorted(labels_seen)}")
        raise RuntimeError("No usable runs — check G2_PREFIX against the labels above.")
    if save_csv:
        P.to_csv(outdir / "probe_geometry.csv", index=False)

    print(f"\n=== Probe encoding geometry, g2 vs null, across {len(P)} runs ===")

    def rep(col, fmt="{:.3f}"):
        med = hier_median(P, col)
        lo, hi = P[col].quantile(.25), P[col].quantile(.75)
        print(f"  {col:18s} median {fmt.format(med)}   IQR [{fmt.format(lo)}, {fmt.format(hi)}]")
        return med

    print("\n  [gain — cannot distinguish reorganization from uniform amplification]")
    tv = rep("total_var_ratio")
    n_gt1 = int((P.total_var_ratio > 1).sum())
    print(f"    runs with ratio > 1: {n_gt1}/{len(P)}")

    print("\n  [shape — invariant to uniform gain]")
    prn = rep("pr_null"); pr2 = rep("pr_g2"); rep("pr_ratio")
    ang = rep("pc1_angle_deg", "{:.1f}")
    n_rot = int((P.pc1_angle_deg > 5).sum())
    print(f"    runs with leading-axis rotation > 5 deg: {n_rot}/{len(P)}")

    print("\n--- paste-ready ---")
    print(f"  total-variance ratio {tv:.2f}; effective dimensionality "
          f"{prn:.1f} -> {pr2:.1f}; leading principal axis rotated by "
          f"{ang:.0f} deg (medians across {len(P)} runs)")
    print(f"\n[done] {outdir / 'probe_geometry.csv'}" if save_csv else "\n[done]")
    return P


P = run_analysis(ROOT, OUTDIR, SAVE_CSV)
