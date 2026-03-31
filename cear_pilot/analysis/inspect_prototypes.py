#!/usr/bin/env python3
# cear_pilot/analysis/inspect_prototypes.py
"""
Extract prototype centers, well depths, and GRU norms from formation checkpoints.
Overlay prototypes on PCA scatter to see if basins align with valence clusters.

Usage:
    python inspect_prototypes.py --sweep_root outputs/alpha_sweep_p1s0_a05_a25_a50
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.2,
    "font.size": 11,
    "figure.dpi": 120,
})
SAVE_DPI = 180


def parse_label(name):
    m = re.match(r"a([\d.]+)_(SSSS|MMMM)_s(\d+)", name)
    if m:
        return {"alpha": float(m.group(1)), "valence": m.group(2), "seed": int(m.group(3))}
    return None


def detect_g_cols(df):
    cols = [c for c in df.columns if re.match(r"g_\d+$", c)]
    cols.sort(key=lambda x: int(x.split("_")[-1]))
    return cols


def load_table(path):
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def find_file(d, stem):
    for ext in (".parquet", ".csv"):
        p = d / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_root", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="")
    args = ap.parse_args()

    sweep_root = Path(args.sweep_root)
    outdir = Path(args.outdir) if args.outdir else sweep_root / "prototype_analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    runs_dir = sweep_root / "runs"

    # ── 1. Extract prototypes from all checkpoints ──────
    print("Extracting prototypes from checkpoints...")
    proto_records = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        parsed = parse_label(run_dir.name)
        if not parsed:
            continue

        ckpt_path = run_dir / "ckpt_final.pt"
        if not ckpt_path.exists():
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt["agent_state"]

        proto_key = "world.prototype_centers"
        depth_key = "world.raw_well_depth"

        if proto_key not in state:
            continue

        protos = state[proto_key].numpy()  # (n_proto, g_dim)
        well_depth = float(np.exp(state[depth_key].numpy())) if depth_key in state else None

        # GRU weight norms
        gru_ih = state.get("world.gru.weight_ih", None)
        gru_hh = state.get("world.gru.weight_hh", None)
        gru_ih_norm = float(torch.norm(gru_ih).item()) if gru_ih is not None else None
        gru_hh_norm = float(torch.norm(gru_hh).item()) if gru_hh is not None else None

        proto_dist = float(np.linalg.norm(protos[0] - protos[1])) if protos.shape[0] >= 2 else None

        for pi in range(protos.shape[0]):
            rec = {
                "alpha": parsed["alpha"],
                "valence": parsed["valence"],
                "seed": parsed["seed"],
                "proto_id": pi,
                "well_depth": well_depth,
                "proto_distance": proto_dist,
                "gru_ih_norm": gru_ih_norm,
                "gru_hh_norm": gru_hh_norm,
            }
            for di in range(protos.shape[1]):
                rec[f"proto_g_{di}"] = float(protos[pi, di])
            proto_records.append(rec)

    proto_df = pd.DataFrame(proto_records)
    proto_df.to_csv(outdir / "prototype_summary.csv", index=False)
    print(f"  {len(proto_records)} prototype records from {len(proto_records)//2} runs")

    # ── 2. Summary table ────────────────────────────────
    print("\n=== Prototype summary by condition ===")
    for alpha in sorted(proto_df["alpha"].unique()):
        a_df = proto_df[proto_df["alpha"] == alpha]
        dist_med = a_df.groupby("seed")["proto_distance"].first().median()
        depth_med = a_df.groupby("seed")["well_depth"].first().median()
        gru_ih = a_df["gru_ih_norm"].median()
        gru_hh = a_df["gru_hh_norm"].median()
        print(f"  alpha={alpha:.2f}: proto_dist={dist_med:.3f}  "
              f"well_depth={depth_med:.3f}  "
              f"gru_ih={gru_ih:.1f}  gru_hh={gru_hh:.1f}")

    # ── 3. Load trajectory data for PCA ─────────────────
    print("\nLoading trajectories for PCA overlay...")
    traj_frames = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        parsed = parse_label(run_dir.name)
        if not parsed:
            continue
        tp = find_file(run_dir, "traj")
        if not tp:
            continue
        tr = load_table(tp)
        tr["alpha_fixed"] = parsed["alpha"]
        tr["valence"] = parsed["valence"]
        tr["seed"] = parsed["seed"]
        traj_frames.append(tr)

    if not traj_frames:
        print("  No trajectory data found")
        return

    traj = pd.concat(traj_frames, ignore_index=True)
    g_cols = detect_g_cols(traj)
    print(f"  {len(traj)} rows, {len(g_cols)} g dims")

    # Fit shared PCA
    late = traj[traj["t"] > 200]
    g_all = late[g_cols].values
    pca = PCA(n_components=2)
    pca.fit(g_all)
    var_exp = pca.explained_variance_ratio_

    # ── 4. Plot: PCA scatter + prototype overlay ────────
    alphas = sorted(traj["alpha_fixed"].unique())
    n_alpha = len(alphas)

    fig, axes = plt.subplots(1, n_alpha, figsize=(5 * n_alpha, 5))
    if n_alpha == 1:
        axes = [axes]

    proto_g_cols = [f"proto_g_{i}" for i in range(len(g_cols))]

    for ai, alpha in enumerate(alphas):
        ax = axes[ai]
        a_late = late[late["alpha_fixed"] == alpha]

        # Scatter g points
        for val, color in [("SSSS", "#185FA5"), ("MMMM", "#D85A30")]:
            v_data = a_late[a_late["valence"] == val]
            if len(v_data) > 2000:
                v_data = v_data.sample(2000, random_state=42)
            g_proj = pca.transform(v_data[g_cols].values)
            ax.scatter(g_proj[:, 0], g_proj[:, 1],
                       c=color, alpha=0.15, s=6, edgecolors="none", label=val)

        # Overlay prototypes
        a_proto = proto_df[proto_df["alpha"] == alpha]
        for seed in a_proto["seed"].unique():
            s_proto = a_proto[a_proto["seed"] == seed]
            for _, row in s_proto.iterrows():
                pvec = np.array([row[c] for c in proto_g_cols]).reshape(1, -1)
                pp = pca.transform(pvec)[0]
                pid = int(row["proto_id"])
                marker = "^" if pid == 0 else "v"
                ax.scatter(pp[0], pp[1], c="black", s=80, marker=marker,
                           edgecolors="white", linewidth=1, zorder=10, alpha=0.7)

        # Legend entries for prototypes
        ax.scatter([], [], c="black", s=80, marker="^", label="Proto 0")
        ax.scatter([], [], c="black", s=80, marker="v", label="Proto 1")

        ax.set_xlabel(f"PC1 ({var_exp[0]:.0%})")
        if ai == 0:
            ax.set_ylabel(f"PC2 ({var_exp[1]:.0%})")
        ax.set_title(f"alpha={alpha}")
        ax.legend(markerscale=1.5, framealpha=0.7, fontsize=8)

    fig.suptitle("PCA of g-space with learned prototype positions",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "pca_with_prototypes.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] pca_with_prototypes.png")

    # ── 5. Plot: Prototype distance by alpha ────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Panel A: proto distance
    ax = axes[0]
    for alpha in alphas:
        a_df = proto_df[proto_df["alpha"] == alpha]
        dists = a_df.groupby("seed")["proto_distance"].first().values
        ax.bar(alphas.index(alpha), np.median(dists),
               yerr=[[np.median(dists) - np.percentile(dists, 25)],
                     [np.percentile(dists, 75) - np.median(dists)]],
               color=["#185FA5", "#1D9E75", "#D85A30", "#E24B4A"][min(ai, 3)],
               alpha=0.7, capsize=4)
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"alpha={a}" for a in alphas])
    ax.set_ylabel("Proto 0-1 distance")
    ax.set_title("Prototype separation")

    # Panel B: well depth
    ax = axes[1]
    for ai, alpha in enumerate(alphas):
        a_df = proto_df[proto_df["alpha"] == alpha]
        depths = a_df.groupby("seed")["well_depth"].first().values
        ax.bar(ai, np.median(depths),
               yerr=[[np.median(depths) - np.percentile(depths, 25)],
                     [np.percentile(depths, 75) - np.median(depths)]],
               color=["#185FA5", "#1D9E75", "#D85A30", "#E24B4A"][min(ai, 3)],
               alpha=0.7, capsize=4)
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"alpha={a}" for a in alphas])
    ax.set_ylabel("Well depth (learned)")
    ax.set_title("Attractor depth")

    # Panel C: GRU weight norms
    ax = axes[2]
    width = 0.35
    for ai, alpha in enumerate(alphas):
        a_df = proto_df[proto_df["alpha"] == alpha]
        ih = a_df.groupby("seed")["gru_ih_norm"].first().median()
        hh = a_df.groupby("seed")["gru_hh_norm"].first().median()
        ax.bar(ai - width/2, ih, width, color="#534AB7", alpha=0.7,
               label="GRU input-hidden" if ai == 0 else "")
        ax.bar(ai + width/2, hh, width, color="#D4537E", alpha=0.7,
               label="GRU hidden-hidden" if ai == 0 else "")
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"alpha={a}" for a in alphas])
    ax.set_ylabel("Weight norm")
    ax.set_title("GRU learned weights")
    ax.legend(fontsize=8)

    fig.suptitle("Learned perspective structure by alpha condition", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "learned_structure_summary.png", dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] learned_structure_summary.png")

    print(f"\nDone. Output in: {outdir}")


if __name__ == "__main__":
    main()