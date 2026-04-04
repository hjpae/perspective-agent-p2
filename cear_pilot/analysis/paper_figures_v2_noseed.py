#!/usr/bin/env python3
"""
Paper figures for ALIFE 2026.

Fig 2: Mixed 0→4→0 — schedule strip + alpha + ||g|| + |γ|
Fig 3: Perspective shapes perception — (a) z_shift 8x, (b) per-dim reshape, (c) perturbation response
Fig 4: Ablation — alpha dynamics + PE + ||g||
Fig S1: Sweep dose-response
Fig S2: Per-block gating

Usage:
    python -m cear_pilot.analysis.paper_figures \
        --sweep_root outputs/sweep \
        --mixed_root outputs/mixed \
        --ablation_root outputs/ablation \
        --outdir paper_figures
"""

from __future__ import annotations
import argparse, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

DPI = 300
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.12, "grid.linewidth": 0.3,
    "font.size": 9, "font.family": "sans-serif",
    "axes.linewidth": 0.5, "axes.labelsize": 10, "axes.titlesize": 10,
    "legend.fontsize": 8,
})

C_ALPHA = "#D85A30"
C_GNORM = "#0F6E56"
C_GAMMA = "#993C1D"
C_AD = "#534AB7"
C_FX = "#0F6E56"
C_FA = "#D85A30"
C_PE = "#534AB7"
ZC = ["#E24B4A", "#D85A30", "#888780", "#1D9E75", "#0F6E56"]

def savefig(fig, path):
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig); print(f"  [fig] {path.name}")

def load_traj(d):
    for n in ["traj.parquet", "traj.csv"]:
        p = d / n
        if p.exists():
            try: return pd.read_parquet(p) if n.endswith(".parquet") else pd.read_csv(p)
            except:
                try: return pd.read_csv(p)
                except: pass
    return None

def load_probe(d):
    p = d / "analysis" / "probe_representation.csv"
    return pd.read_csv(p) if p.exists() else None

def zt_cols(df):
    return sorted([c for c in df.columns if re.match(r"z_t_\d+$", c)],
                  key=lambda x: int(x.split("_")[-1]))

def zraw_cols(df):
    return sorted([c for c in df.columns if re.match(r"z_raw_\d+$", c)],
                  key=lambda x: int(x.split("_")[-1]))

def gm_cols(df):
    return sorted([c for c in df.columns if re.match(r"gamma_\d+$", c)],
                  key=lambda x: int(x.split("_")[-1]))

def late(df, n=20):
    mx = df["episode"].max()
    return df[df["episode"] >= max(0, mx - n + 1)]


# ════════════════════════════════════════════════════
# Fig 2: Mixed 0→4→0 — stacked time series
# ════════════════════════════════════════════════════

def fig2_mixed_history(mixed_root, outdir):
    """4-row stacked: block schedule + alpha + ||g|| + |γ|, with block boundaries."""
    # Collect all 0_4_0 runs (multi-seed)
    dfs = []
    for d in sorted(mixed_root.iterdir()):
        if d.is_dir() and "0_4_0" in d.name:
            tdf = load_traj(d)
            if tdf is not None:
                dfs.append(tdf)
    if not dfs:
        print("  [skip] fig2: no mixed 0_4_0 traj"); return

    ref = dfs[0]
    # Block boundaries
    blk_info = []
    if "block_id" in ref.columns:
        ep_blk = ref.groupby("episode")[["block_id", "n_perturb_setting"]].first()
        changes = ep_blk[ep_blk["block_id"] != ep_blk["block_id"].shift(1)]
        for ep_idx in changes.index:
            blk_info.append((ep_idx, int(changes.loc[ep_idx, "n_perturb_setting"])))

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True,
                             gridspec_kw={"height_ratios": [0.6, 1, 1, 1], "hspace": 0.08})

    # Row 0: Block schedule strip
    ax = axes[0]
    if "n_perturb_setting" in ref.columns:
        ep_nP = ref.groupby("episode")["n_perturb_setting"].first()
        ax.fill_between(ep_nP.index, 0, ep_nP.values, step="post", color="#ddd", alpha=0.6)
        ax.plot(ep_nP.index, ep_nP.values, drawstyle="steps-post", color="#888", lw=1.5)
    ax.set_ylabel("nP", fontsize=9)
    ax.set_ylim(-0.3, max(ref.get("n_perturb_setting", pd.Series([4])).max() + 0.5, 5))
    ax.set_title("Block schedule", fontsize=10, pad=4)
    ax.set_yticks([0, 4])

    # Rows 1-3: alpha, ||g||, |γ|
    metrics = [
        ("alpha", "α (plasticity)", C_ALPHA),
        ("g_norm", "||g||", C_GNORM),
    ]
    # Add |γ| if available
    gc = gm_cols(ref) if gm_cols(ref) else []

    for i, (metric, ylabel, color) in enumerate(metrics):
        ax = axes[i + 1]
        for df in dfs:
            if metric in df.columns:
                ep_m = df.groupby("episode")[metric].mean()
                ax.plot(ep_m.index, ep_m.values, lw=0.3, alpha=0.2, color=color)
        # Median
        if len(dfs) > 1 and metric in dfs[0].columns:
            all_m = pd.concat([df.groupby("episode")[metric].mean() for df in dfs], axis=1)
            med = all_m.median(axis=1)
            q1, q3 = all_m.quantile(0.25, axis=1), all_m.quantile(0.75, axis=1)
            ax.fill_between(med.index, q1, q3, color=color, alpha=0.12)
            ax.plot(med.index, med.values, lw=2, color=color)
        elif metric in dfs[0].columns:
            ep_m = dfs[0].groupby("episode")[metric].mean()
            ax.plot(ep_m.index, ep_m.values, lw=1.5, color=color)
        ax.set_ylabel(ylabel, fontsize=9)
        for bc, nP in blk_info:
            ax.axvline(bc, color="#aaa", lw=0.6, ls="--", alpha=0.5)

    # Row 3: |γ|
    ax = axes[3]
    if gc:
        for df in dfs:
            gc_df = gm_cols(df)
            if gc_df:
                ep_gabs = df.groupby("episode")[gc_df].apply(lambda x: np.abs(x.values).mean())
                ax.plot(ep_gabs.index, ep_gabs.values, lw=0.3, alpha=0.2, color=C_GAMMA)
        if len(dfs) > 1:
            all_gabs = pd.concat([
                df.groupby("episode")[gm_cols(df)].apply(lambda x: np.abs(x.values).mean())
                for df in dfs if gm_cols(df)
            ], axis=1)
            med = all_gabs.median(axis=1)
            ax.plot(med.index, med.values, lw=2, color=C_GAMMA)
        else:
            gc_df = gm_cols(dfs[0])
            if gc_df:
                ep_gabs = dfs[0].groupby("episode")[gc_df].apply(lambda x: np.abs(x.values).mean())
                ax.plot(ep_gabs.index, ep_gabs.values, lw=1.5, color=C_GAMMA)
    ax.set_ylabel("Mean |γ|", fontsize=9)
    ax.set_xlabel("Episode")
    for bc, nP in blk_info:
        ax.axvline(bc, color="#aaa", lw=0.6, ls="--", alpha=0.5)

    # Block labels on top panel
    for bc, nP in blk_info:
        axes[0].axvline(bc, color="#aaa", lw=0.6, ls="--", alpha=0.5)

    savefig(fig, outdir / "fig2_mixed_history.png")


# ════════════════════════════════════════════════════
# Fig 3: Perspective shapes perception
# ════════════════════════════════════════════════════

def fig3_perspective_perception(mixed_root, sweep_root, outdir):
    """(a) z_shift 8x bar, (b) per-dim z_t reshape, (c) perturbation response."""

    # Find probe data
    probe_df = None
    probe_source = None
    for root in [mixed_root, sweep_root]:
        if root is None or not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir(): continue
            pdf = load_probe(d)
            if pdf is not None:
                labels = pdf["g_label"].unique()
                # Prefer one with block labels
                if any("blk" in str(l) for l in labels):
                    probe_df = pdf; probe_source = d.name; break
                elif probe_df is None:
                    probe_df = pdf; probe_source = d.name
        if probe_df is not None and any("blk" in str(l) for l in probe_df["g_label"].unique()):
            break

    if probe_df is None:
        print("  [skip] fig3: no probe data"); return

    # Find perturbation response traj (from sweep nP=4)
    perturb_traj = None
    if sweep_root and sweep_root.exists():
        for d in sorted(sweep_root.iterdir()):
            if d.is_dir() and "nperturb4" in d.name:
                perturb_traj = load_traj(d)
                if perturb_traj is not None: break

    zc = zt_cols(probe_df)
    zr = zraw_cols(probe_df)
    n_z = len(zc)
    if n_z == 0:
        print("  [skip] fig3: no z_t columns"); return

    # Determine g_trained and g_zero labels
    g_labels = sorted(probe_df["g_label"].unique())
    g_trained_candidates = [l for l in g_labels if l != "g_zero"]
    if not g_trained_candidates:
        print("  [skip] fig3: no trained g labels"); return

    # Pick the one with highest z_shift as "g_trained"
    best_label = max(g_trained_candidates,
                     key=lambda l: probe_df[probe_df["g_label"] == l]["z_shift"].mean())

    has_panel_c = perturb_traj is not None
    n_cols = 3 if has_panel_c else 2
    width_ratios = [1, 1.2, 1.3] if has_panel_c else [1, 1.2]

    fig = plt.figure(figsize=(4.5 * n_cols, 4))
    gs = gridspec.GridSpec(1, n_cols, width_ratios=width_ratios, wspace=0.35)

    # ── (a) z_shift: g_trained vs g_zero ──
    ax = fig.add_subplot(gs[0])
    trained_shift = probe_df[probe_df["g_label"] == best_label]["z_shift"].values
    zero_shift = probe_df[probe_df["g_label"] == "g_zero"]["z_shift"].values

    bar_pos = [0, 1]
    means = [trained_shift.mean(), zero_shift.mean()]
    stds = [trained_shift.std(), zero_shift.std()]
    bars = ax.bar(bar_pos, means, yerr=stds, capsize=4,
                  color=[C_AD, "#bbb"], alpha=0.75, edgecolor="white", lw=0.5,
                  error_kw={"lw": 0.8})

    ratio = means[0] / max(means[1], 1e-6)
    # Bracket annotation
    bracket_y = max(means[0] + stds[0], means[1] + stds[1]) + 0.15
    ax.plot([0, 0, 1, 1], [bracket_y - 0.08, bracket_y, bracket_y, bracket_y - 0.08],
            color="#444", lw=0.8)
    ax.text(0.5, bracket_y + 0.05, f"{ratio:.0f}×", ha="center", fontsize=11,
            fontweight="bold", color="#444")

    ax.set_xticks(bar_pos)
    ax.set_xticklabels([f"g (trained)", "g = 0"], fontsize=9)
    ax.set_ylabel("||z_t − z_raw||  (perception shift)")
    ax.set_title("(a) g reshapes perception", fontsize=10)

    # ── (b) per-dim z_t(g_trained) - z_t(g_zero) ──
    ax = fig.add_subplot(gs[1])
    if zc:
        zt_trained = probe_df[probe_df["g_label"] == best_label][zc].mean().values
        zt_zero = probe_df[probe_df["g_label"] == "g_zero"][zc].mean().values
        diff = zt_trained - zt_zero
        colors = [C_AD if v >= 0 else "#7799CC" for v in diff]
        x_pos = np.arange(n_z)
        ax.bar(x_pos, diff, color=colors, alpha=0.7, edgecolor="white", lw=0.3)
        ax.axhline(0, color="#ccc", lw=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"z{i}" for i in range(n_z)], fontsize=7)
        ax.set_xlabel("Latent dimension")
        ax.set_ylabel("Δz_t  (trained − zero)")
        ax.set_title("(b) Per-dimension perceptual reorganization", fontsize=10)

    # ── (c) perturbation event-triggered response ──
    if has_panel_c:
        ax = fig.add_subplot(gs[2])
        ds = perturb_traj.sort_values(["episode", "t"]).reset_index(drop=True)
        pa = ds["perturbation_active"].values
        onsets = [i for i in range(1, len(pa)) if pa[i] == 1 and pa[i-1] == 0]

        if onsets:
            W = 40
            t_ax = np.arange(-W, W + 1)

            def trig(vals):
                tr = []
                for loc in onsets:
                    row = np.full(2*W+1, np.nan)
                    for j in range(max(0, loc-W), min(len(vals), loc+W+1)):
                        row[j-loc+W] = vals[j]
                    tr.append(row)
                a = np.array(tr)
                return np.nanmedian(a, 0), np.nanpercentile(a, 25, 0), np.nanpercentile(a, 75, 0)

            # Plot g_norm response
            m, lo, hi = trig(ds["g_norm"].values)
            ax.fill_between(t_ax, lo, hi, color=C_GNORM, alpha=0.12)
            ax.plot(t_ax, m, color=C_GNORM, lw=2, label="||g||")
            ax.axvline(0, color="#aaa", lw=0.5, ls="--")
            ax.axvspan(0, 15, color=C_ALPHA, alpha=0.04)
            ax.set_xlabel("Steps from perturbation onset")
            ax.set_ylabel("||g||")
            ax.set_title("(c) Perturbation engages perspective", fontsize=10)

            # Alpha on twin axis
            if "alpha" in ds.columns:
                ax2 = ax.twinx()
                m_a, lo_a, hi_a = trig(ds["alpha"].values)
                ax2.plot(t_ax, m_a, color=C_ALPHA, lw=1.5, ls="--", label="α")
                ax2.set_ylabel("α", color=C_ALPHA, fontsize=9)
                ax2.tick_params(axis="y", labelcolor=C_ALPHA)

    savefig(fig, outdir / "fig3_perspective_perception.png")


# ════════════════════════════════════════════════════
# Fig 4: Ablation
# ════════════════════════════════════════════════════

def fig4_ablation(ablation_root, outdir):
    """Alpha panel large + PE + ||g|| smaller."""
    conds = [("adaptive", C_AD, "Adaptive"), ("fixed_010", C_FX, "Fixed (α=0.10)"),
             ("fast_080", C_FA, "Fast (α=0.80)")]
    # Also try fixed_005 naming
    alt_names = {"fixed_010": ["fixed_010", "fixed_005"], "fast_080": ["fast_080"]}

    found = []
    for dirname, color, label in conds:
        dfs = []
        candidates = alt_names.get(dirname, [dirname])
        for d in sorted(ablation_root.iterdir()):
            if not d.is_dir(): continue
            if any(cn in d.name for cn in candidates):
                tdf = load_traj(d)
                if tdf is not None:
                    dfs.append(tdf)
        if dfs:
            found.append((label, color, dfs))

    if len(found) < 2:
        print("  [skip] fig4: need >=2 ablation conditions"); return

    fig = plt.figure(figsize=(13, 3.8))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.5, 1, 1], wspace=0.3)

    panels = [
        (0, "alpha", "α (plasticity)", "(a) Plasticity dynamics"),
        (1, "pred_err", "PE (smoothed)", "(b) Prediction error"),
        (2, "g_norm", "||g||", "(c) Perspective magnitude"),
    ]

    for gi, (metric, ylabel, title) in panels:
        ax = fig.add_subplot(gs[gi])
        for label, color, dfs in found:
            for df in dfs:
                if metric not in df.columns: continue
                ep_m = df.groupby("episode")[metric].mean()
                if metric == "pred_err" and len(ep_m) > 10:
                    ep_m = ep_m.rolling(10, min_periods=1).mean()
                ax.plot(ep_m.index, ep_m.values, lw=0.3, alpha=0.15, color=color)

            if dfs and metric in dfs[0].columns:
                if len(dfs) > 1:
                    series_list = []
                    for df in dfs:
                        s = df.groupby("episode")[metric].mean()
                        if metric == "pred_err" and len(s) > 10:
                            s = s.rolling(10, min_periods=1).mean()
                        series_list.append(s)
                    all_m = pd.concat(series_list, axis=1)
                    med = all_m.median(axis=1)
                    q1 = all_m.quantile(0.25, axis=1)
                    q3 = all_m.quantile(0.75, axis=1)
                    ax.fill_between(med.index, q1, q3, color=color, alpha=0.1)
                    ax.plot(med.index, med.values, lw=2, color=color, label=label)
                else:
                    ep_m = dfs[0].groupby("episode")[metric].mean()
                    if metric == "pred_err" and len(ep_m) > 10:
                        ep_m = ep_m.rolling(10, min_periods=1).mean()
                    ax.plot(ep_m.index, ep_m.values, lw=1.5, color=color, label=label)

        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)

    savefig(fig, outdir / "fig4_ablation.png")


# ════════════════════════════════════════════════════
# Supplementary
# ════════════════════════════════════════════════════

def figS1_sweep(sweep_root, outdir):
    """Dose-response: PE + alpha + |γ| + zone dwell vs nP."""
    runs = {}
    for d in sorted(sweep_root.iterdir()):
        if not d.is_dir() or d.name.startswith("_"): continue
        m = re.search(r"nperturb(\d+)", d.name)
        if not m: continue
        n = int(m.group(1))
        tdf = load_traj(d)
        if tdf is not None:
            runs.setdefault(n, []).append(tdf)

    if len(runs) < 2:
        print("  [skip] figS1: need >=2 sweep conditions"); return

    ns = sorted(runs.keys())
    def agg(fn):
        med, lo, hi = [], [], []
        for n in ns:
            vals = [fn(df) for df in runs[n]]
            med.append(np.median(vals))
            lo.append(np.percentile(vals, 25) if len(vals) > 1 else vals[0])
            hi.append(np.percentile(vals, 75) if len(vals) > 1 else vals[0])
        return np.array(med), np.array(lo), np.array(hi)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))

    ax = axes[0]
    m, lo, hi = agg(lambda df: late(df)["pred_err"].mean())
    ax.fill_between(ns, lo, hi, color=C_PE, alpha=0.15)
    ax.plot(ns, m, "s-", color=C_PE, lw=1.5, markersize=5)
    ax.set_xlabel("nP"); ax.set_ylabel("Late PE"); ax.set_title("(a) Prediction error")

    ax = axes[1]
    m, lo, hi = agg(lambda df: late(df)["alpha"].mean() if "alpha" in df.columns else 0)
    ax.fill_between(ns, lo, hi, color=C_ALPHA, alpha=0.15)
    ax.plot(ns, m, "D-", color=C_ALPHA, lw=1.5, markersize=5)
    ax.set_xlabel("nP"); ax.set_ylabel("Late α"); ax.set_title("(b) Plasticity")

    ax = axes[2]
    def get_gabs(df):
        gc = gm_cols(df)
        return late(df)[gc].abs().mean().mean() if gc else 0.0
    m, lo, hi = agg(get_gabs)
    ax.fill_between(ns, lo, hi, color=C_GAMMA, alpha=0.15)
    ax.plot(ns, m, "o-", color=C_GAMMA, lw=1.5, markersize=5)
    ax.set_xlabel("nP"); ax.set_ylabel("Late |γ|"); ax.set_title("(c) Salience modulation")

    ax = axes[3]
    for zi in range(5):
        m, _, _ = agg(lambda df, z=zi: len(late(df)[late(df)["zone_id"]==z])/max(len(late(df)),1))
        ax.plot(ns, m, "o-", color=ZC[zi], lw=1.2, markersize=4, label=f"Z{zi}")
    ax.set_xlabel("nP"); ax.set_ylabel("Frac"); ax.set_title("(d) Zone dwell")
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle("Fig S1: Perturbation sweep dose-response", fontsize=11, y=1.03)
    savefig(fig, outdir / "figS1_sweep.png")


def figS2_film_blocks(mixed_root, outdir):
    """Per-dim gamma per block (0→4→0)."""
    target = None
    for d in sorted(mixed_root.iterdir()):
        if d.is_dir() and "0_4_0" in d.name:
            tdf = load_traj(d)
            if tdf is not None:
                target = tdf; break
    if target is None:
        print("  [skip] figS2: no traj"); return

    gc = gm_cols(target)
    if not gc or "block_id" not in target.columns:
        print("  [skip] figS2: missing columns"); return

    blocks = sorted(target["block_id"].unique())
    n_dims = len(gc)
    x_pos = np.arange(n_dims)
    width = 0.8 / len(blocks)
    bc = [C_AD, C_FA, C_FX, "#D4537E", "#888"]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    for bi, blk in enumerate(blocks):
        bd = target[target["block_id"] == blk]
        mx = bd["episode"].max()
        bl = bd[bd["episode"] >= max(bd["episode"].min(), mx - 9)]
        gvals = bl[gc].mean().values
        nP = int(bd["n_perturb_setting"].iloc[0])
        offset = (bi - len(blocks)/2 + 0.5) * width
        ax.bar(x_pos + offset, gvals, width, alpha=0.7, color=bc[bi % len(bc)],
               label=f"Block {blk} (nP={nP})", edgecolor="white", lw=0.3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"z{i}" for i in range(n_dims)], fontsize=7)
    ax.axhline(0, color="#ccc", lw=0.5)
    ax.set_ylabel("γ (late block mean)")
    ax.set_title("Fig S2: Salience gating pattern per block (0→4→0)")
    ax.legend(fontsize=7, ncol=3)
    savefig(fig, outdir / "figS2_film_blocks.png")


# ════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_root", type=str, default="")
    ap.add_argument("--mixed_root", type=str, default="")
    ap.add_argument("--ablation_root", type=str, default="")
    ap.add_argument("--outdir", type=str, default="paper_figures")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sr = Path(args.sweep_root) if args.sweep_root else None
    mr = Path(args.mixed_root) if args.mixed_root else None
    ar = Path(args.ablation_root) if args.ablation_root else None

    print(f"Output: {outdir}\n")

    # Main figures
    if mr:
        print("=== Fig 2: Mixed history ===")
        try: fig2_mixed_history(mr, outdir)
        except Exception as e: print(f"  [error] {e}")

    print("\n=== Fig 3: Perspective shapes perception ===")
    try: fig3_perspective_perception(mr, sr, outdir)
    except Exception as e: print(f"  [error] {e}")

    if ar:
        print("\n=== Fig 4: Ablation ===")
        try: fig4_ablation(ar, outdir)
        except Exception as e: print(f"  [error] {e}")

    # Supplementary
    if sr:
        print("\n=== Fig S1: Sweep ===")
        try: figS1_sweep(sr, outdir)
        except Exception as e: print(f"  [error] {e}")

    if mr:
        print("\n=== Fig S2: FiLM per block ===")
        try: figS2_film_blocks(mr, outdir)
        except Exception as e: print(f"  [error] {e}")

    print(f"\nDone. {outdir}")


if __name__ == "__main__":
    main()
