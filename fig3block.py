#%% Fig 3 only: single-panel alpha comparison + stats (whole-block version)
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cear_pilot.analysis.paper_figures_v3 import (
    collect_mixed_runs,
    collect_sweep_runs,
    C_ALPHA,
    C_BL,
)

# ------------------------------------------------------------
# USER SETTING
# ------------------------------------------------------------
ROOT = Path("outputs/phase2_all_20260403_011254")   # <- change if needed
OUTNAME = "fig3_alpha_blocks_grouped.png"

# Colors
C_MIX = C_ALPHA
C_BASE = C_BL
C_PERS = "#5F5F5F"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def mean_alpha_over_episode_range(df, ep_start, ep_end):
    """Mean alpha over an inclusive episode range [ep_start, ep_end]."""
    if "alpha" not in df.columns or "episode" not in df.columns:
        return np.nan
    wd = df[(df["episode"] >= ep_start) & (df["episode"] <= ep_end)].copy()
    if len(wd) == 0:
        return np.nan
    vals = pd.to_numeric(wd["alpha"], errors="coerce").dropna()
    if len(vals) == 0:
        return np.nan
    return float(vals.mean())


def infer_reference_blocks(by_p1_mixed):
    """
    Infer full-episode boundaries for the first and second nP=0 blocks
    from a mixed 0→4→0 trajectory that contains block_id.

    Returns:
      {
        "First": {"block_id": 0, "episode_start": ..., "episode_end": ..., "n_perturb": ...},
        "Second": {"block_id": 2, "episode_start": ..., "episode_end": ..., "n_perturb": ...},
      }
    """
    ref = next((dfs[0] for _, dfs in sorted(by_p1_mixed.items()) if dfs), None)
    if ref is None:
        raise RuntimeError("No reference mixed trajectory available.")
    if "block_id" not in ref.columns or "episode" not in ref.columns:
        raise RuntimeError("Mixed reference trajectory needs block_id and episode columns.")

    ep_blk = ref.groupby("episode")[["block_id"]].first().reset_index()
    if "n_perturb_setting" in ref.columns:
        ep_np = ref.groupby("episode")["n_perturb_setting"].first().reset_index(drop=True)
        ep_blk["n_perturb_setting"] = ep_np.values

    block_ranges = {}
    for blk, sub in ep_blk.groupby("block_id", sort=True):
        blk = int(blk)
        block_ranges[blk] = {
            "block_id": blk,
            "episode_start": int(sub["episode"].min()),
            "episode_end": int(sub["episode"].max()),
            "n_perturb": int(sub["n_perturb_setting"].iloc[0]) if "n_perturb_setting" in sub.columns else None,
        }

    missing = [blk for blk in (0, 2) if blk not in block_ranges]
    if missing:
        raise RuntimeError(f"Reference mixed trajectory missing required blocks: {missing}")

    return {
        "First": block_ranges[0],
        "Second": block_ranges[2],
    }


def seed_level_block_summary(data_by_p1, condition_name, block_name, ep_start, ep_end):
    """
    For each p1 seed:
      - aggregate across p2 seeds by median
      - use the full inclusive episode range [ep_start, ep_end]
    Returns a DataFrame with one row per p1 seed.
    """
    rows = []
    for p1_seed, dfs in sorted(data_by_p1.items()):
        vals = []
        for df in dfs:
            v = mean_alpha_over_episode_range(df, ep_start, ep_end)
            if np.isfinite(v):
                vals.append(float(v))
        if vals:
            rows.append({
                "seed": int(p1_seed),
                "condition": condition_name,
                "block": block_name,
                "alpha_mean": float(np.median(vals)),
                "episode_start": int(ep_start),
                "episode_end": int(ep_end),
            })
    return pd.DataFrame(rows)


def paired_permutation_test(x, y, n_perm=20000, rng=None):
    """
    Paired permutation (sign-flip) test for mean difference.
    H0: mean(x - y) = 0
    Two-sided p-value.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    d = x[mask] - y[mask]
    if len(d) == 0:
        return np.nan, np.nan
    obs = float(np.mean(d))
    if rng is None:
        rng = np.random.default_rng(0)

    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    perm_means = (signs * d[None, :]).mean(axis=1)
    p = float((1 + np.sum(np.abs(perm_means) >= abs(obs))) / (n_perm + 1))
    return obs, p


def holm_adjust(pvals_named):
    """
    pvals_named: list of (name, p)
    returns list of (name, raw_p, holm_p)
    """
    valid = [(n, float(p)) for n, p in pvals_named if np.isfinite(p)]
    if not valid:
        return [(n, p, np.nan) for n, p in pvals_named]

    m = len(valid)
    order = np.argsort([p for _, p in valid])
    ranked = [valid[i] for i in order]

    adj = [None] * m
    running_max = 0.0
    for i, (name, p) in enumerate(ranked):
        val = (m - i) * p
        running_max = max(running_max, val)
        adj[i] = min(running_max, 1.0)

    out_map = {ranked[i][0]: adj[i] for i in range(m)}
    return [(n, p, out_map.get(n, np.nan)) for n, p in pvals_named]


# ------------------------------------------------------------
# Data assembly
# ------------------------------------------------------------
def build_alpha_block_table(root: Path):
    mixed_data = collect_mixed_runs(root)
    sweep_data = collect_sweep_runs(root)

    if "mixed_0_4_0" not in mixed_data or not mixed_data["mixed_0_4_0"]:
        raise RuntimeError("No mixed_0_4_0 runs found.")

    by_p1_mixed = mixed_data["mixed_0_4_0"]
    by_p1_base0 = sweep_data.get(0, {})   # 0→0→0
    by_p1_ref4 = sweep_data.get(4, {})    # 4→4→4

    ref_blocks = infer_reference_blocks(by_p1_mixed)

    dfs = []
    for block_name in ["First", "Second"]:
        ep_start = ref_blocks[block_name]["episode_start"]
        ep_end = ref_blocks[block_name]["episode_end"]

        d = seed_level_block_summary(by_p1_mixed, "Mixed (0→4→0)", block_name, ep_start, ep_end)
        dfs.append(d)

        if by_p1_base0:
            d = seed_level_block_summary(by_p1_base0, "Baseline (0→0→0)", block_name, ep_start, ep_end)
            dfs.append(d)

        if by_p1_ref4:
            d = seed_level_block_summary(by_p1_ref4, "Persistent (4→4→4)", block_name, ep_start, ep_end)
            dfs.append(d)

    if not dfs:
        raise RuntimeError("No valid alpha block summaries found.")

    df = pd.concat(dfs, ignore_index=True)

    conds = list(df["condition"].drop_duplicates())
    blocks = ["First", "Second"]
    needed = {(c, b) for c in conds for b in blocks}

    seed_cells = (
        df.groupby("seed")
          .apply(lambda x: {(r["condition"], r["block"]) for _, r in x.iterrows()})
    )
    common_seeds = [s for s, cells in seed_cells.items() if needed.issubset(cells)]

    df_common = df[df["seed"].isin(common_seeds)].copy()
    if len(df_common) == 0:
        raise RuntimeError("No common seeds across all condition x block cells.")

    df_common["condition"] = pd.Categorical(
        df_common["condition"],
        categories=["Baseline (0→0→0)", "Mixed (0→4→0)", "Persistent (4→4→4)"],
        ordered=True,
    )
    df_common["block"] = pd.Categorical(
        df_common["block"],
        categories=["First", "Second"],
        ordered=True,
    )
    df_common = df_common.sort_values(["condition", "block", "seed"]).reset_index(drop=True)
    return df_common, ref_blocks


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
def plot_grouped_single_panel(df_long, outpath: Path):
    color_map = {
        "Baseline (0→0→0)": "#9A9A9A",   # medium gray
        "Mixed (0→4→0)": C_MIX,          # orange
        "Persistent (4→4→4)": "#1F3A5F", # deep navy
    }

    cond_order = ["Baseline (0→0→0)", "Mixed (0→4→0)", "Persistent (4→4→4)"]
    block_order = ["First", "Second"]

    # compact layout
    group_gap = 0.72
    bar_width = 0.36

    x_positions = []
    x_labels = []
    group_centers = []
    group_pairs = []

    x = 0.0
    for cond in cond_order:
        x_first = x
        x_second = x + bar_width
        x_positions.extend([x_first, x_second])
        x_labels.extend(["First", "Second"])
        group_centers.append((x_first + x_second) / 2)
        group_pairs.append((x_first, x_second))
        x = x_second + group_gap

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.2))
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.30, top=0.88)

    vals_all = df_long["alpha_mean"].astype(float).dropna().values
    y_min = max(0.10, float(np.min(vals_all)) - 0.015)
    y_max = min(0.31, float(np.max(vals_all)) + 0.010)
    y_range = y_max - y_min

    # room below axis for condition labels
    label_y = y_min - y_range * 0.115

    # subtle condition background bands
    for (x_first, x_second) in group_pairs:
        ax.axvspan(
            x_first - bar_width * 0.72,
            x_second + bar_width * 0.72,
            color="#F4F4F4",
            alpha=0.9,
            zorder=0,
        )

    # bars + error bars + points
    for i, cond in enumerate(cond_order):
        for j, block in enumerate(block_order):
            sub = df_long[
                (df_long["condition"] == cond) &
                (df_long["block"] == block)
            ]
            vals = sub["alpha_mean"].astype(float).dropna().values
            if len(vals) == 0:
                continue

            med = float(np.median(vals))
            q1 = float(np.percentile(vals, 25))
            q3 = float(np.percentile(vals, 75))
            err_low = max(med - q1, 0)
            err_high = max(q3 - med, 0)

            xpos = group_pairs[i][j]

            ax.bar(
                xpos,
                med,
                width=bar_width * 0.92,
                color=color_map[cond],
                alpha=0.78,
                edgecolor="white",
                linewidth=0.8,
                yerr=np.array([[err_low], [err_high]]),
                capsize=3.5,
                error_kw={"lw": 0.9},
                zorder=2,
            )

            # deterministic jitter
            if len(vals) > 1:
                jitter = np.linspace(-0.045, 0.045, len(vals))
            else:
                jitter = np.array([0.0])

            ax.scatter(
                np.full(len(vals), xpos) + jitter,
                vals,
                s=30,
                color=color_map[cond],
                alpha=0.95,
                edgecolors="white",
                linewidths=0.5,
                zorder=4,
            )

    # axes / labels
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_positions[0] - 0.45, x_positions[-1] + 0.45)
    ax.set_ylabel("α (plasticity)")
   # ax.set_title("Fig 3. First vs. Second nP=0 block", fontsize=10)

    # use actual ticks for First / Second
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=8)

    # condition labels placed lower so they do not overlap
    for center, cond in zip(group_centers, cond_order):
        ax.text(
            center,
            label_y,
            cond,
            ha="center",
            va="top",
            fontsize=10,
            clip_on=False,
        )

    # separators between condition groups
    for i in range(len(group_pairs) - 1):
        sep = (group_pairs[i][1] + group_pairs[i + 1][0]) / 2
        ax.axvline(sep, color="#D7D7D7", lw=0.7, alpha=0.8, zorder=1)

    ax.grid(axis="y", alpha=0.18, linewidth=0.4)
    ax.tick_params(axis="x", pad=2)

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=color_map["Baseline (0→0→0)"], edgecolor="white", label="Baseline (0→0→0)"),
        Patch(facecolor=color_map["Mixed (0→4→0)"], edgecolor="white", label="Mixed (0→4→0)"),
        Patch(facecolor=color_map["Persistent (4→4→4)"], edgecolor="white", label="Persistent (4→4→4)"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="lower left", frameon=True)

    fig.savefig(outpath, dpi=300, bbox_inches="tight", pad_inches=0.10)
    print(f"[saved] {outpath}")
    plt.close(fig)

# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------
def run_stats(df_long, ref_blocks):
    print("\n=== Fig 3 stats: alpha full blocks ===")
    print(
        "Reference blocks used from mixed_0_4_0: "
        f"First=[{ref_blocks['First']['episode_start']}, {ref_blocks['First']['episode_end']}], "
        f"Second=[{ref_blocks['Second']['episode_start']}, {ref_blocks['Second']['episode_end']}]"
    )
    print(f"Common seeds used: {sorted(df_long['seed'].unique().tolist())}")
    print()

    # ------------------------------------------------------------------
    # Seed-level summary table
    # ------------------------------------------------------------------
    summary = (
        df_long.groupby(["condition", "block"], observed=False)["alpha_mean"]
        .agg(
            median="median",
            q1=lambda s: np.percentile(s, 25),
            q3=lambda s: np.percentile(s, 75),
            mean="mean",
            std="std",
            n="count",
        )
        .reset_index()
    )
    print("Seed-level summaries:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    rng = np.random.default_rng(0)
    pvals_named = []

    # ------------------------------------------------------------------
    # Build per-seed paired table for each condition
    # ------------------------------------------------------------------
    cond_order = ["Baseline", "Mixed", "Persistent"]
    paired_tables = {}

    for cond in cond_order:
        sub = df_long[df_long["condition"] == cond].copy()
        if sub.empty:
            continue

        wide = (
            sub.pivot(index="seed", columns="block", values="alpha_mean")
            .reindex(columns=["First", "Second"])
            .dropna()
            .sort_index()
        )

        if len(wide) == 0:
            continue

        paired_tables[cond] = wide

    # ------------------------------------------------------------------
    # Within-condition paired tests
    # ------------------------------------------------------------------
    print("Within-condition paired permutation tests (Second - First):")
    for cond in cond_order:
        if cond not in paired_tables:
            print(f"  {cond:<10s}  [skip: no paired seeds]")
            continue

        wide = paired_tables[cond]
        first_vals = wide["First"].to_numpy(dtype=float)
        second_vals = wide["Second"].to_numpy(dtype=float)

        obs, p = paired_permutation_test(second_vals, first_vals, rng=rng)
        pvals_named.append((f"{cond}: Second-First", p))

        med_first = float(np.median(first_vals))
        med_second = float(np.median(second_vals))

        print(
            f"  {cond:<10s}  "
            f"median First={med_first:.4f}  median Second={med_second:.4f}  "
            f"delta={obs:+.4f}   p={p:.6g}"
        )
    print()

    # ------------------------------------------------------------------
    # Seed-level delta table: delta = Second - First
    # ------------------------------------------------------------------
    delta_rows = []
    for cond in cond_order:
        if cond not in paired_tables:
            continue

        wide = paired_tables[cond]
        delta = wide["Second"] - wide["First"]

        tmp = pd.DataFrame({
            "seed": delta.index.astype(int),
            "condition": cond,
            "delta": delta.to_numpy(dtype=float),
        })
        delta_rows.append(tmp)

    if not delta_rows:
        print("[warn] No valid paired delta data available.")
        return

    ddelta = pd.concat(delta_rows, ignore_index=True)

    print("Seed-level delta summaries (Second - First):")
    delta_summary = (
        ddelta.groupby("condition", observed=False)["delta"]
        .agg(
            median="median",
            q1=lambda s: np.percentile(s, 25),
            q3=lambda s: np.percentile(s, 75),
            mean="mean",
            std="std",
            n="count",
        )
        .reset_index()
    )
    print(delta_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    # ------------------------------------------------------------------
    # Difference-of-differences across conditions
    # ------------------------------------------------------------------
    print("Difference-of-differences paired permutation tests:")
    pairs = [
        ("Mixed", "Baseline"),
        ("Mixed", "Persistent"),
        ("Baseline", "Persistent"),
    ]

    for a, b in pairs:
        wa = (
            ddelta[ddelta["condition"] == a]
            .set_index("seed")["delta"]
            .sort_index()
        )
        wb = (
            ddelta[ddelta["condition"] == b]
            .set_index("seed")["delta"]
            .sort_index()
        )

        common = wa.index.intersection(wb.index)
        if len(common) == 0:
            print(f"  {a:<10s} vs {b:<10s}   [skip: no common seeds]")
            continue

        xa = wa.loc[common].to_numpy(dtype=float)
        xb = wb.loc[common].to_numpy(dtype=float)

        obs, p = paired_permutation_test(xa, xb, rng=rng)
        pvals_named.append((f"Δ {a}-{b}", p))

        med_a = float(np.median(xa))
        med_b = float(np.median(xb))

        print(
            f"  {a:<10s} vs {b:<10s}   "
            f"medianΔ {a}={med_a:+.4f}  medianΔ {b}={med_b:+.4f}  "
            f"delta-diff={obs:+.4f}   p={p:.6g}"
        )
    print()

    # ------------------------------------------------------------------
    # Holm correction
    # ------------------------------------------------------------------
    print("Holm-adjusted post hoc p-values:")
    for name, raw_p, holm_p in holm_adjust(pvals_named):
        print(f"  {name:<28s} raw={raw_p:.6g}   holm={holm_p:.6g}")
    print()

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------
    print("Interpretation guide:")
    print("  - 'First' and 'Second' refer to the full mixed-reference blocks, not fixed windows.")
    print("  - Within-condition tests ask whether alpha differs between the first and second full blocks.")
    print("  - Δ comparisons ask whether the block-to-block change differs between conditions.")
    print("  - If Holm-adjusted p-values are not < 0.05, treat the results as descriptive trends.")
    print()

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    df_long, ref_blocks = build_alpha_block_table(ROOT)
    outpath = ROOT / OUTNAME
    plot_grouped_single_panel(df_long, outpath)
    run_stats(df_long, ref_blocks)


if __name__ == "__main__":
    main()