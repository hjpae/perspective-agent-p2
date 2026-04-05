#%% Spyder-ready quick plot: Fig 3 with 0-0-0 and 4-4-4 references
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# Import helper functions from your updated v3 file
from cear_pilot.analysis.paper_figures_v3 import (
    collect_mixed_runs,
    collect_sweep_runs,
    hierarchical_curve_summary,
    C_ALPHA,
    C_GNORM,
    C_BL,
)

# ------------------------------------------------------------
# USER SETTING
# ------------------------------------------------------------
ROOT = Path("outputs/phase2_all_20260403_011254")   # <- change if needed

# ------------------------------------------------------------
# Main plotting function
# ------------------------------------------------------------
def plot_fig3_threeway_spyder(root: Path):
    mixed_root = root
    sweep_root = root

    mixed_data = collect_mixed_runs(mixed_root)
    sweep_data = collect_sweep_runs(sweep_root)

    if "mixed_0_4_0" not in mixed_data or not mixed_data["mixed_0_4_0"]:
        raise RuntimeError("No mixed_0_4_0 runs found.")

    by_p1_mixed = mixed_data["mixed_0_4_0"]
    by_p1_base0 = sweep_data.get(0, {})   # 0-0-0 baseline
    by_p1_ref4  = sweep_data.get(4, {})   # 4-4-4 persistent perturbation

    has_base0 = bool(by_p1_base0)
    has_ref4  = bool(by_p1_ref4)

    # Reference trajectory for block schedule / boundaries
    ref = next((dfs[0] for _, dfs in sorted(by_p1_mixed.items()) if dfs), None)
    if ref is None:
        raise RuntimeError("mixed_0_4_0 data is empty.")

    blk_info = []
    if "block_id" in ref.columns and "n_perturb_setting" in ref.columns:
        ep_blk = ref.groupby("episode")[["block_id", "n_perturb_setting"]].first()
        changes = ep_blk[ep_blk["block_id"] != ep_blk["block_id"].shift(1)]
        for ep_idx in changes.index:
            blk_info.append((int(ep_idx), int(changes.loc[ep_idx, "n_perturb_setting"])))

    # --------------------------------------------------------
    # Figure layout
    # --------------------------------------------------------
    fig, axes = plt.subplots(
        3, 1,
        figsize=(7.2, 4.8),
        sharex=True,
        gridspec_kw={"height_ratios": [0.45, 1, 1], "hspace": 0.08}
    )

    # --------------------------------------------------------
    # Row 0: schedule strip from mixed 0->4->0
    # --------------------------------------------------------
    ax = axes[0]
    if "n_perturb_setting" in ref.columns:
        ep_nP = ref.groupby("episode")["n_perturb_setting"].first()
        ax.fill_between(ep_nP.index, 0, ep_nP.values, step="post",
                        color="#dddddd", alpha=0.65)
        ax.plot(ep_nP.index, ep_nP.values, drawstyle="steps-post",
                color="#888888", lw=1.6)

    ax.set_ylabel("nP")
    ax.set_title("Block schedule (nP = 0 → 4 → 0)", fontsize=10, pad=4)
    ax.set_ylim(-0.3, 5.0)
    ax.set_yticks([0, 4])

    for bc, _ in blk_info:
        ax.axvline(bc, color="#aaaaaa", lw=0.7, ls="--", alpha=0.6)

    # --------------------------------------------------------
    # Helper for plotting one metric row
    # --------------------------------------------------------
    def plot_metric_row(ax, metric, ylabel, color_main):
        # Mixed 0->4->0
        med, q1, q3, p1_curves = hierarchical_curve_summary(
            by_p1_mixed,
            lambda df, m=metric: df.groupby("episode")[m].mean()
            if m in df.columns else pd.Series(dtype=float),
        )
        if med is not None:
            for curve in p1_curves:
                ax.plot(curve.index, curve.values, lw=0.3, alpha=0.12, color=color_main)
            ax.fill_between(med.index, q1.values, q3.values, color=color_main, alpha=0.12)
            ax.plot(med.index, med.values, lw=2.2, color=color_main, label="Mixed (0→4→0)")

        # 0-0-0 baseline
        if has_base0:
            med0, q10, q30, p1_0 = hierarchical_curve_summary(
                by_p1_base0,
                lambda df, m=metric: df.groupby("episode")[m].mean()
                if m in df.columns else pd.Series(dtype=float),
            )
            if med0 is not None:
                for curve in p1_0:
                    ax.plot(curve.index, curve.values, lw=0.25, alpha=0.08, color=C_BL)
                ax.fill_between(med0.index, q10.values, q30.values, color=C_BL, alpha=0.07)
                ax.plot(med0.index, med0.values, lw=1.6, color=C_BL, ls="--",
                        label="Baseline (0→0→0)")

        # 4-4-4 persistent reference
        if has_ref4:
            med4, q14, q34, p1_4 = hierarchical_curve_summary(
                by_p1_ref4,
                lambda df, m=metric: df.groupby("episode")[m].mean()
                if m in df.columns else pd.Series(dtype=float),
            )
            if med4 is not None:
                c4 = "#5F5F5F"
                for curve in p1_4:
                    ax.plot(curve.index, curve.values, lw=0.25, alpha=0.08, color=c4)
                ax.fill_between(med4.index, q14.values, q34.values, color=c4, alpha=0.06)
                ax.plot(med4.index, med4.values, lw=1.4, color=c4, ls=":",
                        label="Persistent (4→4→4)")

        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, loc="best")

        for bc, _ in blk_info:
            ax.axvline(bc, color="#aaaaaa", lw=0.7, ls="--", alpha=0.6)

    # --------------------------------------------------------
    # Row 1: alpha
    # --------------------------------------------------------
    plot_metric_row(
        axes[1],
        metric="alpha",
        ylabel="α (plasticity)",
        color_main=C_ALPHA,
    )

    # --------------------------------------------------------
    # Row 2: g_norm
    # --------------------------------------------------------
    plot_metric_row(
        axes[2],
        metric="g_norm",
        ylabel="||g||",
        color_main=C_GNORM,
    )

    axes[2].set_xlabel("Episode")

    outpath = root / "fig3_threeway_spyder.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight", pad_inches=0.12)
    print(f"[saved] {outpath}")
    plt.close(fig)


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
plot_fig3_threeway_spyder(ROOT)