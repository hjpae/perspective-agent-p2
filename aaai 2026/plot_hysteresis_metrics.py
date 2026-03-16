# plot_hysteresis_metrics.py
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

# Reuse your existing implementations
from cear_pilot.analysis.metrics import switch_distribution_stats, dissociation_index
from cear_pilot.analysis.figure_switch_eval import g_signed_score_from_df  # signed g-score used in your switch eval

def _find_traj(run_dir: Path) -> Path:
    candidates = [
        "traj.parquet",
        "traj.csv",
        "train_traj.parquet",
        "train_traj.csv",
    ]
    for name in candidates:
        p = run_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No trajectory file found in {run_dir}. "
        f"Looked for: {candidates}"
    )


def _load_table(path: Path):
    import pandas as pd
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)

def compute_and_plot(
    run_dir: str,
    warmup: int = 150,
    L: int = 40,
    qd_p: float = 0.5,
    early_frac: float = 0.25,
    policy_signal: str = "entropy",
    save_png: bool = True,
    show: bool = True,
):
    run_dir = Path(run_dir)
    df = _load_table(_find_traj(run_dir))

    # Required columns from run_switch_sweep collector
    for col in ["regime", "switch", policy_signal]:
        if col not in df.columns:
            raise KeyError(f"Missing column '{col}' in traj. Did you collect with run_switch_sweep?")

    regime = df["regime"].to_numpy(dtype=int)
    switches = df["switch"].to_numpy(dtype=int)

    # ---- signals
    s_g = g_signed_score_from_df(df, warmup=warmup, regime=regime)

    s_pi_raw = df[policy_signal].to_numpy(dtype=np.float32)
    # z-score policy signal (same convention as figure_switch_eval)
    s_pi = (s_pi_raw - s_pi_raw.mean()) / (s_pi_raw.std() + 1e-6)

    # ---- distribution/shape stats (up/dn/combined)
    stats_g = switch_distribution_stats(
        score=s_g, regime=regime, switches=switches, L=int(L),
        q_p=float(qd_p), early_frac=float(early_frac)
    )
    stats_pi = switch_distribution_stats(
        score=s_pi, regime=regime, switches=switches, L=int(L),
        q_p=float(qd_p), early_frac=float(early_frac)
    )
    dsi = dissociation_index(stats_g, stats_pi)

    out = {
        "run_dir": str(run_dir),
        "warmup": int(warmup),
        "L": int(L),
        "qd_p": float(qd_p),
        "early_frac": float(early_frac),
        "policy_signal": str(policy_signal),
        "stats_g": stats_g,
        "stats_pi": stats_pi,
        "DSI": dsi,
    }

    # ---- save json next to run_dir (handy for paper tables)
    (run_dir / "switch_shape_metrics.json").write_text(json.dumps(out, indent=2, default=_jsonable))
    print(f"[OK] wrote: {run_dir / 'switch_shape_metrics.json'}")

    # ---- plot
    fig = plt.figure(figsize=(10, 6))

    # Panel A: mean curves (median / q-curve) for up vs dn
    ax1 = plt.subplot(2, 2, 1)
    _plot_qcurve(ax1, stats_g, title="g-score: quantile curve (up/dn)")
    ax2 = plt.subplot(2, 2, 2)
    _plot_qcurve(ax2, stats_pi, title=f"{policy_signal}-z: quantile curve (up/dn)")

    # Panel B: bar summary (combined)
    ax3 = plt.subplot(2, 1, 2)
    _plot_summary_bars(ax3, stats_g, stats_pi, dsi=dsi)

    plt.tight_layout()

    if save_png:
        out_png = run_dir / "figs" / f"fig_switch_shape_{policy_signal}.png"
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"[OK] saved: {out_png}")

    if show:
        plt.show()
    else:
        plt.close(fig)

def _jsonable(x):
    # Make numpy types JSON serializable
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)

def _plot_qcurve(ax, stats, title=""):
    up = stats.get("up", {})
    dn = stats.get("dn", {})
    qu = up.get("q_curve", None)
    qd = dn.get("q_curve", None)

    if qu is not None:
        ax.plot(np.asarray(qu), label=f"A -> B (n={up.get('n_events', 0)})")
    if qd is not None:
        ax.plot(np.asarray(qd), label=f"B -> A (n={dn.get('n_events', 0)})")

    ax.set_title(title)
    ax.set_xlabel("tau")
    ax.legend()

def _plot_summary_bars(ax, stats_g, stats_pi, dsi=None, normalize=True):
    def comb(stats, k):
        return stats.get("combined", {}).get(k, None)

    keys = ["NSI_W1", "QD", "Amp_IQR"]
    g_vals = [comb(stats_g, k) for k in keys]
    p_vals = [comb(stats_pi, k) for k in keys]

    g_vals = np.array([np.nan if v is None else float(v) for v in g_vals], dtype=float)
    p_vals = np.array([np.nan if v is None else float(v) for v in p_vals], dtype=float)

    if normalize:
        # metric-wise normalization: max(g, p) -> 1
        denom = np.nanmax(np.vstack([g_vals, p_vals]), axis=0)
        denom = np.where(np.isfinite(denom) & (denom > 0), denom, 1.0)
        g_plot = g_vals / denom
        p_plot = p_vals / denom
        ylab = "Normalized score (per-metric max = 1)"
    else:
        g_plot, p_plot = g_vals, p_vals
        ylab = "Raw score"

    x = np.arange(len(keys))
    w = 0.35
    ax.bar(x - w/2, g_plot, width=w, label="g")
    ax.bar(x + w/2, p_plot, width=w, label="policy")

    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylabel(ylab)
    ax.set_title(f"Distribution/shape metrics (combined) | DSI={('None' if dsi is None else f'{dsi:.3f}')}")
    ax.legend()


# -----------------------
# Spyder entrypoint
# -----------------------
if __name__ == "__main__":
    # EDIT THIS: point to the specific run_dir created by run_switch_sweep
    RUN_DIR = "outputs/runs/seed1/p40"  # put your actual sweep run dir here

    compute_and_plot(
        run_dir=RUN_DIR,
        warmup=150,
        L=40,
        qd_p=0.5,
        early_frac=0.25,
        policy_signal="entropy",
        save_png=True,
        show=True,
    )
