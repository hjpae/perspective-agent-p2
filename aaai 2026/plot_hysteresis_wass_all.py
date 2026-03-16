# script_seed_metrics_wasserstein_dsi.py
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from cear_pilot.analysis.metrics import switch_distribution_stats, dissociation_index
from cear_pilot.analysis.figure_switch_eval import g_signed_score_from_df  # reuse same g-score as your main fig

# -----------------------
# IO helpers
# -----------------------
def _find_traj(run_dir: Path) -> Path:
    candidates = ["traj.csv", "traj.parquet", "train_traj.csv", "train_traj.parquet"]
    for name in candidates:
        p = run_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No traj file found in {run_dir}. Tried: {candidates}")

def _load_df(path: Path):
    import pandas as pd
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)

def _zscore(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-6)

# -----------------------
# metric extraction per seed
# -----------------------
def compute_seed_metrics(run_dir: Path, warmup: int, L: int, policy_signal: str, q_p: float, early_frac: float):
    df = _load_df(_find_traj(run_dir))

    for col in ["regime", "switch", policy_signal]:
        if col not in df.columns:
            raise KeyError(f"[{run_dir.name}] missing '{col}' in traj. (Need run_switch_sweep output.)")

    regime = df["regime"].to_numpy(dtype=int)
    switches = df["switch"].to_numpy(dtype=int)

    # signals
    s_g = g_signed_score_from_df(df, warmup=warmup, regime=regime)
    s_pi = _zscore(df[policy_signal].to_numpy(dtype=np.float32))

    stats_g = switch_distribution_stats(score=s_g, regime=regime, switches=switches, L=L, q_p=q_p, early_frac=early_frac)
    stats_pi = switch_distribution_stats(score=s_pi, regime=regime, switches=switches, L=L, q_p=q_p, early_frac=early_frac)

    dsi = dissociation_index(stats_g, stats_pi)

    # pull combined metrics (you can also use up/dn separately if you want)
    g_comb = stats_g["combined"]
    p_comb = stats_pi["combined"]

    return {
        "run": run_dir.name,
        "n_events": int(g_comb.get("n_events", 0)),
        "g_NSI_W1": g_comb.get("NSI_W1", None),
        "g_QD": g_comb.get("QD", None),
        "g_Amp_IQR": g_comb.get("Amp_IQR", None),
        "pi_NSI_W1": p_comb.get("NSI_W1", None),
        "pi_QD": p_comb.get("QD", None),
        "pi_Amp_IQR": p_comb.get("Amp_IQR", None),
        "DSI": dsi,
    }

# -----------------------
# plotting
# -----------------------
def _mean_sd(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if x.size == 0:
        return np.nan, np.nan, x
    return float(np.mean(x)), float(np.std(x, ddof=1) if x.size > 1 else 0.0), x

def plot_seed_summary(rows, L: int, save_path: str | None = None, show: bool = True):
    # Collect arrays
    g_w1 = [r["g_NSI_W1"] for r in rows]
    p_w1 = [r["pi_NSI_W1"] for r in rows]
    g_qd = [r["g_QD"] for r in rows]
    p_qd = [r["pi_QD"] for r in rows]
    dsi  = [r["DSI"] for r in rows]

    # stats
    gWm, gWs, gW = _mean_sd(g_w1)
    pWm, pWs, pW = _mean_sd(p_w1)
    gQm, gQs, gQ = _mean_sd(g_qd)
    pQm, pQs, pQ = _mean_sd(p_qd)
    Dm,  Ds,  D  = _mean_sd(dsi)

    fig = plt.figure(figsize=(10, 4))

    # Panel 1: NSI_W1 (Wasserstein nonstationarity)
    ax1 = plt.subplot(1, 3, 1)
    x = np.array([0, 1], dtype=float)
    means = [gWm, pWm]
    sds   = [gWs, pWs]
    ax1.bar(x, means, yerr=sds, capsize=4)
    # seed dots
    ax1.scatter(np.full_like(gW, 0.0), gW, s=30)
    ax1.scatter(np.full_like(pW, 1.0), pW, s=30)
    ax1.set_xticks([0, 1]); ax1.set_xticklabels(["g", "policy"])
    ax1.set_title(f"NSI_W1 (Wasserstein) | L={L}")
    ax1.set_ylabel("value")

    # Panel 2: QD (median drift)
    ax2 = plt.subplot(1, 3, 2)
    means = [gQm, pQm]
    sds   = [gQs, pQs]
    ax2.bar(x, means, yerr=sds, capsize=4)
    ax2.scatter(np.full_like(gQ, 0.0), gQ, s=30)
    ax2.scatter(np.full_like(pQ, 1.0), pQ, s=30)
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["g", "policy"])
    ax2.set_title("QD (quantile drift, p=0.5)")
    ax2.set_ylabel("value")

    # Panel 3: DSI (scalar dissociation)
    ax3 = plt.subplot(1, 3, 3)
    ax3.bar([0], [Dm], yerr=[Ds], capsize=4)
    ax3.scatter(np.zeros_like(D), D, s=35)
    ax3.axhline(0.0, linestyle="--", linewidth=1)
    ax3.set_xticks([0]); ax3.set_xticklabels(["DSI"])
    ax3.set_title("Dissociation Index")
    ax3.set_ylabel("value")

    plt.tight_layout()

    if save_path is not None:
        outp = Path(save_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outp, dpi=200, bbox_inches="tight")
        print(f"[OK] saved: {outp}")

    if show:
        plt.show()
    else:
        plt.close(fig)

def main():
    # -----------------------
    # EDIT THESE
    # -----------------------
    RUN_DIRS = [
        "outputs/runs/seed1/p40",
        "outputs/runs/seed2/p40",
        "outputs/runs/seed3/p40",
        "outputs/runs/seed4/p40",
        "outputs/runs/seed5/p40",
    ]

    WARMUP = 150
    L = 40
    POLICY_SIGNAL = "entropy"
    Q_P = 0.5   # within-seed event-ensemble median
    EARLY_FRAC = 0.25

    rows = []
    for rd in map(Path, RUN_DIRS):
        r = compute_seed_metrics(rd, warmup=WARMUP, L=L, policy_signal=POLICY_SIGNAL, q_p=Q_P, early_frac=EARLY_FRAC)
        rows.append(r)

    # quick console dump
    print("\nSeed-wise metrics:")
    for r in rows:
        print(r["run"], "n_events=", r["n_events"], "gW1=", r["g_NSI_W1"], "piW1=", r["pi_NSI_W1"], "DSI=", r["DSI"])

    plot_seed_summary(
        rows,
        L=L,
        save_path=f"outputs/figs/fig_seed_metrics_W1_QD_DSI_L{L}.png",
        show=True,
    )

if __name__ == "__main__":
    main()
