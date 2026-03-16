# plot_hysteresis_agg.py
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from cear_pilot.analysis.metrics import switch_distribution_stats
from cear_pilot.analysis.figure_switch_eval import g_signed_score_from_df  # same g-score as your existing eval

def _find_traj(run_dir: Path) -> Path:
    # support both csv/parquet, and both traj/train_traj naming
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

def _seed_qcurves_for_run(run_dir: Path, warmup: int, L: int, policy_signal: str, qd_p: float, early_frac: float):
    df = _load_df(_find_traj(run_dir))

    # must exist for switch-aligned eval runs
    for col in ["regime", "switch", policy_signal]:
        if col not in df.columns:
            raise KeyError(f"[{run_dir.name}] missing '{col}' in traj. Is this a run_switch_sweep output?")

    regime = df["regime"].to_numpy(dtype=int)
    switches = df["switch"].to_numpy(dtype=int)

    # signals
    s_g = g_signed_score_from_df(df, warmup=warmup, regime=regime)
    s_pi = _zscore(df[policy_signal].to_numpy(dtype=np.float32))

    stats_g = switch_distribution_stats(score=s_g, regime=regime, switches=switches, L=L, q_p=qd_p, early_frac=early_frac)
    stats_pi = switch_distribution_stats(score=s_pi, regime=regime, switches=switches, L=L, q_p=qd_p, early_frac=early_frac)

    # These q_curves are already "event-ensemble quantiles within this seed"
    g_AB = np.asarray(stats_g["up"]["q_curve"], dtype=float)   # A->B
    g_BA = np.asarray(stats_g["dn"]["q_curve"], dtype=float)   # B->A
    p_AB = np.asarray(stats_pi["up"]["q_curve"], dtype=float)
    p_BA = np.asarray(stats_pi["dn"]["q_curve"], dtype=float)

    nAB = int(stats_g["up"].get("n_events", 0))
    nBA = int(stats_g["dn"].get("n_events", 0))
    return (g_AB, g_BA, p_AB, p_BA, nAB, nBA)

def _aggregate_curves(curves: list[np.ndarray], q_lo=0.25, q_hi=0.75):
    """
    curves: list of shape (L,) arrays (one per seed)
    Returns: median curve + (q_lo,q_hi) band across seeds
    """
    X = np.stack(curves, axis=0)  # (S, L)
    med = np.median(X, axis=0)
    lo = np.quantile(X, q_lo, axis=0)
    hi = np.quantile(X, q_hi, axis=0)
    return med, lo, hi

def plot_seed_aggregate(
    run_dirs,
    warmup: int = 150,
    L: int = 40,                 # <-- P=40이면 L=40 권장
    policy_signal: str = "entropy",
    qd_p: float = 0.5,           # within-seed event ensemble quantile (median)
    band_lo: float = 0.25,       # across-seed band
    band_hi: float = 0.75,
    early_frac: float = 0.25,
    save_path: str | None = None,
    show: bool = True,
):
    run_dirs = [Path(p) for p in run_dirs]

    g_AB_list, g_BA_list = [], []
    p_AB_list, p_BA_list = [], []
    nAB_list, nBA_list = [], []

    for rd in run_dirs:
        gAB, gBA, pAB, pBA, nAB, nBA = _seed_qcurves_for_run(
            rd, warmup=warmup, L=L, policy_signal=policy_signal, qd_p=qd_p, early_frac=early_frac
        )
        # Safety: ensure lengths match
        if len(gAB) != L:
            raise ValueError(f"[{rd.name}] q_curve length {len(gAB)} != L={L}. Did you compute with L={L}?")
        g_AB_list.append(gAB); g_BA_list.append(gBA)
        p_AB_list.append(pAB); p_BA_list.append(pBA)
        nAB_list.append(nAB); nBA_list.append(nBA)

    # aggregate across seeds (median + IQR band)
    gAB_med, gAB_lo, gAB_hi = _aggregate_curves(g_AB_list, q_lo=band_lo, q_hi=band_hi)
    gBA_med, gBA_lo, gBA_hi = _aggregate_curves(g_BA_list, q_lo=band_lo, q_hi=band_hi)
    pAB_med, pAB_lo, pAB_hi = _aggregate_curves(p_AB_list, q_lo=band_lo, q_hi=band_hi)
    pBA_med, pBA_lo, pBA_hi = _aggregate_curves(p_BA_list, q_lo=band_lo, q_hi=band_hi)

    tau = np.arange(L)

    fig = plt.figure(figsize=(10, 4))

    # ---- g-score panel
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(tau, gAB_med, label=f"regime A -> B")
    ax1.fill_between(tau, gAB_lo, gAB_hi, alpha=0.2)
    ax1.plot(tau, gBA_med, label=f"regime B -> A")
    ax1.fill_between(tau, gBA_lo, gBA_hi, alpha=0.2)
    ax1.set_title("g-score (seed-median ± IQR)")
    ax1.set_xlabel("tau")
    ax1.legend(loc="upper center")

    # ---- policy panel
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(tau, pAB_med, label=f"regime A -> B")
    ax2.fill_between(tau, pAB_lo, pAB_hi, alpha=0.2)
    ax2.plot(tau, pBA_med, label=f"regime B -> A")
    ax2.fill_between(tau, pBA_lo, pBA_hi, alpha=0.2)
    ax2.set_title(f"{policy_signal}-z (seed-median ± IQR)")
    ax2.set_xlabel("tau")
    ax2.legend(loc="upper center")

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

if __name__ == "__main__":
    # EDIT: put your 5 sweep run_dirs here (one per seed)
    RUN_DIRS = [
        "outputs/runs/seed1/p40",
        "outputs/runs/seed2/p40",
        "outputs/runs/seed3/p40",
        "outputs/runs/seed4/p40",
        "outputs/runs/seed5/p40",
    ]

    plot_seed_aggregate(
        run_dirs=RUN_DIRS,
        warmup=150,
        L=40,
        policy_signal="entropy",
        qd_p=0.5,       # within-seed: median across switch events
        band_lo=0.25,   # across-seed: IQR band
        band_hi=0.75,
        early_frac=0.25,
        save_path="outputs/figs/fig_seed_aggregate_quantilecurves_P40.png",
        show=True,
    )
