# cear_pilot/analysis/figure_switch_eval.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import matplotlib.pyplot as plt

from cear_pilot.analysis.metrics import (
    detect_delay_quantile,
    hysteresis_area,
    transition_lag_half_rise,
    switch_distribution_stats,
    dissociation_index,
)


def load_table(traj_path: Path):
    import pandas as pd
    if traj_path.suffix == ".parquet":
        return pd.read_parquet(traj_path)
    return pd.read_csv(traj_path)


def find_traj(run_dir: Path) -> Path:
    for ext in [".parquet", ".csv"]:
        p = run_dir / f"traj{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No traj.parquet/csv in {run_dir}")


def g_signed_score_from_df(df, warmup: int, regime: np.ndarray) -> np.ndarray:
    """
    Signed projection of g onto the regime-separating direction.
    """
    g_cols = [c for c in df.columns if c.startswith("g_")]
    if len(g_cols) == 0:
        raise ValueError("No g_* columns found in traj.")
    G = df[g_cols].to_numpy(dtype=np.float32)
    T = G.shape[0]

    idx = np.arange(T)
    post = idx >= int(warmup)

    A = post & (regime == 0)
    B = post & (regime == 1)

    # Fallback if one side is empty
    if A.sum() < 10 or B.sum() < 10:
        mu = G[: max(10, min(int(warmup), T))].mean(axis=0)
        return (G - mu[None, :]).sum(axis=-1).astype(np.float32)

    muA = G[A].mean(axis=0)
    muB = G[B].mean(axis=0)

    w = (muB - muA).astype(np.float32)
    w = w / (np.linalg.norm(w) + 1e-8)

    s = (G - muA[None, :]) @ w
    return s.astype(np.float32)


def _segment_boundaries(t: np.ndarray, switch_times_idx: np.ndarray, warmup_t: int) -> List[int]:
    t_arr = t.astype(int)
    b: List[int] = [int(warmup_t)]
    for idx in switch_times_idx.tolist():
        b.append(int(t_arr[idx]))
    b = sorted(set(b))
    b.append(int(t_arr[-1]) + 1)
    return b


def _shade_regimes(ax, t: np.ndarray, regime: np.ndarray, switch_times_idx: np.ndarray, warmup_t: int) -> None:
    t_arr = t.astype(int)
    ax.axvspan(int(t_arr[0]), int(warmup_t), alpha=0.08)

    boundaries = _segment_boundaries(t_arr, switch_times_idx, warmup_t)
    for i in range(len(boundaries) - 1):
        a = boundaries[i]
        b = boundaries[i + 1]
        if b <= a:
            continue

        idx_a = int(np.searchsorted(t_arr, a, side="left"))
        idx_a = min(max(idx_a, 0), len(regime) - 1)
        r = int(regime[idx_a])

        band_alpha = 0.06 if r == 0 else 0.10
        ax.axvspan(a, b, alpha=band_alpha)


def to_jsonable(x):
    """
    Convert numpy types/arrays to JSON-serializable Python types.
    """
    if x is None:
        return None
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_jsonable(v) for v in x]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--pre_window", type=int, default=80)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--consec", type=int, default=3)
    ap.add_argument("--L", type=int, default=60)
    ap.add_argument("--policy_signal", type=str, default="entropy", choices=["entropy", "pi_max", "margin"])

    # new knobs for distribution diagnostics
    ap.add_argument("--early_frac", type=float, default=0.25, help="early window fraction for Amp_IQR")
    ap.add_argument("--qd_p", type=float, default=0.5, help="quantile for QD (robust check)")

    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    df = load_table(find_traj(run_dir))

    # Prefer t_global if present
    t = df["t_global"].to_numpy(dtype=int) if "t_global" in df.columns else df["t"].to_numpy(dtype=int)

    if "regime" not in df.columns or "switch" not in df.columns:
        raise KeyError("traj must contain 'regime' and 'switch' columns. Re-run the collector.")

    regime = df["regime"].to_numpy(dtype=int)
    switches = df["switch"].to_numpy(dtype=int)

    # Read run meta (optional)
    meta_path = run_dir / "meta.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        P = int(meta.get("period", -1))
        W = int(meta.get("warmup", args.warmup))
        T = int(meta.get("T", -1))
    else:
        P, W, T = -1, int(args.warmup), -1

    # Adjust L if period is known
    if P > 0:
        args.L = min(int(args.L), max(2, P - 1))

    # --- scores
    s_g = g_signed_score_from_df(df, warmup=args.warmup, regime=regime)

    if args.policy_signal not in df.columns:
        raise KeyError(f"Missing policy signal column: {args.policy_signal}")
    s_pi_raw = df[args.policy_signal].to_numpy(dtype=np.float32)

    # Normalize policy signal to comparable scale (z-score)
    s_pi = (s_pi_raw - s_pi_raw.mean()) / (s_pi_raw.std() + 1e-6)

    # Switch indices (after warmup)
    switch_times = np.where((switches == 1) & (t >= int(args.warmup)))[0]

    # --- A) detection delay per switch
    delays_g: List[int] = []
    delays_pi: List[int] = []

    for idx in switch_times:
        sw_t = int(idx)

        dg = detect_delay_quantile(
            score=s_g,
            switch_t=sw_t,
            pre_window=args.pre_window,
            alpha=args.alpha,
            consec=args.consec,
        )
        dp = detect_delay_quantile(
            score=s_pi,
            switch_t=sw_t,
            pre_window=args.pre_window,
            alpha=args.alpha,
            consec=args.consec,
        )

        if dg is not None:
            delays_g.append(int(dg))
        if dp is not None:
            delays_pi.append(int(dp))

    # --- B) hysteresis mean curves (legacy plot support)
    hyst_g = hysteresis_area(s_g, regime, switches, L=args.L)
    hyst_pi = hysteresis_area(s_pi, regime, switches, L=args.L)

    # --- B2) transition lag (half-rise time)
    lag_g = transition_lag_half_rise(s_g, regime, switches, L=args.L)
    lag_pi = transition_lag_half_rise(s_pi, regime, switches, L=args.L)

    # --- NEW) distributional diagnostics (shape formalization)
    stats_g = switch_distribution_stats(
        score=s_g,
        regime=regime,
        switches=switches,
        L=args.L,
        q_p=float(args.qd_p),
        early_frac=float(args.early_frac),
    )
    stats_pi = switch_distribution_stats(
        score=s_pi,
        regime=regime,
        switches=switches,
        L=args.L,
        q_p=float(args.qd_p),
        early_frac=float(args.early_frac),
    )
    dsi = dissociation_index(stats_g, stats_pi)

    # --- summary
    def summarize(x: List[int]) -> Optional[Dict[str, float]]:
        if len(x) == 0:
            return None
        return {"n": int(len(x)), "mean": float(np.mean(x)), "median": float(np.median(x))}

    out: Dict[str, Any] = {
        "delay_g": summarize(delays_g),
        "delay_pi": summarize(delays_pi),
        "lag_g": {"up": lag_g["lag_up"], "dn": lag_g["lag_dn"], "L": lag_g["L"]},
        "lag_pi": {"up": lag_pi["lag_up"], "dn": lag_pi["lag_dn"], "L": lag_pi["L"]},
        "hysteresis_g": {
            "n_up": hyst_g["n_up"],
            "n_dn": hyst_g["n_dn"],
            "area": hyst_g["area"],
        },
        "hysteresis_pi": {
            "n_up": hyst_pi["n_up"],
            "n_dn": hyst_pi["n_dn"],
            "area": hyst_pi["area"],
        },
        "dist_stats_g": stats_g,
        "dist_stats_pi": stats_pi,
        "DSI": dsi,
        "policy_signal": args.policy_signal,
        "params": vars(args),
        "meta": {"period": P, "warmup": W, "T": T},
    }

    print(to_jsonable(out))

    # --- plot
    figdir = run_dir / "figs"
    figdir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 7))

    # (1) score time series + regime shading + switch markers
    ax1 = plt.subplot(2, 1, 1)
    _shade_regimes(ax1, t=t, regime=regime, switch_times_idx=switch_times, warmup_t=int(args.warmup))

    ax1.plot(t, s_g, label="g_score")
    ax1.plot(t, s_pi, label=f"{args.policy_signal}_z")

    for sw in switch_times:
        ax1.axvline(int(t[sw]), linewidth=0.7, alpha=0.35)

    ax1.set_title(f"Scores + regime shading | P={P}  warmup={W}  T={T}  (policy={args.policy_signal})")
    ax1.set_xlabel("t")
    ax1.legend()

    # (2) hysteresis mean curves (g)
    ax2 = plt.subplot(2, 2, 3)
    if hyst_g["m_up"] is not None:
        ax2.plot(hyst_g["m_up"], label="A->B")
    if hyst_g["m_dn"] is not None:
        ax2.plot(hyst_g["m_dn"], label="B->A")
    ax2.set_title("g hysteresis (mean)")
    ax2.set_xlabel("tau")
    ax2.legend()

    # (3) hysteresis mean curves (policy)
    ax3 = plt.subplot(2, 2, 4)
    if hyst_pi["m_up"] is not None:
        ax3.plot(hyst_pi["m_up"], label="A->B")
    if hyst_pi["m_dn"] is not None:
        ax3.plot(hyst_pi["m_dn"], label="B->A")
    ax3.set_title(f"{args.policy_signal} hysteresis (mean)")
    ax3.set_xlabel("tau")
    ax3.legend()

    out_png = figdir / f"fig_switch_eval_{args.policy_signal}.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"[OK] Saved: {out_png}")

    # Save json summary (convert numpy objects)
    (run_dir / "switch_eval.json").write_text(json.dumps(to_jsonable(out), indent=2))


if __name__ == "__main__":
    main()

#%% old code
# # cear_pilot/analysis/figure_switch_eval.py
# # -*- coding: utf-8 -*-

# from __future__ import annotations

# import argparse
# import json
# from pathlib import Path
# from typing import Dict, Any, Optional, List

# import numpy as np
# import matplotlib.pyplot as plt

# from cear_pilot.analysis.metrics import detect_delay_quantile, hysteresis_area, transition_lag_half_rise


# def load_table(traj_path: Path):
#     import pandas as pd
#     if traj_path.suffix == ".parquet":
#         return pd.read_parquet(traj_path)
#     return pd.read_csv(traj_path)


# def find_traj(run_dir: Path) -> Path:
#     for ext in [".parquet", ".csv"]:
#         p = run_dir / f"traj{ext}"
#         if p.exists():
#             return p
#     raise FileNotFoundError(f"No traj.parquet/csv in {run_dir}")


# # def g_score_from_df(df, warmup: int) -> np.ndarray:
# #     g_cols = [c for c in df.columns if c.startswith("g_")]
# #     if len(g_cols) == 0:
# #         raise ValueError("No g_* columns found in traj.")
# #     G = df[g_cols].to_numpy(dtype=np.float32)

# #     # Use warmup mean as baseline (pre-mean)
# #     w = max(10, int(warmup))
# #     w = min(w, G.shape[0])
# #     mu = G[:w].mean(axis=0)
# #     return np.linalg.norm(G - mu[None, :], axis=-1).astype(np.float32)


# def g_signed_score_from_df(df, warmup: int, regime: np.ndarray, buffer: int = 2) -> np.ndarray:
#     g_cols = [c for c in df.columns if c.startswith("g_")]
#     G = df[g_cols].to_numpy(dtype=np.float32)
#     T = G.shape[0]

#     # Use only post-warmup points, and exclude a small buffer around switches if desired
#     idx = np.arange(T)
#     post = idx >= int(warmup)

#     # Regime masks in post-warmup
#     A = post & (regime == 0)
#     B = post & (regime == 1)

#     # Fallback if one side is empty
#     if A.sum() < 10 or B.sum() < 10:
#         mu = G[:max(10, min(int(warmup), T))].mean(axis=0)
#         return (G - mu[None, :]).sum(axis=-1).astype(np.float32)

#     muA = G[A].mean(axis=0)
#     muB = G[B].mean(axis=0)

#     w = (muB - muA).astype(np.float32)
#     w = w / (np.linalg.norm(w) + 1e-8)

#     # Signed projection (centered at muA)
#     s = (G - muA[None, :]) @ w
#     return s.astype(np.float32)


# def _segment_boundaries(t: np.ndarray, switch_times_idx: np.ndarray, warmup_t: int) -> List[int]:
#     # Build boundaries in "time" coordinates, not indices
#     t_arr = t.astype(int)

#     b: List[int] = [int(warmup_t)]
#     for idx in switch_times_idx.tolist():
#         b.append(int(t_arr[idx]))

#     b = sorted(set(b))
#     b.append(int(t_arr[-1]) + 1)
#     return b


# def _shade_regimes(ax, t: np.ndarray, regime: np.ndarray, switch_times_idx: np.ndarray, warmup_t: int) -> None:
#     # Shade warmup region
#     t_arr = t.astype(int)
#     ax.axvspan(int(t_arr[0]), int(warmup_t), alpha=0.08)

#     # Shade post-warmup regimes as alternating bands
#     boundaries = _segment_boundaries(t_arr, switch_times_idx, warmup_t)

#     for i in range(len(boundaries) - 1):
#         a = boundaries[i]
#         b = boundaries[i + 1]
#         if b <= a:
#             continue

#         # Get regime label at the start boundary a
#         idx_a = int(np.searchsorted(t_arr, a, side="left"))
#         idx_a = min(max(idx_a, 0), len(regime) - 1)
#         r = int(regime[idx_a])

#         # Slightly different opacity for A vs B
#         band_alpha = 0.06 if r == 0 else 0.10
#         ax.axvspan(a, b, alpha=band_alpha)


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--run_dir", type=str, required=True)
#     ap.add_argument("--warmup", type=int, default=150)
#     ap.add_argument("--pre_window", type=int, default=80)
#     ap.add_argument("--alpha", type=float, default=0.05)
#     ap.add_argument("--consec", type=int, default=3)
#     ap.add_argument("--L", type=int, default=60)
#     ap.add_argument("--policy_signal", type=str, default="entropy", choices=["entropy", "pi_max", "margin"])
#     args = ap.parse_args()

#     run_dir = Path(args.run_dir)
#     df = load_table(find_traj(run_dir))

#     # Prefer t_global if present (collector should store it)
#     t = df["t_global"].to_numpy(dtype=int) if "t_global" in df.columns else df["t"].to_numpy(dtype=int)

#     if "regime" not in df.columns or "switch" not in df.columns:
#         raise KeyError("traj must contain 'regime' and 'switch' columns. Re-run the collector.")

#     regime = df["regime"].to_numpy(dtype=int)
#     switches = df["switch"].to_numpy(dtype=int)

#     # Read run meta (optional)
#     meta_path = run_dir / "meta.json"
#     meta = {}
#     if meta_path.exists():
#         meta = json.loads(meta_path.read_text())
#         P = int(meta.get("period", -1))
#         W = int(meta.get("warmup", args.warmup))
#         T = int(meta.get("T", -1))
#     else:
#         P, W, T = -1, int(args.warmup), -1
    
#     # After reading P from meta
#     if P > 0:
#         args.L = min(int(args.L), max(2, P - 1))

#     # --- scores
#     s_g = g_signed_score_from_df(df, warmup=args.warmup, regime=regime)

#     if args.policy_signal not in df.columns:
#         raise KeyError(f"Missing policy signal column: {args.policy_signal}")
#     s_pi_raw = df[args.policy_signal].to_numpy(dtype=np.float32)

#     # Normalize policy signal to a comparable scale (z-score)
#     s_pi = (s_pi_raw - s_pi_raw.mean()) / (s_pi_raw.std() + 1e-6)

#     # Switch indices (after warmup)
#     switch_times = np.where((switches == 1) & (t >= int(args.warmup)))[0]

#     # --- A) detection delay per switch
#     delays_g: List[int] = []
#     delays_pi: List[int] = []

#     for idx in switch_times:
#         sw_t = int(idx)

#         dg = detect_delay_quantile(
#             score=s_g,
#             switch_t=sw_t,
#             pre_window=args.pre_window,
#             alpha=args.alpha,
#             consec=args.consec,
#         )
#         dp = detect_delay_quantile(
#             score=s_pi,
#             switch_t=sw_t,
#             pre_window=args.pre_window,
#             alpha=args.alpha,
#             consec=args.consec,
#         )

#         if dg is not None:
#             delays_g.append(int(dg))
#         if dp is not None:
#             delays_pi.append(int(dp))

#     # --- B) hysteresis area
#     hyst_g = hysteresis_area(s_g, regime, switches, L=args.L)
#     hyst_pi = hysteresis_area(s_pi, regime, switches, L=args.L)
    
#     # --- B2) transition lag (half-rise time)
#     lag_g = transition_lag_half_rise(s_g, regime, switches, L=args.L)
#     lag_pi = transition_lag_half_rise(s_pi, regime, switches, L=args.L)

#     # --- summary
#     def summarize(x: List[int]) -> Optional[Dict[str, float]]:
#         if len(x) == 0:
#             return None
#         return {"n": int(len(x)), "mean": float(np.mean(x)), "median": float(np.median(x))}

#     out: Dict[str, Any] = {
#         "delay_g": summarize(delays_g),
#         "delay_pi": summarize(delays_pi),
#         # "hysteresis_g": {"area": hyst_g["area"], "n_up": hyst_g["n_up"], "n_dn": hyst_g["n_dn"]},
#         # "hysteresis_pi": {"area": hyst_pi["area"], "n_up": hyst_pi["n_up"], "n_dn": hyst_pi["n_dn"]},
#         "lag_g": {"up": lag_g["lag_up"], "dn": lag_g["lag_dn"], "L": lag_g["L"]},
#         "lag_pi": {"up": lag_pi["lag_up"], "dn": lag_pi["lag_dn"], "L": lag_pi["L"]},
#         "policy_signal": args.policy_signal,
#         "params": vars(args),
#         "meta": {"period": P, "warmup": W, "T": T},
#     }
#     print(out)

#     # --- plot
#     figdir = run_dir / "figs"
#     figdir.mkdir(parents=True, exist_ok=True)

#     plt.figure(figsize=(11, 7))

#     # (1) score time series + regime shading + switch markers
#     ax1 = plt.subplot(2, 1, 1)

#     _shade_regimes(ax1, t=t, regime=regime, switch_times_idx=switch_times, warmup_t=int(args.warmup))

#     ax1.plot(t, s_g, label="g_score")
#     ax1.plot(t, s_pi, label=f"{args.policy_signal}_z")

#     for sw in switch_times:
#         ax1.axvline(int(t[sw]), linewidth=0.7, alpha=0.35)

#     # # Period label
#     # ax1.text(
#     #     0.01, 0.90,
#     #     f"lag_g(up/dn)={lag_g['lag_up']} / {lag_g['lag_dn']}\n"
#     #     f"lag_pi(up/dn)={lag_pi['lag_up']} / {lag_pi['lag_dn']}",
#     #     transform=ax1.transAxes,
#     #     ha="left", va="top", fontsize=9, alpha=0.9
#     # )

#     ax1.set_title(f"Scores + regime shading | P={P}  warmup={W}  T={T}  (policy={args.policy_signal})")
#     ax1.set_xlabel("t")
#     ax1.legend()

#     # (2) hysteresis mean curves (g)
#     ax2 = plt.subplot(2, 2, 3)
#     if hyst_g["m_up"] is not None:
#         ax2.plot(hyst_g["m_up"], label="A->B")
#     if hyst_g["m_dn"] is not None:
#         ax2.plot(hyst_g["m_dn"], label="B->A")
#     ax2.set_title(f"g hysteresis")
#     ax2.set_xlabel("tau")
#     ax2.legend()

#     # (3) hysteresis mean curves (policy)
#     ax3 = plt.subplot(2, 2, 4)
#     if hyst_pi["m_up"] is not None:
#         ax3.plot(hyst_pi["m_up"], label="A->B")
#     if hyst_pi["m_dn"] is not None:
#         ax3.plot(hyst_pi["m_dn"], label="B->A")
#     ax3.set_title(f"{args.policy_signal} hysteresis")
#     ax3.set_xlabel("tau")
#     ax3.legend()

#     out_png = figdir / f"fig_switch_eval_{args.policy_signal}.png"
#     plt.tight_layout()
#     plt.savefig(out_png, dpi=200, bbox_inches="tight")
#     print(f"[OK] Saved: {out_png}")

#     # Save json summary
#     (run_dir / "switch_eval.json").write_text(json.dumps(out, indent=2))


# if __name__ == "__main__":
#     main()
