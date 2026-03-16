# cear_pilot/analysis/metrics.py
# -*- coding: utf-8 -*-
"""
Metrics for "order parameter" behavior:
- drift in g
- recovery time after perturbation
- silhouette score by zone (in embedding space)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Any

import numpy as np


def g_columns(df) -> list[str]:
    return [c for c in df.columns if c.startswith("g_")]


def s_columns(df) -> list[str]:
    return [c for c in df.columns if c.startswith("s_")]


def obs_columns(df) -> list[str]:
    return [c for c in df.columns if c.startswith("obs_")]


def drift_norm(G: np.ndarray) -> np.ndarray:
    """
    G: (T, D)
    returns stepwise ||g_t - g_{t-1}||, length T (with 0 at t=0)
    """
    d = np.zeros((G.shape[0],), dtype=np.float32)
    if G.shape[0] <= 1:
        return d
    d[1:] = np.linalg.norm(G[1:] - G[:-1], axis=-1)
    return d


def recovery_time(
    G: np.ndarray,
    t0: int,
    window: int = 20,
    threshold: float = 0.15,
) -> Optional[int]:
    """
    Define recovery as: distance to pre-perturb mean <= threshold * pre-perturb std
    using a pre window [t0-window, t0).
    Returns number of steps after t0 until recovered, or None.
    """
    T = G.shape[0]
    a = max(0, t0 - window)
    b = max(0, t0)

    if b - a < 5:
        return None

    pre = G[a:b]
    mu = pre.mean(axis=0)
    sig = pre.std(axis=0) + 1e-6

    # distance in standardized space
    def dist(g):
        return np.linalg.norm((g - mu) / sig)

    for t in range(t0, T):
        if dist(G[t]) <= threshold:
            return t - t0
    return None


def silhouette_by_zone(emb: np.ndarray, zone: np.ndarray) -> Optional[float]:
    """
    emb: (N, k), zone: (N,)
    """
    try:
        from sklearn.metrics import silhouette_score
        # Need at least 2 labels
        if len(np.unique(zone)) < 2:
            return None
        return float(silhouette_score(emb, zone))
    except Exception:
        return None


def detect_delay_quantile(
    score: np.ndarray,
    switch_t: int,
    pre_window: int = 80,
    alpha: float = 0.05,
    consec: int = 3,
) -> Optional[int]:
    """
    Change-point detection delay.
    threshold = (1-alpha) quantile of pre-window score.
    delay = first t>=switch_t where score[t:t+consec] all exceed threshold.

    Returns delay steps, or None if not detected.
    """
    T = len(score)
    a = max(0, switch_t - pre_window)
    b = max(0, switch_t)
    if b - a < 10:
        return None

    thr = float(np.quantile(score[a:b], 1.0 - alpha))
    for t in range(switch_t, T - consec + 1):
        if np.all(score[t:t+consec] > thr):
            return int(t - switch_t)
    return None


def hysteresis_area(
    score: np.ndarray,
    regime: np.ndarray,
    switches: np.ndarray,
    L: int = 60,
) -> Dict[str, Any]:
    """
    Compute hysteresis (A->B vs B->A) after warmup using local windows.

    - regime[t] in {0,1}
    - switches[t]=1 at the switch time (same length as score)
    - For each switch time t0, take window [t0, t0+L)
      and collect score segments separately for A->B and B->A.
    - Mean trajectories m_up, m_dn; area = mean(|m_up - m_dn|)

    Returns dict with area and mean curves.
    """
    T = len(score)
    idx = np.where(switches.astype(int) == 1)[0].tolist()
    seg_up = []  # 0->1
    seg_dn = []  # 1->0

    for t0 in idx:
        if t0 + L > T:
            continue
        r0 = int(regime[t0-1]) if t0 - 1 >= 0 else int(regime[t0])
        r1 = int(regime[t0])
        seg = score[t0:t0+L].astype(np.float32)

        if r0 == 0 and r1 == 1:
            seg_up.append(seg)
        elif r0 == 1 and r1 == 0:
            seg_dn.append(seg)

    def mean_or_none(segs: List[np.ndarray]) -> Optional[np.ndarray]:
        if len(segs) == 0:
            return None
        return np.stack(segs, axis=0).mean(axis=0)

    m_up = mean_or_none(seg_up)
    m_dn = mean_or_none(seg_dn)

    out: Dict[str, Any] = {
        "n_up": len(seg_up),
        "n_dn": len(seg_dn),
        "m_up": m_up,
        "m_dn": m_dn,
        "area": None,
    }
    if m_up is not None and m_dn is not None:
        out["area"] = float(np.mean(np.abs(m_up - m_dn)))
    return out

def transition_lag_half_rise(
    score: np.ndarray,
    regime: np.ndarray,
    switches: np.ndarray,
    L: int,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    """
    Compute transition lag (half-rise time) separately for A->B and B->A switches.

    For each switch time t0:
      - pre baseline: mean(score[t0-L : t0])   (clipped to valid range)
      - post target:  mean(score[t0 : t0+L])   (clipped)
      - half level: baseline + 0.5*(target-baseline)
      - lag: smallest tau>=0 such that score[t0+tau] crosses the half level in the correct direction.

    Returns:
      {
        "lag_up": {"n":..., "mean":..., "median":...} or None,
        "lag_dn": {"n":..., "mean":..., "median":...} or None,
        "raw_up": [...],
        "raw_dn": [...],
      }
    """
    T = len(score)
    idx = np.where(switches.astype(int) == 1)[0].tolist()

    raw_up: List[int] = []
    raw_dn: List[int] = []

    for t0 in idx:
        if t0 <= 1 or t0 >= T - 2:
            continue

        r0 = int(regime[t0 - 1]) if t0 - 1 >= 0 else int(regime[t0])
        r1 = int(regime[t0])

        a0 = max(0, t0 - L)
        a1 = t0
        b0 = t0
        b1 = min(T, t0 + L)

        if (a1 - a0) < max(3, L // 4) or (b1 - b0) < max(3, L // 4):
            continue

        baseline = float(np.mean(score[a0:a1]))
        target = float(np.mean(score[b0:b1]))
        delta = target - baseline
        if abs(delta) < eps:
            continue

        half = baseline + 0.5 * delta

        # Find first crossing in the correct direction
        seg = score[b0:b1]
        lag = None
        if delta > 0:
            # rising: first time >= half
            hits = np.where(seg >= half)[0]
        else:
            # falling: first time <= half
            hits = np.where(seg <= half)[0]

        if hits.size > 0:
            lag = int(hits[0])
            if r0 == 0 and r1 == 1:
                raw_up.append(lag)
            elif r0 == 1 and r1 == 0:
                raw_dn.append(lag)

    def summarize(xs: List[int]) -> Optional[Dict[str, float]]:
        if len(xs) == 0:
            return None
        return {"n": int(len(xs)), "mean": float(np.mean(xs)), "median": float(np.median(xs))}

    return {
        "lag_up": summarize(raw_up),
        "lag_dn": summarize(raw_dn),
        "raw_up": raw_up,
        "raw_dn": raw_dn,
        "L": int(L),
    }

# -----------------------------
# Switch / distribution diagnostics (shape formalization)
# -----------------------------

def _collect_switch_segments(
    score: np.ndarray,
    regime: np.ndarray,
    switches: np.ndarray,
    L: int,
) -> Dict[str, Any]:
    """
    Collect event-aligned segments after each regime switch.

    Returns:
      {
        "seg_up": np.ndarray (K_up, L) or None,
        "seg_dn": np.ndarray (K_dn, L) or None,
        "n_up": int,
        "n_dn": int,
      }
    """
    T = len(score)
    idx = np.where(switches.astype(int) == 1)[0].tolist()
    seg_up: List[np.ndarray] = []
    seg_dn: List[np.ndarray] = []

    for t0 in idx:
        if t0 + L > T:
            continue
        r0 = int(regime[t0 - 1]) if t0 - 1 >= 0 else int(regime[t0])
        r1 = int(regime[t0])
        seg = score[t0 : t0 + L].astype(np.float32)

        if r0 == 0 and r1 == 1:
            seg_up.append(seg)
        elif r0 == 1 and r1 == 0:
            seg_dn.append(seg)

    def stack_or_none(segs: List[np.ndarray]) -> Optional[np.ndarray]:
        if len(segs) == 0:
            return None
        return np.stack(segs, axis=0)  # (K, L)

    X_up = stack_or_none(seg_up)
    X_dn = stack_or_none(seg_dn)

    return {
        "seg_up": X_up,
        "seg_dn": X_dn,
        "n_up": 0 if X_up is None else int(X_up.shape[0]),
        "n_dn": 0 if X_dn is None else int(X_dn.shape[0]),
    }


def _wasserstein_1d_empirical(a: np.ndarray, b: np.ndarray, q_grid: int = 129) -> Optional[float]:
    """
    1D Wasserstein-1 between empirical distributions (uniform weights), no scipy.
    Approximates integral of |Q_a(p) - Q_b(p)| dp on a fixed quantile grid.

    Returns None if inputs are too small.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3 or b.size < 3:
        return None

    q = np.linspace(0.0, 1.0, int(q_grid), dtype=np.float64)
    qa = np.quantile(a, q, method="linear")
    qb = np.quantile(b, q, method="linear")
    return float(np.mean(np.abs(qa - qb)))


def nonstationarity_W1(
    X: Optional[np.ndarray],
) -> Dict[str, Any]:
    """
    Distribution Nonstationarity Index via Wasserstein-1.

    X: (K, L) event-aligned matrix; at each tau, distribution is X[:, tau] across events.

    NSI_W1 = sum_{tau=1..L-1} W1(P_tau, P_{tau-1})

    Returns:
      { "NSI_W1": float|None, "w1_increments": np.ndarray|None }
    """
    if X is None or X.shape[0] < 3 or X.shape[1] < 3:
        return {"NSI_W1": None, "w1_increments": None}

    K, L = X.shape
    inc = np.zeros((L - 1,), dtype=np.float32)

    ok = True
    for tau in range(1, L):
        d = _wasserstein_1d_empirical(X[:, tau], X[:, tau - 1])
        if d is None:
            ok = False
            break
        inc[tau - 1] = float(d)

    if not ok:
        return {"NSI_W1": None, "w1_increments": None}

    return {"NSI_W1": float(np.sum(inc)), "w1_increments": inc}


def quantile_drift(
    X: Optional[np.ndarray],
    p: float = 0.5,
) -> Dict[str, Any]:
    """
    Robust check:
      q_p(tau) = quantile_p over events at each tau
      QD = sum_{tau=1..L-1} |q_p(tau) - q_p(tau-1)|

    Returns:
      { "QD": float|None, "q_curve": np.ndarray|None }
    """
    if X is None or X.shape[0] < 3 or X.shape[1] < 3:
        return {"QD": None, "q_curve": None}

    q_curve = np.quantile(X, float(p), axis=0, method="linear").astype(np.float32)  # (L,)
    qd = float(np.sum(np.abs(q_curve[1:] - q_curve[:-1])))
    return {"QD": qd, "q_curve": q_curve}


def amplitude_IQR(
    X: Optional[np.ndarray],
    early_frac: float = 0.25,
    agg: str = "median",
) -> Dict[str, Any]:
    """
    Event-level dispersion amplitude.

    A(tau) = IQR_k( X(k,tau) ) = Q75 - Q25 across events.
    Aggregate amplitude over early taus only.

    Returns:
      { "Amp_IQR": float|None, "iqr_curve": np.ndarray|None, "tau_early": int|None }
    """
    if X is None or X.shape[0] < 3 or X.shape[1] < 3:
        return {"Amp_IQR": None, "iqr_curve": None, "tau_early": None}

    K, L = X.shape
    q25 = np.quantile(X, 0.25, axis=0, method="linear")
    q75 = np.quantile(X, 0.75, axis=0, method="linear")
    iqr = (q75 - q25).astype(np.float32)

    tau_e = max(2, int(np.floor(float(early_frac) * L)))
    tau_e = min(tau_e, L)
    early = iqr[:tau_e]

    if str(agg).lower().strip() == "mean":
        amp = float(np.mean(early))
    else:
        amp = float(np.median(early))

    return {"Amp_IQR": amp, "iqr_curve": iqr, "tau_early": int(tau_e)}


def switch_distribution_stats(
    score: np.ndarray,
    regime: np.ndarray,
    switches: np.ndarray,
    L: int,
    q_p: float = 0.5,
    early_frac: float = 0.25,
) -> Dict[str, Any]:
    """
    Compute distributional diagnostics for transition-aligned switch segments.

    Returns:
      {
        "up": {n_events, NSI_W1, QD, Amp_IQR, ...curves...},
        "dn": {...},
        "combined": {NSI_W1, QD, Amp_IQR, n_events},
        "L": int,
      }
    """
    segs = _collect_switch_segments(score, regime, switches, L=L)
    X_up = segs["seg_up"]
    X_dn = segs["seg_dn"]

    def stats_one(X: Optional[np.ndarray]) -> Dict[str, Any]:
        w1 = nonstationarity_W1(X)
        qd = quantile_drift(X, p=q_p)
        amp = amplitude_IQR(X, early_frac=early_frac, agg="median")

        return {
            "n_events": 0 if X is None else int(X.shape[0]),
            "NSI_W1": w1["NSI_W1"],
            "QD": qd["QD"],
            "Amp_IQR": amp["Amp_IQR"],
            # curves are useful for plotting/debug; if you want smaller json, drop these
            "w1_increments": None if w1["w1_increments"] is None else w1["w1_increments"],
            "q_curve": None if qd["q_curve"] is None else qd["q_curve"],
            "iqr_curve": None if amp["iqr_curve"] is None else amp["iqr_curve"],
            "tau_early": amp["tau_early"],
        }

    up = stats_one(X_up)
    dn = stats_one(X_dn)

    def mean_if_present(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None and b is None:
            return None
        if a is None:
            return float(b)
        if b is None:
            return float(a)
        return float(0.5 * (float(a) + float(b)))

    combined = {
        "NSI_W1": mean_if_present(up["NSI_W1"], dn["NSI_W1"]),
        "QD": mean_if_present(up["QD"], dn["QD"]),
        "Amp_IQR": mean_if_present(up["Amp_IQR"], dn["Amp_IQR"]),
        "n_events": int(up["n_events"] + dn["n_events"]),
    }

    return {"up": up, "dn": dn, "combined": combined, "L": int(L)}


def dissociation_index(
    stats_g: Dict[str, Any],
    stats_pi: Dict[str, Any],
    eps: float = 1e-9,
) -> Optional[float]:
    """
    Optional scalar summary:
      DSI = log(NS_g/NS_pi) + log(A_pi/A_g)

    Uses combined NSI_W1 and combined Amp_IQR.
    """
    Ng = stats_g.get("combined", {}).get("NSI_W1", None)
    Np = stats_pi.get("combined", {}).get("NSI_W1", None)
    Ag = stats_g.get("combined", {}).get("Amp_IQR", None)
    Ap = stats_pi.get("combined", {}).get("Amp_IQR", None)

    if Ng is None or Np is None or Ag is None or Ap is None:
        return None

    return float(np.log((Ng + eps) / (Np + eps)) + np.log((Ap + eps) / (Ag + eps)))
