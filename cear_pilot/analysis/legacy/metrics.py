# cear_pilot/analysis/metrics.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional, Tuple, Dict, List, Any

import numpy as np


def g_columns(df) -> list[str]:
    return [c for c in df.columns if c.startswith("g_")]


def s_columns(df) -> list[str]:
    return [c for c in df.columns if c.startswith("s_")]


def obs_columns(df) -> list[str]:
    return [c for c in df.columns if c.startswith("obs_")]


def drift_norm(G: np.ndarray) -> np.ndarray:
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
    T = G.shape[0]
    a = max(0, t0 - window)
    b = max(0, t0)

    if b - a < 5:
        return None

    pre = G[a:b]
    mu = pre.mean(axis=0)
    sig = pre.std(axis=0) + 1e-6

    def dist(g):
        return np.linalg.norm((g - mu) / sig)

    for t in range(t0, T):
        if dist(G[t]) <= threshold:
            return t - t0
    return None


def silhouette_by_zone(emb: np.ndarray, zone: np.ndarray) -> Optional[float]:
    try:
        from sklearn.metrics import silhouette_score
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
    T = len(score)
    idx = np.where(switches.astype(int) == 1)[0].tolist()
    seg_up = []
    seg_dn = []

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

    def mean_or_none(segs):
        if len(segs) == 0:
            return None
        return np.stack(segs, axis=0).mean(axis=0)

    m_up = mean_or_none(seg_up)
    m_dn = mean_or_none(seg_dn)

    out = {"n_up": len(seg_up), "n_dn": len(seg_dn), "m_up": m_up, "m_dn": m_dn, "area": None}
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
    T = len(score)
    idx = np.where(switches.astype(int) == 1)[0].tolist()

    raw_up = []
    raw_dn = []

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
        seg = score[b0:b1]
        if delta > 0:
            hits = np.where(seg >= half)[0]
        else:
            hits = np.where(seg <= half)[0]

        if hits.size > 0:
            lag = int(hits[0])
            if r0 == 0 and r1 == 1:
                raw_up.append(lag)
            elif r0 == 1 and r1 == 0:
                raw_dn.append(lag)

    def summarize(xs):
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


def _collect_switch_segments(score: np.ndarray, regime: np.ndarray, switches: np.ndarray, L: int):
    T = len(score)
    idx = np.where(switches.astype(int) == 1)[0].tolist()
    seg_up = []
    seg_dn = []

    for t0 in idx:
        if t0 + L > T:
            continue
        r0 = int(regime[t0 - 1]) if t0 - 1 >= 0 else int(regime[t0])
        r1 = int(regime[t0])
        seg = score[t0:t0+L].astype(np.float32)

        if r0 == 0 and r1 == 1:
            seg_up.append(seg)
        elif r0 == 1 and r1 == 0:
            seg_dn.append(seg)

    def stack_or_none(segs):
        if len(segs) == 0:
            return None
        return np.stack(segs, axis=0)

    X_up = stack_or_none(seg_up)
    X_dn = stack_or_none(seg_dn)

    return {
        "seg_up": X_up,
        "seg_dn": X_dn,
        "n_up": 0 if X_up is None else int(X_up.shape[0]),
        "n_dn": 0 if X_dn is None else int(X_dn.shape[0]),
    }


def _wasserstein_1d_empirical(a: np.ndarray, b: np.ndarray, q_grid: int = 129) -> Optional[float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3 or b.size < 3:
        return None

    q = np.linspace(0.0, 1.0, int(q_grid), dtype=np.float64)
    qa = np.quantile(a, q, method="linear")
    qb = np.quantile(b, q, method="linear")
    return float(np.mean(np.abs(qa - qb)))


def nonstationarity_W1(X: Optional[np.ndarray]) -> Dict[str, Any]:
    if X is None or X.shape[0] < 3 or X.shape[1] < 3:
        return {"NSI_W1": None, "w1_increments": None}

    _, L = X.shape
    inc = np.zeros((L - 1,), dtype=np.float32)

    for tau in range(1, L):
        d = _wasserstein_1d_empirical(X[:, tau], X[:, tau - 1])
        if d is None:
            return {"NSI_W1": None, "w1_increments": None}
        inc[tau - 1] = float(d)

    return {"NSI_W1": float(np.sum(inc)), "w1_increments": inc}


def quantile_drift(X: Optional[np.ndarray], p: float = 0.5) -> Dict[str, Any]:
    if X is None or X.shape[0] < 3 or X.shape[1] < 3:
        return {"QD": None, "q_curve": None}

    q_curve = np.quantile(X, float(p), axis=0, method="linear").astype(np.float32)
    qd = float(np.sum(np.abs(q_curve[1:] - q_curve[:-1])))
    return {"QD": qd, "q_curve": q_curve}


def amplitude_IQR(X: Optional[np.ndarray], early_frac: float = 0.25, agg: str = "median") -> Dict[str, Any]:
    if X is None or X.shape[0] < 3 or X.shape[1] < 3:
        return {"Amp_IQR": None, "iqr_curve": None, "tau_early": None}

    _, L = X.shape
    q25 = np.quantile(X, 0.25, axis=0, method="linear")
    q75 = np.quantile(X, 0.75, axis=0, method="linear")
    iqr = (q75 - q25).astype(np.float32)

    tau_e = max(2, int(np.floor(float(early_frac) * L)))
    tau_e = min(tau_e, L)
    early = iqr[:tau_e]

    amp = float(np.mean(early)) if str(agg).lower().strip() == "mean" else float(np.median(early))
    return {"Amp_IQR": amp, "iqr_curve": iqr, "tau_early": int(tau_e)}


def switch_distribution_stats(
    score: np.ndarray,
    regime: np.ndarray,
    switches: np.ndarray,
    L: int,
    q_p: float = 0.5,
    early_frac: float = 0.25,
) -> Dict[str, Any]:
    segs = _collect_switch_segments(score, regime, switches, L=L)
    X_up = segs["seg_up"]
    X_dn = segs["seg_dn"]

    def stats_one(X):
        w1 = nonstationarity_W1(X)
        qd = quantile_drift(X, p=q_p)
        amp = amplitude_IQR(X, early_frac=early_frac, agg="median")
        return {
            "n_events": 0 if X is None else int(X.shape[0]),
            "NSI_W1": w1["NSI_W1"],
            "QD": qd["QD"],
            "Amp_IQR": amp["Amp_IQR"],
            "w1_increments": None if w1["w1_increments"] is None else w1["w1_increments"],
            "q_curve": None if qd["q_curve"] is None else qd["q_curve"],
            "iqr_curve": None if amp["iqr_curve"] is None else amp["iqr_curve"],
            "tau_early": amp["tau_early"],
        }

    up = stats_one(X_up)
    dn = stats_one(X_dn)

    def mean_if_present(a, b):
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


def dissociation_index(stats_g: Dict[str, Any], stats_pi: Dict[str, Any], eps: float = 1e-9) -> Optional[float]:
    Ng = stats_g.get("combined", {}).get("NSI_W1", None)
    Np = stats_pi.get("combined", {}).get("NSI_W1", None)
    Ag = stats_g.get("combined", {}).get("Amp_IQR", None)
    Ap = stats_pi.get("combined", {}).get("Amp_IQR", None)

    if Ng is None or Np is None or Ag is None or Ap is None:
        return None

    return float(np.log((Ng + eps) / (Np + eps)) + np.log((Ap + eps) / (Ag + eps)))


# -------------------------------------------------------------------
# Phase 2 encounter / rupture metrics
# -------------------------------------------------------------------

def encounter_indices(df) -> np.ndarray:
    if "encounter_event" not in df.columns:
        return np.array([], dtype=int)
    return np.where(df["encounter_event"].to_numpy().astype(int) == 1)[0]


def rupture_indices(df) -> np.ndarray:
    if "rupture" not in df.columns:
        return np.array([], dtype=int)
    return np.where(df["rupture"].to_numpy().astype(int) == 1)[0]


def _extract_G(df) -> np.ndarray:
    cols = g_columns(df)
    if len(cols) == 0:
        raise ValueError("No g_* columns found.")
    return df[cols].to_numpy(dtype=np.float32)


def encounter_aligned_mean(
    df,
    event_col: str = "encounter_event",
    value_col: str = "pi_entropy",
    L_pre: int = 5,
    L_post: int = 15,
) -> Dict[str, Any]:
    if event_col not in df.columns or value_col not in df.columns:
        return {"n": 0, "mean": None, "segments": None}

    x = df[value_col].to_numpy(dtype=np.float32)
    idx = np.where(df[event_col].to_numpy().astype(int) == 1)[0]

    segs = []
    for t0 in idx:
        a = t0 - L_pre
        b = t0 + L_post + 1
        if a < 0 or b > len(x):
            continue
        segs.append(x[a:b])

    if len(segs) == 0:
        return {"n": 0, "mean": None, "segments": None}

    X = np.stack(segs, axis=0)
    return {"n": int(X.shape[0]), "mean": X.mean(axis=0), "segments": X}


def encounter_induced_g_shift(
    df,
    event_col: str = "encounter_event",
    horizon: int = 1,
) -> Dict[str, Any]:
    if event_col not in df.columns:
        return {"n": 0, "values": None, "mean": None, "median": None}

    G = _extract_G(df)
    idx = np.where(df[event_col].to_numpy().astype(int) == 1)[0]

    vals = []
    for t0 in idx:
        t1 = t0 + int(horizon)
        if t1 >= G.shape[0]:
            continue
        vals.append(float(np.linalg.norm(G[t1] - G[t0])))

    if len(vals) == 0:
        return {"n": 0, "values": None, "mean": None, "median": None}

    vals = np.asarray(vals, dtype=np.float32)
    return {
        "n": int(vals.shape[0]),
        "values": vals,
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
    }


def false_alarm_reactivity(
    df,
    encounter_col: str = "encounter_event",
    rupture_col: str = "rupture",
    horizon: int = 5,
) -> Dict[str, Any]:
    if encounter_col not in df.columns or rupture_col not in df.columns:
        return {"n": 0, "mean_g_shift": None, "mean_entropy_jump": None}

    G = _extract_G(df)
    H = df["pi_entropy"].to_numpy(dtype=np.float32) if "pi_entropy" in df.columns else None

    enc_idx = np.where(df[encounter_col].to_numpy().astype(int) == 1)[0]
    rupt = df[rupture_col].to_numpy().astype(int)

    g_vals = []
    h_vals = []

    for t0 in enc_idx:
        t1 = min(len(rupt), t0 + int(horizon) + 1)
        has_rupture = np.any(rupt[t0:t1] == 1)
        if has_rupture:
            continue

        if t0 + 1 < G.shape[0]:
            g_vals.append(float(np.linalg.norm(G[t0 + 1] - G[t0])))
        if H is not None and t0 + 1 < len(H):
            h_vals.append(float(H[t0 + 1] - H[t0]))

    return {
        "n": int(len(g_vals)),
        "mean_g_shift": None if len(g_vals) == 0 else float(np.mean(g_vals)),
        "mean_entropy_jump": None if len(h_vals) == 0 else float(np.mean(h_vals)),
    }


def recovery_half_life_from_rows(
    df,
    rupture_col: str = "rupture",
    pre_window: int = 20,
    threshold: float = 0.15,
) -> Dict[str, Any]:
    if rupture_col not in df.columns:
        return {"n": 0, "mean": None, "median": None, "values": []}

    G = _extract_G(df)
    idx = np.where(df[rupture_col].to_numpy().astype(int) == 1)[0]

    vals = []
    for t0 in idx:
        rt = recovery_time(G, t0=t0, window=pre_window, threshold=threshold)
        if rt is not None:
            vals.append(int(rt))

    if len(vals) == 0:
        return {"n": 0, "mean": None, "median": None, "values": []}

    return {
        "n": int(len(vals)),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "values": vals,
    }


def context_sensitive_engagement(
    df,
    fragility_threshold: float = 0.5,
) -> Dict[str, Any]:
    needed = {"on_encounter", "action_env", "fragility"}
    if not needed.issubset(set(df.columns)):
        return {"n_low": 0, "n_high": 0, "engage_rate_low": None, "engage_rate_high": None, "gap": None}

    m = df["on_encounter"].to_numpy().astype(int) == 1
    if not np.any(m):
        return {"n_low": 0, "n_high": 0, "engage_rate_low": None, "engage_rate_high": None, "gap": None}

    A = df.loc[m, "action_env"].to_numpy().astype(int)
    F = df.loc[m, "fragility"].to_numpy(dtype=np.float32)

    # crude engagement proxy: not-STAY
    engage = (A != 4).astype(np.float32)

    low = F < float(fragility_threshold)
    high = F >= float(fragility_threshold)

    low_rate = None if np.sum(low) == 0 else float(np.mean(engage[low]))
    high_rate = None if np.sum(high) == 0 else float(np.mean(engage[high]))

    gap = None
    if low_rate is not None and high_rate is not None:
        gap = float(high_rate - low_rate)

    return {
        "n_low": int(np.sum(low)),
        "n_high": int(np.sum(high)),
        "engage_rate_low": low_rate,
        "engage_rate_high": high_rate,
        "gap": gap,
    }