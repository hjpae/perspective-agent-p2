# cear_pilot/analysis/phase2_maturity_analysis.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt


def load_table(run_dir: Path):
    import pandas as pd
    p_parq = run_dir / "traj.parquet"
    p_csv = run_dir / "traj.csv"
    if p_parq.exists():
        return pd.read_parquet(p_parq)
    if p_csv.exists():
        return pd.read_csv(p_csv)
    raise FileNotFoundError(f"No traj.parquet or traj.csv under {run_dir}")


def load_meta(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def to_jsonable(x):
    if x is None:
        return None
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_jsonable(v) for v in x]
    if isinstance(x, tuple):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def g_columns(df):
    return sorted([c for c in df.columns if c.startswith("g_")], key=lambda x: int(x.split("_")[1]))


def get_G(df) -> np.ndarray:
    cols = g_columns(df)
    if len(cols) == 0:
        raise ValueError("No g_* columns in dataframe.")
    return df[cols].to_numpy(dtype=np.float32)


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    if len(x) == 0:
        return x
    w = max(1, int(w))
    out = np.zeros_like(x, dtype=np.float32)
    for i in range(len(x)):
        a = max(0, i - w + 1)
        out[i] = float(np.mean(x[a:i+1]))
    return out


def _decision_prob_by_reliability_bins(df, bins: np.ndarray):
    m = df["on_encounter"].to_numpy().astype(int) == 1
    if np.sum(m) < 20:
        return None

    rel = df.loc[m, "recent_reliability"].to_numpy(dtype=np.float32)
    engage = df.loc[m, "engage"].to_numpy(dtype=np.float32)
    wait = df.loc[m, "wait"].to_numpy(dtype=np.float32)
    avoid = df.loc[m, "avoid"].to_numpy(dtype=np.float32)
    sample = df.loc[m, "sample"].to_numpy(dtype=np.float32)

    inds = np.digitize(rel, bins) - 1
    centers = []
    engage_curve = []
    wait_curve = []
    avoid_curve = []
    sample_curve = []
    counts = []

    for b in range(len(bins) - 1):
        mm = inds == b
        if np.sum(mm) > 0:
            centers.append(0.5 * (bins[b] + bins[b + 1]))
            engage_curve.append(float(np.mean(engage[mm])))
            wait_curve.append(float(np.mean(wait[mm])))
            avoid_curve.append(float(np.mean(avoid[mm])))
            sample_curve.append(float(np.mean(sample[mm])))
            counts.append(int(np.sum(mm)))

    if len(centers) < 3:
        return None

    return {
        "centers": np.asarray(centers, dtype=np.float32),
        "engage": np.asarray(engage_curve, dtype=np.float32),
        "wait": np.asarray(wait_curve, dtype=np.float32),
        "avoid": np.asarray(avoid_curve, dtype=np.float32),
        "sample": np.asarray(sample_curve, dtype=np.float32),
        "counts": np.asarray(counts, dtype=np.int32),
    }


def compute_plasticity(df) -> Dict[str, Any]:
    """
    New plasticity:
    Not engage-only. Measure whether the decision profile changes as reliability changes.

    We want:
    - engage to rise with reliability
    - wait/sample/avoid to shift in complementary ways
    """
    bins = np.linspace(-1.0, 1.0, 9)
    curves = _decision_prob_by_reliability_bins(df, bins)
    if curves is None:
        return {
            "plasticity_raw": None,
            "plasticity_score": None,
            "engage_low": None,
            "engage_high": None,
            "wait_low": None,
            "wait_high": None,
            "profile_shift_L1": None,
        }

    x = curves["centers"]
    engage = curves["engage"]
    wait = curves["wait"]
    avoid = curves["avoid"]
    sample = curves["sample"]

    def safe_corr(a, b):
        if len(a) < 3 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    corr_engage = safe_corr(x, engage)   # should be positive
    corr_wait = safe_corr(x, wait)       # often negative or curved
    corr_avoid = safe_corr(x, avoid)     # often negative
    corr_sample = safe_corr(x, sample)   # may be negative or peaked

    # profile shift: average L1 distance between low-reliability and high-reliability decision profile
    lo = np.argmin(x)
    hi = np.argmax(x)
    p_lo = np.array([engage[lo], wait[lo], avoid[lo], sample[lo]], dtype=np.float32)
    p_hi = np.array([engage[hi], wait[hi], avoid[hi], sample[hi]], dtype=np.float32)
    profile_shift = float(np.sum(np.abs(p_hi - p_lo)))

    # stronger weight on engage/avoid, but include overall profile shift
    raw = 0.50 * corr_engage - 0.25 * corr_avoid + 0.25 * min(1.0, profile_shift)
    score = float(np.clip((raw + 1.0) / 2.0, 0.0, 1.0))

    return {
        "plasticity_raw": float(raw),
        "plasticity_score": score,
        "corr_engage": float(corr_engage),
        "corr_wait": float(corr_wait),
        "corr_avoid": float(corr_avoid),
        "corr_sample": float(corr_sample),
        "profile_shift_L1": profile_shift,

        "engage_low": float(engage[lo]),
        "engage_high": float(engage[hi]),
        "wait_low": float(wait[lo]),
        "wait_high": float(wait[hi]),
        "avoid_low": float(avoid[lo]),
        "avoid_high": float(avoid[hi]),
        "sample_low": float(sample[lo]),
        "sample_high": float(sample[hi]),

        "centers": x,
        "engage_curve": engage,
        "wait_curve": wait,
        "avoid_curve": avoid,
        "sample_curve": sample,
        "counts": curves["counts"],
    }


def compute_stability(df, pre_window: int = 8, post_window: int = 8) -> Dict[str, Any]:
    if "rupture" not in df.columns:
        return {"overcorrection_index": None, "stability_score": None}

    rup_idx = np.where(df["rupture"].to_numpy().astype(int) == 1)[0]
    if len(rup_idx) == 0:
        return {"overcorrection_index": None, "stability_score": None}

    # Use full decision profile, not just engage
    engage = df["engage"].to_numpy(dtype=np.float32)
    wait = df["wait"].to_numpy(dtype=np.float32)
    avoid = df["avoid"].to_numpy(dtype=np.float32)
    sample = df["sample"].to_numpy(dtype=np.float32)

    vals = []
    engage_vals = []
    for t0 in rup_idx:
        a0 = max(0, t0 - pre_window)
        a1 = t0
        b0 = t0 + 1
        b1 = min(len(engage), t0 + 1 + post_window)
        if (a1 - a0) < 2 or (b1 - b0) < 2:
            continue

        pre = np.array([
            np.mean(engage[a0:a1]),
            np.mean(wait[a0:a1]),
            np.mean(avoid[a0:a1]),
            np.mean(sample[a0:a1]),
        ], dtype=np.float32)

        post = np.array([
            np.mean(engage[b0:b1]),
            np.mean(wait[b0:b1]),
            np.mean(avoid[b0:b1]),
            np.mean(sample[b0:b1]),
        ], dtype=np.float32)

        vals.append(float(np.sum(np.abs(post - pre))))
        engage_vals.append(float(abs(post[0] - pre[0])))

    if len(vals) == 0:
        return {"overcorrection_index": None, "stability_score": None}

    oci = float(np.mean(vals))
    oci_engage = float(np.mean(engage_vals))

    # lower overcorrection = higher stability
    stability = float(np.clip(1.0 - min(1.0, oci), 0.0, 1.0))

    return {
        "overcorrection_index": oci,
        "overcorrection_engage_only": oci_engage,
        "stability_score": stability,
        "n_events": int(len(vals)),
        "values": vals,
    }


def compute_g_recalibration(df, window: int = 10) -> Dict[str, Any]:
    G = get_G(df)
    if G.shape[0] < 3:
        return {"g_shift_mean": None}

    d = np.zeros((G.shape[0],), dtype=np.float32)
    d[1:] = np.linalg.norm(G[1:] - G[:-1], axis=-1)

    return {
        "g_shift_mean": float(np.mean(d)),
        "g_shift_median": float(np.median(d)),
        "g_shift_curve": d,
        "g_shift_rolling": rolling_mean(d, window),
    }


def classify_profile(plasticity_score: Optional[float], stability_score: Optional[float]) -> str:
    if plasticity_score is None or stability_score is None:
        return "undetermined"

    p_hi = plasticity_score >= 0.55
    s_hi = stability_score >= 0.55

    if p_hi and s_hi:
        return "mature_integrative"
    if (not p_hi) and s_hi:
        return "rigid"
    if p_hi and (not s_hi):
        return "hyper_reactive"
    return "disorganized"


def make_plots(df, run_dir: Path, summary: Dict[str, Any]) -> None:
    fig_dir = run_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1) reliability vs decision profile
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    p = summary["plasticity"]

    if p["centers"] is not None:
        x = np.asarray(p["centers"], dtype=np.float32)
        ax.plot(x, p["engage_curve"], marker="o", label="engage")
        ax.plot(x, p["wait_curve"], marker="o", label="wait")
        ax.plot(x, p["avoid_curve"], marker="o", label="avoid")
        ax.plot(x, p["sample_curve"], marker="o", label="sample")
        ax.legend(frameon=False)
    ax.set_xlabel("recent_reliability")
    ax.set_ylabel("decision probability on encounter")
    ax.set_title("Reliability-conditioned decision profile")
    fig.tight_layout()
    fig.savefig(fig_dir / "maturity_reliability_vs_decision_profile.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 2) 2D maturity profile map
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    pscore = summary["plasticity"]["plasticity_score"]
    sscore = summary["stability"]["stability_score"]

    ax.axvline(0.55, linestyle="--", linewidth=1)
    ax.axhline(0.55, linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.18, 0.82, "rigid", ha="center", va="center")
    ax.text(0.78, 0.82, "mature\nintegrative", ha="center", va="center")
    ax.text(0.78, 0.18, "hyper-\nreactive", ha="center", va="center")
    ax.text(0.18, 0.18, "disorganized", ha="center", va="center")

    if pscore is not None and sscore is not None:
        ax.scatter([pscore], [sscore], s=60)
    ax.set_xlabel("plasticity score")
    ax.set_ylabel("stability score")
    ax.set_title("Phase 2 maturity profile")
    fig.tight_layout()
    fig.savefig(fig_dir / "maturity_profile_map.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 3) g PCA
    try:
        from sklearn.decomposition import PCA
        G = get_G(df)
        Y = PCA(n_components=2).fit_transform(G)

        fig, ax = plt.subplots(figsize=(6, 5))
        if "recent_reliability" in df.columns:
            c = df["recent_reliability"].to_numpy(dtype=np.float32)
            sc = ax.scatter(Y[:, 0], Y[:, 1], c=c, s=8, alpha=0.6)
            fig.colorbar(sc, ax=ax, label="recent_reliability")
        else:
            ax.scatter(Y[:, 0], Y[:, 1], s=8, alpha=0.6)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Latent g trajectory (PCA)")
        fig.tight_layout()
        fig.savefig(fig_dir / "maturity_g_pca.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass


def summarize_run(df) -> Dict[str, Any]:
    plasticity = compute_plasticity(df)
    stability = compute_stability(df)
    gstats = compute_g_recalibration(df)

    summary: Dict[str, Any] = {
        "n_steps": int(len(df)),
        "n_encounter_steps": int(df["on_encounter"].sum()) if "on_encounter" in df.columns else 0,
        "n_encounter_events": int(df["encounter_event"].sum()) if "encounter_event" in df.columns else 0,
        "n_ruptures": int(df["rupture"].sum()) if "rupture" in df.columns else 0,

        "engage_rate": float(df["engage"].mean()) if "engage" in df.columns else None,
        "avoid_rate": float(df["avoid"].mean()) if "avoid" in df.columns else None,
        "wait_rate": float(df["wait"].mean()) if "wait" in df.columns else None,
        "sample_rate": float(df["sample"].mean()) if "sample" in df.columns else None,

        "plasticity": plasticity,
        "stability": stability,
        "g": gstats,
        "profile": classify_profile(plasticity["plasticity_score"], stability["stability_score"]),
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    df = load_table(run_dir)
    meta = load_meta(run_dir)

    summary = summarize_run(df)
    summary["meta"] = meta

    make_plots(df, run_dir, summary)

    out_json = run_dir / "phase2_maturity_summary.json"
    out_json.write_text(json.dumps(to_jsonable(summary), indent=2))

    print(json.dumps(to_jsonable(summary), indent=2))
    print(f"[OK] Saved maturity summary: {out_json}")
    print(f"[OK] Saved figures under: {run_dir / 'figs'}")


if __name__ == "__main__":
    main()