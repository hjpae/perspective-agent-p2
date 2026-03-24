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
    import numpy as np
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


def compute_plasticity(df) -> Dict[str, Any]:
    m = df["on_encounter"].to_numpy().astype(int) == 1
    if np.sum(m) < 10:
        return {"plasticity_raw": None, "plasticity_score": None, "engage_low": None, "engage_high": None}

    engage = df.loc[m, "engage"].to_numpy(dtype=np.float32)
    rel = df.loc[m, "recent_reliability"].to_numpy(dtype=np.float32)

    q_lo = np.quantile(rel, 0.25)
    q_hi = np.quantile(rel, 0.75)

    lo_mask = rel <= q_lo
    hi_mask = rel >= q_hi

    engage_low = None if np.sum(lo_mask) == 0 else float(np.mean(engage[lo_mask]))
    engage_high = None if np.sum(hi_mask) == 0 else float(np.mean(engage[hi_mask]))

    if np.std(rel) < 1e-8 or np.std(engage) < 1e-8:
        corr = 0.0
    else:
        corr = float(np.corrcoef(rel, engage)[0, 1])

    # desirable: positive correlation between reliability evidence and engagement
    score = float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))
    return {
        "plasticity_raw": corr,
        "plasticity_score": score,
        "engage_low": engage_low,
        "engage_high": engage_high,
        "delta_engage_hi_lo": None if engage_low is None or engage_high is None else float(engage_high - engage_low),
    }


def compute_stability(df, pre_window: int = 8, post_window: int = 8) -> Dict[str, Any]:
    if "rupture" not in df.columns:
        return {"overcorrection_index": None, "stability_score": None}

    rup_idx = np.where(df["rupture"].to_numpy().astype(int) == 1)[0]
    if len(rup_idx) == 0:
        return {"overcorrection_index": None, "stability_score": None}

    engage = df["engage"].to_numpy(dtype=np.float32)

    vals = []
    for t0 in rup_idx:
        a0 = max(0, t0 - pre_window)
        a1 = t0
        b0 = t0 + 1
        b1 = min(len(engage), t0 + 1 + post_window)
        if (a1 - a0) < 2 or (b1 - b0) < 2:
            continue

        pre = float(np.mean(engage[a0:a1]))
        post = float(np.mean(engage[b0:b1]))
        vals.append(abs(post - pre))

    if len(vals) == 0:
        return {"overcorrection_index": None, "stability_score": None}

    oci = float(np.mean(vals))
    stability = float(np.clip(1.0 - oci, 0.0, 1.0))
    return {
        "overcorrection_index": oci,
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

    # 1) reliability vs engagement
    fig, ax = plt.subplots(figsize=(7, 4.5))
    m = df["on_encounter"].to_numpy().astype(int) == 1
    if np.sum(m) > 0:
        rel = df.loc[m, "recent_reliability"].to_numpy(dtype=np.float32)
        engage = df.loc[m, "engage"].to_numpy(dtype=np.float32)

        if len(rel) > 0:
            bins = np.linspace(-1.0, 1.0, 9)
            inds = np.digitize(rel, bins) - 1
            xs, ys = [], []
            for b in range(len(bins) - 1):
                mm = inds == b
                if np.sum(mm) > 0:
                    xs.append(0.5 * (bins[b] + bins[b+1]))
                    ys.append(np.mean(engage[mm]))
            if len(xs) > 0:
                ax.plot(xs, ys, marker="o")
    ax.set_xlabel("recent_reliability")
    ax.set_ylabel("engage rate on encounter")
    ax.set_title("Reliability-conditioned engagement")
    fig.tight_layout()
    fig.savefig(fig_dir / "maturity_reliability_vs_engagement.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 2) 2D maturity profile map
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    p = summary["plasticity"]["plasticity_score"]
    s = summary["stability"]["stability_score"]

    ax.axvline(0.55, linestyle="--", linewidth=1)
    ax.axhline(0.55, linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.18, 0.82, "rigid", ha="center", va="center")
    ax.text(0.78, 0.82, "mature\nintegrative", ha="center", va="center")
    ax.text(0.78, 0.18, "hyper-\nreactive", ha="center", va="center")
    ax.text(0.18, 0.18, "disorganized", ha="center", va="center")

    if p is not None and s is not None:
        ax.scatter([p], [s], s=60)
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