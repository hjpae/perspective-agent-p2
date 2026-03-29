# cear_pilot/analysis/plot_train_phase2.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SAVE_DPI = 160


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def find_existing(run_dir: Path, stem: str) -> Optional[Path]:
    candidates = [run_dir / f"{stem}.parquet", run_dir / f"{stem}.csv"]
    for p in candidates:
        if p.exists():
            return p
    return None


def savefig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def safe_plot(ax, df: pd.DataFrame, xcol: str, ycol: str, label: str, marker: str = "o") -> bool:
    if xcol in df.columns and ycol in df.columns:
        ax.plot(df[xcol], df[ycol], marker=marker, label=label)
        return True
    return False


def pick_top_var_cols(df: pd.DataFrame, prefix: str, n: int) -> List[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return []
    scored = []
    for c in cols:
        try:
            v = float(df[c].astype(float).var(ddof=0))
        except Exception:
            continue
        scored.append((v, c))
    scored.sort(reverse=True)
    return [c for _, c in scored[:n]]


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    if len(x) == 0:
        return x
    w = max(int(w), 1)
    if w == 1:
        return x.copy()
    s = pd.Series(x)
    return s.rolling(window=w, min_periods=1).mean().to_numpy()


def detect_g_cols(traj: pd.DataFrame) -> List[str]:
    cols = []
    for c in traj.columns:
        if c.startswith("g_"):
            try:
                _ = int(c.split("_")[-1])
                cols.append(c)
            except Exception:
                pass
    cols.sort(key=lambda x: int(x.split("_")[-1]))
    return cols


def add_delta_g(traj: pd.DataFrame) -> pd.DataFrame:
    if "delta_g" in traj.columns:
        return traj

    g_cols = detect_g_cols(traj)
    if not g_cols:
        traj = traj.copy()
        traj["delta_g"] = np.nan
        return traj

    traj = traj.copy().sort_values(["episode", "t"]).reset_index(drop=True)

    deltas = np.zeros(len(traj), dtype=np.float32)
    for ep, idx in traj.groupby("episode").groups.items():
        ix = np.array(list(idx))
        g = traj.loc[ix, g_cols].astype(float).to_numpy()
        d = np.zeros((len(ix),), dtype=np.float32)
        if len(ix) >= 2:
            d[1:] = np.linalg.norm(g[1:] - g[:-1], axis=1)
        deltas[ix] = d

    traj["delta_g"] = deltas
    return traj


def plot_episode_overview(ep: pd.DataFrame, outdir: Path) -> None:
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()

    used = False
    used |= safe_plot(ax, ep, "episode", "mean_pred_loss", "mean_pred_loss", marker="o")
    used |= safe_plot(ax, ep, "episode", "mean_energy", "mean_energy", marker="s")
    used |= safe_plot(ax, ep, "episode", "mean_alpha", "mean_alpha", marker="^")

    ax.set_xlabel("episode")
    ax.set_ylabel("value")
    ax.set_title("Phase 2: prediction / energy / alpha")
    if used:
        ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, outdir / "fig1_episode_overview.png")


def plot_latent_movement(ep: pd.DataFrame, outdir: Path) -> None:
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()

    used = False
    used |= safe_plot(ax, ep, "episode", "g_shift", "g_shift", marker="o")
    used |= safe_plot(ax, ep, "episode", "mean_delta_g", "mean_delta_g", marker="s")
    used |= safe_plot(ax, ep, "episode", "switches", "switches", marker="^")

    ax.set_xlabel("episode")
    ax.set_ylabel("value")
    ax.set_title("Phase 2: latent movement / basin switching")
    if used:
        ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, outdir / "fig2_latent_movement.png")


def plot_context_channels(ep: pd.DataFrame, outdir: Path) -> None:
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()

    used = False
    used |= safe_plot(ax, ep, "episode", "mean_entropy", "mean_entropy", marker="o")
    used |= safe_plot(ax, ep, "episode", "mean_abs_c", "mean_abs_c", marker="s")
    used |= safe_plot(ax, ep, "episode", "mean_err_norm", "mean_err_norm", marker="^")

    ax.set_xlabel("episode")
    ax.set_ylabel("value")
    ax.set_title("Phase 2: context / uncertainty traces")
    if used:
        ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, outdir / "fig3_context_channels.png")


def plot_selected_err_channels(ep: pd.DataFrame, outdir: Path, n_dims: int) -> List[str]:
    err_cols = pick_top_var_cols(ep, "mean_err_", n_dims)

    fig = plt.figure(figsize=(9, 5))
    ax = plt.gca()
    for c in err_cols:
        ax.plot(ep["episode"], ep[c], marker="o", label=c.replace("mean_err_", ""))

    ax.set_xlabel("episode")
    ax.set_ylabel("value")
    ax.set_title("Phase 2: selected evidence channels (episode means)")
    if err_cols:
        ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, outdir / "fig4_selected_err_channels.png")
    return err_cols


def plot_alpha_histogram(traj: pd.DataFrame, outdir: Path) -> None:
    if "alpha" not in traj.columns:
        return

    fig = plt.figure(figsize=(7, 5))
    ax = plt.gca()
    ax.hist(traj["alpha"].astype(float).to_numpy(), bins=40)
    ax.set_xlabel("alpha")
    ax.set_ylabel("count")
    ax.set_title("Trajectory-level alpha histogram")
    ax.grid(True, alpha=0.3)
    savefig(fig, outdir / "fig5_alpha_histogram.png")


def plot_alpha_vs_context(traj: pd.DataFrame, outdir: Path) -> None:
    if "t" not in traj.columns or "alpha" not in traj.columns:
        return

    fig = plt.figure(figsize=(9, 5))
    ax = plt.gca()

    t = traj["t"].astype(float).to_numpy()
    alpha = traj["alpha"].astype(float).to_numpy()
    ax.plot(t, rolling_mean(alpha, 15), label="alpha (rolling mean)")

    if "event_now" in traj.columns:
        ax.plot(t, rolling_mean(traj["event_now"].astype(float).to_numpy(), 15), label="event_now (rolling mean)")
    if "consequence_now" in traj.columns:
        ax.plot(t, rolling_mean(traj["consequence_now"].astype(float).to_numpy(), 15), label="consequence_now (rolling mean)")
    if "c_state" in traj.columns:
        ax.plot(t, rolling_mean(np.abs(traj["c_state"].astype(float).to_numpy()), 15), label="|c_state| (rolling mean)")

    ax.set_xlabel("t")
    ax.set_ylabel("value")
    ax.set_title("Alpha vs context traces")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, outdir / "fig6_alpha_vs_context.png")


def plot_alpha_by_context_boxplot(traj: pd.DataFrame, outdir: Path) -> None:
    if "alpha" not in traj.columns:
        return

    alpha = traj["alpha"].astype(float)
    labels = []
    values = []

    def add_group(name: str, mask: pd.Series):
        arr = alpha[mask].to_numpy()
        if len(arr) > 0:
            labels.append(name)
            values.append(arr)

    add_group("all", pd.Series([True] * len(traj), index=traj.index))

    if "event_now" in traj.columns:
        add_group("event_now", traj["event_now"].astype(float) > 0)
    if "consequence_now" in traj.columns:
        add_group("consequence_now", traj["consequence_now"].astype(float) > 0)
    if "err_supportive_flag" in traj.columns:
        add_group("supportive_flag", traj["err_supportive_flag"].astype(float) > 0)
    if "err_misleading_flag" in traj.columns:
        add_group("misleading_flag", traj["err_misleading_flag"].astype(float) > 0)

    if not values:
        return

    fig = plt.figure(figsize=(9, 5))
    ax = plt.gca()
    ax.boxplot(values, tick_labels=labels, showfliers=False)
    ax.set_xlabel("context")
    ax.set_ylabel("alpha")
    ax.set_title("Alpha by context")
    ax.grid(True, alpha=0.3)
    savefig(fig, outdir / "fig7_alpha_by_context_boxplot.png")


def event_aligned_average(
    traj: pd.DataFrame,
    trigger_col: str,
    value_col: str,
    window_before: int,
    window_after: int,
) -> Optional[pd.DataFrame]:
    if trigger_col not in traj.columns or value_col not in traj.columns:
        return None

    use = traj[["episode", "t", trigger_col, value_col]].copy()
    use["episode"] = use["episode"].astype(int)
    use["t"] = use["t"].astype(int)
    use[trigger_col] = use[trigger_col].astype(float)
    use[value_col] = use[value_col].astype(float)

    chunks = []
    for ep, g in use.groupby("episode"):
        g = g.sort_values("t").reset_index(drop=True)
        trigger_idx = g.index[g[trigger_col] > 0].tolist()
        for idx in trigger_idx:
            t0 = int(g.loc[idx, "t"])
            w = g[(g["t"] >= t0 - window_before) & (g["t"] <= t0 + window_after)].copy()
            if len(w) == 0:
                continue
            w["rel_t"] = w["t"] - t0
            chunks.append(w[["rel_t", value_col]])

    if not chunks:
        return None

    cat = pd.concat(chunks, axis=0, ignore_index=True)
    out = cat.groupby("rel_t")[value_col].mean().reset_index().sort_values("rel_t").reset_index(drop=True)
    return out


def plot_event_and_consequence_aligned(traj: pd.DataFrame, outdir: Path, window_before: int, window_after: int) -> None:
    specs = [
        ("event_now", "alpha", "fig8_event_aligned_alpha.png", "Event-aligned alpha"),
        ("event_now", "delta_g", "fig9_event_aligned_delta_g.png", "Event-aligned delta_g"),
        ("consequence_now", "alpha", "fig10_consequence_aligned_alpha.png", "Consequence-aligned alpha"),
        ("consequence_now", "delta_g", "fig11_consequence_aligned_delta_g.png", "Consequence-aligned delta_g"),
    ]

    for trig, val, fname, title in specs:
        aligned = event_aligned_average(traj, trig, val, window_before, window_after)
        if aligned is None or len(aligned) == 0:
            continue

        fig = plt.figure(figsize=(8, 5))
        ax = plt.gca()
        ax.plot(aligned["rel_t"], aligned[val], marker="o")
        ax.axvline(0.0)
        ax.set_xlabel("relative t")
        ax.set_ylabel(val)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        savefig(fig, outdir / fname)


def plot_flag_distributions(traj: pd.DataFrame, outdir: Path) -> None:
    needed = ["alpha", "delta_g", "err_supportive_flag", "err_misleading_flag"]
    if not all(c in traj.columns for c in needed):
        return

    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()

    groups = []
    labels = []

    sup_mask = traj["err_supportive_flag"].astype(float) > 0
    mis_mask = traj["err_misleading_flag"].astype(float) > 0

    if sup_mask.any():
        groups.append(traj.loc[sup_mask, "alpha"].astype(float).to_numpy())
        labels.append("alpha@supportive")
    if mis_mask.any():
        groups.append(traj.loc[mis_mask, "alpha"].astype(float).to_numpy())
        labels.append("alpha@misleading")
    if sup_mask.any():
        groups.append(traj.loc[sup_mask, "delta_g"].astype(float).to_numpy())
        labels.append("delta_g@supportive")
    if mis_mask.any():
        groups.append(traj.loc[mis_mask, "delta_g"].astype(float).to_numpy())
        labels.append("delta_g@misleading")

    if not groups:
        return

    ax.boxplot(groups, tick_labels=labels, showfliers=False)
    ax.set_ylabel("value")
    ax.set_title("Supportive / misleading flag distributions")
    ax.grid(True, alpha=0.3)
    savefig(fig, outdir / "fig12_supportive_misleading_distributions.png")


def compute_variance_summary(traj: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if "alpha" not in traj.columns:
        return out

    alpha = traj["alpha"].astype(float)

    out["alpha_var_all"] = float(alpha.var(ddof=0))

    if "event_now" in traj.columns:
        mask = traj["event_now"].astype(float) > 0
        if mask.any():
            out["alpha_var_event_now"] = float(alpha[mask].var(ddof=0))
            out["alpha_var_event_ratio_vs_all"] = float(out["alpha_var_event_now"] / max(out["alpha_var_all"], 1e-12))

    if "consequence_now" in traj.columns:
        mask = traj["consequence_now"].astype(float) > 0
        if mask.any():
            out["alpha_var_consequence_now"] = float(alpha[mask].var(ddof=0))
            out["alpha_var_consequence_ratio_vs_all"] = float(out["alpha_var_consequence_now"] / max(out["alpha_var_all"], 1e-12))

    return out


def plot_variance_comparison(traj: pd.DataFrame, outdir: Path) -> Dict[str, float]:
    stats = compute_variance_summary(traj)
    if not stats:
        return stats

    labels = []
    values = []

    for k in ["alpha_var_all", "alpha_var_event_now", "alpha_var_consequence_now"]:
        if k in stats:
            labels.append(k.replace("alpha_var_", ""))
            values.append(stats[k])

    if values:
        fig = plt.figure(figsize=(7, 5))
        ax = plt.gca()
        ax.bar(labels, values)
        ax.set_ylabel("variance")
        ax.set_title("Alpha variance: all vs event/consequence windows")
        ax.grid(True, alpha=0.3)
        savefig(fig, outdir / "fig13_alpha_variance_comparison.png")

    return stats


def plot_episode_end_positions(ep: pd.DataFrame, outdir: Path) -> None:
    if "episode" not in ep.columns:
        return
    if "final_x" not in ep.columns and "final_c" not in ep.columns:
        return

    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()

    used = False
    used |= safe_plot(ax, ep, "episode", "final_x", "final_x", marker="o")
    used |= safe_plot(ax, ep, "episode", "final_c", "final_c", marker="s")

    ax.set_xlabel("episode")
    ax.set_ylabel("value")
    ax.set_title("Episode-end state traces")
    if used:
        ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, outdir / "fig14_episode_end_state.png")


def build_summary(ep: pd.DataFrame, traj: Optional[pd.DataFrame], meta: dict, err_cols: List[str], var_stats: Dict[str, float]) -> dict:
    out = {
        "episodes": int(len(ep)),
        "schedule_pattern": meta.get("env_cfg", {}).get("schedule_pattern"),
        "valence_sequence": meta.get("env_cfg", {}).get("valence_sequence"),
        "selected_err_channels": list(err_cols),
    }

    for c in [
        "mean_alpha",
        "mean_pred_loss",
        "mean_energy",
        "g_shift",
        "mean_delta_g",
        "switches",
        "mean_err_norm",
    ]:
        if c in ep.columns and len(ep) > 0:
            out[f"{c}_overall_mean"] = float(ep[c].astype(float).mean())

    if traj is not None and len(traj) > 0:
        if "alpha" in traj.columns:
            alpha = traj["alpha"].astype(float)
            out["alpha_min"] = float(alpha.min())
            out["alpha_max"] = float(alpha.max())
            out["alpha_mean"] = float(alpha.mean())
            out["alpha_std"] = float(alpha.std(ddof=0))

        for c in ["event_now", "consequence_now", "err_supportive_flag", "err_misleading_flag"]:
            if c in traj.columns:
                out[f"{c}_mean"] = float(traj[c].astype(float).mean())

    out.update(var_stats)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Phase 2 training diagnostics.")
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--dims", type=int, default=4)
    ap.add_argument("--window_before", type=int, default=8)
    ap.add_argument("--window_after", type=int, default=12)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    outdir = run_dir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    ep_path = find_existing(run_dir, "episode_summary")
    if ep_path is None:
        raise FileNotFoundError(f"Could not find episode_summary under {run_dir}")

    traj_path = find_existing(run_dir, "traj")
    if traj_path is None:
        raise FileNotFoundError(f"Could not find traj under {run_dir}. These analyses require traj.parquet/csv.")

    ep = load_table(ep_path).sort_values("episode").reset_index(drop=True)
    traj = load_table(traj_path).sort_values(["episode", "t"]).reset_index(drop=True)
    traj = add_delta_g(traj)

    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    plot_episode_overview(ep, outdir)
    plot_latent_movement(ep, outdir)
    plot_context_channels(ep, outdir)
    err_cols = plot_selected_err_channels(ep, outdir, int(args.dims))
    plot_alpha_histogram(traj, outdir)
    plot_alpha_vs_context(traj, outdir)
    plot_alpha_by_context_boxplot(traj, outdir)
    plot_event_and_consequence_aligned(traj, outdir, int(args.window_before), int(args.window_after))
    plot_flag_distributions(traj, outdir)
    var_stats = plot_variance_comparison(traj, outdir)
    plot_episode_end_positions(ep, outdir)

    summary = build_summary(ep, traj, meta, err_cols, var_stats)
    (outdir / "plot_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Saved figures to: {outdir}")


if __name__ == "__main__":
    main()