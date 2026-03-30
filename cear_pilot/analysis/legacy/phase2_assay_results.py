# cear_pilot/analysis/phase2_assay_results.py
# -*- coding: utf-8 -*-

## figure generator for phase2_g_assay results

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %%
# -----------------------------
# Config
# -----------------------------
BASE = Path("outputs/runs")   # change if needed
RUN_PREFIX = "phase2_g_sweeps_"

# Optional manual override:
# SWEEP_DIR = Path("outputs/runs/phase2_g_sweeps_test1")
SWEEP_DIR = None

SAVE_FIGS = True
FIG_OUT = Path("outputs/figures_phase2")
FIG_OUT.mkdir(parents=True, exist_ok=True)


# %%
# -----------------------------
# Helpers
# -----------------------------
def find_latest_sweep_dir(base: Path, prefix: str) -> Path:
    cands = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix)])
    if not cands:
        raise FileNotFoundError(f"No sweep directories found under {base} with prefix {prefix}")
    return cands[-1]


def load_all_trajs(sweep_dir: Path) -> pd.DataFrame:
    traj_paths = sorted(sweep_dir.glob("a_*/*/traj.parquet"))
    if not traj_paths:
        raise FileNotFoundError(f"No traj.parquet files found under {sweep_dir}")

    dfs = []
    for p in traj_paths:
        m = re.search(r"a_([^/\\]+)[/\\]([^/\\]+)[/\\]traj\.parquet$", str(p))
        if not m:
            continue
        alpha_tag = m.group(1)
        cond_tag = m.group(2)

        df = pd.read_parquet(p)
        df["alpha_tag"] = alpha_tag
        df["condition_tag"] = cond_tag
        df["source_file"] = str(p)
        dfs.append(df)

    if not dfs:
        raise RuntimeError("Found traj.parquet paths, but failed to parse any.")
    out = pd.concat(dfs, ignore_index=True)

    # numeric alpha
    out["alpha"] = out["alpha_tag"].astype(float)

    # nicer condition ordering
    cond_order = ["ssss", "mmmm", "s_s_s_m", "sss_m", "s_s_m_m", "ss_mm"]
    out["condition_tag"] = pd.Categorical(out["condition_tag"], categories=cond_order, ordered=True)

    return out


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # consequence rows
    df["is_cons"] = (df["consequence_now"] == 1).astype(int)
    df["is_cons_sup"] = ((df["consequence_now"] == 1) & (df["consequence_valence_hidden"] == "supportive")).astype(int)
    df["is_cons_mis"] = ((df["consequence_now"] == 1) & (df["consequence_valence_hidden"] == "misleading")).astype(int)

    # event rows
    df["is_event"] = (df["event_now"] == 1).astype(int)
    df["is_event_sup"] = ((df["event_now"] == 1) & (df["event_valence_hidden"] == "supportive")).astype(int)
    df["is_event_mis"] = ((df["event_now"] == 1) & (df["event_valence_hidden"] == "misleading")).astype(int)

    # simple behavior probes
    df["is_right"] = (df["action"] == 3).astype(int)
    df["is_left"] = (df["action"] == 2).astype(int)
    df["is_stay"] = (df["action"] == 4).astype(int)

    return df


def summarize_condition_alpha(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (alpha, cond), g in df.groupby(["alpha", "condition_tag"], observed=True):
        g_cons_sup = g[g["is_cons_sup"] == 1]
        g_cons_mis = g[g["is_cons_mis"] == 1]

        row = {
            "alpha": float(alpha),
            "condition_tag": str(cond),

            # uptake magnitudes
            "dg_sup": float(g_cons_sup["delta_g"].mean()) if len(g_cons_sup) else np.nan,
            "dg_mis": float(g_cons_mis["delta_g"].mean()) if len(g_cons_mis) else np.nan,

            # c-state at consequence moments
            "c_sup": float(g_cons_sup["c_state"].mean()) if len(g_cons_sup) else np.nan,
            "c_mis": float(g_cons_mis["c_state"].mean()) if len(g_cons_mis) else np.nan,

            # behavior over whole rollout
            "p_right": float(g["is_right"].mean()),
            "p_left": float(g["is_left"].mean()),
            "p_stay": float(g["is_stay"].mean()),
            "mean_entropy": float(g["entropy"].mean()),

            # progression
            "final_x_mean": float(g.groupby("episode")["x"].tail(1).mean()),

            # counts
            "n_steps": int(len(g)),
            "n_cons_sup": int(g["is_cons_sup"].sum()),
            "n_cons_mis": int(g["is_cons_mis"].sum()),
        }
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["condition_tag", "alpha"]).reset_index(drop=True)
    return out


def plot_alpha_sweep(summary: pd.DataFrame, conditions=None):
    if conditions is None:
        conditions = ["ssss", "mmmm", "s_s_s_m", "sss_m", "s_s_m_m", "ss_mm"]

    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()

    for cond in conditions:
        s = summary[summary["condition_tag"] == cond].sort_values("alpha")
        if len(s) == 0:
            continue
        ax.plot(s["alpha"], s["dg_mis"], marker="o", label=cond)

    ax.set_xlabel("alpha")
    ax.set_ylabel("mean delta_g at misleading consequence")
    ax.set_title("Fig 1. Misleading uptake curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if SAVE_FIGS:
        fig.savefig(FIG_OUT / "fig1_misleading_uptake_curve.png", dpi=180, bbox_inches="tight")
    plt.show()


def plot_alpha_sweep_supportive(summary: pd.DataFrame, conditions=None):
    if conditions is None:
        conditions = ["ssss", "mmmm", "s_s_s_m", "sss_m", "s_s_m_m", "ss_mm"]

    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()

    for cond in conditions:
        s = summary[summary["condition_tag"] == cond].sort_values("alpha")
        if len(s) == 0:
            continue
        ax.plot(s["alpha"], s["dg_sup"], marker="o", label=cond)

    ax.set_xlabel("alpha")
    ax.set_ylabel("mean delta_g at supportive consequence")
    ax.set_title("Fig 2. Supportive uptake curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if SAVE_FIGS:
        fig.savefig(FIG_OUT / "fig2_supportive_uptake_curve.png", dpi=180, bbox_inches="tight")
    plt.show()


def plot_schedule_effect(summary: pd.DataFrame, alpha_value=0.10):
    s = summary[np.isclose(summary["alpha"], alpha_value)].copy()
    s = s.sort_values("condition_tag")

    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()

    x = np.arange(len(s))
    width = 0.35

    ax.bar(x - width / 2, s["dg_sup"], width=width, label="supportive")
    ax.bar(x + width / 2, s["dg_mis"], width=width, label="misleading")

    ax.set_xticks(x)
    ax.set_xticklabels(s["condition_tag"], rotation=30, ha="right")
    ax.set_ylabel("mean delta_g at consequence")
    ax.set_title(f"Fig 3. Schedule effect at alpha={alpha_value:.2f}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    if SAVE_FIGS:
        fig.savefig(FIG_OUT / f"fig3_schedule_effect_alpha_{alpha_value:.2f}.png", dpi=180, bbox_inches="tight")
    plt.show()


def plot_heatmap(summary, value_col="dg_mis", title="Fig 4. Misleading uptake heatmap"):
    piv = summary.pivot(index="condition_tag", columns="alpha", values=value_col)

    fig = plt.figure(figsize=(7, 4.5))
    ax = plt.gca()
    im = ax.imshow(piv.values, aspect="auto")

    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in piv.columns])
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels(list(piv.index))

    ax.set_xlabel("alpha")
    ax.set_ylabel("condition")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

    if SAVE_FIGS:
        fig.savefig(FIG_OUT / f"{value_col}_heatmap.png", dpi=180, bbox_inches="tight")
    plt.show()


def inspect_one_run(df: pd.DataFrame, alpha=0.10, condition="sss_m", episode=0):
    """
    Quick smoke-test trajectory view for a single episode.
    This is not a main paper figure, just a sanity plot.
    """
    d = df[
        np.isclose(df["alpha"], alpha)
        & (df["condition_tag"].astype(str) == condition)
        & (df["episode"] == episode)
    ].copy()

    if len(d) == 0:
        print("No matching rows for inspect_one_run.")
        return

    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()

    ax.plot(d["t"], d["delta_g"], label="delta_g")
    ax.plot(d["t"], d["c_state"], label="c_state")

    ev = d[d["event_now"] == 1]
    cs = d[d["consequence_now"] == 1]

    if len(ev):
        ax.scatter(ev["t"], ev["delta_g"], marker="o", s=40, label="event")
    if len(cs):
        ax.scatter(cs["t"], cs["delta_g"], marker="x", s=50, label="consequence")

    ax.set_xlabel("t")
    ax.set_title(f"Sanity plot: alpha={alpha:.2f}, condition={condition}, episode={episode}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if SAVE_FIGS:
        fig.savefig(FIG_OUT / f"sanity_alpha_{alpha:.2f}_{condition}_ep{episode}.png", dpi=180, bbox_inches="tight")
    plt.show()


# %%
# -----------------------------
# Load
# -----------------------------
if SWEEP_DIR is None:
    SWEEP_DIR = find_latest_sweep_dir(BASE, RUN_PREFIX)

print("Using sweep dir:", SWEEP_DIR)

df = load_all_trajs(SWEEP_DIR)
df = add_derived_columns(df)

summary = summarize_condition_alpha(df)

print(summary[[
    "alpha", "condition_tag",
    "dg_sup", "dg_mis",
    "p_right", "mean_entropy",
    "final_x_mean", "n_cons_sup", "n_cons_mis"
]].sort_values(["condition_tag", "alpha"]))


# %%
# -----------------------------
# Main figures
# -----------------------------
plot_alpha_sweep(summary)
plot_alpha_sweep_supportive(summary)
plot_schedule_effect(summary, alpha_value=0.10)
plot_heatmap(summary, value_col="dg_mis", title="Fig 4. Misleading uptake heatmap")
plot_heatmap(summary, value_col="dg_sup", title="Supplement. Supportive uptake heatmap")


# %%
# -----------------------------
# Optional sanity check
# -----------------------------
inspect_one_run(df, alpha=0.10, condition="sss_m", episode=0)