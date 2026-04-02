#!/usr/bin/env python3
"""
Probe-based representation analysis: same input, different g → different z_t?

Loads a checkpoint + trajectory, extracts g states from different history blocks,
then feeds fixed probe observations through the encoder with each g to measure
how salience gating reorganizes perception.

Usage:
    python -m cear_pilot.analysis.probe_representation \
        --run_dir outputs/mixed_20260401/mixed_0_4_0
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

DPI = 150
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                      "axes.grid": True, "grid.alpha": 0.2, "font.size": 11})


def load_agent(ckpt_path, device):
    from cear_pilot.models.agent import CEARAgent, AgentConfig
    from cear_pilot.models.encoder import EncoderConfig
    from cear_pilot.models.world_latent import WorldLatentConfig
    from cear_pilot.models.state_head import StateHeadConfig
    from cear_pilot.models.policy import PolicyConfig

    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt["meta"]

    enc_cfg = EncoderConfig()
    enc_cfg.__dict__.update(meta["agent_cfg"]["encoder"])
    world_cfg = WorldLatentConfig()
    world_cfg.__dict__.update(meta["agent_cfg"]["world"])
    state_cfg = StateHeadConfig()
    state_cfg.__dict__.update(meta["agent_cfg"]["state"])
    policy_cfg = PolicyConfig()
    policy_cfg.__dict__.update(meta["agent_cfg"]["policy"])

    agent_cfg = AgentConfig(encoder=enc_cfg, world=world_cfg,
                            state=state_cfg, policy=policy_cfg, device=device)
    agent = CEARAgent(agent_cfg)
    agent.load_state_dict(ckpt["agent_state"], strict=False)
    agent.to(device).eval()
    return agent, meta


def load_traj(run_dir):
    for name in ["traj.parquet", "traj.csv"]:
        p = run_dir / name
        if p.exists():
            return pd.read_parquet(p) if name.endswith(".parquet") else pd.read_csv(p)
    return None


def g_cols(df):
    return sorted([c for c in df.columns if re.match(r"g_\d+$", c)],
                  key=lambda x: int(x.split("_")[-1]))

def indexed_cols(df: pd.DataFrame, prefix: str):
    """Return columns like f'{prefix}<int>' sorted by index."""
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    cols = []
    for c in df.columns:
        m = pat.match(c)
        if m:
            cols.append((int(m.group(1)), c))
    cols.sort(key=lambda x: x[0])
    return [c for _, c in cols]
    
def extract_block_g(df, gc, block_id, n_late_eps=10):
    """Get mean g from the late portion of a given block."""
    blk = df[df["block_id"] == block_id]
    if len(blk) == 0:
        return None
    mx = blk["episode"].max()
    late = blk[blk["episode"] >= max(blk["episode"].min(), mx - n_late_eps + 1)]
    return late[gc].mean().values.astype(np.float32)


def extract_perturb_g(df, gc, low_n=0, high_n=4, n_late_eps=10):
    """For non-mixed runs: get g from episodes with low vs high perturbation."""
    if "n_perturb_setting" in df.columns:
        low = df[df["n_perturb_setting"] <= low_n]
        high = df[df["n_perturb_setting"] >= high_n]
    else:
        # Fallback: use early vs late episodes
        mx = df["episode"].max()
        mid = mx // 2
        low = df[df["episode"] < mid]
        high = df[df["episode"] >= mid]

    g_low = low[gc].mean().values.astype(np.float32) if len(low) > 0 else None
    g_high = high[gc].mean().values.astype(np.float32) if len(high) > 0 else None
    return g_low, g_high


def collect_zone_probes(env_cfg_dict, n_per_zone=3, seed=9999):
    """Sample fixed probe observations from each zone."""
    from cear_pilot.envs.nzone_phase2 import NZonePhase2Config, NZonePhase2Env

    cfg = NZonePhase2Config()
    for k, v in env_cfg_dict.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.n_perturbations = 0  # no perturbation during probe collection

    env = NZonePhase2Env(config=cfg)
    rng = np.random.default_rng(seed)

    # Zone boundaries: 4/5/5/5/4 columns → zones 0-4
    zone_x_ranges = [(0, 3), (4, 8), (9, 13), (14, 18), (19, 22)]
    probes = []

    for zi, (x_lo, x_hi) in enumerate(zone_x_ranges):
        for _ in range(n_per_zone):
            x = rng.integers(x_lo, x_hi + 1)
            y = rng.integers(0, cfg.height)
            env.x, env.y = x, y
            env.t = 0
            env._perturbation_active = False
            env._perturbation_trace = 0.0
            env._rng = np.random.default_rng(seed + zi * 100 + x * 10 + y)
            obs = env._observe()
            probes.append({"zone_id": zi, "x": x, "y": y, "obs": obs.copy()})

    return probes


def analyze_probes(agent, probes, g_dict, device):
    """For each probe × each g condition, compute z_raw, z_t, gamma, beta."""
    results = []

    for pi, probe in enumerate(probes):
        x_t = torch.tensor(probe["obs"], dtype=torch.float32, device=device).unsqueeze(0)

        # z_raw (g-independent)
        with torch.no_grad():
            z_raw = torch.tanh(agent.enc.obs_enc.mlp(x_t)).cpu().numpy()[0]

        for g_label, g_vec in g_dict.items():
            g_t = torch.tensor(g_vec, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                z_t = agent.enc.obs_enc(x_t, g_t=g_t).cpu().numpy()[0]

                gamma_np = np.zeros(len(z_raw))
                beta_np = np.zeros(len(z_raw))
                if hasattr(agent.enc.obs_enc, "film"):
                    gb = agent.enc.obs_enc.film(g_t)
                    gam, bet = gb.chunk(2, dim=-1)
                    gamma_np = gam.cpu().numpy()[0]
                    beta_np = bet.cpu().numpy()[0]

            row = {
                "probe_idx": pi,
                "zone_id": probe["zone_id"],
                "x": probe["x"], "y": probe["y"],
                "g_label": g_label,
                "z_raw_norm": float(np.linalg.norm(z_raw)),
                "z_t_norm": float(np.linalg.norm(z_t)),
                "z_shift": float(np.linalg.norm(z_t - z_raw)),
                "gamma_mean": float(gamma_np.mean()),
                "gamma_abs_mean": float(np.abs(gamma_np).mean()),
                "beta_mean": float(beta_np.mean()),
            }
            for di in range(len(z_raw)):
                row[f"z_raw_{di}"] = float(z_raw[di])
                row[f"z_t_{di}"] = float(z_t[di])
                row[f"gamma_{di}"] = float(gamma_np[di])
                row[f"beta_{di}"] = float(beta_np[di])
            results.append(row)

    return pd.DataFrame(results)


def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {path.name}")


def make_figures(results, g_labels, outdir):
    zt_cols = indexed_cols(results, "z_t_")
    gamma_dim_cols = indexed_cols(results, "gamma_")
    beta_dim_cols = indexed_cols(results, "beta_")
    n_z = len(zt_cols)
    if n_z == 0:
        raise ValueError("No z_t_<dim> columns found in probe results.")

    # ── Fig 1: z_t shift magnitude by zone and g condition ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: z_shift per zone per g
    ax = axes[0]
    zones = sorted(results["zone_id"].unique())
    zone_colors = ["#E24B4A", "#D85A30", "#888780", "#1D9E75", "#0F6E56"]
    width = 0.8 / len(g_labels)
    for gi, gl in enumerate(g_labels):
        gd = results[results["g_label"] == gl]
        means = [gd[gd["zone_id"] == z]["z_shift"].mean() for z in zones]
        offset = (gi - len(g_labels) / 2 + 0.5) * width
        ax.bar(np.array(zones) + offset, means, width, alpha=0.7, label=gl)
    ax.set_xticks(zones)
    ax.set_xticklabels([f"Z{z}" for z in zones])
    ax.set_ylabel("||z_t - z_raw|| (perception shift)")
    ax.set_title("How much does g change perception?")
    ax.legend(fontsize=8)

    # Panel B: cross-g z_t distance
    ax = axes[1]
    if len(g_labels) >= 2:
        gA, gB = g_labels[0], g_labels[1]
        cross_dists = []
        for pi in results["probe_idx"].unique():
            zA = results[(results["probe_idx"] == pi) & (results["g_label"] == gA)]
            zB = results[(results["probe_idx"] == pi) & (results["g_label"] == gB)]
            if len(zA) > 0 and len(zB) > 0:
                zt_cols = [f"z_t_{d}" for d in range(n_z)]
                vA = zA[zt_cols].values[0]
                vB = zB[zt_cols].values[0]
                zi = int(zA["zone_id"].iloc[0])
                cross_dists.append({"zone_id": zi, "dist": float(np.linalg.norm(vA - vB))})
        if cross_dists:
            cdf = pd.DataFrame(cross_dists)
            for z in zones:
                zd = cdf[cdf["zone_id"] == z]["dist"]
                if len(zd) > 0:
                    ax.bar(z, zd.mean(), color=zone_colors[z % 5], alpha=0.7)
                    ax.scatter([z] * len(zd), zd.values, c="black", s=10, alpha=0.3, zorder=5)
        ax.set_xticks(zones)
        ax.set_xticklabels([f"Z{z}" for z in zones])
        ax.set_ylabel(f"||z_t({gA}) - z_t({gB})||")
        ax.set_title("Same input, different perspective → different perception")

    savefig(fig, outdir / "probe_z_shift.png")

    # ── Fig 2: per-dim gating comparison ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: gamma per dim per g condition (averaged over zones)
    ax = axes[0]
    gamma_cols = [f"gamma_{d}" for d in range(n_z)]
    for gi, gl in enumerate(g_labels):
        gd = results[results["g_label"] == gl]
        gvals = gd[gamma_cols].mean().values
        ax.bar(np.arange(n_z) + (gi - len(g_labels)/2 + 0.5) * 0.3, gvals, 0.28,
               alpha=0.7, label=gl)
    ax.set_xticks(range(n_z))
    ax.set_xticklabels([f"z{d}" for d in range(n_z)], fontsize=8)
    ax.set_ylabel("γ (FiLM scale)")
    ax.set_title("Gating pattern by perspective")
    ax.axhline(0, color="#ccc", lw=0.5)
    ax.legend(fontsize=8)

    # Panel B: z_t heatmap (probes × dims, for two g conditions)
    ax = axes[1]
    if len(g_labels) >= 2:
        gA, gB = g_labels[0], g_labels[1]
        zt_cols = [f"z_t_{d}" for d in range(n_z)]
        dfA = results[results["g_label"] == gA].sort_values("probe_idx")[zt_cols].values
        dfB = results[results["g_label"] == gB].sort_values("probe_idx")[zt_cols].values
        diff = dfA - dfB  # (n_probes, n_z)
        im = ax.imshow(diff, aspect="auto", cmap="RdBu_r", vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
        ax.set_xlabel("z dimension")
        ax.set_ylabel("Probe index")
        ax.set_title(f"z_t({gA}) - z_t({gB}) per probe")
        plt.colorbar(im, ax=ax, shrink=0.8)

    savefig(fig, outdir / "probe_gating_detail.png")

    # ── Summary ──
    print(f"\n  Probe analysis summary:")
    for gl in g_labels:
        gd = results[results["g_label"] == gl]
        print(f"    {gl}: mean z_shift={gd['z_shift'].mean():.4f}  "
              f"mean |γ|={gd['gamma_abs_mean'].mean():.4f}")
    if len(g_labels) >= 2:
        print(f"    Cross-g z_t distance ({g_labels[0]} vs {g_labels[1]}):")
        for z in zones:
            zA = results[(results["zone_id"] == z) & (results["g_label"] == g_labels[0])]
            zB = results[(results["zone_id"] == z) & (results["g_label"] == g_labels[1])]
            if len(zA) > 0 and len(zB) > 0:
                zt_cols = [f"z_t_{d}" for d in range(n_z)]
                dists = []
                for pi in zA["probe_idx"].unique():
                    a = zA[zA["probe_idx"] == pi][zt_cols].values
                    b = zB[zB["probe_idx"] == pi][zt_cols].values
                    if len(a) > 0 and len(b) > 0:
                        dists.append(float(np.linalg.norm(a[0] - b[0])))
                if dists:
                    print(f"      Z{z}: mean={np.mean(dists):.4f} max={np.max(dists):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--n_probes_per_zone", type=int, default=5)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    outdir = run_dir / "analysis"
    outdir.mkdir(exist_ok=True)

    # Load
    ckpt_path = run_dir / "ckpt_final.pt"
    if not ckpt_path.exists():
        print(f"ERROR: {ckpt_path} not found")
        return

    agent, meta = load_agent(str(ckpt_path), args.device)
    df = load_traj(run_dir)
    if df is None:
        print("ERROR: No trajectory")
        return

    gc = g_cols(df)
    print(f"  Loaded: {len(df)} rows, {len(gc)} g dims")

    # Extract g states
    g_dict = {}
    has_blocks = "block_id" in df.columns and df["block_id"].nunique() > 1

    if has_blocks:
        blocks = sorted(df["block_id"].unique())
        for blk in blocks:
            g_vec = extract_block_g(df, gc, blk)
            if g_vec is not None:
                nP = int(df[df["block_id"] == blk]["n_perturb_setting"].iloc[0])
                g_dict[f"blk{blk}_nP{nP}"] = g_vec
    else:
        g_low, g_high = extract_perturb_g(df, gc)
        if g_low is not None:
            g_dict["low_perturb"] = g_low
        if g_high is not None:
            g_dict["high_perturb"] = g_high

    # Add g_zero
    g_dict["g_zero"] = np.zeros(len(gc), dtype=np.float32)

    print(f"  G conditions: {list(g_dict.keys())}")
    for k, v in g_dict.items():
        print(f"    {k}: ||g||={np.linalg.norm(v):.4f}")

    # Collect probes
    env_cfg = meta.get("env_cfg", {})
    probes = collect_zone_probes(env_cfg, n_per_zone=args.n_probes_per_zone)
    print(f"  Collected {len(probes)} probes across {len(set(p['zone_id'] for p in probes))} zones")

    # Analyze
    results = analyze_probes(agent, probes, g_dict, args.device)
    results.to_csv(outdir / "probe_representation.csv", index=False)
    print(f"  [csv] probe_representation.csv")

    # Figures
    g_labels = [k for k in g_dict.keys() if k != "g_zero"]
    if len(g_labels) < 2:
        g_labels = list(g_dict.keys())
    make_figures(results, g_labels, outdir)


if __name__ == "__main__":
    main()
