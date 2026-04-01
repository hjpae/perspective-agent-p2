#!/usr/bin/env bash
# quick_regime_test.sh
#
# Quick test: regime-coupled dynamics.
# 1 phase1 seed × 2 valences × 1 phase2 seed × 1 alpha = 2 runs
# 180 episodes × 300 steps each, carry_g=True
#
# Usage:
#   chmod +x quick_regime_test.sh
#   ./quick_regime_test.sh outputs/runs/p1_s0/ckpt_final.pt cuda

set -euo pipefail

P1_CKPT="${1:?Usage: $0 <phase1_ckpt> [device]}"
DEVICE="${2:-cuda}"

ALPHA=0.10
SEED=0
EPISODES=180
MAX_STEPS=300
SCHEDULE="1-1-1-1"
C_DECAY=0.985

REGIME_MU_SCALE=0.30
REGIME_SIGMA_SCALE=0.40
DISTORTION_SCALE=0.55

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="outputs/regime_test_${STAMP}"
mkdir -p "${OUT_ROOT}/logs"

echo "============================================"
echo "Regime-coupled dynamics test"
echo "  alpha:              ${ALPHA}"
echo "  regime_mu_scale:    ${REGIME_MU_SCALE}"
echo "  regime_sigma_scale: ${REGIME_SIGMA_SCALE}"
echo "  episodes:           ${EPISODES}"
echo "  carry_g:            TRUE"
echo "  output:             ${OUT_ROOT}"
echo "============================================"

echo ""
echo "=== Training SSSS ==="
python -m cear_pilot.training.train_phase2 \
  --phase1_ckpt "${P1_CKPT}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --episodes "${EPISODES}" \
  --max_steps "${MAX_STEPS}" \
  --update_mode fixed \
  --alpha_fixed "${ALPHA}" \
  --valence_sequence "SSSS" \
  --schedule_pattern "${SCHEDULE}" \
  --c_decay "${C_DECAY}" \
  --distortion_scale "${DISTORTION_SCALE}" \
  --regime_mu_scale "${REGIME_MU_SCALE}" \
  --regime_sigma_scale "${REGIME_SIGMA_SCALE}" \
  --save_traj \
  --carry_g_between_episodes \
  --freeze_encoder \
  --freeze_state \
  --freeze_policy \
  --no-freeze_decoder \
  --outdir "${OUT_ROOT}/runs/a${ALPHA}_SSSS_s${SEED}" \
  2>&1 | tee "${OUT_ROOT}/logs/ssss.log"

echo ""
echo "=== Training MMMM ==="
python -m cear_pilot.training.train_phase2 \
  --phase1_ckpt "${P1_CKPT}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --episodes "${EPISODES}" \
  --max_steps "${MAX_STEPS}" \
  --update_mode fixed \
  --alpha_fixed "${ALPHA}" \
  --valence_sequence "MMMM" \
  --schedule_pattern "${SCHEDULE}" \
  --c_decay "${C_DECAY}" \
  --distortion_scale "${DISTORTION_SCALE}" \
  --regime_mu_scale "${REGIME_MU_SCALE}" \
  --regime_sigma_scale "${REGIME_SIGMA_SCALE}" \
  --save_traj \
  --carry_g_between_episodes \
  --freeze_encoder \
  --freeze_state \
  --freeze_policy \
  --no-freeze_decoder \
  --outdir "${OUT_ROOT}/runs/a${ALPHA}_MMMM_s${SEED}" \
  2>&1 | tee "${OUT_ROOT}/logs/mmmm.log"

echo ""
echo "=== Quick comparison ==="
python3 - "${OUT_ROOT}" <<'PYEOF'
import sys, re
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(sys.argv[1]) / "runs"
g_cols = None
centroids = {}

for run_dir in sorted(root.iterdir()):
    if not run_dir.is_dir():
        continue
    traj_path = run_dir / "traj.parquet"
    if not traj_path.exists():
        traj_path = run_dir / "traj.csv"
    if not traj_path.exists():
        continue

    df = pd.read_parquet(traj_path) if traj_path.suffix == ".parquet" else pd.read_csv(traj_path)

    if g_cols is None:
        g_cols = sorted([c for c in df.columns if re.match(r"g_\d+$", c)],
                        key=lambda x: int(x.split("_")[-1]))

    name = run_dir.name
    max_ep = df["episode"].max()
    late = df[df["episode"] >= max(0, max_ep - 9)]
    g_late = late[g_cols].values
    g_norm = np.linalg.norm(g_late, axis=1)
    g_mean = g_late.mean(axis=0)
    centroids[name] = g_mean

    pe_col = None
    for c in ["pred_err", "mean_pred_loss"]:
        if c in df.columns:
            pe_col = c
            break

    print(f"\n{name}:")
    print(f"  ||g|| late: mean={g_norm.mean():.4f} std={g_norm.std():.4f}")
    print(f"  g centroid: [{', '.join(f'{v:.4f}' for v in g_mean[:4])}...]")
    if pe_col:
        pe_early = df[df["episode"] < 10][pe_col].values
        pe_late = late[pe_col].values
        print(f"  PE early: {pe_early.mean():.4f}  PE late: {pe_late.mean():.4f}")

    ep_norms = []
    for ep in sorted(df["episode"].unique()):
        ep_data = df[df["episode"] == ep]
        g_end = ep_data[ep_data["t"] == ep_data["t"].max()][g_cols].values
        if len(g_end) > 0:
            ep_norms.append(float(np.linalg.norm(g_end[0])))
    step = max(1, len(ep_norms) // 9)
    print(f"  ||g|| trajectory: {[f'{ep_norms[i]:.2f}' for i in range(0, len(ep_norms), step)]}")

print("\n" + "=" * 50)
print("CROSS-CONDITION COMPARISON")
print("=" * 50)

keys = sorted(centroids.keys())
if len(keys) >= 2:
    c0, c1 = centroids[keys[0]], centroids[keys[1]]
    dist = np.linalg.norm(c0 - c1)
    cos = np.dot(c0, c1) / (np.linalg.norm(c0) * np.linalg.norm(c1) + 1e-8)
    print(f"\n  {keys[0]} vs {keys[1]}:")
    print(f"  Centroid L2 distance: {dist:.4f}")
    print(f"  Cosine similarity:    {cos:.4f}")
    print(f"  Per-dim max diff:     {np.max(np.abs(c0 - c1)):.4f}")
    print()
    if dist > 0.5:
        print("  >>> STRONG separation! Regime coupling is working.")
    elif dist > 0.1:
        print("  >>> Moderate separation. Regime coupling has some effect.")
    else:
        print("  >>> Weak separation. May need stronger regime_mu_scale.")

PYEOF

echo ""
echo "============================================"
echo "Test complete. Output: ${OUT_ROOT}"
echo "============================================"
