#!/usr/bin/env bash
# quick_scratch_test.sh
#
# Train from scratch: AAAI-equivalent dynamics + regime-coupled env.
# No Phase 1 pretrain. All components trainable.
#
# Usage:
#   chmod +x quick_scratch_test.sh
#   ./quick_scratch_test.sh cuda

set -euo pipefail

DEVICE="${1:-cuda}"

SEED=0
EPISODES=300
MAX_STEPS=300
ALPHA=0.10
SCHEDULE="1-1-1-1"
C_DECAY=0.985
DISTORTION_SCALE=0.55
REGIME_MU_SCALE=0.30
REGIME_SIGMA_SCALE=0.40

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="outputs/scratch_test_${STAMP}"
mkdir -p "${OUT_ROOT}/logs"

echo "============================================"
echo "From-scratch training"
echo "  energy_mode:  none (AAAI equivalent)"
echo "  regime:       mu=${REGIME_MU_SCALE} sigma=${REGIME_SIGMA_SCALE}"
echo "  episodes:     ${EPISODES}"
echo "  freezing:     NONE (all trainable)"
echo "  output:       ${OUT_ROOT}"
echo "============================================"

for VAL in SSSS MMMM; do
  echo ""
  echo "=== Training ${VAL} (from scratch) ==="
  python -m cear_pilot.training.train_phase2 \
    --from_scratch \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --episodes "${EPISODES}" \
    --max_steps "${MAX_STEPS}" \
    --update_mode fixed \
    --alpha_fixed "${ALPHA}" \
    --energy_mode none \
    --dyn_eta 0.0 \
    --confine_lambda 0.0 \
    --valence_sequence "${VAL}" \
    --schedule_pattern "${SCHEDULE}" \
    --c_decay "${C_DECAY}" \
    --distortion_scale "${DISTORTION_SCALE}" \
    --regime_mu_scale "${REGIME_MU_SCALE}" \
    --regime_sigma_scale "${REGIME_SIGMA_SCALE}" \
    --save_traj \
    --carry_g_between_episodes \
    --no-freeze_encoder \
    --no-freeze_state \
    --no-freeze_policy \
    --no-freeze_decoder \
    --w_energy 0.0 \
    --w_basin 0.0 \
    --w_sep 0.0 \
    --outdir "${OUT_ROOT}/runs/${VAL}_s${SEED}" \
    2>&1 | tee "${OUT_ROOT}/logs/${VAL}.log"
done

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
    for ext in [".parquet", ".csv"]:
        p = run_dir / f"traj{ext}"
        if p.exists():
            df = pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p)
            break
    else:
        continue

    if g_cols is None:
        g_cols = sorted([c for c in df.columns if re.match(r"g_\d+$", c)],
                        key=lambda x: int(x.split("_")[-1]))

    name = run_dir.name
    max_ep = df["episode"].max()
    late = df[df["episode"] >= max(0, max_ep - 19)]
    g_late = late[g_cols].values
    centroids[name] = g_late.mean(axis=0)

    pe_col = next((c for c in ["pred_err", "mean_pred_loss"] if c in df.columns), None)

    print(f"\n{name}:")
    print(f"  ||g|| late: mean={np.linalg.norm(g_late, axis=1).mean():.4f}")
    if pe_col:
        pe_early = df[df["episode"] < 20][pe_col].mean()
        pe_late = late[pe_col].mean()
        print(f"  PE: early={pe_early:.4f} → late={pe_late:.4f} (delta={pe_early-pe_late:+.4f})")

print("\n" + "="*50)
keys = sorted(centroids.keys())
if len(keys) >= 2:
    dist = np.linalg.norm(centroids[keys[0]] - centroids[keys[1]])
    cos = np.dot(centroids[keys[0]], centroids[keys[1]]) / (
        np.linalg.norm(centroids[keys[0]]) * np.linalg.norm(centroids[keys[1]]) + 1e-8)
    print(f"  Centroid distance: {dist:.4f}")
    print(f"  Cosine: {cos:.4f}")
    print(f"\n  Then run: python do_g_quick.py --regime_root {sys.argv[1]}")
PYEOF

echo ""
echo "============================================"
echo "Done. Output: ${OUT_ROOT}"
echo "============================================"
