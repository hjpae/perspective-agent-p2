#!/usr/bin/env bash
set -euo pipefail

# Usage:
# cd /workspace/perspective-agent-p2
# export PYTHONPATH=$(pwd):$PYTHONPATH
# bash run_train_phase2_ssss.sh outputs/runs/p1_s0/ckpt_final.pt cuda

CKPT_PATH="${1:-}"
DEVICE="${2:-cuda}"

if [[ -z "$CKPT_PATH" ]]; then
  echo "Usage: bash run_train_phase2_ssss.sh /path/to/ckpt_final.pt [device]"
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="outputs/runs/phase2_train_ssss_alpha5_${STAMP}"

mkdir -p "$OUTDIR"

echo "[1/2] Running persistent-g Phase 2 formation rollout..."
python cear_pilot/training/train_phase2.py \
  --ckpt "$CKPT_PATH" \
  --device "$DEVICE" \
  --episodes 100 \
  --max_steps 300 \
  --alpha_fixed 0.50 \
  --schedule_pattern 1-1-1-1 \
  --valence_sequence SSSS \
  --seed 0 \
  --greedy \
  --save_traj \
  --outdir "$OUTDIR"

echo "[2/2] Plotting formation figures..."
python cear_pilot/analysis/plot_train_phase2.py \
  --run_dir "$OUTDIR" \
  --dims 4

echo "Done. Outputs are in: $OUTDIR"
