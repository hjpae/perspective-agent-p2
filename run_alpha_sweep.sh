#!/usr/bin/env bash
# run_alpha_sweep.sh
# Alpha sweep experiment: 3 alpha × 2 valence × 5 seeds = 30 runs
# Then collect trajectories and run comparison analysis.
#
# Usage:
#   chmod +x run_alpha_sweep.sh
#   ./run_alpha_sweep.sh /path/to/phase1_checkpoint.pt [device]

set -euo pipefail

# ── Config ──────────────────────────────────────────────
PHASE1_CKPT="${1:?Usage: $0 <phase1_ckpt> [device]}"
DEVICE="${2:-cuda}"

ALPHAS=(0.05 0.25 0.50)
VALENCES=("SSSS" "MMMM")
SEEDS=(0 1 2 3 4)

EPISODES=180
MAX_STEPS=300
SCHEDULE="1-1-1-1"

# Phase 2 env changes
C_DECAY=0.985
DISTORTION_SCALE=0.55

# Output root
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="outputs/alpha_sweep_${TIMESTAMP}"
mkdir -p "${OUT_ROOT}"

echo "============================================"
echo "Alpha sweep experiment"
echo "  Phase 1 ckpt: ${PHASE1_CKPT}"
echo "  Device:       ${DEVICE}"
echo "  Alphas:       ${ALPHAS[*]}"
echo "  Valences:     ${VALENCES[*]}"
echo "  Seeds:        ${SEEDS[*]}"
echo "  Output:       ${OUT_ROOT}"
echo "  c_decay:      ${C_DECAY}"
echo "  distortion:   ${DISTORTION_SCALE}"
echo "============================================"
echo ""

# Save experiment config
cat > "${OUT_ROOT}/experiment_config.json" << EOF
{
  "phase1_ckpt": "${PHASE1_CKPT}",
  "device": "${DEVICE}",
  "alphas": [0.05, 0.25, 0.50],
  "valences": ["SSSS", "MMMM"],
  "seeds": [0, 1, 2, 3, 4],
  "episodes": ${EPISODES},
  "max_steps": ${MAX_STEPS},
  "schedule_pattern": "${SCHEDULE}",
  "c_decay": ${C_DECAY},
  "distortion_scale": ${DISTORTION_SCALE}
}
EOF

# ── Phase 1: Train all 30 runs ─────────────────────────
echo "=== PHASE: Training ==="

RUN_COUNT=0
TOTAL_RUNS=$(( ${#ALPHAS[@]} * ${#VALENCES[@]} * ${#SEEDS[@]} ))

for ALPHA in "${ALPHAS[@]}"; do
  for VAL in "${VALENCES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      RUN_COUNT=$((RUN_COUNT + 1))
      LABEL="a${ALPHA}_${VAL}_s${SEED}"
      RUN_DIR="${OUT_ROOT}/runs/${LABEL}"

      echo ""
      echo "[${RUN_COUNT}/${TOTAL_RUNS}] alpha=${ALPHA} valence=${VAL} seed=${SEED}"
      echo "  -> ${RUN_DIR}"

      python -m cear_pilot.training.train_phase2 \
        --phase1_ckpt "${PHASE1_CKPT}" \
        --device "${DEVICE}" \
        --seed "${SEED}" \
        --episodes "${EPISODES}" \
        --max_steps "${MAX_STEPS}" \
        --update_mode fixed \
        --alpha_fixed "${ALPHA}" \
        --valence_sequence "${VAL}" \
        --schedule_pattern "${SCHEDULE}" \
        --c_decay "${C_DECAY}" \
        --distortion_scale "${DISTORTION_SCALE}" \
        --outdir "${RUN_DIR}" \
        --save_traj \
        --freeze_encoder \
        --freeze_state \
        --freeze_policy \
        --freeze_decoder \
        2>&1 | tee "${RUN_DIR}/train.log"

    done
  done
done

echo ""
echo "=== All ${TOTAL_RUNS} training runs complete ==="

# ── Phase 2: Collect trajectories (greedy, for clean eval) ──
echo ""
echo "=== PHASE: Collection (greedy) ==="

RUN_COUNT=0
for ALPHA in "${ALPHAS[@]}"; do
  for VAL in "${VALENCES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      RUN_COUNT=$((RUN_COUNT + 1))
      LABEL="a${ALPHA}_${VAL}_s${SEED}"
      RUN_DIR="${OUT_ROOT}/runs/${LABEL}"
      COLLECT_DIR="${OUT_ROOT}/collect/${LABEL}"

      # Find the checkpoint
      CKPT="${RUN_DIR}/ckpt_final.pt"
      if [ ! -f "${CKPT}" ]; then
        # train_phase2 appends timestamp to outdir
        CKPT=$(find "${RUN_DIR}"* -name "ckpt_final.pt" 2>/dev/null | head -1)
      fi

      if [ -z "${CKPT}" ] || [ ! -f "${CKPT}" ]; then
        echo "[${RUN_COUNT}/${TOTAL_RUNS}] SKIP ${LABEL} - no checkpoint found"
        continue
      fi

      echo "[${RUN_COUNT}/${TOTAL_RUNS}] Collecting ${LABEL}"

      python -m cear_pilot.experiments.collect_phase2 \
        --ckpt "${CKPT}" \
        --device "${DEVICE}" \
        --episodes 40 \
        --seed 1000 \
        --greedy \
        --outdir "${COLLECT_DIR}" \
        --valence_sequence "${VAL}" \
        --schedule_pattern "${SCHEDULE}" \
        --max_steps "${MAX_STEPS}" \
        2>&1 | tee "${COLLECT_DIR}/collect.log"

    done
  done
done

echo ""
echo "=== Collection complete ==="

# ── Phase 3: Run comparison analysis ───────────────────
echo ""
echo "=== PHASE: Analysis ==="

ANALYSIS_DIR="${OUT_ROOT}/analysis"
mkdir -p "${ANALYSIS_DIR}"

python compare_alpha_sweep.py \
  --sweep_root "${OUT_ROOT}" \
  --outdir "${ANALYSIS_DIR}" \
  2>&1 | tee "${ANALYSIS_DIR}/analysis.log"

echo ""
echo "============================================"
echo "Experiment complete!"
echo "  Runs:     ${OUT_ROOT}/runs/"
echo "  Collect:  ${OUT_ROOT}/collect/"
echo "  Analysis: ${ANALYSIS_DIR}/"
echo "============================================"
