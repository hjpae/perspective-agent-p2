#!/usr/bin/env bash
set -euo pipefail

CKPT="${1:-outputs/runs/p1_s0/ckpt_final.pt}"
DEVICE="${DEVICE:-cuda}"
EPISODES="${EPISODES:-100}"
PRINT_EVERY_STEPS="${PRINT_EVERY_STEPS:-1000}"
SEED="${SEED:-0}"

# Core assay set
# tag pattern valence
CONDITIONS=(
  "ssss 1-1-1-1 SSSS"
  "mmmm 1-1-1-1 MMMM"
  "s_s_s_m 1-1-1-1 SSSM"
  "sss_m 3-1 SSSM"
  "s_s_m_m 1-1-1-1 SSMM"
  "ss_mm 2-2 SSMM"
)

ALPHAS=(0.03 0.10 0.24)

STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_OUT="outputs/runs/phase2_g_sweeps_${STAMP}"
mkdir -p "${BASE_OUT}"

echo "[phase2_g_sweeps] ckpt=${CKPT}"
echo "[phase2_g_sweeps] out=${BASE_OUT}"
echo "[phase2_g_sweeps] episodes=${EPISODES}"
echo "[phase2_g_sweeps] device=${DEVICE}"

for alpha in "${ALPHAS[@]}"; do
  for spec in "${CONDITIONS[@]}"; do
    read -r tag pattern valence <<< "${spec}"

    RUN_OUT="${BASE_OUT}/a_${alpha}/${tag}"
    mkdir -p "${RUN_OUT}"

    echo
    echo "============================================================"
    echo "[RUN] alpha=${alpha} tag=${tag} pattern=${pattern} valence=${valence}"
    echo "============================================================"

    python -m cear_pilot.experiments.phase2_g_assay \
      --ckpt "${CKPT}" \
      --device "${DEVICE}" \
      --episodes "${EPISODES}" \
      --seed "${SEED}" \
      --alpha_fixed "${alpha}" \
      --schedule_pattern "${pattern}" \
      --valence_sequence "${valence}" \
      --print_every_steps "${PRINT_EVERY_STEPS}" \
      --outdir "${RUN_OUT}"
  done
done

echo
echo "[phase2_g_sweeps] done: ${BASE_OUT}"