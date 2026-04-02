#!/usr/bin/env bash
# run_phase2.sh
#
# Usage:
#   ./run_phase2.sh <phase1_ckpt> [device]          # full sweep (0-6 perturbations)
#   ./run_phase2.sh <phase1_ckpt> [device] --smoke   # smoke test only (3 episodes)
#
# Override defaults with env vars:
#   EPISODES=200 SEED=42 ./run_phase2.sh <ckpt> cuda
#
#chmod +x run_phase2.sh
#./run_phase2.sh outputs/runs/p1_s0/ckpt_final.pt cuda

set -euo pipefail

P1_CKPT="${1:?Usage: $0 <phase1_ckpt> [device] [--smoke]}"
DEVICE="${2:-cuda}"
MODE="sweep"
if [[ "${3:-}" == "--smoke" ]]; then
    MODE="smoke"
fi

SEED="${SEED:-0}"
EPISODES="${EPISODES:-150}"
MAX_STEPS="${MAX_STEPS:-300}"
PRINT_EVERY="${PRINT_EVERY:-10}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="outputs/sweep_${STAMP}"
mkdir -p "${OUT_ROOT}"

echo "============================================"
echo "Phase 2 run"
echo "  Mode:          ${MODE}"
echo "  Phase 1 ckpt:  ${P1_CKPT}"
echo "  Device:        ${DEVICE}"
echo "  Episodes:      ${EPISODES}"
echo "  Seed:          ${SEED}"
echo "  Output:        ${OUT_ROOT}"
echo "============================================"

if [[ "${MODE}" == "smoke" ]]; then
    echo ""
    echo "=== SMOKE TEST (10 episodes) ==="
    python -m cear_pilot.training.train_phase2 \
      --phase1_ckpt "${P1_CKPT}" \
      --device "${DEVICE}" \
      --seed "${SEED}" \
      --episodes 10 \
      --max_steps "${MAX_STEPS}" \
      --n_perturbations 4 \
      --save_traj \
      --print_every 1 \
      --outdir "${OUT_ROOT}/_smoke"

    echo ""
    echo "SMOKE TEST DONE. Output: ${OUT_ROOT}/_smoke"
    exit 0
fi

# ── Full perturbation sweep ──
for N in 0 1 2 3 4 5 6; do
    RUNDIR="${OUT_ROOT}/nperturb${N}_s${SEED}"
    echo ""
    echo "=== n_perturbations=${N} (${EPISODES} episodes) ==="

    python -m cear_pilot.training.train_phase2 \
      --phase1_ckpt "${P1_CKPT}" \
      --device "${DEVICE}" \
      --seed "${SEED}" \
      --episodes "${EPISODES}" \
      --max_steps "${MAX_STEPS}" \
      --n_perturbations "${N}" \
      --save_traj \
      --print_every "${PRINT_EVERY}" \
      --outdir "${RUNDIR}"

    echo "[DONE] n=${N} → ${RUNDIR}"
done

# ── Analysis ──
echo ""
echo "=== ANALYSIS ==="
python -m cear_pilot.analysis.analyze_phase2 --sweep_root "${OUT_ROOT}"

echo ""
echo "============================================"
echo "Complete. Results: ${OUT_ROOT}"
echo "============================================"