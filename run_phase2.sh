#!/usr/bin/env bash
# run_phase2.sh
#
# Usage:
#   ./run_phase2.sh <phase1_ckpt> [device] [mode]
#
# Modes:
#   all       - smoke + sweep + mixed + ablation sequentially
#   sweep     - perturbation count 0-6
#   mixed     - block schedule within one agent
#   ablation  - adaptive vs fixed vs fast alpha
#   smoke     - 3-episode quick check
#
# Examples:
#   chmod +x run_phase2.sh
#   ./run_phase2.sh outputs/runs/p1_s0/ckpt_final.pt cuda all
#   EPISODES=200 SEED=42 ./run_phase2.sh outputs/runs/p1_s0/ckpt_final.pt cuda sweep

set -euo pipefail

P1_CKPT="${1:?Usage: $0 <phase1_ckpt> [device] [mode: all|sweep|mixed|ablation|smoke>]}"
DEVICE="${2:-cuda}"
MODE="${3:-all}"

SEED="${SEED:-0}"
EPISODES="${EPISODES:-150}"
MAX_STEPS="${MAX_STEPS:-300}"
PE="${PRINT_EVERY:-30}"

STAMP="$(date +%Y%m%d_%H%M%S)"
ROOT_OUT="outputs/phase2_${STAMP}"
mkdir -p "${ROOT_OUT}"

echo "============================================"
echo "  Mode:    ${MODE}"
echo "  Ckpt:    ${P1_CKPT}"
echo "  Device:  ${DEVICE}"
echo "  Seed:    ${SEED}"
echo "  Root:    ${ROOT_OUT}"
echo "============================================"

COMMON="--phase1_ckpt ${P1_CKPT} --device ${DEVICE} --seed ${SEED} --max_steps ${MAX_STEPS} --save_traj"

run_smoke() {
    local OUT="${ROOT_OUT}/smoke"
    mkdir -p "${OUT}"

    echo ""
    echo "############################################"
    echo "# SMOKE"
    echo "############################################"

    python -m cear_pilot.training.train_phase2 ${COMMON} \
      --episodes 3 --n_perturbations 2 --print_every 1 \
      --outdir "${OUT}/_smoke"

    echo "SMOKE DONE: ${OUT}/_smoke"
}

run_sweep() {
    local OUT="${ROOT_OUT}/sweep"
    mkdir -p "${OUT}"

    echo ""
    echo "############################################"
    echo "# SWEEP"
    echo "############################################"

    for N in 0 1 2 3 4 5 6; do
        echo ""
        echo "=== n_perturbations=${N} ==="
        python -m cear_pilot.training.train_phase2 ${COMMON} \
          --episodes "${EPISODES}" \
          --n_perturbations "${N}" \
          --print_every "${PE}" \
          --outdir "${OUT}/nperturb${N}_s${SEED}"
    done

    echo ""
    echo "=== Analyze sweep ==="
    python -m cear_pilot.analysis.analyze_phase2 --sweep_root "${OUT}"

    echo ""
    for d in "${OUT}"/nperturb*; do
        if [ -d "${d}" ]; then
            echo "=== Probe representation: $(basename "${d}") ==="
            python -m cear_pilot.analysis.probe_representation \
              --run_dir "${d}" --device "${DEVICE}"
        fi
    done
}

run_mixed() {
    local OUT="${ROOT_OUT}/mixed"
    mkdir -p "${OUT}"

    echo ""
    echo "############################################"
    echo "# MIXED"
    echo "############################################"

    echo ""
    echo "=== Mixed schedule: 0:50, 4:50, 0:50 ==="
    python -m cear_pilot.training.train_phase2 ${COMMON} \
      --mixed_schedule "0:50,4:50,0:50" \
      --print_every "${PE}" \
      --outdir "${OUT}/mixed_0_4_0"

    echo ""
    echo "=== Mixed schedule: 0:50, 6:50, 0:50 ==="
    python -m cear_pilot.training.train_phase2 ${COMMON} \
      --mixed_schedule "0:50,6:50,0:50" \
      --print_every "${PE}" \
      --outdir "${OUT}/mixed_0_6_0"

    echo ""
    echo "=== Mixed schedule: 2:30, 0:30, 4:30, 0:30, 6:30 ==="
    python -m cear_pilot.training.train_phase2 ${COMMON} \
      --mixed_schedule "2:30,0:30,4:30,0:30,6:30" \
      --print_every "${PE}" \
      --outdir "${OUT}/mixed_ramp"

    echo ""
    for d in "${OUT}"/mixed_*; do
        if [ -d "${d}" ]; then
            echo "=== Analyze: $(basename "${d}") ==="
            python -m cear_pilot.analysis.analyze_phase2 --root "${d}"

            echo ""
            echo "=== Probe representation: $(basename "${d}") ==="
            python -m cear_pilot.analysis.probe_representation \
              --run_dir "${d}" --device "${DEVICE}"
        fi
    done
}

run_ablation() {
    local OUT="${ROOT_OUT}/ablation"
    mkdir -p "${OUT}"

    echo ""
    echo "############################################"
    echo "# ABLATION"
    echo "############################################"

    echo ""
    echo "=== Adaptive alpha (self-modulating) ==="
    python -m cear_pilot.training.train_phase2 ${COMMON} \
      --episodes "${EPISODES}" \
      --n_perturbations 4 \
      --update_mode adaptive \
      --alpha_min 0.03 \
      --alpha_max 0.30 \
      --print_every "${PE}" \
      --outdir "${OUT}/adaptive"

    echo ""
    echo "=== Fixed alpha = 0.05 ==="
    python -m cear_pilot.training.train_phase2 ${COMMON} \
      --episodes "${EPISODES}" \
      --n_perturbations 4 \
      --update_mode fixed \
      --alpha_fixed 0.05 \
      --print_every "${PE}" \
      --outdir "${OUT}/fixed_010"

    echo ""
    echo "=== Fast alpha = 0.80 (near-instant update) ==="
    python -m cear_pilot.training.train_phase2 ${COMMON} \
      --episodes "${EPISODES}" \
      --n_perturbations 4 \
      --update_mode fixed \
      --alpha_fixed 0.80 \
      --print_every "${PE}" \
      --outdir "${OUT}/fast_080"

    echo ""
    echo "=== Analyze ablation ==="
    python -m cear_pilot.analysis.analyze_phase2 --ablation_root "${OUT}"

    echo ""
    for d in "${OUT}"/adaptive "${OUT}"/fixed_* "${OUT}"/fast_*; do
        if [ -d "${d}" ]; then
            echo "=== Probe representation: $(basename "${d}") ==="
            python -m cear_pilot.analysis.probe_representation \
              --run_dir "${d}" --device "${DEVICE}"
        fi
    done
}

case "${MODE}" in
    all)
        run_mixed
        run_ablation
        ;;
    smoke)
        run_smoke
        ;;
    sweep)
        run_sweep
        ;;
    mixed)
        run_mixed
        ;;
    ablation)
        run_ablation
        ;;
    *)
        echo "Unknown mode: ${MODE}"
        echo "Use: all | sweep | mixed | ablation | smoke"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "Done. Root output: ${ROOT_OUT}"
echo "============================================"