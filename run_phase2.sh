#!/usr/bin/env bash
# run_phase2.sh
#
# Usage:
#   ./run_phase2.sh [device] [mode]
#
# Modes:
#   all       - full sweep + mixed + ablation
#   sweep     - perturbation count 0-6
#   mixed     - block schedule within one agent
#   ablation  - adaptive vs fixed vs fast alpha
#   smoke     - smoke-sized ALL:
#               runs sweep + mixed + ablation
#               across all p1 seeds and all p2 seeds
#
# Existing Phase1 checkpoints expected at:
#   outputs/runs/p1_s0/ckpt_final.pt
#   outputs/runs/p1_s1/ckpt_final.pt
#   outputs/runs/p1_s2/ckpt_final.pt
#   outputs/runs/p1_s3/ckpt_final.pt
#   outputs/runs/p1_s4/ckpt_final.pt
#
# Parallelism:
#   - Parallelizes over Phase2 seeds only
#   - Default MAX_JOBS=2
#
# Examples:
#   chmod +x run_phase2.sh
#   ./run_phase2.sh cuda smoke
#   ./run_phase2.sh cuda all
#   MAX_JOBS=2 ./run_phase2.sh cuda smoke

set -euo pipefail

DEVICE="${1:-cuda}"
MODE="${2:-all}"

PHASE1_SEEDS=(${PHASE1_SEEDS:-0 1 2 3 4})
PHASE2_SEEDS=(${PHASE2_SEEDS:-0 1 2 3 4 5})

PHASE1_ROOT="${PHASE1_ROOT:-outputs/runs}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
MAX_JOBS="${MAX_JOBS:-2}"

# Full-run defaults
FULL_EPISODES="${EPISODES:-150}"
FULL_MAX_STEPS="${MAX_STEPS:-300}"
FULL_PE="${PRINT_EVERY:-30}"

# Smoke defaults
SMOKE_EPISODES="${SMOKE_EPISODES:-3}"
SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-300}"
SMOKE_PE="${SMOKE_PRINT_EVERY:-1}"

STAMP="$(date +%Y%m%d_%H%M%S)"
ROOT_OUT="outputs/phase2_${MODE}_${STAMP}"
LOG_ROOT="${ROOT_OUT}/logs"
mkdir -p "${ROOT_OUT}" "${LOG_ROOT}"

case "${MODE}" in
    smoke)
        RUN_EPISODES="${SMOKE_EPISODES}"
        RUN_MAX_STEPS="${SMOKE_MAX_STEPS}"
        RUN_PE="${SMOKE_PE}"
        RUN_TAG="smoke"
        ;;
    all|sweep|mixed|ablation)
        RUN_EPISODES="${FULL_EPISODES}"
        RUN_MAX_STEPS="${FULL_MAX_STEPS}"
        RUN_PE="${FULL_PE}"
        RUN_TAG="full"
        ;;
    *)
        echo "Unknown mode: ${MODE}"
        echo "Use: all | sweep | mixed | ablation | smoke"
        exit 1
        ;;
esac

echo "============================================"
echo "  Mode:         ${MODE}"
echo "  Run tag:      ${RUN_TAG}"
echo "  Device:       ${DEVICE}"
echo "  Phase1 root:  ${PHASE1_ROOT}"
echo "  Phase1 seeds: ${PHASE1_SEEDS[*]}"
echo "  Phase2 seeds: ${PHASE2_SEEDS[*]}"
echo "  Episodes:     ${RUN_EPISODES}"
echo "  Max steps:    ${RUN_MAX_STEPS}"
echo "  Print every:  ${RUN_PE}"
echo "  Max jobs:     ${MAX_JOBS}"
echo "  Root out:     ${ROOT_OUT}"
echo "============================================"

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

should_skip_dir() {
    local outdir="$1"

    if [ "${SKIP_EXISTING}" != "1" ]; then
        return 1
    fi

    if [ -f "${outdir}/metrics.json" ] || \
       [ -f "${outdir}/summary.json" ] || \
       [ -f "${outdir}/results.json" ] || \
       [ -f "${outdir}/done.txt" ] || \
       [ -f "${outdir}/ckpt_final.pt" ]; then
        return 0
    fi

    return 1
}

phase2_common_args() {
    local p1_ckpt="$1"
    echo "--phase1_ckpt ${p1_ckpt} --device ${DEVICE} --max_steps ${RUN_MAX_STEPS} --save_traj"
}

# -------------------------------------------------------
# Parallel helpers
# -------------------------------------------------------
PIDS=()
PID_LABELS=()

reset_job_queue() {
    PIDS=()
    PID_LABELS=()
}

reap_finished_jobs() {
    local new_pids=()
    local new_labels=()
    local i

    for i in "${!PIDS[@]}"; do
        local pid="${PIDS[$i]}"
        local label="${PID_LABELS[$i]}"

        if kill -0 "${pid}" 2>/dev/null; then
            new_pids+=("${pid}")
            new_labels+=("${label}")
        else
            if wait "${pid}"; then
                echo "[OK] finished: ${label}"
            else
                echo "[ERROR] failed: ${label}"
                exit 1
            fi
        fi
    done

    PIDS=("${new_pids[@]}")
    PID_LABELS=("${new_labels[@]}")
}

wait_for_slot() {
    while [ "${#PIDS[@]}" -ge "${MAX_JOBS}" ]; do
        reap_finished_jobs
        sleep 2
    done
}

wait_for_all() {
    while [ "${#PIDS[@]}" -gt 0 ]; do
        reap_finished_jobs
        sleep 2
    done
}

# -------------------------------------------------------
# Per-seed workers
# -------------------------------------------------------

worker_sweep_seed() {
    local p1_seed="$1"
    local p1_ckpt="$2"
    local s="$3"

    local seed_root="${ROOT_OUT}/from_p1_s${p1_seed}/sweep/seed${s}"
    local log_file="${LOG_ROOT}/from_p1_s${p1_seed}_sweep_seed${s}.log"

    mkdir -p "${seed_root}"

    {
        echo ""
        echo "############################################"
        echo "# SWEEP worker | p1_seed=${p1_seed} | p2_seed=${s}"
        echo "############################################"

        for n in 0 1 2 3 4 5 6; do
            local run_dir="${seed_root}/nperturb${n}"

            echo ""
            echo "=== n_perturbations=${n} ==="

            if should_skip_dir "${run_dir}"; then
                echo "Skip existing: ${run_dir}"
                continue
            fi

            mkdir -p "${run_dir}"
            python -m cear_pilot.training.train_phase2 $(phase2_common_args "${p1_ckpt}") \
              --seed "${s}" \
              --episodes "${RUN_EPISODES}" \
              --n_perturbations "${n}" \
              --print_every "${RUN_PE}" \
              --outdir "${run_dir}"
            touch "${run_dir}/done.txt"
        done

        echo ""
        echo "=== Analyze sweep | p2_seed=${s} ==="
        python -m cear_pilot.analysis.analyze_phase2 --sweep_root "${seed_root}"

        echo ""
        for d in "${seed_root}"/nperturb*; do
            if [ -d "${d}" ]; then
                echo "=== Probe representation: $(basename "${d}") ==="
                python -m cear_pilot.analysis.probe_representation \
                  --run_dir "${d}" --device "${DEVICE}"
            fi
        done
    } 2>&1 | tee "${log_file}"
}

worker_mixed_seed() {
    local p1_seed="$1"
    local p1_ckpt="$2"
    local s="$3"

    local seed_root="${ROOT_OUT}/from_p1_s${p1_seed}/mixed/seed${s}"
    local log_file="${LOG_ROOT}/from_p1_s${p1_seed}_mixed_seed${s}.log"

    mkdir -p "${seed_root}"

    {
        echo ""
        echo "############################################"
        echo "# MIXED worker | p1_seed=${p1_seed} | p2_seed=${s}"
        echo "############################################"

        if ! should_skip_dir "${seed_root}/mixed_0_4_0"; then
            mkdir -p "${seed_root}/mixed_0_4_0"
            python -m cear_pilot.training.train_phase2 $(phase2_common_args "${p1_ckpt}") \
              --seed "${s}" \
              --mixed_schedule "0:50,4:50,0:50" \
              --print_every "${RUN_PE}" \
              --outdir "${seed_root}/mixed_0_4_0"
            touch "${seed_root}/mixed_0_4_0/done.txt"
        else
            echo "Skip existing: ${seed_root}/mixed_0_4_0"
        fi

        if ! should_skip_dir "${seed_root}/mixed_0_6_0"; then
            mkdir -p "${seed_root}/mixed_0_6_0"
            python -m cear_pilot.training.train_phase2 $(phase2_common_args "${p1_ckpt}") \
              --seed "${s}" \
              --mixed_schedule "0:50,6:50,0:50" \
              --print_every "${RUN_PE}" \
              --outdir "${seed_root}/mixed_0_6_0"
            touch "${seed_root}/mixed_0_6_0/done.txt"
        else
            echo "Skip existing: ${seed_root}/mixed_0_6_0"
        fi

        if ! should_skip_dir "${seed_root}/mixed_ramp"; then
            mkdir -p "${seed_root}/mixed_ramp"
            python -m cear_pilot.training.train_phase2 $(phase2_common_args "${p1_ckpt}") \
              --seed "${s}" \
              --mixed_schedule "2:30,0:30,4:30,0:30,6:30" \
              --print_every "${RUN_PE}" \
              --outdir "${seed_root}/mixed_ramp"
            touch "${seed_root}/mixed_ramp/done.txt"
        else
            echo "Skip existing: ${seed_root}/mixed_ramp"
        fi

        echo ""
        for d in "${seed_root}"/mixed_*; do
            if [ -d "${d}" ]; then
                echo "=== Analyze: $(basename "${d}") ==="
                python -m cear_pilot.analysis.analyze_phase2 --root "${d}"

                echo "=== Probe representation: $(basename "${d}") ==="
                python -m cear_pilot.analysis.probe_representation \
                  --run_dir "${d}" --device "${DEVICE}"
            fi
        done
    } 2>&1 | tee "${log_file}"
}

worker_ablation_seed() {
    local p1_seed="$1"
    local p1_ckpt="$2"
    local s="$3"

    local seed_root="${ROOT_OUT}/from_p1_s${p1_seed}/ablation/seed${s}"
    local log_file="${LOG_ROOT}/from_p1_s${p1_seed}_ablation_seed${s}.log"

    mkdir -p "${seed_root}"

    {
        echo ""
        echo "############################################"
        echo "# ABLATION worker | p1_seed=${p1_seed} | p2_seed=${s}"
        echo "############################################"

        if ! should_skip_dir "${seed_root}/adaptive"; then
            mkdir -p "${seed_root}/adaptive"
            python -m cear_pilot.training.train_phase2 $(phase2_common_args "${p1_ckpt}") \
              --seed "${s}" \
              --episodes "${RUN_EPISODES}" \
              --n_perturbations 4 \
              --update_mode adaptive \
              --alpha_min 0.03 \
              --alpha_max 0.30 \
              --print_every "${RUN_PE}" \
              --outdir "${seed_root}/adaptive"
            touch "${seed_root}/adaptive/done.txt"
        else
            echo "Skip existing: ${seed_root}/adaptive"
        fi

        if ! should_skip_dir "${seed_root}/fixed_005"; then
            mkdir -p "${seed_root}/fixed_005"
            python -m cear_pilot.training.train_phase2 $(phase2_common_args "${p1_ckpt}") \
              --seed "${s}" \
              --episodes "${RUN_EPISODES}" \
              --n_perturbations 4 \
              --update_mode fixed \
              --alpha_fixed 0.05 \
              --print_every "${RUN_PE}" \
              --outdir "${seed_root}/fixed_005"
            touch "${seed_root}/fixed_005/done.txt"
        else
            echo "Skip existing: ${seed_root}/fixed_005"
        fi

        if ! should_skip_dir "${seed_root}/fast_080"; then
            mkdir -p "${seed_root}/fast_080"
            python -m cear_pilot.training.train_phase2 $(phase2_common_args "${p1_ckpt}") \
              --seed "${s}" \
              --episodes "${RUN_EPISODES}" \
              --n_perturbations 4 \
              --update_mode fixed \
              --alpha_fixed 0.80 \
              --print_every "${RUN_PE}" \
              --outdir "${seed_root}/fast_080"
            touch "${seed_root}/fast_080/done.txt"
        else
            echo "Skip existing: ${seed_root}/fast_080"
        fi

        echo ""
        echo "=== Analyze ablation | p2_seed=${s} ==="
        python -m cear_pilot.analysis.analyze_phase2 --ablation_root "${seed_root}"

        echo ""
        for d in "${seed_root}"/adaptive "${seed_root}"/fixed_* "${seed_root}"/fast_*; do
            if [ -d "${d}" ]; then
                echo "=== Probe representation: $(basename "${d}") ==="
                python -m cear_pilot.analysis.probe_representation \
                  --run_dir "${d}" --device "${DEVICE}"
            fi
        done
    } 2>&1 | tee "${log_file}"
}

# -------------------------------------------------------
# Mode runners for one Phase1 seed
# -------------------------------------------------------

run_sweep_for_one_p1() {
    local p1_seed="$1"
    local p1_ckpt="$2"

    echo ""
    echo "############################################"
    echo "# SWEEP | from_p1_s${p1_seed}"
    echo "############################################"

    reset_job_queue
    for s in "${PHASE2_SEEDS[@]}"; do
        wait_for_slot
        worker_sweep_seed "${p1_seed}" "${p1_ckpt}" "${s}" &
        PIDS+=("$!")
        PID_LABELS+=("sweep:p1s${p1_seed}:seed${s}")
    done
    wait_for_all
}

run_mixed_for_one_p1() {
    local p1_seed="$1"
    local p1_ckpt="$2"

    echo ""
    echo "############################################"
    echo "# MIXED | from_p1_s${p1_seed}"
    echo "############################################"

    reset_job_queue
    for s in "${PHASE2_SEEDS[@]}"; do
        wait_for_slot
        worker_mixed_seed "${p1_seed}" "${p1_ckpt}" "${s}" &
        PIDS+=("$!")
        PID_LABELS+=("mixed:p1s${p1_seed}:seed${s}")
    done
    wait_for_all
}

run_ablation_for_one_p1() {
    local p1_seed="$1"
    local p1_ckpt="$2"

    echo ""
    echo "############################################"
    echo "# ABLATION | from_p1_s${p1_seed}"
    echo "############################################"

    reset_job_queue
    for s in "${PHASE2_SEEDS[@]}"; do
        wait_for_slot
        worker_ablation_seed "${p1_seed}" "${p1_ckpt}" "${s}" &
        PIDS+=("$!")
        PID_LABELS+=("ablation:p1s${p1_seed}:seed${s}")
    done
    wait_for_all
}

run_smoke_for_one_p1() {
    local p1_seed="$1"
    local p1_ckpt="$2"

    echo ""
    echo "############################################"
    echo "# SMOKE-ALL | from_p1_s${p1_seed}"
    echo "############################################"

    # Smoke = short versions of all modes touched by MODE=all
    run_sweep_for_one_p1 "${p1_seed}" "${p1_ckpt}"
    run_mixed_for_one_p1 "${p1_seed}" "${p1_ckpt}"
    run_ablation_for_one_p1 "${p1_seed}" "${p1_ckpt}"
}

# -------------------------------------------------------
# Main
# -------------------------------------------------------

for p1s in "${PHASE1_SEEDS[@]}"; do
    p1_ckpt="${PHASE1_ROOT}/p1_s${p1s}/ckpt_final.pt"

    echo ""
    echo "============================================"
    echo "Phase1 seed: ${p1s}"
    echo "Checkpoint:  ${p1_ckpt}"
    echo "============================================"

    if [ ! -f "${p1_ckpt}" ]; then
        echo "ERROR: Missing checkpoint: ${p1_ckpt}"
        exit 1
    fi

    case "${MODE}" in
        all)
            run_sweep_for_one_p1 "${p1s}" "${p1_ckpt}"
            run_mixed_for_one_p1 "${p1s}" "${p1_ckpt}"
            run_ablation_for_one_p1 "${p1s}" "${p1_ckpt}"
            ;;
        smoke)
            run_smoke_for_one_p1 "${p1s}" "${p1_ckpt}"
            ;;
        sweep)
            run_sweep_for_one_p1 "${p1s}" "${p1_ckpt}"
            ;;
        mixed)
            run_mixed_for_one_p1 "${p1s}" "${p1_ckpt}"
            ;;
        ablation)
            run_ablation_for_one_p1 "${p1s}" "${p1_ckpt}"
            ;;
    esac
done

echo ""
echo "============================================"
echo "Done. Root output: ${ROOT_OUT}"
echo "Logs: ${LOG_ROOT}"
echo "============================================"