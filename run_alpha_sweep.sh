#!/usr/bin/env bash
# run_alpha_sweep.sh
#
# Full Phase 2 alpha sweep:
#   3 alpha × 2 valence × 6 seeds = 36 runs
#   training (parallel, max 2 jobs)
#   -> greedy collection (parallel, max 2 jobs)
#   -> analysis
#
# Usage:
#   chmod +x run_alpha_sweep.sh
#   ./run_alpha_sweep.sh /path/to/phase1_ckpt.pt [device]
#
# Example:
#chmod +x run_alpha_sweep.sh
#./run_alpha_sweep.sh outputs/runs/p1_s0/ckpt_final.pt cuda

set -euo pipefail

PHASE1_CKPT="${1:?Usage: $0 <phase1_ckpt> [device]}"
DEVICE="${2:-cuda}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# -------------------------------------------------------
# Experiment config
# -------------------------------------------------------
ALPHAS=(0.05 0.25 0.50)
VALENCES=("SSSS" "MMMM")
SEEDS=(0 1 2 3 4 5)

EPISODES=180
COLLECT_EPISODES=40
MAX_STEPS=300
SCHEDULE="1-1-1-1"

C_DECAY=0.985
DISTORTION_SCALE=0.55
EVENT_DELAY=10
MIN_EVENT_GAP=16

MAX_JOBS=2
COLLECT_SEED=1000

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="outputs/alpha_sweep_${STAMP}"
RUNS_DIR="${OUT_ROOT}/runs"
COLLECT_DIR="${OUT_ROOT}/collect"
ANALYSIS_DIR="${OUT_ROOT}/analysis"
LOGS_DIR="${OUT_ROOT}/logs"

mkdir -p "${RUNS_DIR}" "${COLLECT_DIR}" "${ANALYSIS_DIR}" "${LOGS_DIR}"

TOTAL_RUNS=$(( ${#ALPHAS[@]} * ${#VALENCES[@]} * ${#SEEDS[@]} ))

# -------------------------------------------------------
# Sanity checks
# -------------------------------------------------------
echo "============================================"
echo "Phase 2 alpha sweep"
echo "  repo root:      ${ROOT_DIR}"
echo "  phase1 ckpt:    ${PHASE1_CKPT}"
echo "  device:         ${DEVICE}"
echo "  out root:       ${OUT_ROOT}"
echo "  total runs:     ${TOTAL_RUNS}"
echo "  max jobs:       ${MAX_JOBS}"
echo "  alphas:         ${ALPHAS[*]}"
echo "  valences:       ${VALENCES[*]}"
echo "  seeds:          ${SEEDS[*]}"
echo "  episodes:       ${EPISODES}"
echo "  collect eps:    ${COLLECT_EPISODES}"
echo "  max steps:      ${MAX_STEPS}"
echo "  schedule:       ${SCHEDULE}"
echo "  c_decay:        ${C_DECAY}"
echo "  distortion:     ${DISTORTION_SCALE}"
echo "  event_delay:    ${EVENT_DELAY}"
echo "  min_event_gap:  ${MIN_EVENT_GAP}"
echo "============================================"
echo ""

if [ ! -f "${PHASE1_CKPT}" ]; then
  echo "[ERROR] phase1 checkpoint not found: ${PHASE1_CKPT}"
  exit 1
fi

python - <<'PY'
import importlib
mods = [
    "cear_pilot.training.train_phase2",
    "cear_pilot.experiments.collect_phase2",
    "cear_pilot.analysis.compare_alpha_sweep",
]
for m in mods:
    importlib.import_module(m)
print("[OK] import sanity check passed")
PY

cat > "${OUT_ROOT}/experiment_config.json" << EOF
{
  "phase1_ckpt": "${PHASE1_CKPT}",
  "device": "${DEVICE}",
  "alphas": [0.05, 0.25, 0.50],
  "valences": ["SSSS", "MMMM"],
  "seeds": [0, 1, 2, 3, 4, 5],
  "episodes": ${EPISODES},
  "collect_episodes": ${COLLECT_EPISODES},
  "max_steps": ${MAX_STEPS},
  "schedule_pattern": "${SCHEDULE}",
  "c_decay": ${C_DECAY},
  "distortion_scale": ${DISTORTION_SCALE},
  "event_delay_steps": ${EVENT_DELAY},
  "min_event_gap": ${MIN_EVENT_GAP},
  "max_parallel_jobs": ${MAX_JOBS},
  "collect_seed": ${COLLECT_SEED}
}
EOF

# -------------------------------------------------------
# Parallel helpers
# -------------------------------------------------------
PIDS=()
PID_LABELS=()

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

resolve_latest_run_dir() {
  local label="$1"
  local resolved
  resolved="$(ls -dt "${RUNS_DIR}/${label}"* 2>/dev/null | head -1 || true)"
  if [ -n "${resolved}" ] && [ -d "${resolved}" ]; then
    printf '%s\n' "${resolved}"
    return 0
  fi
  return 1
}

# -------------------------------------------------------
# Phase 1: Training
# -------------------------------------------------------
echo ""
echo "=== PHASE 1/3: Training ==="

RUN_COUNT=0

for ALPHA in "${ALPHAS[@]}"; do
  for VAL in "${VALENCES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      RUN_COUNT=$((RUN_COUNT + 1))

      LABEL="a${ALPHA}_${VAL}_s${SEED}"
      TRAIN_BASE="${RUNS_DIR}/${LABEL}"
      TRAIN_LOG="${LOGS_DIR}/${LABEL}_train.log"

      wait_for_slot

      (
        echo "============================================"
        echo "[TRAIN ${RUN_COUNT}/${TOTAL_RUNS}] ${LABEL}"
        echo "  out base: ${TRAIN_BASE}"
        echo "  log:      ${TRAIN_LOG}"
        echo "============================================"

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
          --event_delay_steps "${EVENT_DELAY}" \
          --min_event_gap "${MIN_EVENT_GAP}" \
          --save_traj \
          --freeze_encoder \
          --freeze_state \
          --freeze_policy \
          --freeze_decoder \
          --outdir "${TRAIN_BASE}" \
          2>&1 | tee "${TRAIN_LOG}"
      ) &

      PIDS+=("$!")
      PID_LABELS+=("train:${LABEL}")
    done
  done
done

wait_for_all

echo ""
echo "=== Training complete ==="

# -------------------------------------------------------
# Phase 2: Collection
# -------------------------------------------------------
echo ""
echo "=== PHASE 2/3: Greedy collection ==="

RUN_COUNT=0

for ALPHA in "${ALPHAS[@]}"; do
  for VAL in "${VALENCES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      RUN_COUNT=$((RUN_COUNT + 1))

      LABEL="a${ALPHA}_${VAL}_s${SEED}"
      COLLECT_RUN_DIR="${COLLECT_DIR}/${LABEL}"
      COLLECT_LOG="${LOGS_DIR}/${LABEL}_collect.log"

      ACTUAL_RUN_DIR="$(resolve_latest_run_dir "${LABEL}" || true)"
      if [ -z "${ACTUAL_RUN_DIR}" ] || [ ! -d "${ACTUAL_RUN_DIR}" ]; then
        echo "[WARN] [COLLECT ${RUN_COUNT}/${TOTAL_RUNS}] ${LABEL} -> no train dir found, skipping"
        continue
      fi

      CKPT="${ACTUAL_RUN_DIR}/ckpt_final.pt"
      if [ ! -f "${CKPT}" ]; then
        echo "[WARN] [COLLECT ${RUN_COUNT}/${TOTAL_RUNS}] ${LABEL} -> no ckpt_final.pt, skipping"
        continue
      fi

      mkdir -p "${COLLECT_RUN_DIR}"

      wait_for_slot

      (
        echo "============================================"
        echo "[COLLECT ${RUN_COUNT}/${TOTAL_RUNS}] ${LABEL}"
        echo "  train dir: ${ACTUAL_RUN_DIR}"
        echo "  ckpt:      ${CKPT}"
        echo "  out dir:   ${COLLECT_RUN_DIR}"
        echo "  log:       ${COLLECT_LOG}"
        echo "============================================"

        python -m cear_pilot.experiments.collect_phase2 \
          --ckpt "${CKPT}" \
          --device "${DEVICE}" \
          --episodes "${COLLECT_EPISODES}" \
          --seed "${COLLECT_SEED}" \
          --greedy \
          --outdir "${COLLECT_RUN_DIR}" \
          --valence_sequence "${VAL}" \
          --schedule_pattern "${SCHEDULE}" \
          --max_steps "${MAX_STEPS}" \
          2>&1 | tee "${COLLECT_LOG}"
      ) &

      PIDS+=("$!")
      PID_LABELS+=("collect:${LABEL}")
    done
  done
done

wait_for_all

echo ""
echo "=== Collection complete ==="

# -------------------------------------------------------
# Phase 3: Analysis
# -------------------------------------------------------
echo ""
echo "=== PHASE 3/3: Analysis ==="

ANALYSIS_LOG="${LOGS_DIR}/analysis.log"

python -m cear_pilot.analysis.compare_alpha_sweep \
  --sweep_root "${OUT_ROOT}" \
  --outdir "${ANALYSIS_DIR}" \
  2>&1 | tee "${ANALYSIS_LOG}"

echo ""
echo "============================================"
echo "ALPHA SWEEP COMPLETE"
echo "  root:      ${OUT_ROOT}"
echo "  runs:      ${RUNS_DIR}"
echo "  collect:   ${COLLECT_DIR}"
echo "  analysis:  ${ANALYSIS_DIR}"
echo "  logs:      ${LOGS_DIR}"
echo "============================================"