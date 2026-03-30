#!/usr/bin/env bash
# run_alpha_sweep_v2.sh
#
# Fine alpha sweep:
#   4 alpha × 2 valence × 6 phase2 seeds × 5 phase1 seeds = 240 train runs
#
# Modes:
#   Full:
# chmod +x run_alpha_sweep_v2.sh
# ./run_alpha_sweep_v2.sh ALL cuda
#
#   Smoke:
# chmod +x run_alpha_sweep_v2.sh
# ./run_alpha_sweep_v2.sh ALL cuda --smoke
#
# Notes:
#   - In both full and smoke modes, ALL means using all 5 Phase 1 checkpoints.
#   - Smoke mode still sweeps all Phase 1 checkpoints; it only reduces episode count.
#   - Expects:
#       outputs/runs/p1_s0/ckpt_final.pt
#       outputs/runs/p1_s1/ckpt_final.pt
#       outputs/runs/p1_s2/ckpt_final.pt
#       outputs/runs/p1_s3/ckpt_final.pt
#       outputs/runs/p1_s4/ckpt_final.pt
#
# Analyses:
#   (1) compare_alpha_sweep
#   (2) reanalyze_gscore
#   (3) analyze_g_pca
#
# Optional:
#   export REVERSAL_ROOT=outputs/reversal_YYYYMMDD_HHMMSS
#   ./run_alpha_sweep_v2.sh ALL cuda

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <phase1_ckpt_or_ALL> [device] [--smoke]"
  exit 1
fi

P1_ARG="$1"
DEVICE="${2:-cuda}"

SMOKE=0
for arg in "$@"; do
  if [ "${arg}" = "--smoke" ]; then
    SMOKE=1
  fi
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# Optional reversal root for PCA reversal panel
REVERSAL_ROOT="${REVERSAL_ROOT:-}"

# -------------------------------------------------------
# Sweep config
# -------------------------------------------------------
ALPHAS=(0.05 0.10 0.20 0.30)
VALENCES=("SSSS" "MMMM")
P2_SEEDS=(0 1 2 3 4 5)

SCHEDULE="1-1-1-1"
C_DECAY=0.985
DISTORTION_SCALE=0.55
EVENT_DELAY=10
MIN_EVENT_GAP=16
MAX_JOBS=2

if [ "${SMOKE}" -eq 1 ]; then
  EPISODES=4
  MAX_STEPS=10
  RUN_TAG="smoke"
else
  EPISODES=180
  MAX_STEPS=300
  RUN_TAG="full"
fi

# -------------------------------------------------------
# Resolve analysis entrypoints
# -------------------------------------------------------
COMPARE_CMD=()
GSCORE_CMD=()
PCA_CMD=()

if python - <<'PY' >/dev/null 2>&1
import importlib
importlib.import_module("cear_pilot.analysis.compare_alpha_sweep")
PY
then
  COMPARE_CMD=(python -m cear_pilot.analysis.compare_alpha_sweep)
else
  if [ -f "${ROOT_DIR}/compare_alpha_sweep.py" ]; then
    COMPARE_CMD=(python "${ROOT_DIR}/compare_alpha_sweep.py")
  else
    echo "[ERROR] compare_alpha_sweep entrypoint not found"
    exit 1
  fi
fi

if python - <<'PY' >/dev/null 2>&1
import importlib
importlib.import_module("cear_pilot.analysis.reanalyze_gscore")
PY
then
  GSCORE_CMD=(python -m cear_pilot.analysis.reanalyze_gscore)
else
  if [ -f "${ROOT_DIR}/reanalyze_gscore.py" ]; then
    GSCORE_CMD=(python "${ROOT_DIR}/reanalyze_gscore.py")
  else
    echo "[WARN] reanalyze_gscore entrypoint not found as module or local script"
    GSCORE_CMD=()
  fi
fi

if python - <<'PY' >/dev/null 2>&1
import importlib
importlib.import_module("cear_pilot.analysis.analyze_g_pca")
PY
then
  PCA_CMD=(python -m cear_pilot.analysis.analyze_g_pca)
else
  if [ -f "${ROOT_DIR}/analyze_g_pca.py" ]; then
    PCA_CMD=(python "${ROOT_DIR}/analyze_g_pca.py")
  else
    echo "[WARN] analyze_g_pca entrypoint not found as module or local script"
    PCA_CMD=()
  fi
fi

# -------------------------------------------------------
# Resolve Phase 1 checkpoints
# -------------------------------------------------------
P1_CKPTS=()
P1_LABELS=()

if [ "${P1_ARG}" = "ALL" ]; then
  for s in 0 1 2 3 4; do
    ckpt="outputs/runs/p1_s${s}/ckpt_final.pt"
    if [ -f "${ckpt}" ]; then
      P1_CKPTS+=("${ckpt}")
      P1_LABELS+=("p1s${s}")
    else
      echo "[WARN] missing ${ckpt} — skipping p1_s${s}"
    fi
  done

  if [ "${#P1_CKPTS[@]}" -eq 0 ]; then
    echo "[ERROR] no Phase 1 checkpoints found"
    exit 1
  fi
else
  if [ ! -f "${P1_ARG}" ]; then
    echo "[ERROR] checkpoint not found: ${P1_ARG}"
    exit 1
  fi

  P1_CKPTS+=("${P1_ARG}")

  base_dir="$(basename "$(dirname "${P1_ARG}")")"
  if [[ "${base_dir}" =~ ^p1_s([0-9]+)$ ]]; then
    P1_LABELS+=("p1s${BASH_REMATCH[1]}")
  else
    P1_LABELS+=("p1single")
  fi
fi

TOTAL_RUNS=$(( ${#ALPHAS[@]} * ${#VALENCES[@]} * ${#P2_SEEDS[@]} * ${#P1_CKPTS[@]} ))

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="outputs/alpha_sweep_v2_${RUN_TAG}_${STAMP}"
RUNS_DIR="${OUT_ROOT}/runs"
LOGS_DIR="${OUT_ROOT}/logs"

ANALYSIS1_DIR="${OUT_ROOT}/analysis"
ANALYSIS2_DIR="${OUT_ROOT}/reanalysis"
ANALYSIS3_DIR="${OUT_ROOT}/pca_analysis"

mkdir -p "${RUNS_DIR}" "${LOGS_DIR}" "${ANALYSIS1_DIR}" "${ANALYSIS2_DIR}" "${ANALYSIS3_DIR}"

echo "============================================"
echo "Alpha sweep v2 (${RUN_TAG})"
echo "  phase1 labels:   ${P1_LABELS[*]}"
echo "  alphas:          ${ALPHAS[*]}"
echo "  valences:        ${VALENCES[*]}"
echo "  phase2 seeds:    ${P2_SEEDS[*]}"
echo "  total train:     ${TOTAL_RUNS}"
echo "  parallel jobs:   ${MAX_JOBS}"
echo "  episodes:        ${EPISODES}"
echo "  max steps:       ${MAX_STEPS}"
echo "  output root:     ${OUT_ROOT}"
if [ -n "${REVERSAL_ROOT}" ]; then
  echo "  reversal root:   ${REVERSAL_ROOT}"
else
  echo "  reversal root:   (none)"
fi
echo "============================================"
echo ""

cat > "${OUT_ROOT}/experiment_config.json" << EOF
{
  "mode": "${RUN_TAG}",
  "smoke": ${SMOKE},
  "phase1_ckpts": [$(printf '"%s",' "${P1_CKPTS[@]}" | sed 's/,$//')],
  "phase1_labels": [$(printf '"%s",' "${P1_LABELS[@]}" | sed 's/,$//')],
  "alphas": [$(printf '%s,' "${ALPHAS[@]}" | sed 's/,$//')],
  "valences": ["SSSS", "MMMM"],
  "phase2_seeds": [$(printf '%s,' "${P2_SEEDS[@]}" | sed 's/,$//')],
  "episodes": ${EPISODES},
  "max_steps": ${MAX_STEPS},
  "schedule_pattern": "${SCHEDULE}",
  "c_decay": ${C_DECAY},
  "distortion_scale": ${DISTORTION_SCALE},
  "event_delay_steps": ${EVENT_DELAY},
  "min_event_gap": ${MIN_EVENT_GAP},
  "max_parallel_jobs": ${MAX_JOBS},
  "total_train_runs": ${TOTAL_RUNS},
  "analysis_1_dir": "${ANALYSIS1_DIR}",
  "analysis_2_dir": "${ANALYSIS2_DIR}",
  "analysis_3_dir": "${ANALYSIS3_DIR}",
  "reversal_root": "${REVERSAL_ROOT}"
}
EOF

# -------------------------------------------------------
# Import sanity check
# -------------------------------------------------------
python - <<'PY'
import importlib

required = [
    "cear_pilot.training.train_phase2",
]

optional = [
    "cear_pilot.analysis.compare_alpha_sweep",
    "cear_pilot.analysis.reanalyze_gscore",
    "cear_pilot.analysis.analyze_g_pca",
]

for m in required:
    importlib.import_module(m)

for m in optional:
    try:
        importlib.import_module(m)
        print(f"[OK] optional import: {m}")
    except Exception:
        print(f"[WARN] optional import unavailable: {m}")

print("[OK] import sanity check passed")
PY

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

# -------------------------------------------------------
# Training
# -------------------------------------------------------
echo "=== PHASE: Training ==="

RUN_COUNT=0

for P1_IDX in "${!P1_CKPTS[@]}"; do
  P1_CKPT="${P1_CKPTS[$P1_IDX]}"
  P1_LABEL="${P1_LABELS[$P1_IDX]}"

  for ALPHA in "${ALPHAS[@]}"; do
    for VAL in "${VALENCES[@]}"; do
      for P2_SEED in "${P2_SEEDS[@]}"; do
        RUN_COUNT=$((RUN_COUNT + 1))

        LABEL="a${ALPHA}_${VAL}_s${P2_SEED}_${P1_LABEL}"
        OUTDIR_BASE="${RUNS_DIR}/${LABEL}"
        LOG_FILE="${LOGS_DIR}/${LABEL}_train.log"

        wait_for_slot

        (
          echo "============================================"
          echo "[TRAIN ${RUN_COUNT}/${TOTAL_RUNS}] ${LABEL}"
          echo "  phase1_ckpt: ${P1_CKPT}"
          echo "  out base:    ${OUTDIR_BASE}"
          echo "  log:         ${LOG_FILE}"
          echo "============================================"

          python -m cear_pilot.training.train_phase2 \
            --phase1_ckpt "${P1_CKPT}" \
            --device "${DEVICE}" \
            --seed "${P2_SEED}" \
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
            --outdir "${OUTDIR_BASE}" \
            2>&1 | tee "${LOG_FILE}"
        ) &

        PIDS+=("$!")
        PID_LABELS+=("${LABEL}")
      done
    done
  done
done

wait_for_all

echo ""
echo "=== Training complete (${TOTAL_RUNS} runs) ==="

# -------------------------------------------------------
# Analysis (1): compare_alpha_sweep
# -------------------------------------------------------
echo ""
echo "=== PHASE: Analysis (1) compare_alpha_sweep ==="

"${COMPARE_CMD[@]}" \
  --sweep_root "${OUT_ROOT}" \
  --outdir "${ANALYSIS1_DIR}" \
  --source runs \
  2>&1 | tee "${LOGS_DIR}/analysis_compare_alpha_sweep.log"

# -------------------------------------------------------
# Analysis (2): reanalyze_gscore
# -------------------------------------------------------
if [ "${#GSCORE_CMD[@]}" -gt 0 ]; then
  echo ""
  echo "=== PHASE: Analysis (2) reanalyze_gscore ==="

  "${GSCORE_CMD[@]}" \
    --sweep_root "${OUT_ROOT}" \
    --outdir "${ANALYSIS2_DIR}" \
    --source runs \
    2>&1 | tee "${LOGS_DIR}/analysis_reanalyze_gscore.log"
else
  echo ""
  echo "[WARN] skipping Analysis (2): reanalyze_gscore entrypoint not found"
fi

# -------------------------------------------------------
# Analysis (3): analyze_g_pca
# -------------------------------------------------------
if [ "${#PCA_CMD[@]}" -gt 0 ]; then
  echo ""
  echo "=== PHASE: Analysis (3) analyze_g_pca ==="

  if [ -n "${REVERSAL_ROOT}" ]; then
    "${PCA_CMD[@]}" \
      --sweep_root "${OUT_ROOT}" \
      --reversal_root "${REVERSAL_ROOT}" \
      --outdir "${ANALYSIS3_DIR}" \
      --source runs \
      2>&1 | tee "${LOGS_DIR}/analysis_analyze_g_pca.log"
  else
    "${PCA_CMD[@]}" \
      --sweep_root "${OUT_ROOT}" \
      --outdir "${ANALYSIS3_DIR}" \
      --source runs \
      2>&1 | tee "${LOGS_DIR}/analysis_analyze_g_pca.log"
  fi
else
  echo ""
  echo "[WARN] skipping Analysis (3): analyze_g_pca entrypoint not found"
fi

echo ""
echo "============================================"
echo "Experiment complete"
echo "  mode:             ${RUN_TAG}"
echo "  total train runs: ${TOTAL_RUNS}"
echo "  runs:             ${RUNS_DIR}"
echo "  analysis (1):     ${ANALYSIS1_DIR}"
echo "  analysis (2):     ${ANALYSIS2_DIR}"
echo "  analysis (3):     ${ANALYSIS3_DIR}"
echo "  logs:             ${LOGS_DIR}"
echo "============================================"