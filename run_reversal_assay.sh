#!/usr/bin/env bash
# run_reversal_assay.sh
# Take agents trained on SSSS, test them on MMMM (and vice versa).
# No further training, just collect trajectories with frozen weights.
#
# Usage:
#   ./run_reversal_assay.sh outputs/alpha_sweep_v2_YYYYMMDD [device]

# chmod +x run_reversal_assay.sh
# ./run_reversal_assay.sh cuda

set -euo pipefail

SWEEP_ROOT="${1:?Usage: $0 <sweep_root> [device]}"
DEVICE="${2:-cuda}"

EVAL_EPISODES=40
EVAL_SEED=0
MAX_STEPS=300
SCHEDULE="1-1-1-1"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_ROOT="${SWEEP_ROOT}/reversal_${TIMESTAMP}"
RUNS_OUT_DIR="${OUT_ROOT}/runs"
LOGS_DIR="${OUT_ROOT}/logs"

# Analysis output dirs
ANALYSIS_COMPARE_DIR="${OUT_ROOT}/analysis_compare_alpha_sweep"
ANALYSIS_GSCORE_DIR="${OUT_ROOT}/analysis_reanalyze_gscore"
ANALYSIS_PCA_DIR="${OUT_ROOT}/analysis_pca"

mkdir -p \
  "${OUT_ROOT}" \
  "${RUNS_OUT_DIR}" \
  "${LOGS_DIR}" \
  "${ANALYSIS_COMPARE_DIR}" \
  "${ANALYSIS_GSCORE_DIR}" \
  "${ANALYSIS_PCA_DIR}"

echo "============================================"
echo "Reversal assay"
echo "  Formation root: ${SWEEP_ROOT}"
echo "  Output:         ${OUT_ROOT}"
echo "  Device:         ${DEVICE}"
echo "============================================"

# -------------------------------------------------------
# Sanity checks
# -------------------------------------------------------
RUNS_DIR="${SWEEP_ROOT}/runs"
if [ ! -d "${RUNS_DIR}" ]; then
    echo "ERROR: ${RUNS_DIR} not found"
    exit 1
fi

python - <<'PY'
import importlib

mods = [
    "cear_pilot.experiments.collect_phase2",
    "cear_pilot.analysis.compare_alpha_sweep",
    "cear_pilot.analysis.reanalyze_gscore",
    "cear_pilot.analysis.analyze_g_pca",
]
for m in mods:
    importlib.import_module(m)
print("[OK] import sanity check passed")
PY

# -------------------------------------------------------
# Count formation runs
# -------------------------------------------------------
RUN_COUNT=0
TOTAL_RUNS=0

for RUN_DIR in "${RUNS_DIR}"/a*; do
    [ -d "${RUN_DIR}" ] || continue
    CKPT="${RUN_DIR}/ckpt_final.pt"
    [ -f "${CKPT}" ] || continue
    TOTAL_RUNS=$((TOTAL_RUNS + 1))
done

echo "Found ${TOTAL_RUNS} formation checkpoints"
echo ""

if [ "${TOTAL_RUNS}" -eq 0 ]; then
    echo "ERROR: no formation checkpoints found under ${RUNS_DIR}"
    exit 1
fi

# -------------------------------------------------------
# Reversal collection
# -------------------------------------------------------
echo "=== Collecting reversal trajectories ==="

for RUN_DIR in "${RUNS_DIR}"/a*; do
    [ -d "${RUN_DIR}" ] || continue

    DIRNAME=$(basename "${RUN_DIR}")
    CKPT="${RUN_DIR}/ckpt_final.pt"
    [ -f "${CKPT}" ] || continue

    RUN_COUNT=$((RUN_COUNT + 1))

    # Parse formation valence from directory name
    if echo "${DIRNAME}" | grep -q "_SSSS_"; then
        FORM_VAL="SSSS"
        TEST_VAL="MMMM"
    elif echo "${DIRNAME}" | grep -q "_MMMM_"; then
        FORM_VAL="MMMM"
        TEST_VAL="SSSS"
    else
        echo "[${RUN_COUNT}/${TOTAL_RUNS}] SKIP ${DIRNAME} — unknown valence"
        continue
    fi

    # Preserve alpha / p2 seed / p1 seed info, add reversal tag
    # e.g. a0.10_SSSS_s2_p1s4 -> a0.10_SSSS_s2_p1s4_rev_MMMM
    LABEL="${DIRNAME}_rev_${TEST_VAL}"
    COLLECT_DIR="${RUNS_OUT_DIR}/${LABEL}"
    LOG_FILE="${LOGS_DIR}/${LABEL}.log"

    echo "[${RUN_COUNT}/${TOTAL_RUNS}] ${DIRNAME}: ${FORM_VAL} -> ${TEST_VAL}"
    mkdir -p "${COLLECT_DIR}"

    python -m cear_pilot.experiments.collect_phase2 \
        --ckpt "${CKPT}" \
        --device "${DEVICE}" \
        --episodes "${EVAL_EPISODES}" \
        --seed "${EVAL_SEED}" \
        --greedy \
        --outdir "${COLLECT_DIR}" \
        --valence_sequence "${TEST_VAL}" \
        --schedule_pattern "${SCHEDULE}" \
        --max_steps "${MAX_STEPS}" \
        2>&1 | tee "${LOG_FILE}"
done

echo ""
echo "=== Reversal collection complete ==="

# -------------------------------------------------------
# Formation baseline collection
# -------------------------------------------------------
echo ""
echo "=== Collecting formation baselines ==="

RUN_COUNT=0
for RUN_DIR in "${RUNS_DIR}"/a*; do
    [ -d "${RUN_DIR}" ] || continue

    DIRNAME=$(basename "${RUN_DIR}")
    CKPT="${RUN_DIR}/ckpt_final.pt"
    [ -f "${CKPT}" ] || continue

    RUN_COUNT=$((RUN_COUNT + 1))

    if echo "${DIRNAME}" | grep -q "_SSSS_"; then
        FORM_VAL="SSSS"
    elif echo "${DIRNAME}" | grep -q "_MMMM_"; then
        FORM_VAL="MMMM"
    else
        echo "[${RUN_COUNT}/${TOTAL_RUNS}] SKIP ${DIRNAME} — unknown valence"
        continue
    fi

    LABEL="${DIRNAME}_baseline"
    COLLECT_DIR="${RUNS_OUT_DIR}/${LABEL}"
    LOG_FILE="${LOGS_DIR}/${LABEL}.log"

    echo "[${RUN_COUNT}/${TOTAL_RUNS}] ${DIRNAME}: baseline ${FORM_VAL}"
    mkdir -p "${COLLECT_DIR}"

    python -m cear_pilot.experiments.collect_phase2 \
        --ckpt "${CKPT}" \
        --device "${DEVICE}" \
        --episodes "${EVAL_EPISODES}" \
        --seed "${EVAL_SEED}" \
        --greedy \
        --outdir "${COLLECT_DIR}" \
        --valence_sequence "${FORM_VAL}" \
        --schedule_pattern "${SCHEDULE}" \
        --max_steps "${MAX_STEPS}" \
        2>&1 | tee "${LOG_FILE}"
done

echo ""
echo "=== Formation baseline collection complete ==="

# -------------------------------------------------------
# Analysis (1): compare_alpha_sweep
#   - formation sweep only
# -------------------------------------------------------
echo ""
echo "=== Running analysis (1): compare_alpha_sweep ==="

python cear_pilot/analysis/compare_alpha_sweep.py \
    --sweep_root "${SWEEP_ROOT}" \
    --outdir "${ANALYSIS_COMPARE_DIR}" \
    --source runs \
    2>&1 | tee "${LOGS_DIR}/analysis_compare_alpha_sweep.log"

# -------------------------------------------------------
# Analysis (2): reanalyze_gscore
#   - formation sweep only
# -------------------------------------------------------
echo ""
echo "=== Running analysis (2): reanalyze_gscore ==="

python cear_pilot/analysis/reanalyze_gscore.py \
    --sweep_root "${SWEEP_ROOT}" \
    --outdir "${ANALYSIS_GSCORE_DIR}" \
    --source runs \
    2>&1 | tee "${LOGS_DIR}/analysis_reanalyze_gscore.log"

# -------------------------------------------------------
# Analysis (3): analyze_g_pca
#   - formation + reversal together
# -------------------------------------------------------
echo ""
echo "=== Running analysis (3): analyze_g_pca ==="

python cear_pilot/analysis/analyze_g_pca.py \
    --sweep_root "${SWEEP_ROOT}" \
    --reversal_root "${OUT_ROOT}" \
    --outdir "${ANALYSIS_PCA_DIR}" \
    --source runs \
    2>&1 | tee "${LOGS_DIR}/analysis_analyze_g_pca.log"

echo ""
echo "============================================"
echo "Reversal assay complete!"
echo "  Reversal runs:    ${RUNS_OUT_DIR}/"
echo "  Analysis (1):     ${ANALYSIS_COMPARE_DIR}/"
echo "  Analysis (2):     ${ANALYSIS_GSCORE_DIR}/"
echo "  Analysis (3):     ${ANALYSIS_PCA_DIR}/"
echo "  Logs:             ${LOGS_DIR}/"
echo "============================================"