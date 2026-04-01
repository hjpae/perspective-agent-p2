#!/usr/bin/env bash
# run_experiment_3_4.sh
#
# Re-run only:
#   (3) analyze_g_pca
#   (4) analyze_reversal
#
# Keeps the same defaults/settings as the original script:
#   - MAX_JOBS=2
#   - same reversal config
#   - same schedule/max_steps defaults
#   - same module/file fallback resolution
#
# Usage:
#   chmod +x run_experiment_3_4.sh
#   ./run_experiment_3_4.sh <OUT_ROOT> [device] [--skip-reversal]
#
# Examples:
#chmod +x run_experiment_3_4.sh
#./run_experiment_3_4.sh outputs/experiment_full_20260330_180220 cuda

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <OUT_ROOT> [device] [--skip-reversal]"
  exit 1
fi

OUT_ROOT="$1"
DEVICE="${2:-cuda}"

SKIP_REVERSAL=0
for arg in "$@"; do
  if [ "${arg}" = "--skip-reversal" ]; then
    SKIP_REVERSAL=1
  fi
done

if [ ! -d "${OUT_ROOT}" ]; then
  echo "[ERROR] OUT_ROOT not found: ${OUT_ROOT}"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

RUNS_DIR="${OUT_ROOT}/runs"
REVERSAL_DIR="${OUT_ROOT}/reversal"
LOGS_DIR="${OUT_ROOT}/logs"
ANALYSIS3_DIR="${OUT_ROOT}/analysis_pca"
ANALYSIS4_DIR="${OUT_ROOT}/analysis_reversal"

mkdir -p "${REVERSAL_DIR}" "${LOGS_DIR}" "${ANALYSIS3_DIR}" "${ANALYSIS4_DIR}"

# -------------------------------------------------------
# Defaults kept same as original script
# -------------------------------------------------------
SCHEDULE="1-1-1-1"
MAX_JOBS=2

# Reversal config (same defaults)
REVERSAL_WARMUP=5
REVERSAL_EPISODES=10
REVERSAL_SEED=0
MAX_STEPS=300

# Try to inherit prior config if present
CONFIG_JSON="${OUT_ROOT}/experiment_config.json"
if [ -f "${CONFIG_JSON}" ]; then
  if command -v python >/dev/null 2>&1; then
    eval "$(
      python - <<PY
import json
from pathlib import Path

cfg_path = Path(r"${CONFIG_JSON}")
cfg = json.loads(cfg_path.read_text())

def emit(name, value):
    if isinstance(value, str):
        print(f'{name}="{value}"')
    else:
        print(f'{name}={value}')

emit("MAX_JOBS", cfg.get("max_parallel_jobs", 2))
emit("REVERSAL_WARMUP", cfg.get("reversal_warmup_episodes", 5))
emit("REVERSAL_EPISODES", cfg.get("reversal_episodes", 10))
emit("REVERSAL_SEED", cfg.get("reversal_seed", 0))
emit("MAX_STEPS", cfg.get("max_steps", 300))
emit("SCHEDULE", cfg.get("schedule_pattern", "1-1-1-1"))
PY
    )"
  fi
fi

# -------------------------------------------------------
# Resolve analysis entrypoints
# -------------------------------------------------------
PCA_CMD=()
REVERSAL_CMD=()

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
    echo "[WARN] analyze_g_pca entrypoint not found"
    PCA_CMD=()
  fi
fi

if python - <<'PY' >/dev/null 2>&1
import importlib
importlib.import_module("cear_pilot.analysis.analyze_reversal")
PY
then
  REVERSAL_CMD=(python -m cear_pilot.analysis.analyze_reversal)
else
  if [ -f "${ROOT_DIR}/analyze_reversal.py" ]; then
    REVERSAL_CMD=(python "${ROOT_DIR}/analyze_reversal.py")
  else
    echo "[WARN] analyze_reversal entrypoint not found"
    REVERSAL_CMD=()
  fi
fi

# -------------------------------------------------------
# Import sanity check
# -------------------------------------------------------
python - <<'PY'
import importlib

required = [
    "cear_pilot.experiments.collect_reversal",
]

optional = [
    "cear_pilot.analysis.analyze_g_pca",
    "cear_pilot.analysis.analyze_reversal",
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

echo "============================================"
echo "Analysis 3 & 4 rerun"
echo "  OUT_ROOT:           ${OUT_ROOT}"
echo "  runs:               ${RUNS_DIR}"
echo "  reversal:           ${REVERSAL_DIR}"
echo "  analysis pca:       ${ANALYSIS3_DIR}"
echo "  analysis reversal:  ${ANALYSIS4_DIR}"
echo "  device:             ${DEVICE}"
echo "  parallel jobs:      ${MAX_JOBS}"
echo "  max steps:          ${MAX_STEPS}"
echo "  reversal warmup:    ${REVERSAL_WARMUP}"
echo "  reversal episodes:  ${REVERSAL_EPISODES}"
echo "  reversal seed:      ${REVERSAL_SEED}"
echo "  skip reversal:      ${SKIP_REVERSAL}"
echo "============================================"

if [ ! -d "${RUNS_DIR}" ]; then
  echo "[ERROR] runs directory not found: ${RUNS_DIR}"
  exit 1
fi

# ═══════════════════════════════════════════════════════
# STEP A: Reversal assay (optional / regenerated if needed)
# ═══════════════════════════════════════════════════════
if [ "${SKIP_REVERSAL}" -eq 0 ]; then
  echo ""
  echo "=== STEP A: Reversal assay ==="

  RUN_COUNT=0
  TOTAL_REV=0

  for RUN_DIR_CANDIDATE in "${RUNS_DIR}"/a*; do
    [ -d "${RUN_DIR_CANDIDATE}" ] || continue
    [ -f "${RUN_DIR_CANDIDATE}/ckpt_final.pt" ] || continue
    TOTAL_REV=$((TOTAL_REV + 1))
  done

  echo "Found ${TOTAL_REV} checkpoints for reversal"

  for RUN_DIR_CANDIDATE in "${RUNS_DIR}"/a*; do
    [ -d "${RUN_DIR_CANDIDATE}" ] || continue

    DIRNAME=$(basename "${RUN_DIR_CANDIDATE}")
    CKPT="${RUN_DIR_CANDIDATE}/ckpt_final.pt"
    [ -f "${CKPT}" ] || continue

    RUN_COUNT=$((RUN_COUNT + 1))

    if echo "${DIRNAME}" | grep -q "_SSSS_"; then
      NATIVE_VAL="SSSS"
    elif echo "${DIRNAME}" | grep -q "_MMMM_"; then
      NATIVE_VAL="MMMM"
    else
      echo "[${RUN_COUNT}/${TOTAL_REV}] SKIP ${DIRNAME} — unknown valence"
      continue
    fi

    REV_OUTDIR="${REVERSAL_DIR}/${DIRNAME}"
    LOG_FILE="${LOGS_DIR}/${DIRNAME}_reversal.log"

    # Skip if reversal result already seems present
    if [ -d "${REV_OUTDIR}" ] && [ -n "$(find "${REV_OUTDIR}" -type f 2>/dev/null)" ]; then
      echo "[${RUN_COUNT}/${TOTAL_REV}] SKIP ${DIRNAME} — existing reversal output found"
      continue
    fi

    wait_for_slot

    (
      echo "[REVERSAL ${RUN_COUNT}/${TOTAL_REV}] ${DIRNAME}: ${NATIVE_VAL} → warmup → switch"

      python -m cear_pilot.experiments.collect_reversal \
        --ckpt "${CKPT}" \
        --device "${DEVICE}" \
        --native_valence "${NATIVE_VAL}" \
        --warmup_episodes "${REVERSAL_WARMUP}" \
        --reversal_episodes "${REVERSAL_EPISODES}" \
        --seed "${REVERSAL_SEED}" \
        --max_steps "${MAX_STEPS}" \
        --schedule_pattern "${SCHEDULE}" \
        --outdir "${REV_OUTDIR}" \
        --greedy \
        2>&1 | tee "${LOG_FILE}"
    ) &

    PIDS+=("$!")
    PID_LABELS+=("rev:${DIRNAME}")
  done

  wait_for_all

  echo ""
  echo "=== Reversal complete ==="
else
  echo ""
  echo "=== STEP A skipped: reversal regeneration disabled ==="
fi

# ═══════════════════════════════════════════════════════
# STEP B: Analysis (3) and (4)
# ═══════════════════════════════════════════════════════
echo ""
echo "=== STEP B: Analysis ==="

# ── Analysis (3): analyze_g_pca ──
if [ "${#PCA_CMD[@]}" -gt 0 ]; then
  echo ""
  echo "--- Analysis (3): analyze_g_pca ---"

  "${PCA_CMD[@]}" \
    --sweep_root "${OUT_ROOT}" \
    --outdir "${ANALYSIS3_DIR}" \
    --source runs \
    2>&1 | tee "${LOGS_DIR}/analysis_3_pca.log"
else
  echo ""
  echo "[WARN] skipping Analysis (3): analyze_g_pca not found"
fi

# ── Analysis (4): analyze_reversal ──
if [ "${#REVERSAL_CMD[@]}" -gt 0 ]; then
  echo ""
  echo "--- Analysis (4): analyze_reversal ---"

  "${REVERSAL_CMD[@]}" \
    --formation_root "${OUT_ROOT}" \
    --reversal_root "${REVERSAL_DIR}" \
    --outdir "${ANALYSIS4_DIR}" \
    2>&1 | tee "${LOGS_DIR}/analysis_4_reversal.log"
else
  echo ""
  echo "[WARN] skipping Analysis (4): analyze_reversal not found"
fi

echo ""
echo "============================================"
echo "ANALYSIS 3 & 4 COMPLETE"
echo "  OUT_ROOT:           ${OUT_ROOT}"
echo "  reversal:           ${REVERSAL_DIR}"
echo "  analysis pca:       ${ANALYSIS3_DIR}"
echo "  analysis reversal:  ${ANALYSIS4_DIR}"
echo "  logs:               ${LOGS_DIR}"
echo "============================================"