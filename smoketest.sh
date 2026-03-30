#!/usr/bin/env bash
# smoketest.sh
#
# Minimal end-to-end smoke test for Phase 2 pipeline:
#   train_phase2 -> collect_phase2 -> compare_alpha_sweep
#
# Checks:
#   1) imports / module paths work
#   2) phase1 checkpoint can be loaded
#   3) train_phase2 writes timestamped run dir
#   4) ckpt_final.pt / episode_summary / traj are produced
#   5) collect_phase2 can load the checkpoint and save outputs
#   6) compare_alpha_sweep can discover both runs/ and collect/
#
# Usage:
#   chmod +x smoketest.sh
#   ./smoketest.sh /path/to/phase1_ckpt.pt [device]
#
# Example:
#   ./smoketest.sh outputs/runs/p1_s0/ckpt_final.pt cuda

set -euo pipefail

PHASE1_CKPT="${1:?Usage: $0 <phase1_ckpt> [device]}"
DEVICE="${2:-cuda}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
SMOKE_ROOT="outputs/smoketest_${STAMP}"
LOG_DIR="${SMOKE_ROOT}/logs"
RUNS_DIR="${SMOKE_ROOT}/runs"
COLLECT_DIR="${SMOKE_ROOT}/collect"
ANALYSIS_DIR="${SMOKE_ROOT}/analysis"

mkdir -p "${LOG_DIR}" "${RUNS_DIR}" "${COLLECT_DIR}" "${ANALYSIS_DIR}"

echo "============================================"
echo "Phase 2 smoke test"
echo "  repo root:    ${ROOT_DIR}"
echo "  phase1 ckpt:  ${PHASE1_CKPT}"
echo "  device:       ${DEVICE}"
echo "  out root:     ${SMOKE_ROOT}"
echo "============================================"

if [ ! -f "${PHASE1_CKPT}" ]; then
  echo "[ERROR] phase1 checkpoint not found: ${PHASE1_CKPT}"
  exit 1
fi

# -------------------------------------------------------
# 0) Quick module/import sanity check
# -------------------------------------------------------
echo ""
echo "=== [0/4] Import sanity check ==="
python - <<'PY'
import importlib
mods = [
    "cear_pilot.training.train_phase2",
    "cear_pilot.experiments.collect_phase2",
    "cear_pilot.analysis.compare_alpha_sweep",
    "cear_pilot.envs.nzone_phase2",
    "cear_pilot.models.agent",
]
for m in mods:
    importlib.import_module(m)
print("[OK] imports passed")
PY

# -------------------------------------------------------
# 1) Tiny training run
# -------------------------------------------------------
# NOTE:
# train_phase2 appends its own timestamp to --outdir base.
# We therefore pass a base prefix and then search for the actual dir.
echo ""
echo "=== [1/4] Tiny train run ==="

LABEL="a0.25_SSSS_s0"
TRAIN_BASE="${RUNS_DIR}/${LABEL}"

python -m cear_pilot.training.train_phase2 \
  --phase1_ckpt "${PHASE1_CKPT}" \
  --device "${DEVICE}" \
  --seed 0 \
  --episodes 3 \
  --max_steps 80 \
  --print_every 1 \
  --update_mode fixed \
  --alpha_fixed 0.25 \
  --schedule_pattern 1-1-1-1 \
  --valence_sequence SSSS \
  --c_decay 0.985 \
  --distortion_scale 0.55 \
  --event_delay_steps 10 \
  --min_event_gap 16 \
  --save_traj \
  --freeze_encoder \
  --freeze_state \
  --freeze_policy \
  --freeze_decoder \
  --outdir "${TRAIN_BASE}" \
  2>&1 | tee "${LOG_DIR}/train.log"

ACTUAL_TRAIN_DIR=$(
  ls -dt "${TRAIN_BASE}"* 2>/dev/null | head -1 || true
)

if [ -z "${ACTUAL_TRAIN_DIR}" ] || [ ! -d "${ACTUAL_TRAIN_DIR}" ]; then
  echo "[ERROR] could not resolve actual train dir from base: ${TRAIN_BASE}"
  exit 1
fi

echo "[OK] actual train dir: ${ACTUAL_TRAIN_DIR}"

CKPT="${ACTUAL_TRAIN_DIR}/ckpt_final.pt"
TRAIN_META="${ACTUAL_TRAIN_DIR}/meta.json"

if [ ! -f "${CKPT}" ]; then
  echo "[ERROR] missing checkpoint: ${CKPT}"
  exit 1
fi

if [ ! -f "${TRAIN_META}" ]; then
  echo "[ERROR] missing meta.json: ${TRAIN_META}"
  exit 1
fi

TRAIN_EP=""
TRAIN_TRAJ=""
for ext in parquet csv; do
  [ -f "${ACTUAL_TRAIN_DIR}/episode_summary.${ext}" ] && TRAIN_EP="${ACTUAL_TRAIN_DIR}/episode_summary.${ext}"
  [ -f "${ACTUAL_TRAIN_DIR}/traj.${ext}" ] && TRAIN_TRAJ="${ACTUAL_TRAIN_DIR}/traj.${ext}"
done

if [ -z "${TRAIN_EP}" ]; then
  echo "[ERROR] missing episode_summary.(parquet|csv)"
  exit 1
fi

if [ -z "${TRAIN_TRAJ}" ]; then
  echo "[ERROR] missing traj.(parquet|csv)"
  exit 1
fi

echo "[OK] training outputs found"
echo "     ckpt: ${CKPT}"
echo "     ep:   ${TRAIN_EP}"
echo "     traj: ${TRAIN_TRAJ}"

# -------------------------------------------------------
# 2) Tiny greedy collection
# -------------------------------------------------------
echo ""
echo "=== [2/4] Tiny greedy collection ==="

COLLECT_RUN_DIR="${COLLECT_DIR}/${LABEL}"

python -m cear_pilot.experiments.collect_phase2 \
  --ckpt "${CKPT}" \
  --device "${DEVICE}" \
  --episodes 2 \
  --seed 1000 \
  --greedy \
  --schedule_pattern 1-1-1-1 \
  --valence_sequence SSSS \
  --max_steps 80 \
  --outdir "${COLLECT_RUN_DIR}" \
  2>&1 | tee "${LOG_DIR}/collect.log"

if [ ! -d "${COLLECT_RUN_DIR}" ]; then
  echo "[ERROR] missing collect dir: ${COLLECT_RUN_DIR}"
  exit 1
fi

COLLECT_META="${COLLECT_RUN_DIR}/meta.json"
if [ ! -f "${COLLECT_META}" ]; then
  echo "[ERROR] missing collect meta.json: ${COLLECT_META}"
  exit 1
fi

COLLECT_EP=""
COLLECT_TRAJ=""
for ext in parquet csv; do
  [ -f "${COLLECT_RUN_DIR}/episode_summary.${ext}" ] && COLLECT_EP="${COLLECT_RUN_DIR}/episode_summary.${ext}"
  [ -f "${COLLECT_RUN_DIR}/traj.${ext}" ] && COLLECT_TRAJ="${COLLECT_RUN_DIR}/traj.${ext}"
done

if [ -z "${COLLECT_EP}" ]; then
  echo "[ERROR] missing collect episode_summary.(parquet|csv)"
  exit 1
fi

if [ -z "${COLLECT_TRAJ}" ]; then
  echo "[ERROR] missing collect traj.(parquet|csv)"
  exit 1
fi

echo "[OK] collection outputs found"
echo "     ep:   ${COLLECT_EP}"
echo "     traj: ${COLLECT_TRAJ}"

# -------------------------------------------------------
# 3) Analysis on the smoke root
# -------------------------------------------------------
echo ""
echo "=== [3/4] Analysis ==="

python -m cear_pilot.analysis.compare_alpha_sweep \
  --sweep_root "${SMOKE_ROOT}" \
  --outdir "${ANALYSIS_DIR}" \
  --source runs \
  2>&1 | tee "${LOG_DIR}/analysis.log"

SUMMARY_TABLE="${ANALYSIS_DIR}/summary_table.csv"
if [ ! -f "${SUMMARY_TABLE}" ]; then
  echo "[ERROR] missing analysis summary table: ${SUMMARY_TABLE}"
  exit 1
fi

echo "[OK] analysis summary produced: ${SUMMARY_TABLE}"

# -------------------------------------------------------
# 4) Final compact report
# -------------------------------------------------------
echo ""
echo "=== [4/4] Smoke test summary ==="
echo "train dir:    ${ACTUAL_TRAIN_DIR}"
echo "collect dir:  ${COLLECT_RUN_DIR}"
echo "analysis dir: ${ANALYSIS_DIR}"
echo ""
echo "Artifacts:"
echo "  - ${CKPT}"
echo "  - ${TRAIN_EP}"
echo "  - ${TRAIN_TRAJ}"
echo "  - ${COLLECT_EP}"
echo "  - ${COLLECT_TRAJ}"
echo "  - ${SUMMARY_TABLE}"
echo ""
echo "SMOKETEST PASSED"