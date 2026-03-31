#!/usr/bin/env bash
# run_experiment_full.sh
#
# Full experiment pipeline:
#   Stage 1: Training with carry_g=True
#            4 alpha × 2 valence × 6 p2 seeds × N p1 seeds
#   Stage 2: Reversal assay (warmup in native → switch to opposite)
#   Stage 3: Analysis (formation + reversal + PCA)
#
# Modes:
#   Full:
#     chmod +x run_experiment_full.sh
#     ./run_experiment_full.sh ALL cuda
#
#   Smoke:
#     ./run_experiment_full.sh ALL cuda --smoke
#
#   Single Phase 1 seed:
#     ./run_experiment_full.sh outputs/runs/p1_s0/ckpt_final.pt cuda
#
# Expects (when using ALL):
#   outputs/runs/p1_s0/ckpt_final.pt
#   outputs/runs/p1_s1/ckpt_final.pt
#   outputs/runs/p1_s2/ckpt_final.pt
#   outputs/runs/p1_s3/ckpt_final.pt
#   outputs/runs/p1_s4/ckpt_final.pt
#
# Analyses:
#   (1) compare_alpha_sweep   — formation sweep figures
#   (2) reanalyze_gscore      — g-score event-triggered + per-event separation
#   (3) analyze_g_pca         — PCA scatter/trajectories/basin separation
#   (4) analyze_reversal      — reversal g-score, residue, PCA migration, PE

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

# Reversal config
REVERSAL_WARMUP=5
REVERSAL_EPISODES=10
REVERSAL_SEED=0

if [ "${SMOKE}" -eq 1 ]; then
  EPISODES=4
  MAX_STEPS=10
  REVERSAL_WARMUP=2
  REVERSAL_EPISODES=3
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
REVERSAL_CMD=()

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
    echo "[WARN] reanalyze_gscore entrypoint not found"
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

TOTAL_TRAIN=$(( ${#ALPHAS[@]} * ${#VALENCES[@]} * ${#P2_SEEDS[@]} * ${#P1_CKPTS[@]} ))

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="outputs/experiment_${RUN_TAG}_${STAMP}"
RUNS_DIR="${OUT_ROOT}/runs"
REVERSAL_DIR="${OUT_ROOT}/reversal"
LOGS_DIR="${OUT_ROOT}/logs"

ANALYSIS1_DIR="${OUT_ROOT}/analysis_formation"
ANALYSIS2_DIR="${OUT_ROOT}/analysis_gscore"
ANALYSIS3_DIR="${OUT_ROOT}/analysis_pca"
ANALYSIS4_DIR="${OUT_ROOT}/analysis_reversal"

mkdir -p "${RUNS_DIR}" "${REVERSAL_DIR}" "${LOGS_DIR}" \
         "${ANALYSIS1_DIR}" "${ANALYSIS2_DIR}" "${ANALYSIS3_DIR}" "${ANALYSIS4_DIR}"

echo "============================================"
echo "Full experiment (${RUN_TAG})"
echo "  phase1 labels:     ${P1_LABELS[*]}"
echo "  alphas:            ${ALPHAS[*]}"
echo "  valences:          ${VALENCES[*]}"
echo "  phase2 seeds:      ${P2_SEEDS[*]}"
echo "  total train:       ${TOTAL_TRAIN}"
echo "  carry_g:           TRUE"
echo "  parallel jobs:     ${MAX_JOBS}"
echo "  episodes:          ${EPISODES}"
echo "  max steps:         ${MAX_STEPS}"
echo "  reversal warmup:   ${REVERSAL_WARMUP}"
echo "  reversal episodes: ${REVERSAL_EPISODES}"
echo "  output root:       ${OUT_ROOT}"
echo "============================================"
echo ""

cat > "${OUT_ROOT}/experiment_config.json" << EOF
{
  "mode": "${RUN_TAG}",
  "smoke": ${SMOKE},
  "carry_g_during_training": true,
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
  "total_train_runs": ${TOTAL_TRAIN},
  "reversal_warmup_episodes": ${REVERSAL_WARMUP},
  "reversal_episodes": ${REVERSAL_EPISODES},
  "reversal_seed": ${REVERSAL_SEED},
  "analysis_1_dir": "${ANALYSIS1_DIR}",
  "analysis_2_dir": "${ANALYSIS2_DIR}",
  "analysis_3_dir": "${ANALYSIS3_DIR}",
  "analysis_4_dir": "${ANALYSIS4_DIR}"
}
EOF

# -------------------------------------------------------
# Import sanity check
# -------------------------------------------------------
python - <<'PY'
import importlib

required = [
    "cear_pilot.training.train_phase2",
    "cear_pilot.experiments.collect_reversal",
]

optional = [
    "cear_pilot.analysis.compare_alpha_sweep",
    "cear_pilot.analysis.reanalyze_gscore",
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

# ═══════════════════════════════════════════════════════
# STAGE 1/3: Training with carry_g=True
# ═══════════════════════════════════════════════════════
echo "=== STAGE 1/3: Training (carry_g=True) ==="

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
          echo "[TRAIN ${RUN_COUNT}/${TOTAL_TRAIN}] ${LABEL}"
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
            --carry_g_between_episodes \
            --freeze_encoder \
            --freeze_state \
            --freeze_policy \
            --freeze_decoder \
            --outdir "${OUTDIR_BASE}" \
            2>&1 | tee "${LOG_FILE}"
        ) &

        PIDS+=("$!")
        PID_LABELS+=("train:${LABEL}")
      done
    done
  done
done

wait_for_all

echo ""
echo "=== Training complete (${TOTAL_TRAIN} runs) ==="

# ═══════════════════════════════════════════════════════
# STAGE 2/3: Reversal assay
# ═══════════════════════════════════════════════════════
echo ""
echo "=== STAGE 2/3: Reversal assay ==="

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

  # Parse native valence
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

# ═══════════════════════════════════════════════════════
# STAGE 3/3: Analysis
# ═══════════════════════════════════════════════════════
echo ""
echo "=== STAGE 3/3: Analysis ==="

# ── Analysis (1): compare_alpha_sweep ──
echo ""
echo "--- Analysis (1): compare_alpha_sweep ---"

"${COMPARE_CMD[@]}" \
  --sweep_root "${OUT_ROOT}" \
  --outdir "${ANALYSIS1_DIR}" \
  --source runs \
  2>&1 | tee "${LOGS_DIR}/analysis_1_compare.log"

# ── Analysis (2): reanalyze_gscore ──
if [ "${#GSCORE_CMD[@]}" -gt 0 ]; then
  echo ""
  echo "--- Analysis (2): reanalyze_gscore ---"

  "${GSCORE_CMD[@]}" \
    --sweep_root "${OUT_ROOT}" \
    --outdir "${ANALYSIS2_DIR}" \
    --source runs \
    2>&1 | tee "${LOGS_DIR}/analysis_2_gscore.log"
else
  echo ""
  echo "[WARN] skipping Analysis (2): reanalyze_gscore not found"
fi

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
echo "EXPERIMENT COMPLETE"
echo "  mode:               ${RUN_TAG}"
echo "  carry_g:            TRUE"
echo "  total train runs:   ${TOTAL_TRAIN}"
echo "  total reversal:     ${TOTAL_REV}"
echo "  runs:               ${RUNS_DIR}"
echo "  reversal:           ${REVERSAL_DIR}"
echo "  analysis formation: ${ANALYSIS1_DIR}"
echo "  analysis gscore:    ${ANALYSIS2_DIR}"
echo "  analysis pca:       ${ANALYSIS3_DIR}"
echo "  analysis reversal:  ${ANALYSIS4_DIR}"
echo "  logs:               ${LOGS_DIR}"
echo "============================================"