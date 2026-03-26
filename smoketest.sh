#!/usr/bin/env bash
set -euo pipefail

# Smoke test for CEAR Phase 1 -> Phase 2 pipeline
# run this from the project root

DEVICE="${DEVICE:-cpu}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ROOT="${RUN_ROOT:-outputs/runs}"

mkdir -p "${RUN_ROOT}"

latest_ckpt() {
  ls -t "${RUN_ROOT}"/*/ckpt_final.pt 2>/dev/null | head -n 1
}

echo "[1/4] Phase 1 smoke training"
P1_BEFORE="$(latest_ckpt || true)"

${PYTHON_BIN} -m cear_pilot.training.train_phase1 \
  --device "${DEVICE}" \
  --seed 11 \
  --steps 400 \
  --warmup_steps 100 \
  --print_every 100 \
  --save_ckpt_every 400 \
  --log_traj

P1_CKPT="$(latest_ckpt)"
if [[ -z "${P1_CKPT}" ]]; then
  echo "ERROR: Phase 1 checkpoint not found."
  exit 1
fi
if [[ "${P1_CKPT}" == "${P1_BEFORE}" ]]; then
  echo "WARN: latest phase1 checkpoint path did not change; using ${P1_CKPT}"
fi
echo "Phase 1 checkpoint: ${P1_CKPT}"

echo "[2/4] Phase 1 smoke collect"
${PYTHON_BIN} -m cear_pilot.experiments.run_collect_phase1 \
  --ckpt "${P1_CKPT}" \
  --device "${DEVICE}" \
  --episodes 5 \
  --seed 111 \
  --greedy

echo "[3/4] Phase 2 smoke training"
P2_BEFORE="$(latest_ckpt || true)"

${PYTHON_BIN} -m cear_pilot.training.train_phase2 \
  --phase1_ckpt "${P1_CKPT}" \
  --device "${DEVICE}" \
  --seed 22 \
  --steps 600 \
  --print_every 100 \
  --save_ckpt_every 600 \
  --log_traj \
  --formation_profile even_m4 \
  --update_mode adaptive \
  --alpha_min 0.03 \
  --alpha_max 0.30 \
  --w_smooth 1.0

P2_CKPT="$(latest_ckpt)"
if [[ -z "${P2_CKPT}" ]]; then
  echo "ERROR: Phase 2 checkpoint not found."
  exit 1
fi
if [[ "${P2_CKPT}" == "${P2_BEFORE}" ]]; then
  echo "WARN: latest phase2 checkpoint path did not change; using ${P2_CKPT}"
fi
echo "Phase 2 checkpoint: ${P2_CKPT}"

echo "[4/4] Phase 2 smoke collect"
${PYTHON_BIN} -m cear_pilot.experiments.run_collect_phase2 \
  --ckpt "${P2_CKPT}" \
  --device "${DEVICE}" \
  --episodes 5 \
  --seed 222 \
  --greedy

echo "Smoke test completed."