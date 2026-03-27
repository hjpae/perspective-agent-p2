#!/usr/bin/env bash
set -euo pipefail

CKPT="${1:-outputs/runs/p1_s0/ckpt_final.pt}"
DEVICE="${DEVICE:-cuda}"
EPISODES="${EPISODES:-100}"
SEED="${SEED:-0}"

PATTERN="3-1"
VALENCE="SSSM"
COND_TAG="sss_m"

STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_OUT="outputs/runs/phase2_sss_m_alpha_sanity_${STAMP}"
FIG_OUT="outputs/figures_phase2_sss_m_alpha_sanity_${STAMP}"

mkdir -p "${BASE_OUT}"
mkdir -p "${FIG_OUT}"

ALPHAS=(0.10 0.20 0.30 0.40 0.50 0.60)

echo "[sss_m_alpha_sanity] ckpt=${CKPT}"
echo "[sss_m_alpha_sanity] device=${DEVICE}"
echo "[sss_m_alpha_sanity] episodes=${EPISODES}"
echo "[sss_m_alpha_sanity] run_out=${BASE_OUT}"
echo "[sss_m_alpha_sanity] fig_out=${FIG_OUT}"

for alpha in "${ALPHAS[@]}"; do
  RUN_OUT="${BASE_OUT}/a_${alpha}/${COND_TAG}"
  mkdir -p "${RUN_OUT}"

  echo
  echo "============================================================"
  echo "[RUN] alpha=${alpha} pattern=${PATTERN} valence=${VALENCE}"
  echo "============================================================"

  python -m cear_pilot.experiments.phase2_g_assay \
    --ckpt "${CKPT}" \
    --device "${DEVICE}" \
    --episodes "${EPISODES}" \
    --seed "${SEED}" \
    --alpha_fixed "${alpha}" \
    --schedule_pattern "${PATTERN}" \
    --valence_sequence "${VALENCE}" \
    --print_every_steps 1000 \
    --outdir "${RUN_OUT}"
done

echo
echo "============================================================"
echo "[PLOT] generating sanity plots"
echo "============================================================"

python - << EOF
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

import cear_pilot.analysis.phase2_assay_results as A

A.SAVE_FIGS = True
A.FIG_OUT = Path("${FIG_OUT}")
A.FIG_OUT.mkdir(parents=True, exist_ok=True)

sweep_dir = Path("${BASE_OUT}")
df = A.load_all_trajs(sweep_dir)
df = A.add_derived_columns(df)
summary = A.summarize_condition_alpha(df)

print(summary.sort_values(["condition_tag", "alpha"]))

# Main sweep summaries for this one condition
A.plot_alpha_sweep(summary, conditions=["sss_m"])
A.plot_alpha_sweep_supportive(summary, conditions=["sss_m"])
A.plot_heatmap(summary[summary["condition_tag"] == "sss_m"], value_col="dg_mis", title="sss_m misleading uptake")
A.plot_heatmap(summary[summary["condition_tag"] == "sss_m"], value_col="dg_sup", title="sss_m supportive uptake")

# One sanity trajectory plot per alpha, episode 0
for alpha in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]:
    A.inspect_one_run(df, alpha=alpha, condition="sss_m", episode=0)

# Also save a csv summary for quick inspection
summary.to_csv(A.FIG_OUT / "summary_sss_m_alpha_sweep.csv", index=False)
print(f"[saved] figures -> {A.FIG_OUT}")
EOF

echo
echo "[sss_m_alpha_sanity] done"
echo "[sss_m_alpha_sanity] runs: ${BASE_OUT}"
echo "[sss_m_alpha_sanity] figs: ${FIG_OUT}"