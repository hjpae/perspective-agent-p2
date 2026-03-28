# smoketest_phase2_landscape.sh
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
PROJECT_ROOT="$(pwd)"

if [[ ! -d "$PROJECT_ROOT/cear_pilot" && -d "$SCRIPT_DIR/cear_pilot" ]]; then
  PROJECT_ROOT="$SCRIPT_DIR"
fi

CKPT="${1:-}"
DEVICE="${2:-cpu}"
OUTROOT="${3:-$PROJECT_ROOT/outputs/runs}"

if [[ -z "$CKPT" ]]; then
  echo "Usage: bash smoketest_phase2_landscape.sh /path/to/phase1_ckpt.pt [device] [outroot]"
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
TRAIN_DIR="$OUTROOT/phase2_landscape_smoketest_$STAMP"
COLLECT_DIR="$TRAIN_DIR/collect"

python "$PROJECT_ROOT/cear_pilot/training/train_phase2.py" \
  --phase1_ckpt "$CKPT" \
  --device "$DEVICE" \
  --episodes 20 \
  --max_steps 180 \
  --schedule_pattern 1-1-1-1 \
  --valence_sequence SSSS \
  --update_mode fixed \
  --alpha_fixed 0.20 \
  --energy_mode prototype_wells \
  --dyn_eta 0.10 \
  --confine_lambda 0.01 \
  --n_prototypes 2 \
  --well_depth 0.60 \
  --well_width 1.25 \
  --freeze_encoder \
  --freeze_state \
  --freeze_policy \
  --save_traj \
  --greedy \
  --outdir "$TRAIN_DIR"

python "$PROJECT_ROOT/cear_pilot/experiments/collect_phase2.py" \
  --ckpt "$TRAIN_DIR/ckpt_final.pt" \
  --device "$DEVICE" \
  --episodes 8 \
  --max_steps 180 \
  --schedule_pattern 1-1-1-1 \
  --valence_sequence SSSS \
  --greedy \
  --do_g_at 60 \
  --do_g_mode shock \
  --do_g_scale 0.75 \
  --outdir "$COLLECT_DIR"

python "$PROJECT_ROOT/cear_pilot/analysis/plot_phase2_landscape.py" \
  --run_dir "$COLLECT_DIR"

echo "Project root: $PROJECT_ROOT"
echo "Train dir: $TRAIN_DIR"
echo "Collect dir: $COLLECT_DIR"
