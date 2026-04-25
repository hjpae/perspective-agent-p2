# run_stance.py

import sys
from pathlib import Path

RUNS_ROOT = "outputs/phase2_all_20260403_011254"
SCHEDULE = "mixed_0_4_0"
N_LATE_EPS = 10
SAVE_JSON = ""   # 예: "results.json" (안 쓰면 "")

# ----------------------------------------

# stance_diagnostic.py가 같은 폴더에 있다고 가정
SCRIPT_NAME = "stance_diagnostic_v2.py"

# sys.argv 세팅 (CLI 흉내)
sys.argv = [
    SCRIPT_NAME,
    "--runs_root", RUNS_ROOT,
    "--schedule", SCHEDULE,
    "--n_late_eps", str(N_LATE_EPS),
]

if SAVE_JSON:
    sys.argv += ["--save_json", SAVE_JSON]

# 실제 실행
import stance_diagnostic
stance_diagnostic.main()