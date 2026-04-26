# run_stance.py
import sys
from pathlib import Path

RUNS_ROOT = "outputs/phase2_all_20260403_011254"
SCHEDULE = "mixed_0_4_0"
N_LATE_EPS = 10

SCRIPT_NAME = "stance_diagnostic_v2.py"

sys.argv = [
    SCRIPT_NAME,
    "--runs_root", RUNS_ROOT,
    "--schedule", SCHEDULE,
    "--n_late_eps", str(N_LATE_EPS),
]

import stance_diagnostic_v2
stance_diagnostic_v2.main()

#%% 
import sys
from pathlib import Path

RUNS_ROOT = "outputs/phase2_all_20260403_011254"
SCHEDULE = "mixed_0_6_0"
N_LATE_EPS = 10

SCRIPT_NAME = "stance_diagnostic_v2.py"

sys.argv = [
    SCRIPT_NAME,
    "--runs_root", RUNS_ROOT,
    "--schedule", SCHEDULE,
    "--n_late_eps", str(N_LATE_EPS),
]

import stance_diagnostic_v2
stance_diagnostic_v2.main()