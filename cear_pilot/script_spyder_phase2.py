# -*- coding: utf-8 -*-
"""
Spyder/local launcher for Phase 2 analysis.

Assumes training/collection were done elsewhere (e.g. Vast.ai), and that
run directories have been synced locally under outputs/runs/.
"""

from pathlib import Path
import os
import sys
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def run_module(module: str, args: list[str]):
    cmd = [sys.executable, "-m", module] + args
    print("\nRunning:", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ------------------------------------------------------------------
# EDIT THIS
# ------------------------------------------------------------------
RUN_DIRS = [
    # Example:
    # RUNS_DIR / "probe_seed0",
    # RUNS_DIR / "probe_seed1",
]

FRAGILITY_THRESHOLD = 0.50
ENCOUNTER_HORIZON = 5
ENCOUNTER_PRE = 5
ENCOUNTER_POST = 15
RECOVERY_PRE = 20
RECOVERY_POST = 30


if __name__ == "__main__":
    if len(RUN_DIRS) == 0:
        raise ValueError("Populate RUN_DIRS with one or more collected probe directories.")

    for run_dir in RUN_DIRS:
        run_dir = Path(run_dir)
        assert run_dir.exists(), f"Missing run dir: {run_dir}"

        run_module(
            "cear_pilot.analysis.phase2_analysis",
            [
                "--run_dir", str(run_dir),
                "--fragility_threshold", str(FRAGILITY_THRESHOLD),
                "--encounter_horizon", str(ENCOUNTER_HORIZON),
                "--encounter_pre", str(ENCOUNTER_PRE),
                "--encounter_post", str(ENCOUNTER_POST),
                "--recovery_pre_window", str(RECOVERY_PRE),
                "--recovery_post_window", str(RECOVERY_POST),
            ],
        )

    print("\n[OK] Phase 2 analysis complete.")
