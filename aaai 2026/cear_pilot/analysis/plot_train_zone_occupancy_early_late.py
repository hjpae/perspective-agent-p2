# plot_train_zone_occupancy_early_late.py
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RUN_DIRS = [
    Path("outputs/runs/seed1"),
    Path("outputs/runs/seed2"),
    Path("outputs/runs/seed3"),
    Path("outputs/runs/seed4"),
    Path("outputs/runs/seed5"),
]

EARLY = (0, 20)         # inclusive
LATE  = (180, 200)      # inclusive
ZONES = [0, 1, 2]

def zone_occupancy(df: pd.DataFrame, ep_range):
    lo, hi = ep_range
    sub = df[(df["episode"] >= lo) & (df["episode"] <= hi)]
    counts = sub["zone_id"].value_counts(normalize=True)
    return np.array([counts.get(z, 0.0) for z in ZONES], dtype=float)

early_list, late_list = [], []

for rd in RUN_DIRS:
    df = pd.read_csv(rd / "train_traj.csv")
    early_list.append(zone_occupancy(df, EARLY))
    late_list.append(zone_occupancy(df, LATE))

early = np.stack(early_list)  # (n_seed, 3)
late  = np.stack(late_list)

early_mean, early_std = early.mean(0), early.std(0)
late_mean,  late_std  = late.mean(0),  late.std(0)

x = np.arange(len(ZONES))
w = 0.35

plt.figure(figsize=(6,4))
plt.bar(x - w/2, early_mean, w, yerr=early_std, capsize=4, label="Early (ep 0–20)")
plt.bar(x + w/2, late_mean,  w, yerr=late_std,  capsize=4, label="Late (ep 180–200)")
plt.xticks(x, [f"Z{z}" for z in ZONES])
plt.ylabel("Zone occupancy (fraction of steps)")
plt.ylim(0, 1.0)
plt.legend()
plt.tight_layout()
plt.show()
