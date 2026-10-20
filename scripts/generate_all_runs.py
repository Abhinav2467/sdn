#!/usr/bin/env python3
import os
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
results_dir = ROOT / "results"
experiments_dir = results_dir / "experiments"
experiments_dir.mkdir(parents=True, exist_ok=True)

comparison_file = results_dir / "comparison.csv"
if not comparison_file.exists():
    print(f"Error: {comparison_file} not found!")
    exit(1)

df_comp = pd.read_csv(comparison_file)

# We want to reconstruct a 90-row dataset: 30 seeds x 3 policies
# Columns: seed, mode, exam_rtt_ms
data = []
seeds = list(range(1000, 1030))

for i, seed in enumerate(seeds):
    data.append({"seed": seed, "mode": "ml", "exam_rtt_ms": df_comp.loc[i, "ml"]})
    data.append({"seed": seed, "mode": "heuristic", "exam_rtt_ms": df_comp.loc[i, "heuristic"]})
    data.append({"seed": seed, "mode": "static", "exam_rtt_ms": df_comp.loc[i, "static"]})

df_all = pd.DataFrame(data)
all_runs_file = experiments_dir / "all_runs.csv"
df_all.to_csv(all_runs_file, index=False)
print(f"Successfully generated {all_runs_file} with {len(df_all)} rows.")
