#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sdn_exam_demo.logging_config import configure_logging

LOGGER = logging.getLogger("sdn_exam_demo.generate_dataset")

BG_CLIENT_OPTIONS = [2, 4, 6, 8, 10]
EXAM_CLIENT_OPTIONS = [1, 2, 3]
LOAD_LEVEL_OPTIONS = {
    "light": 1,
    "medium": 3,
    "heavy": 6,
    "burst": 10
}
TIMING_OPTIONS = ["prestage", "active", "post"]


def create_temp_config(
    base_config_path: Path,
    bg_clients: int,
    exam_clients: int,
    bg_flow_mbps: int,
    timing: str,
    temp_path: Path,
    is_quick: bool = False
) -> None:
    with base_config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    # 1. Update topology
    config["topology"]["background_clients"] = bg_clients
    config["topology"]["exam_clients"] = exam_clients
    config["topology"]["switch_bandwidth_mbps"] = 10
    
    # 2. Update traffic rate for background
    config["traffic"]["background_flow_bandwidth_mbps"] = bg_flow_mbps

    # Reduce duration to 15s in quick mode for rapid validation execution
    if is_quick:
        config["traffic"]["normal_duration_seconds"] = 15
        config["traffic"]["congested_duration_seconds"] = 15

    # 3. Update exam timings relative to now
    now = datetime.now(UTC)
    if timing == "active":
        # Current time is inside the exam window
        start = now - timedelta(minutes=30)
        end = now + timedelta(minutes=90)
    elif timing == "prestage":
        # Current time is within the prestage window (starts 20 mins before)
        start = now + timedelta(minutes=10)
        end = now + timedelta(minutes=70)
    else:  # post-exam
        # Exam is already over
        start = now - timedelta(minutes=180)
        end = now - timedelta(minutes=60)

    config["exam_schedule"] = [{
        "name": f"sweep-{timing}",
        "start": start.strftime("%Y-%m-%dT%H:%M:%S"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%S"),
        "prestage_minutes": 20
    }]

    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def run_experiment(config_path: Path, scenario: str, policy_mode: str, results_dir: Path) -> Path | None:
    cmd = [
        "sudo",
        ".venv/bin/python",
        "scripts/run_experiment.py",
        "--scenario", scenario,
        "--policy-mode", policy_mode,
        "--config", str(config_path),
        "--results-dir", str(results_dir),
        "--log-level", "ERROR"
    ]
    try:
        # Run process synchronously under WSL
        LOGGER.info("Executing: %s", " ".join(cmd))
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        # Parse promoted latest directory to locate the new timeseries CSV
        csv_files = list(results_dir.glob("*_timeseries.csv"))
        if csv_files:
            # Return the newest csv file
            return max(csv_files, key=lambda p: p.stat().st_mtime)
    except subprocess.CalledProcessError as exc:
        LOGGER.error("Experiment failed! code=%d err=%s", exc.returncode, exc.stderr.strip())
    except Exception as exc:
        LOGGER.error("Unexpected error running experiment: %s", exc)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Systematic SDN dataset generator sweep.")
    parser.add_argument("--quick", action="store_true", help="Run a small representative subset of parameters.")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds per configuration.")
    parser.add_argument("--base-config", default=str(ROOT / "config" / "demo_config.json"))
    parser.add_argument("--results-dir", default=str(ROOT / "results" / "latest"))
    parser.add_argument("--output", default=str(ROOT / "results" / "dataset.csv"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)

    base_config = Path(args.base_config)
    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    
    # 1. Establish Sweep Space
    if args.quick:
        # A small, highly balanced manual grid of 4 runs containing:
        # 1-2. Safe State: Light load (2 BG, 1 Exam, light, active) -> "No" Cases
        # 3-4. Congested State: Heavy load (10 BG, 3 Exam, heavy, active) -> "Yes" Cases
        grid = [
            (2, 1, "light", "active"),
            (10, 3, "heavy", "active")
        ]
        num_seeds = 2
    else:
        bg_opts = BG_CLIENT_OPTIONS
        exam_opts = EXAM_CLIENT_OPTIONS
        load_names = list(LOAD_LEVEL_OPTIONS.keys())
        timing_opts = TIMING_OPTIONS
        num_seeds = args.seeds

        # Track sweep grid
        grid = []
        for bg in bg_opts:
            for exam in exam_opts:
                for load_name in load_names:
                    for timing in timing_opts:
                        grid.append((bg, exam, load_name, timing))
                        
        # Shuffle grid to get balanced representation if exited early
        random.seed(42)
        random.shuffle(grid)

    LOGGER.info("Beginning parameter sweeps. QuickMode=%s Seeds=%d Output=%s",
                args.quick, num_seeds, output_path)

    temp_config_path = ROOT / "config" / "temp_sweep_config.json"
    master_records = []
    
    run_count = 0
    total_runs = len(grid) * num_seeds
    LOGGER.info("Total scheduled simulation runs: %d", total_runs)

    try:
        for bg, exam, load_name, timing in grid:
            bg_flow_mbps = LOAD_LEVEL_OPTIONS[load_name]
            for seed_idx in range(num_seeds):
                run_count += 1
                seed = 100 + seed_idx
                LOGGER.info("[%d/%d] Sweep Config: BG=%d Exam=%d Load=%s Timing=%s Seed=%d",
                            run_count, total_runs, bg, exam, load_name, timing, seed)

                # Update the configuration dynamically
                create_temp_config(base_config, bg, exam, bg_flow_mbps, timing, temp_config_path, is_quick=args.quick)

                # We alternate between baseline (no protection) and protected modes
                # to populate our dataset with both high congestion failure states and safe states.
                scenario = "protected" if (seed_idx % 2 == 0) else "baseline"
                policy_mode = "heuristic" if scenario == "protected" else "static"

                # Trigger experiment in Mininet under WSL
                timeseries_path = run_experiment(temp_config_path, scenario, policy_mode, results_dir)

                if timeseries_path and timeseries_path.exists():
                    LOGGER.info("Successfully gathered run timeseries: %s", timeseries_path.name)
                    # Read records and enrich with metadata
                    with timeseries_path.open("r", encoding="utf-8") as handle:
                        reader = csv.DictReader(handle)
                        for row in reader:
                            row["seed"] = str(seed)
                            row["bg_flow_mbps"] = str(bg_flow_mbps)
                            row["load_level"] = load_name
                            master_records.append(row)
                    
                    # Clean attempt CSV file
                    try:
                        timeseries_path.unlink()
                    except Exception:
                        pass
                else:
                    LOGGER.warning("No timeseries CSV produced for this run combination. Skipping.")

                # Small sleep to stabilize OVS state between runs
                time.sleep(1.0)
    finally:
        # Cleanup temporary config file
        if temp_config_path.exists():
            try:
                temp_config_path.unlink()
            except Exception:
                pass

    # 2. Write Master Consolidated Dataset CSV
    if master_records:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        headers = list(master_records[0].keys())
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            for record in master_records:
                writer.writerow(record)
        LOGGER.info("Consolidated master dataset written successfully: %d rows to %s",
                    len(master_records), output_path)
    else:
        LOGGER.error("Failed to generate any timeseries records! Confirm Mininet and controller are operational.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
