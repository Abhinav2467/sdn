#!/usr/bin/env python3
from __future__ import annotations


import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sdn_exam_demo.logging_config import configure_logging
from sdn_exam_demo.results import FIELDNAMES, failure_reason_from_logs, newest_failed_log

LOGGER = logging.getLogger("sdn_exam_demo.plot_results")


def plot_aggregate_results(data: pd.DataFrame, output_dir: Path) -> None:
    LOGGER.info("Plotting standard aggregate results...")
    for metric in sorted(data["metric"].unique()):
        subset = data[data["metric"] == metric]
        pivot = subset.pivot_table(index="scenario", columns="subject", values="value", aggfunc="mean")
        ax = pivot.plot(kind="bar", rot=0, title=f"{metric.upper()} by Scenario")
        ax.set_ylabel(metric)
        ax.figure.tight_layout()
        target = output_dir / f"{metric}.png"
        ax.figure.savefig(target)
        plt.close(ax.figure)
        LOGGER.info("Wrote aggregate plot to %s", target)


def plot_timeseries_results(timeseries_files: list[Path], output_dir: Path) -> None:
    LOGGER.info("Plotting time-series transient performance results...")
    frames = []
    for csv_path in timeseries_files:
        try:
            df = pd.read_csv(csv_path)
            # Add an elapsed interval index (each row is 5 seconds)
            df["elapsed_interval"] = range(len(df))
            df["elapsed_seconds"] = df["elapsed_interval"] * 5
            frames.append(df)
        except Exception as exc:
            LOGGER.warning("Could not read time-series CSV %s: %s", csv_path, exc)
            
    if not frames:
        return
        
    data = pd.concat(frames, ignore_index=True)
    
    # Define plotting metrics
    metrics_to_plot = {
        "exam_rtt_ms": "Exam RTT (ms)",
        "measured_throughput_mbps": "Estimated Network Throughput (Mbps)",
        "packet_loss_percent": "Exam Packet Loss (%)",
        "current_admitted_background_flow_limit": "Background Flow Admission Limit"
    }
    
    # We group by scenario and policy_mode to plot comparative timelines
    data["strategy"] = data["scenario"].astype(str) + " (" + data["policy_mode"].astype(str) + ")"
    
    for col, label in metrics_to_plot.items():
        if col not in data.columns:
            continue
            
        plt.figure(figsize=(10, 5))
        plt.title(f"Transient {label} Over Simulation Duration")
        
        # Plot mean performance with confidence shadow if multiple runs exist
        for strategy in data["strategy"].unique():
            subset = data[data["strategy"] == strategy]
            # Group by elapsed time and compute mean & std
            grouped = subset.groupby("elapsed_seconds")[col].agg(["mean", "std", "count"])
            
            plt.plot(grouped.index, grouped["mean"], label=strategy, marker="o", linewidth=2)
            
            # Confidence interval shadow
            if grouped["count"].max() > 1:
                # 95% Confidence Interval error margin
                ci = 1.96 * (grouped["std"] / np.sqrt(grouped["count"]))
                plt.fill_between(grouped.index, grouped["mean"] - ci, grouped["mean"] + ci, alpha=0.15)
                
        plt.xlabel("Elapsed Time (Seconds)")
        plt.ylabel(label)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        
        target = output_dir / f"timeseries_{col}.png"
        plt.savefig(target)
        plt.close()
        LOGGER.info("Wrote transient performance plot to %s", target)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot SDN exam traffic experiment results.")
    parser.add_argument("--results-dir", default=str(ROOT / "results" / "latest"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)

    results_dir = Path(args.results_dir)
    output_dir = results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import numpy as np

    # Check for timeseries CSV files first
    timeseries_files = sorted(results_dir.glob("*_timeseries.csv"))
    if timeseries_files:
        plot_timeseries_results(timeseries_files, output_dir)
        
    # Check for standard aggregate CSV files
    csv_files = sorted(path for path in results_dir.glob("*.csv") 
                       if not path.name.endswith("_timeseries.csv") and path.name != "failure_summary.csv")
                       
    if csv_files:
        frames = []
        for csv_path in csv_files:
            df = pd.read_csv(csv_path)
            # Find corresponding log file to extract the policy mode
            log_path = csv_path.with_suffix(".log")
            policy_mode = "static"
            if log_path.exists():
                try:
                    for line in log_path.read_text(encoding="utf-8").splitlines():
                        if line.startswith("policy_mode="):
                            policy_mode = line.split("=")[1].strip()
                            break
                except Exception as exc:
                    LOGGER.warning("Could not read policy_mode from %s: %s", log_path, exc)
            df["scenario"] = df["scenario"].astype(str) + " (" + policy_mode + ")"
            frames.append(df)
            
        data = pd.concat(frames, ignore_index=True)
        missing = sorted(set(FIELDNAMES) - set(data.columns))
        if missing:
            raise SystemExit(f"Result CSV files are missing required column(s): {', '.join(missing)}")
        plot_aggregate_results(data, output_dir)
        
    if not timeseries_files and not csv_files:
        failure_reason = failure_reason_from_logs(results_dir)
        if failure_reason:
            raise SystemExit(f"No result CSV files found in {results_dir}. Latest run failed: {failure_reason}")
        failed_log = newest_failed_log(results_dir)
        if failed_log:
            failed_reason = failure_reason_from_logs(failed_log.parent) or str(failed_log)
            raise SystemExit(f"No result CSV files found in {results_dir}. Newest failed attempt: {failed_reason}")
        raise SystemExit(f"No result CSV files found in {results_dir}; no failed-run diagnostics were found either.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

