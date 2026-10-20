#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import r2_score, f1_score, mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sdn_exam_demo.logging_config import configure_logging

LOGGER = logging.getLogger("sdn_exam_demo.compare_models")

def main() -> int:
    parser = argparse.ArgumentParser(description="Compare ML models for SDN admission control.")
    parser.add_argument("--dataset", default=str(ROOT / "results" / "dataset.csv"))
    parser.add_argument("--results-dir", default=str(ROOT / "results" / "model_comparison"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)

    dataset_path = Path(args.dataset)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("=== Machine Learning Model Comparison ===")

    if not dataset_path.exists():
        LOGGER.error("Dataset not found at %s!", dataset_path)
        return 1

    df = pd.read_csv(dataset_path)

    # Data Engineering (Identical to train_ml_model.py)
    df["exam_window_active"] = df["exam_window_state"].apply(lambda s: 1.0 if s == "active" else 0.0)
    df["exam_window_prestage"] = df["exam_window_state"].apply(lambda s: 1.0 if s == "prestage" else 0.0)

    delta_exam = 0.25 * df["active_exam_clients"]
    delta_bg = 0.15 * df["active_background_clients"]
    delta_rtt = 0.6 * np.minimum(df["exam_rtt_ms"] / 0.15, 2.0)
    delta_loss = 0.8 * np.minimum(df["packet_loss_percent"] / 2.0, 2.0)
    delta_queue = 0.8 * (df["queue_utilization_percent"] / 100.0)

    delta_time = np.zeros(len(df))
    active_mask = df["exam_window_active"] == 1.0
    prestage_mask = df["exam_window_prestage"] == 1.0
    delta_time[active_mask] = 0.8
    delta_time[prestage_mask] = 0.4 * (1.0 - df.loc[prestage_mask, "time_to_exam_seconds"] / 1200.0)

    raw_limit = 4.0 - (delta_exam + delta_bg + delta_rtt + delta_loss + delta_queue + delta_time)
    y_clean = np.clip(raw_limit, 1.0, 4.0)
    y_clean = y_clean - 0.01 * df["measured_throughput_mbps"]
    y_clean = np.clip(y_clean, 1.0, 4.0)

    np.random.seed(42)
    y_noisy = y_clean + np.random.normal(0, 0.20, size=len(y_clean))
    y = np.clip(y_noisy, 1.0, 4.0)
    y = pd.Series(y, index=df.index, name="target_limit")

    df["future_label"] = np.where(y_clean < 2.5, 0, 1)
    labels = df["future_label"].copy()

    feature_cols = [
        "active_exam_clients", "active_background_clients", "measured_throughput_mbps",
        "exam_rtt_ms", "packet_loss_percent", "iperf_flow_count",
        "queue_utilization_percent", "exam_window_active", "exam_window_prestage",
        "time_to_exam_seconds"
    ]
    continuous_features = [
        "exam_rtt_ms", "measured_throughput_mbps", "packet_loss_percent", "time_to_exam_seconds"
    ]

    X = df[feature_cols].copy()

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), continuous_features)],
        remainder="passthrough"
    )

    # Models to compare
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42),
        "Gradient Boosting": HistGradientBoostingRegressor(max_iter=100, max_depth=5, min_samples_leaf=10, random_state=42),
        "SVM (RBF)": SVR(kernel='rbf', C=1.0, epsilon=0.1),
        "Neural Network (MLP)": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    }

    X_train, X_holdout, y_train, y_holdout, label_train, label_holdout = train_test_split(
        X, y, labels, test_size=0.2, stratify=labels, random_state=42
    )

    np.random.seed(99)
    y_holdout_noisy = y_holdout + np.random.normal(0, 0.20, size=len(y_holdout))
    y_holdout_noisy = np.clip(y_holdout_noisy, 1.0, 4.0)
    label_holdout_noisy = np.where(y_holdout_noisy < 2.5, 0, 1)

    results = {}

    for name, regressor in models.items():
        LOGGER.info("Training %s...", name)
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("scaler", StandardScaler()), ("regressor", regressor)])
        # SVM and MLP strongly benefit from scaling all features, so we add a full scaler after the column transformer just in case
        
        pipeline.fit(X_train, y_train)
        holdout_preds = pipeline.predict(X_holdout)

        r2 = float(r2_score(y_holdout_noisy, holdout_preds))
        mae = float(mean_absolute_error(y_holdout_noisy, holdout_preds))
        rmse = float(np.sqrt(mean_squared_error(y_holdout_noisy, holdout_preds)))
        
        class_preds = np.where(holdout_preds < 2.5, 0, 1)
        f1 = float(f1_score(label_holdout_noisy, class_preds, pos_label=0))

        results[name] = {
            "R2": r2,
            "F1": f1,
            "MAE": mae,
            "RMSE": rmse
        }
        LOGGER.info("%s - R2: %.4f | F1: %.4f", name, r2, f1)

    # Export metrics
    metrics_path = results_dir / "model_comparison_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    LOGGER.info("Comparison metrics saved to %s", metrics_path)

    # Generate Plots
    names = list(results.keys())
    r2_scores = [results[n]["R2"] for n in names]
    f1_scores = [results[n]["F1"] for n in names]

    # Plot R2
    plt.figure(figsize=(10, 6))
    plt.bar(names, r2_scores, color=['royalblue', 'darkorange', 'forestgreen', 'firebrick'])
    plt.title("Model Comparison: Regression R2 Score")
    plt.ylabel("R2 Score (Higher is better)")
    plt.ylim(0, 1.0)
    for i, v in enumerate(r2_scores):
        plt.text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(plots_dir / "model_comparison_r2.png")
    plt.close()

    # Plot F1
    plt.figure(figsize=(10, 6))
    plt.bar(names, f1_scores, color=['royalblue', 'darkorange', 'forestgreen', 'firebrick'])
    plt.title("Model Comparison: Congestion Classification F1 Score")
    plt.ylabel("F1 Score (Higher is better)")
    plt.ylim(0, 1.0)
    for i, v in enumerate(f1_scores):
        plt.text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(plots_dir / "model_comparison_f1.png")
    plt.close()

    LOGGER.info("Comparison plots saved to %s", plots_dir)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
