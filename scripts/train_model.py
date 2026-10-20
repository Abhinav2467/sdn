#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sdn_exam_demo.logging_config import configure_logging

LOGGER = logging.getLogger("sdn_exam_demo.train_model")


def prepare_features_and_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    # 1. Feature Map Construction
    df["exam_window_active"] = df["exam_window_state"].apply(lambda s: 1.0 if s == "active" else 0.0)
    df["exam_window_prestage"] = df["exam_window_state"].apply(lambda s: 1.0 if s == "prestage" else 0.0)
    
    feature_cols = [
        "active_exam_clients",
        "active_background_clients",
        "measured_throughput_mbps",
        "exam_rtt_ms",
        "packet_loss_percent",
        "iperf_flow_count",
        "queue_utilization_percent",
        "exam_window_active",
        "exam_window_prestage",
        "time_to_exam_seconds"
    ]
    
    X = df[feature_cols].copy()
    
    # 2. Safety Target Limit (y)
    # The optimal limit is the current limit if the next interval stays under target RTT (future_label == 1),
    # otherwise it should be throttled to the safety floor (min_background_new_flows = 5).
    y = df["current_admitted_background_flow_limit"].copy()
    violation_mask = df["future_label"] == 0
    y[violation_mask] = 5.0
    
    # 3. Formulate unique simulation run groups to prevent temporal leakage
    groups = df["scenario"].astype(str) + "_" + df["seed"].astype(str) + "_" + df["load_level"].astype(str) + "_" + df["exam_window_state"].astype(str)
    
    return X, y, groups


def train_and_evaluate_model(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    model_type: str = "rf"
) -> tuple[Any, dict[str, float]]:
    if model_type == "rf":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = DecisionTreeRegressor(random_state=42)
        
    logo = LeaveOneGroupOut()
    mae_list = []
    rmse_list = []
    r2_list = []
    
    # We perform group-based cross validation (Leave-One-Group-Out) to verify generalization
    # but to save training speed we limit logo iterations if groups count is very large.
    max_folds = min(10, logo.get_n_splits(groups=groups))
    LOGGER.info("Performing %d-fold Group-based Cross Validation...", max_folds)
    
    unique_groups = groups.unique()
    np.random.seed(42)
    selected_groups = np.random.choice(unique_groups, size=max_folds, replace=False)
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    for group_val in selected_groups:
        test_idx = groups == group_val
        train_idx = ~test_idx
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        if len(X_train) == 0 or len(X_test) == 0:
            continue
            
        fold_model = RandomForestRegressor(n_estimators=50, random_state=42) if model_type == "rf" else DecisionTreeRegressor(random_state=42)
        fold_model.fit(X_train, y_train)
        preds = fold_model.predict(X_test)
        
        mae_list.append(mean_absolute_error(y_test, preds))
        rmse_list.append(np.sqrt(mean_squared_error(y_test, preds)))
        r2_list.append(r2_score(y_test, preds))
        
    # Fit final model on all data
    LOGGER.info("Fitting final %s model on entire dataset...", model_type.upper())
    model.fit(X, y)
    
    metrics = {
        "cv_mean_mae": float(np.mean(mae_list)) if mae_list else 0.0,
        "cv_mean_rmse": float(np.mean(rmse_list)) if rmse_list else 0.0,
        "cv_mean_r2": float(np.mean(r2_list)) if r2_list else 0.0
    }
    
    return model, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="SDN ML Model Trainer & Explainability.")
    parser.add_argument("--dataset", default=str(ROOT / "results" / "dataset.csv"))
    parser.add_argument("--model-dir", default=str(ROOT / "src" / "sdn_exam_demo" / "models"))
    parser.add_argument("--results-dir", default=str(ROOT / "results" / "latest"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)

    dataset_path = Path(args.dataset)
    model_dir = Path(args.model_dir)
    results_dir = Path(args.results_dir)
    
    if not dataset_path.exists():
        LOGGER.error("Dataset not found at %s. Run generate_dataset.py first!", dataset_path)
        return 1
        
    LOGGER.info("Loading dataset from %s", dataset_path)
    df = pd.read_csv(dataset_path)
    
    if len(df) < 5:
        LOGGER.error("Dataset has too few records (%d) to train a model.", len(df))
        return 1
        
    # 1. Feature and Target Preparation
    LOGGER.info("Extracting features, safety targets, and simulation groups...")
    X, y, groups = prepare_features_and_targets(df)
    
    # 2. Train Models
    # Interpretability Baseline (Decision Tree)
    LOGGER.info("Training baseline Decision Tree...")
    dt_model, dt_metrics = train_and_evaluate_model(X, y, groups, model_type="dt")
    LOGGER.info("Decision Tree metrics: %s", dt_metrics)

    # Primary Model (Random Forest)
    LOGGER.info("Training primary Random Forest...")
    rf_model, rf_metrics = train_and_evaluate_model(X, y, groups, model_type="rf")
    LOGGER.info("Random Forest metrics: %s", rf_metrics)
    
    # 3. Feature Importance Plot
    LOGGER.info("Generating XAI Feature Importance explanations...")
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)
    
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.title("Random Forest Telemetry Feature Importances")
    plt.barh(range(len(indices)), importances[indices], align="center", color="royalblue")
    plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    
    importance_plot_path = plots_dir / "feature_importance.png"
    plt.savefig(importance_plot_path)
    plt.close()
    LOGGER.info("Feature importance plot saved to %s", importance_plot_path)
    
    # 4. Save joblib Model Artifact
    import joblib
    model_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = model_dir / "admitter_model.joblib"
    joblib.dump(rf_model, model_save_path)
    LOGGER.info("Serialized trained Random Forest model successfully to %s", model_save_path)
    
    # 5. Export metrics JSON
    metrics_path = results_dir / "model_metrics.json"
    metrics_data = {
        "decision_tree": dt_metrics,
        "random_forest": rf_metrics,
        "model_path": str(model_save_path),
        "dataset_rows": len(df)
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_data, handle, indent=2)
    LOGGER.info("Saved model evaluation metrics to %s", metrics_path)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
