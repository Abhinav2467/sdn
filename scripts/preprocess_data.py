#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sdn_exam_demo.logging_config import configure_logging

LOGGER = logging.getLogger("sdn_exam_demo.preprocess_data")

def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess and verify SDN exam traffic dataset.")
    parser.add_argument("--dataset", default=str(ROOT / "results" / "dataset.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "processed"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("=== Phase 1: Dataset Verification & Preprocessing ===")

    # Step 1.1: Data Ingestion & Integrity Check
    if not dataset_path.exists():
        LOGGER.error("Dataset not found at %s!", dataset_path)
        return 1

    LOGGER.info("Loading dataset from %s...", dataset_path)
    df = pd.read_csv(dataset_path)

    # Completeness Check
    row_count = len(df)
    LOGGER.info("Completeness Check: Loaded %d rows from dataset.", row_count)
    if abs(row_count - 2700) <= 100:
        LOGGER.info("[OK] Row count matches target parameter grid (approx. 2700 rows).")
    else:
        LOGGER.warning("[WARNING] Row count (%d) deviates significantly from the target (approx. 2700 rows).", row_count)

    # Quality Audit
    null_counts = df.isnull().sum().sum()
    duplicate_count = df.duplicated().sum()
    LOGGER.info("Quality Audit: Found %d missing values (NaNs).", null_counts)
    LOGGER.info("Quality Audit: Found %d duplicate rows.", duplicate_count)

    if null_counts > 0:
        LOGGER.info("Cleaning missing values by dropping corrupted rows...")
        df = df.dropna()
    if duplicate_count > 0:
        LOGGER.info("Removing duplicate rows...")
        df = df.drop_duplicates()
        
    LOGGER.info("[OK] Quality audit completed. Total clean rows: %d", len(df))

    # Step 1.3: Feature Engineering & Formatting (moved up to support target calculation)
    LOGGER.info("Engineering features and formatting columns...")
    df["exam_window_active"] = df["exam_window_state"].apply(lambda s: 1.0 if s == "active" else 0.0)
    df["exam_window_prestage"] = df["exam_window_state"].apply(lambda s: 1.0 if s == "prestage" else 0.0)

    # Real-world Multi-dimensional Network Congestion and Flow Budget Engineering:
    # Instead of a single-feature shortcut, the optimal admission limit y is calculated
    # using a multi-dimensional resource budget based on all available network telemetry.
    LOGGER.info("Computing multi-dimensional network congestion and flow budget...")
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

    # Compute optimal continuous budget limits
    raw_limit = 4.0 - (delta_exam + delta_bg + delta_rtt + delta_loss + delta_queue + delta_time)
    y = np.clip(raw_limit, 1.0, 4.0)
    y = y - 0.01 * df["measured_throughput_mbps"]
    y = np.clip(y, 1.0, 4.0)
    
    # Establish binary classification target: 0 represents Congestion Impending/Restrictive (< 2.5),
    # 1 represents Stable/Permissive (>= 2.5)
    df["future_label"] = np.where(y < 2.5, 0, 1)

    # Step 1.2: Target Label Distribution Analysis
    counts = df["future_label"].value_counts()
    ratios = df["future_label"].value_counts(normalize=True)
    LOGGER.info("Target Label Distribution (future_label):")
    for val, cnt in counts.items():
        pct = ratios[val] * 100
        state = "Stable/Permissive (1)" if val == 1 else "Congestion/Restrictive (0)"
        LOGGER.info("  Class %d [%s]: %d rows (%.2f%%)", val, state, cnt, pct)

    skewed = False
    for val, pct in ratios.items():
        if pct > 0.80:
            skewed = True
            LOGGER.warning("[WARNING] High class imbalance detected (%.2f%% for class %d).", pct * 100, val)

    if not skewed:
        LOGGER.info("[OK] Class distribution is balanced and suitable for stratified cross-validation.")

    # 2. Continuous vs Discrete Feature Identification
    continuous_features = [
        "exam_rtt_ms",
        "measured_throughput_mbps",
        "packet_loss_percent",
        "time_to_exam_seconds"
    ]
    discrete_features = [
        "active_exam_clients",
        "active_background_clients",
        "iperf_flow_count",
        "queue_utilization_percent",
        "exam_window_active",
        "exam_window_prestage"
    ]

    all_features = continuous_features + discrete_features
    X = df[all_features].copy()
    y = pd.Series(y, index=df.index, name="target_limit")

    # Data Scaling (Fit scaler on continuous features)
    LOGGER.info("Applying StandardScaler to continuous features: %s", continuous_features)
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[continuous_features] = scaler.fit_transform(X[continuous_features])

    # Save scaler for prediction pipeline
    scaler_path = output_dir / "feature_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    LOGGER.info("Saved continuous feature StandardScaler to %s", scaler_path)

    # Stratified Split based on future_label classes
    LOGGER.info("Performing Stratified Shuffle Split (80% Train, 20% Holdout)...")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
    for train_index, val_index in sss.split(X_scaled, df["future_label"]):
        X_train, X_val = X_scaled.iloc[train_index], X_scaled.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        label_train, label_val = df["future_label"].iloc[train_index], df["future_label"].iloc[val_index]

    # Save preprocessed partitions
    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_val.to_csv(output_dir / "X_val.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False, header=True)
    y_val.to_csv(output_dir / "y_val.csv", index=False, header=True)
    
    # Save the original future_label labels to check fold stratification metrics later
    label_train.to_csv(output_dir / "label_train.csv", index=False, header=True)
    label_val.to_csv(output_dir / "label_val.csv", index=False, header=True)

    LOGGER.info("[OK] Dataset preprocessed successfully!")
    LOGGER.info("  Training Set shape  : %s (Target label ratio: %.2f%% stable)", X_train.shape, label_train.mean() * 100)
    LOGGER.info("  Validation Set shape: %s (Target label ratio: %.2f%% stable)", X_val.shape, label_val.mean() * 100)
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
