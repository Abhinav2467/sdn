#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, matthews_corrcoef, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
dataset_path = ROOT / "results" / "dataset.csv"
model_path = ROOT / "src" / "sdn_exam_demo" / "models" / "admitter_model.joblib"
metrics_path = ROOT / "results" / "processed" / "model_metrics.json"

def main():
    print("=== Computing Holdout Classification Metrics ===")
    
    if not dataset_path.exists():
        print(f"Error: {dataset_path} not found!")
        return 1
        
    df = pd.read_csv(dataset_path)
    
    # 1. Feature and target reconstruction (identical to train_ml_model.py)
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

    # Split
    X_train, X_holdout, y_train, y_holdout, label_train, label_holdout = train_test_split(
        X, y, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # Noise on holdout
    np.random.seed(99)
    y_holdout_noisy = y_holdout + np.random.normal(0, 0.20, size=len(y_holdout))
    y_holdout_noisy = np.clip(y_holdout_noisy, 1.0, 4.0)
    label_holdout_noisy = np.where(y_holdout_noisy < 2.5, 0, 1)

    # Load trained pipeline
    pipeline = joblib.load(model_path)
    holdout_preds = pipeline.predict(X_holdout)

    # Classification metrics
    holdout_class_preds = np.where(holdout_preds < 2.5, 0, 1)
    
    # Restrictive is class 0, Permissive is class 1
    cm = confusion_matrix(label_holdout_noisy, holdout_class_preds)
    tn, fp, fn, tp = cm.ravel()
    
    total = len(label_holdout_noisy)
    restrictive_count = int((label_holdout_noisy == 0).sum())
    permissive_count = int((label_holdout_noisy == 1).sum())
    
    bal_acc = balanced_accuracy_score(label_holdout_noisy, holdout_class_preds)
    mcc = matthews_corrcoef(label_holdout_noisy, holdout_class_preds)
    f1 = f1_score(label_holdout_noisy, holdout_class_preds, pos_label=0)
    prec = precision_score(label_holdout_noisy, holdout_class_preds, pos_label=0)
    rec = recall_score(label_holdout_noisy, holdout_class_preds, pos_label=0)

    print("\nClass Distribution:")
    print(f"  Restrictive (y < 2.5): {restrictive_count} ({restrictive_count/total*100:.2f}%)")
    print(f"  Permissive (y >= 2.5): {permissive_count} ({permissive_count/total*100:.2f}%)")

    print("\nConfusion Matrix:")
    print(f"  TN (Correct Restrictive): {tn}")
    print(f"  FP (False Permissive): {fp}")
    print(f"  FN (False Restrictive): {fn}")
    print(f"  TP (Correct Permissive): {tp}")

    print("\nHoldout Performance Metrics:")
    print(f"  F1-Score (Class 0): {f1:.4f}")
    print(f"  Precision (Class 0): {prec:.4f}")
    print(f"  Recall (Class 0): {rec:.4f}")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    print(f"  Matthews Correlation Coefficient (MCC): {mcc:.4f}")

    # Load and update metrics json
    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)
        
    metrics_data["random_forest"]["class_distribution"] = {
        "restrictive_count": restrictive_count,
        "permissive_count": permissive_count,
        "restrictive_pct": float(restrictive_count / total),
        "permissive_pct": float(permissive_count / total)
    }
    metrics_data["random_forest"]["confusion_matrix"] = {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp)
    }
    metrics_data["random_forest"]["balanced_accuracy"] = float(bal_acc)
    metrics_data["random_forest"]["matthews_correlation_coefficient"] = float(mcc)
    
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    # Also save to results/latest/
    with open(ROOT / "results" / "latest" / "model_metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=2)
        
    print(f"\n[OK] Successfully updated model metrics files with holdout classification parameters.")
    return 0

if __name__ == "__main__":
    exit(main())
