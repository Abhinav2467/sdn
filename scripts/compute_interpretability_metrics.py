#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd
import joblib
import shap
from sklearn.inspection import permutation_importance
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
processed_dir = ROOT / "results" / "processed"

def main():
    print("=== Phase 8: Compute Interpretability Metrics (SHAP & Permutation) ===")
    
    # Load model and data
    model_path = ROOT / "src" / "sdn_exam_demo" / "models" / "admitter_model.joblib"
    if not model_path.exists():
        print("FAIL: ML model not found!")
        return 1
        
    pipeline = joblib.load(model_path)
    X_train_path = processed_dir / "X_train.csv"
    X_val_path = processed_dir / "X_val.csv"
    y_val_path = processed_dir / "y_val.csv"
    scaler_path = processed_dir / "feature_scaler.joblib"
    
    if not (X_train_path.exists() and X_val_path.exists() and y_val_path.exists() and scaler_path.exists()):
        print("FAIL: Preprocessed datasets or scaler are missing!")
        return 1
        
    X_train = pd.read_csv(X_train_path)
    X_val = pd.read_csv(X_val_path)
    y_val = pd.read_csv(y_val_path).iloc[:, 0]
    scaler = joblib.load(scaler_path)

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
    feature_names = continuous_features + discrete_features

    # UNSCALE features first because the pipeline preprocessor ColumnTransformer expects RAW values
    X_train_unscaled = X_train.copy()
    X_train_unscaled[continuous_features] = scaler.inverse_transform(X_train[continuous_features])
    
    X_val_unscaled = X_val.copy()
    X_val_unscaled[continuous_features] = scaler.inverse_transform(X_val[continuous_features])

    # Extract pipeline preprocessor and regressor
    preprocessor = pipeline.named_steps["preprocessor"]
    regressor = pipeline.named_steps["regressor"]
    
    X_train_transformed = preprocessor.transform(X_train_unscaled)
    X_val_transformed = preprocessor.transform(X_val_unscaled)

    # 1. SHAP Quantitative Analysis
    print("Computing SHAP TreeExplainer values...")
    explainer = shap.TreeExplainer(regressor)
    np.random.seed(42)
    sample_idx = np.random.choice(len(X_val_transformed), size=100, replace=False)
    X_sample = X_val_transformed[sample_idx]
    shap_values = explainer.shap_values(X_sample)

    # Compute mean absolute SHAP per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    std_abs_shap = np.abs(shap_values).std(axis=0)

    # Compute percentile ranks
    ranks = pd.Series(mean_abs_shap, index=feature_names).rank(ascending=False)
    total_shap = mean_abs_shap.sum()

    shap_results = {}
    for i, col in enumerate(feature_names):
        shap_results[col] = {
            "mean_abs_shap": float(mean_abs_shap[i]),
            "std_abs_shap": float(std_abs_shap[i]),
            "rank": int(ranks[col]),
            "pct_contribution": float(mean_abs_shap[i] / total_shap * 100) if total_shap > 0 else 0.0
        }

    shap_out_path = processed_dir / "shap_quantitative.json"
    with shap_out_path.open("w", encoding="utf-8") as f:
        json.dump(shap_results, f, indent=2)
    print(f"[OK] SHAP Quantitative results saved to {shap_out_path}")

    # 2. Permutation Importance Analysis
    print("Computing Permutation Importance...")
    # Permutation importance must be run on the full pipeline, using preprocessed unscaled dataframes
    perm_importance = permutation_importance(pipeline, X_val_unscaled, y_val, n_repeats=10, random_state=42)
    perm_results = {}
    for i, col in enumerate(X_val.columns):
        perm_results[col] = {
            "mean_importance": float(perm_importance.importances_mean[i]),
            "std_importance": float(perm_importance.importances_std[i])
        }

    perm_out_path = processed_dir / "permutation_importance.json"
    with perm_out_path.open("w", encoding="utf-8") as f:
        json.dump(perm_results, f, indent=2)
    print(f"[OK] Permutation Importance results saved to {perm_out_path}")

    return 0

if __name__ == "__main__":
    main()
