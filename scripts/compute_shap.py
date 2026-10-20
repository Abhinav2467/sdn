#!/usr/bin/env python3
import logging
import sys
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
processed_dir = ROOT / "results" / "processed"
results_dir = ROOT / "results"
plots_dir = results_dir / "latest" / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
LOGGER = logging.getLogger("sdn_exam_demo.compute_shap")

def main():
    LOGGER.info("=== Phase 6: SHAP Analysis & Explainability ===")
    
    # 1. Load trained model pipeline and training partition
    model_path = ROOT / "src" / "sdn_exam_demo" / "models" / "admitter_model.joblib"
    if not model_path.exists():
        LOGGER.error("ML model not found!")
        return 1
        
    pipeline = joblib.load(model_path)
    X_train = pd.read_csv(processed_dir / "X_train.csv")
    
    # 2. Extract pipeline steps
    preprocessor = pipeline.named_steps["preprocessor"]
    regressor = pipeline.named_steps["regressor"]
    
    # Preprocess training features to numpy array (matching column transformer outputs)
    X_train_transformed = preprocessor.transform(X_train)
    
    # Reconstruct exact feature names after ColumnTransformer preprocessing:
    # StandardScaler transforms continuous_features in the first block, remainder passes through discrete_features
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
    
    # 3. Calculate SHAP values using TreeExplainer
    LOGGER.info("Initializing SHAP TreeExplainer and calculating Shapley values...")
    explainer = shap.TreeExplainer(regressor)
    
    # Sample 100 rows for fast and robust calculations
    sample_idx = np.random.choice(len(X_train_transformed), size=100, replace=False)
    X_sample = X_train_transformed[sample_idx]
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_sample)
    
    # Serialize SHAP values numpy array
    shap_values_path = processed_dir / "shap_values.npy"
    np.save(shap_values_path, shap_values)
    LOGGER.info(f"[OK] SHAP values calculated and serialized successfully to {shap_values_path}")
    
    # 4. Generate SHAP summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.title("SHAP Feature Attribution: Drivers of SDN Flow Admission Limits", fontsize=14, pad=15)
    plt.tight_layout()
    
    shap_plot_path = plots_dir / "shap_summary.png"
    plt.savefig(shap_plot_path)
    plt.close()
    LOGGER.info(f"[OK] SHAP summary plot generated and saved to {shap_plot_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
