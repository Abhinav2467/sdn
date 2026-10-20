#!/usr/bin/env python3
import json
import logging
import os
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
processed_dir = ROOT / "results" / "processed"
results_dir = ROOT / "results"
results_dir.mkdir(parents=True, exist_ok=True)
model_path = ROOT / "src" / "sdn_exam_demo" / "models" / "admitter_model.joblib"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
LOGGER = logging.getLogger("sdn_exam_demo.compare_topologies")

def main():
    LOGGER.info("=== Phase 5: Topology Variation & Generalization ===")
    
    # 1. Load ML Model Pipeline
    if not model_path.exists():
        LOGGER.error("ML model not found!")
        return 1
        
    pipeline = joblib.load(model_path)
    LOGGER.info("Successfully loaded ML model pipeline.")
    
    # Check if preprocessed validation split and scaler exist
    X_val_path = processed_dir / "X_val.csv"
    y_val_path = processed_dir / "y_val.csv"
    scaler_path = processed_dir / "feature_scaler.joblib"
    
    if not (X_val_path.exists() and y_val_path.exists() and scaler_path.exists()):
        LOGGER.error("Preprocessed validation split or scaler missing!")
        return 1
        
    X_val = pd.read_csv(X_val_path)
    y_val = pd.read_csv(y_val_path).iloc[:, 0]
    scaler = joblib.load(scaler_path)
    
    continuous_features = [
        "exam_rtt_ms",
        "measured_throughput_mbps",
        "packet_loss_percent",
        "time_to_exam_seconds"
    ]
    
    # Unscale validation features to restore original ranges before pipeline scaling
    X_val_unscaled = X_val.copy()
    X_val_unscaled[continuous_features] = scaler.inverse_transform(X_val[continuous_features])
    
    topologies = ["linear", "tree", "fattree"]
    results = {}
    
    for topo in topologies:
        config_file = ROOT / "config" / f"topology_{topo}.json"
        if not config_file.exists():
            LOGGER.error(f"Topology configuration for {topo} not found!")
            return 1
            
        with open(config_file, "r") as handle:
            topo_meta = json.load(handle)
            
        LOGGER.info(f"Evaluating model generalization on: Topology {topo.upper()} ({topo_meta.get('switches', 1)} switches)")
        
        # Apply topology-specific multi-hop queuing delay shifts to unscaled validation RTT
        X_topo = X_val_unscaled.copy()
        if topo == "linear":
            shift = 0.0
            r2_guardrail = 0.75
        elif topo == "tree":
            shift = 0.05
            r2_guardrail = 0.70
        else: # fattree
            shift = 0.12
            r2_guardrail = 0.55
            
        X_topo["exam_rtt_ms"] = X_topo["exam_rtt_ms"] + shift
        
        # Predict using final pipeline
        predictions = pipeline.predict(X_topo)
        
        # Calculate generalization R2 metric
        r2 = r2_score(y_val, predictions)
        
        # Congestion Accuracy:
        # Measures the fraction of restrictive states (y_val < 2.5) predicted within operational tolerance (+- 0.25)
        restrictive_mask = y_val < 2.5
        total_congested = np.sum(restrictive_mask)
        correct_congested = np.sum(restrictive_mask & (np.abs(predictions - y_val) <= 0.25))
        congestion_accuracy = correct_congested / total_congested if total_congested > 0 else 1.0
        
        passed = r2 >= r2_guardrail
        
        results[topo] = {
            "r2_score": float(r2),
            "congestion_accuracy": float(congestion_accuracy),
            "mean_simulated_rtt": float(np.mean(X_topo["exam_rtt_ms"])),
            "mean_predicted_limit": float(np.mean(predictions)),
            "status": "PASSED" if passed else "FAILED"
        }
        
        LOGGER.info(f"  {topo.upper()} -> R2 Score: {r2:.4f}, Congestion Accuracy: {congestion_accuracy:.2%}, Status: {'PASSED' if passed else 'FAILED'}")
        
    # 3. Export Topology Comparison Report
    comparison_path = results_dir / "topology_comparison.json"
    with open(comparison_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
        
    LOGGER.info(f"[OK] Topology comparison report written successfully to {comparison_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
