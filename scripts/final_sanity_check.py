#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def audit():
    print("=== SDN EXAM TRAFFIC DEMO FINAL SANITY CHECK ===")
    errors = []
    
    # 1. Dataset verification
    dataset = ROOT / "results" / "dataset.csv"
    if not dataset.exists():
        errors.append("dataset.csv is missing!")
    else:
        rows = len(open(dataset).readlines())
        print(f"[OK] dataset.csv exists and has {rows} rows (Expected: 2701).")
        if rows != 2701:
            errors.append(f"dataset.csv has {rows} rows instead of 2701!")
            
    # 2. Model files verification
    model = ROOT / "src" / "sdn_exam_demo" / "models" / "admitter_model.joblib"
    if not model.exists() or model.stat().st_size == 0:
        errors.append("admitter_model.joblib is missing or empty!")
    else:
        print(f"[OK] admitter_model.joblib is present ({model.stat().st_size / 1024:.2f} KB).")
        
    schema = ROOT / "src" / "sdn_exam_demo" / "models" / "schema.json"
    if not schema.exists():
        errors.append("schema.json is missing!")
    else:
        print("[OK] schema.json feature definition exists.")
        
    # 3. Model metrics validation
    metrics = ROOT / "results" / "processed" / "model_metrics.json"
    if not metrics.exists():
        errors.append("model_metrics.json is missing!")
    else:
        try:
            with open(metrics) as handle:
                m = json.load(handle)
                r2 = m["random_forest"]["cv_mean_r2"]
                f1 = m["random_forest"]["holdout_f1"]
                print(f"[OK] model_metrics.json exists. CV Mean R2: {r2:.4f}, Holdout F1: {f1:.4f}")
                if not (0.75 <= r2 <= 0.95):
                    errors.append(f"CV Mean R2 score ({r2:.4f}) violates the strict [0.75, 0.95] overfitting guardrail!")
                if not (0.85 <= f1 <= 0.98):
                    errors.append(f"Holdout F1 score ({f1:.4f}) violates the [0.85, 0.98] target!")
        except Exception as exc:
            errors.append(f"Failed to parse model_metrics.json: {exc}")
            
    # 4. Statistical significance verification
    stat_file = ROOT / "results" / "statistical_tests.json"
    if not stat_file.exists():
        errors.append("statistical_tests.json is missing!")
    else:
        try:
            with open(stat_file) as handle:
                s = json.load(handle)
                p_static = s["mann_whitney_u"]["ml_vs_static"]["p_value"]
                p_heur = s["mann_whitney_u"]["ml_vs_heuristic"]["p_value"]
                print(f"[OK] statistical_tests.json exists. p_static={p_static:.2e}, p_heuristic={p_heur:.2e}")
                if p_static >= 0.05 or p_heur >= 0.05:
                    errors.append("U-test p-value is not statistically significant (p >= 0.05)!")
        except Exception as exc:
            errors.append(f"Failed to parse statistical_tests.json: {exc}")
            
    # 5. Topology comparison verification
    topo_file = ROOT / "results" / "topology_comparison.json"
    if not topo_file.exists():
        errors.append("topology_comparison.json is missing!")
    else:
        try:
            with open(topo_file) as handle:
                t = json.load(handle)
                print("[OK] topology_comparison.json exists.")
                for topo in ["linear", "tree", "fattree"]:
                    r2_t = t[topo]["r2_score"]
                    print(f"  Topology {topo.upper()} Generalization R2: {r2_t:.4f}")
                    if r2_t <= 0.50:
                        errors.append(f"Topology {topo.upper()} generalization R2 is too low ({r2_t:.4f})!")
        except Exception as exc:
            errors.append(f"Failed to parse topology_comparison.json: {exc}")
            
    # 6. Plots verification
    shap_plot = ROOT / "results" / "latest" / "plots" / "shap_summary.png"
    if not shap_plot.exists():
        errors.append("shap_summary.png explainability plot is missing!")
    else:
        print("[OK] SHAP explainability plot exists.")
        
    comp_plot = ROOT / "results" / "latest" / "plots" / "comparison_rtt.png"
    if not comp_plot.exists():
        errors.append("comparison_rtt.png performance distribution plot is missing!")
    else:
        print("[OK] Comparative boxplot performance distribution exists.")
        
    # 7. Documentation verification
    shap_doc = ROOT / "results" / "shap_interpretation.md"
    if not shap_doc.exists():
        errors.append("shap_interpretation.md documentation is missing!")
    else:
        print("[OK] shap_interpretation.md interpretability analysis exists.")
        
    bib = ROOT / "papers" / "references.bib"
    if not bib.exists():
        errors.append("references.bib bibliography is missing!")
    else:
        print("[OK] references.bib bibliography entries exist.")
        
    print("------------------------------------------------")
    if errors:
        print("SANITY CHECK RESULT: FAILED!")
        for error in errors:
            print(f"  - ERROR: {error}")
        return 1
    else:
        print("SANITY CHECK RESULT: 100% PASSED!")
        return 0

if __name__ == "__main__":
    sys.exit(audit())
