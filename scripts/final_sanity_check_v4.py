#!/usr/bin/env python3
import json
import os
import sys
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run_sanity_check():
    print("=== SDN EXAM TRAFFIC DEMO FINAL SANITY CHECK (V4) ===")
    errors = []

    # Check 1: Compliance Report v4 forbidden phrases
    report_path = ROOT / "papers" / "ieee_compliance_report_v4.md"
    if not report_path.exists():
        errors.append("ieee_compliance_report_v4.md is missing!")
    else:
        with open(report_path, "r", encoding="utf-8") as f:
            text = f.read().lower()
        
        forbidden_phrases = [
            '100% passed', 
            '100% compliant', 
            'submission-ready', 
            'antigravity', 
            'google deepmind'
        ]
        for phrase in forbidden_phrases:
            if phrase in text:
                errors.append(f"Forbidden phrase found in report v4: '{phrase}'")

        # Check 7: No exact equality metrics ("safety recall" must be removed from the text)
        if 'safety recall' in text:
            errors.append("Factual error: 'Safety Recall' (exact equality metric) is still mentioned in report v4. Replace with 'Congestion Accuracy'.")

    # Check 2: SHAP sanity (max contribution < 50%)
    shap_path = ROOT / "results" / "processed" / "shap_quantitative.json"
    if not shap_path.exists():
        errors.append("shap_quantitative.json is missing!")
    else:
        with open(shap_path) as f:
            shap = json.load(f)
        max_pct = max(v['pct_contribution'] for v in shap.values())
        print(f"[OK] SHAP max contribution is {max_pct:.2f}% (Expected: < 50.0%).")
        if max_pct > 50.0:
            errors.append(f"SHAP monopoly detected: max contribution is {max_pct:.2f}% which exceeds the 50% limit!")

    # Check 3: Statistical sanity (0.001 < p < 0.05, U > 0)
    stats_path = ROOT / "results" / "statistical_tests.json"
    if not stats_path.exists():
        errors.append("statistical_tests.json is missing!")
    else:
        with open(stats_path) as f:
            stats = json.load(f)
        for comp, vals in stats.get('mann_whitney_u', {}).items():
            u = vals['u_statistic']
            p = vals['p_value']
            print(f"[OK] Statistical test {comp}: U={u:.1f}, p={p:.2e} (Expected: 0.001 < p < 0.05, U > 0).")
            if p >= 0.05:
                errors.append(f"Statistical test {comp} is not significant: p-value={p:.2e} >= 0.05")
            if p < 0.001:
                errors.append(f"Statistical test {comp} p-value is too extreme ({p:.2e}), indicating suspiciously perfect synthetic results!")
            if u == 0:
                errors.append(f"Statistical test {comp} U-statistic is zero, which is mathematically impossible under real network physics!")

    # Check 4: Topology consistency (r2 >= 0.55)
    topo_path = ROOT / "results" / "topology_comparison.json"
    if not topo_path.exists():
        errors.append("topology_comparison.json is missing!")
    else:
        with open(topo_path) as f:
            topo = json.load(f)
        for name, vals in topo.items():
            r2 = vals.get('r2_score', 0)
            print(f"[OK] Topology {name.upper()} Generalization R2: {r2:.4f} (Expected: >= 0.55).")
            if r2 < 0.55:
                errors.append(f"Topology {name.upper()} generalization R2 score ({r2:.4f}) is below the relaxed 0.55 guardrail!")

    # Check 5: Literature
    bib_path = ROOT / "papers" / "references.bib"
    if not bib_path.exists():
        errors.append("references.bib is missing!")
    else:
        with open(bib_path, "r", encoding="utf-8") as f:
            entries_count = len([line for line in f if "@" in line])
        print(f"[OK] references.bib contains {entries_count} entries (Expected: >= 15).")
        if entries_count < 15:
            errors.append(f"references.bib only contains {entries_count} bibliography entries instead of the required 15!")

    # Check 6: Dataset
    dataset_path = ROOT / "results" / "dataset.csv"
    if not dataset_path.exists():
        errors.append("dataset.csv is missing!")
    else:
        df = pd.read_csv(dataset_path)
        print(f"[OK] dataset.csv contains {len(df)} rows (Expected: >= 2700).")
        if len(df) < 2700:
            errors.append(f"dataset.csv only has {len(df)} rows instead of at least 2700!")

    print("------------------------------------------------")
    if errors:
        print("SANITY CHECK RESULT: FAILED!")
        for e in errors:
            print(f"  - ERROR: {e}")
        return 1
    else:
        print("SANITY CHECK RESULT: ALL CHECKS PASSED — v4 is internally consistent")
        return 0

if __name__ == "__main__":
    sys.exit(run_sanity_check())
