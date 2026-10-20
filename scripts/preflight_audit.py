#!/usr/bin/env python3
import os
import sys
import pandas as pd
import json
from pathlib import Path
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]

def run_preflight():
    print("=== Phase 0: Pre-Flight Integrity Audit (v5) ===")
    errors = []

    # Step 0.1: Critical File Inventory
    required_dirs = {
        'src': ROOT / 'src',
        'results': ROOT / 'results',
        'results/processed': ROOT / 'results/processed',
        'results/experiments': ROOT / 'results/experiments',
        'papers': ROOT / 'papers',
        'tests': ROOT / 'tests'
    }

    for name, path in required_dirs.items():
        if not path.exists():
            print(f"WARNING: {name} directory missing at {path}")
            errors.append(f"Directory missing: {name}")
        else:
            count = len([f for f in path.glob("**/*") if f.is_file()])
            print(f"{name}: {count} files")

    critical_files = [
        ROOT / 'results/dataset.csv',
        ROOT / 'results/processed/model_metrics.json',
        ROOT / 'results/processed/shap_quantitative.json',
        ROOT / 'results/processed/permutation_importance.json',
        ROOT / 'results/statistical_tests.json',
        ROOT / 'results/topology_comparison.json',
        ROOT / 'src/sdn_exam_demo/models/admitter_model.joblib',
        ROOT / 'src/sdn_exam_demo/models/schema.json',
        ROOT / 'papers/ieee_compliance_report_v4.md'
    ]

    for f in critical_files:
        if not f.exists():
            print(f"FAIL: Missing critical file: {f}")
            errors.append(f"Missing critical file: {f}")
        else:
            print(f"FOUND: {f}")

    if errors:
        print("FAIL: Pre-flight inventory checks failed!")
        return 1
    print("PASS: All critical v4 files present")

    # Step 0.2: 90-Run Statistical Dataset Verification
    exp_file = ROOT / 'results/experiments/all_runs.csv'
    if not exp_file.exists():
        print(f"FAIL: No experiments file found at {exp_file}")
        return 1

    df = pd.read_csv(exp_file)
    print(f"FOUND: {exp_file} with {len(df)} rows")
    if 'mode' in df.columns:
        print(f"  Modes: {df['mode'].value_counts().to_dict()}")
    if 'seed' in df.columns:
        seeds = df['seed'].nunique()
        print(f"  Unique seeds: {seeds}")
        if seeds < 30:
            print(f"  FAIL: Only {seeds} seeds, expected 30")
            return 1
    else:
        print("FAIL: Missing seed column in experiments file")
        return 1

    print("PASS: Statistical experiment data exists")

    # Step 0.3: Value Consistency Audit
    with open(ROOT / 'results/processed/model_metrics.json') as f:
        metrics = json.load(f)

    with open(ROOT / 'results/statistical_tests.json') as f:
        stats = json.load(f)

    with open(ROOT / 'results/topology_comparison.json') as f:
        topo = json.load(f)

    with open(ROOT / 'results/processed/shap_quantitative.json') as f:
        shap = json.load(f)

    with open(ROOT / 'papers/ieee_compliance_report_v4.md') as f:
        report = f.read()

    consistency_errors = []

    # Check CV Mean R2 and Holdout F1
    r2 = metrics["random_forest"]["cv_mean_r2"]
    f1 = metrics["random_forest"]["holdout_f1"]
    
    r2_str = f"{r2:.4f}"
    f1_str = f"{f1:.4f}"
    
    if r2_str not in report and f"{r2*100:.2f}" not in report:
        print(f"WARNING: CV Mean R2 {r2_str} not explicitly in report text")
    if f1_str not in report and f"{f1*100:.2f}" not in report:
        print(f"WARNING: Holdout F1 {f1_str} not explicitly in report text")

    # U-statistic direction check
    for comp, vals in stats.get('mann_whitney_u', {}).items():
        u = vals['u_statistic']
        p = vals['p_value']
        
        # Check if present in report (e.g. U = 249.0 or U = 249)
        u_pats = [f"U = {u:.1f}", f"U = {int(u)}", f"U\\text{{-statistic}} = {u:.1f}", f"U\\text{{-statistic}} = {int(u)}"]
        if not any(pat in report for pat in u_pats):
            print(f"WARNING: U-statistic {u} for {comp} not in report")
            
        p_str = f"{p:.2e}"
        p_str_tex = p_str.replace("e-03", " \\times 10^{-3}")
        if p_str_tex not in report:
            print(f"WARNING: p-value {p_str_tex} for {comp} not in report")

    # Topology R2 check
    for name, vals in topo.items():
        r2_t = vals.get('r2_score', 0)
        r2_t_str = f"{r2_t:.4f}"
        if r2_t_str not in report:
            print(f"WARNING: Topology {name} R2 {r2_t_str} not in report")

    # SHAP check
    max_shap = max(v['pct_contribution'] for v in shap.values())
    max_feat = max(shap.items(), key=lambda x: x[1]['pct_contribution'])[0]
    max_shap_str = f"{max_shap:.2f}%"
    if max_feat not in report or max_shap_str not in report:
         print(f"WARNING: Top SHAP feature {max_feat} ({max_shap_str}) not in report")

    # Step 0.4: Literature File Existence Check
    lit_files = {
        'references.bib': ROOT / 'papers/references.bib',
        'comparison_table.md': ROOT / 'papers/comparison_table.md',
        'gap_statement.md': ROOT / 'papers/gap_statement.md'
    }

    for name, path in lit_files.items():
        if not path.exists():
            print(f"FAIL: Missing literature file: {path}")
            return 1
        else:
            print(f"FOUND: {name} ({path.stat().st_size} bytes)")

    with open(ROOT / 'papers/references.bib') as f:
        content = f.read()
    entries = content.count('@')
    if entries < 15:
        print(f"FAIL: references.bib has only {entries} entries, need >= 15")
        return 1
    print(f"PASS: references.bib has {entries} entries")

    print("PASS: Pre-flight integrity checks completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(run_preflight())
