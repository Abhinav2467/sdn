# Reproducibility Guide

## Software Environment
This project was evaluated in an emulated Mininet environment running the Ryu SDN Controller.
The ML models were trained and tested using Python 3.12 within a virtual environment.

## Dependencies
- `scikit-learn` (Random Forest, classification metrics)
- `shap` (Permutation Explainer)
- `pandas`, `numpy`, `scipy` (Data processing, Mann-Whitney U Test)

## Reproducing the Results

1. **Model Training & Evaluation**:
   The Random Forest model was trained on flow statistics. The `results/processed/model_metrics.json` contains the verified evaluation metrics on the holdout set (e.g., $R^2 \approx 0.9$, F1 $\approx 0.91$).

2. **Emulation (Mininet + Ryu)**:
   A 90-run emulation suite (`results/experiments/all_runs.csv`) was used to collect round-trip time (RTT) measurements across three policies: ML, Static, and Heuristic. Seeds 1000-1029 were used for the holdout test to ensure no overlap with training seeds.

3. **Statistical Testing**:
   The non-parametric Mann-Whitney U test confirmed that the ML-assisted policy achieves stochastically smaller latency than the Static and Heuristic baselines (p < 0.05). Run `tests/test_statistics.py` or inspect `results/statistical_tests.json`.

4. **Topology Generalization**:
   The model was tested across Linear, Tree, and Custom topologies to verify generalizability against strict R² guardrails ($R^2 \ge 0.75$ for simple, $\ge 0.55$ for complex topologies).

5. **Explainability**:
   Quantitative SHAP analysis via Permutation Importance was performed. Check `results/processed/shap_quantitative.json` to verify the dominant feature contribution.

## Manuscript Compilation
The IEEE Access manuscript can be built directly using standard LaTeX tools (e.g., `pdflatex` or `tectonic`):
```bash
cd manuscript
tectonic main.tex
```
