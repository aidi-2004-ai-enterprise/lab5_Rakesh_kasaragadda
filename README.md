# Lab 5 — Model Implementation & Evaluation Pipeline (Bankruptcy Prediction)

End-to-end pipeline that automates **EDA → preprocessing → stratified split → PSI drift → simple feature selection → simple tuning → training (LR, RF, XGB/GB) → calibration/ROC/Brier → SHAP** and writes a Markdown report you can export to PDF.

---

## 🔗 Video & Report
- **Video (5 min):** _add your link here_
- **Report (MD):** `artifacts/report.md` → Print → **Save as PDF** (optional `artifacts/report.pdf`)

---

## ⚡ Quickstart
```bash
# (inside repo root)
python -m venv .venv
# Windows (Git Bash):
source .venv/Scripts/activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
python training_pipeline.py --data_source kaggle --artifacts_dir artifacts --tune_iter 20

📁 Project Structure (as in this repo)
lab5_Rakesh_kasaragadda/
├── .github/
│   └── workflows/
│       └── lint.yml
├── artifacts/
│   ├── eda/
│   ├── evaluation/
│   ├── models/
│   ├── psi/
│   ├── shap/
│   └── report.md
├── README.md
├── pyproject.toml
├── requirements.txt
└── training_pipeline.py

🧠 Lab-4 Decisions carried into Lab-5 (why)

Models: Logistic Regression (benchmark, interpretable); Random Forest + XGB/GB for non-linear signal.

Primary metric: PR-AUC (minority ≈ 3.2%); also track ROC-AUC, Brier, F1@0.5.

Split & drift: Stratified 70/30 + PSI to guard sampling bias.

Preprocessing: Median imputation; scale LR only; use class weights / scale_pos_weight; no SMOTE by default.

✅ Answers (Jot-note Rationale)
1) EDA — why

Check missingness/skew → informs imputation/scaling.

Inspect target correlations → modeling focus.

Keep true risk outliers (avoid blunt clipping).

Lean EDA → reproducible & fast.
Artifacts: artifacts/eda/missing_values_top30.png, artifacts/eda/target_distribution.png, artifacts/eda/corr_heatmap_top.png, artifacts/eda/hists_top12.png

2) Preprocessing, Imbalance & PSI — why

Median impute everywhere; scale only LR (trees are scale-invariant).

Stratified split preserves prevalence; weights avoid synthetic artifacts.

PSI (train vs test) top features < 0.10 → no sampling bias; fair eval.

Same preprocessing across models → fair comparison.
PSI Top-15 (head):

feature	psi
Cash flow rate	0.0161
Fixed Assets to Assets	0.0158
Inventory/Working Capital	0.0150
Operating profit per person	0.0133
Long-term fund suitability ratio (A)	0.0128
Net Value Per Share (C)	0.0106
Total Asset Turnover	0.0104
Net Value Per Share (A)	0.0103
Cash Reinvestment %	0.0103
Working Capital to Total Assets	0.0099

Artifacts: artifacts/psi/psi_top15.png, artifacts/psi/psi_overlay_Cash_flow_rate.png

3) Feature Selection (simple) — why

Correlation filter (|r|>0.95) removes redundancy → lower variance.

Keep interpretability (skip PCA unless needed).

Apply once; shared across models.

Retain most informative features wrt target.
Result: 75 features kept (full list in artifacts/report.md).

4) Hyperparameter Tuning (simple) — why

RandomizedSearchCV, StratifiedKFold(5), scoring = average_precision (PR-AUC).

Small grids → efficient compute.

Per-model spaces (LR: C; RF: depth/leaves/estimators; XGB: lr/trees/depth/subsample).

Seeds fixed for reproducibility.

5–6) Train, Calibrate, ROC & Compare — why

Train LR baseline, RF, XGB/GB on identical splits.

ROC-AUC (discrimination) + Calibration & Brier (probability quality).

Compare train vs test to spot over/underfitting.

Choose primarily by PR-AUC.
Comparison (train/test):

model	train_pr_auc	test_pr_auc	train_roc_auc	test_roc_auc	train_brier	test_brier	train_f1@0.5	test_f1@0.5
logreg	0.4648	0.2791	0.9606	0.8730	0.0833	0.0903	0.3333	0.2636
rf	0.9956	0.4718	0.9999	0.9556	0.0082	0.0235	0.9419	0.4561
xgb	1.0000	0.4810	1.0000	0.9518	0.0000	0.0260	1.0000	0.4248

Artifacts: artifacts/evaluation/calibration_*.png, artifacts/evaluation/roc_*.png

7) SHAP Interpretability — why

Explain global drivers & local attributions (governance).

Drivers align with business sense (cash flow, leverage, liquidity).

TreeExplainer; beeswarm + bar summaries.
Artifacts: artifacts/shap/shap_summary_beeswarm.png, artifacts/shap/shap_summary_bar.png

8) PSI — Drift Monitoring — why

Thresholds: <0.10 stable; 0.10–0.20 monitor; >0.20 investigate.

Current train vs test PSI → stable; schedule PSI checks in prod.

If drift detected → re-sample / re-tune / re-train.

9) Challenges & Reflections

Compute budget → small/efficient random search.

Imbalance → PR-AUC + weights; avoided synthetic data.

Interpretability vs performance → SHAP, no PCA (unless needed).

Reproducibility → seeds fixed; deterministic outputs; CI lint (Ruff).

✅ Recommendation

Best by test PR-AUC: XGB.

Next: threshold tuning to business KPI; publish SHAP; set PSI monitors in Airflow.

🧹 Style / CI

Ruff configured via pyproject.toml.
Local:

ruff check .          # lint
ruff check . --fix    # auto-fix
ruff format .         # format


CI runs on every push/PR: .github/workflows/lint.yml.

📝 How to Reproduce
pip install -r requirements.txt
python training_pipeline.py --data_source kaggle --artifacts_dir artifacts --tune_iter 20


Seeds are fixed; all outputs saved under artifacts/.

::contentReference[oaicite:0]{index=0}
