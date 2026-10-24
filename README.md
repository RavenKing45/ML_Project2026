# An Empirical Study of Class-Imbalance Handling Techniques for Credit Card Fraud Detection

This repository compares six class-imbalance handling techniques (baseline, random
oversampling, random undersampling, SMOTE, SMOTE followed by random undersampling, and class
weighting) across logistic regression, random forest and LightGBM on the ULB credit-card fraud
dataset, using a leakage-free protocol with average precision (PR-AUC) as the primary metric.
A FastAPI backend and a React + TypeScript dashboard present the saved results read-only.

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate   # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e .
```

Put the Kaggle file at `data/raw/creditcard.csv`
(<https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>). Its SHA-256 is pinned in
`configs/dataset.yaml` and checked on load. Without it, use `--source sample` (synthetic data,
for development only).

| step | command | output |
|------|---------|--------|
| data summary | `python scripts/data_summary.py` | counts per split (printed) |
| EDA | `python scripts/run_eda.py --source raw` | `artifacts/results/eda.json`, `artifacts/figures/eda_*.png` |
| resampling audit | `python scripts/run_resampling_audit.py --source raw` | `artifacts/results/resampling_audit.json` (class counts per technique, ratio and CV fold; training split only) |
| experiments | `python scripts/run_experiments.py --source raw [--quick] [--models ...] [--techniques ...] [--force]` | `artifacts/results/cells/<model>__<technique>.json` + `_oof.npy`, `artifacts/results/fold_scores.csv`, `artifacts/results/run_meta.json`, `artifacts/models/cells/*.joblib` (git-ignored); finished cells are skipped unless `--force` |
| merge machines | `python scripts/merge_results.py --inputs DIR [DIR ...] [--output artifacts/results]` | `cv_summary.json`, de-duplicated `fold_scores.csv`; lists missing cells |
| checks | `ruff check . && pytest -q` | lint and tests (synthetic sample only) |

Full setup (experiment, backend, dashboard) will be added as those parts land.
