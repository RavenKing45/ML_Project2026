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
| test evaluation (once) | `python scripts/evaluate_test.py --source raw [--cells model__technique ...] [--rerun-reason "<why>"]` | `artifacts/results/comparison.json`, `curves.json`, `test_eval_details.json`, `TEST_SET_OPENED.json` (guard: a second run needs `--rerun-reason`, logged in `docs/decisions.md`); `--source sample` is exempt and writes to `artifacts/results/sample_test_eval/` |
| temporal check (sensitivity only) | `python scripts/run_temporal_check.py --source raw [--model lightgbm]` | `artifacts/results/temporal_check.json`; raw runs only after the test evaluation |
| statistics | `python scripts/run_stats.py --source raw` | `artifacts/results/stats.json` (Wilcoxon vs baseline with Holm, Friedman, Nemenyi CD) and `docs/findings.md` (auto tables; the team-written "Interpretation" section is kept on re-runs); `--source sample` writes both to `artifacts/results/sample_test_eval/` |
| report figures | `python scripts/make_figures.py --source raw` | PR / ROC curves, PR-AUC with CIs, confusion matrices, threshold effect, CD diagrams, resampling counts, fold-score box plots in `artifacts/figures/` (`artifacts/figures/sample/` for the sample); then re-run `run_stats.py` to link them |
| export for the backend | `python scripts/export_artifacts.py --source raw --default-cell <model>__<technique> [--rerun-reason "<why>"]` | `artifacts/models/manifest.json`, `feature_reference.json`, `deploy/<id>.joblib` (+ `.calibrator.joblib`; git-ignored), `artifacts/demo/presets.json`, `data/sample/demo_batch.csv`. Deploys `deploy.cells` in experiment.yaml (6 LightGBM + logreg for the default's technique); the team picks the default cell from the CV results, the script never does; run after the test evaluation (recorded in the guard). `--source sample` writes to `artifacts/models/sample/` and `artifacts/demo/sample/` |
| checks | `ruff check . && pytest -q` | lint and tests (synthetic sample only) |

## Backend (FastAPI)

The backend only reads saved artifacts; it never trains. Install and run it from the repository
root (`requirements-dev.txt` already includes the backend requirements):

```bash
pip install -e . -r backend/requirements.txt
USE_EXAMPLE_ARTIFACTS=true uvicorn backend.app.main:app --reload   # PowerShell: $env:USE_EXAMPLE_ARTIFACTS="true"; uvicorn backend.app.main:app --reload
```

Open <http://localhost:8000/docs> for the interactive docs. Endpoints (under `/api/v1`):
`GET /health`, `/models`, `/results`, `/eda`, `/presets`; `POST /predict` (one payment, with an
explanation and every LightGBM technique's score), `POST /predict/batch` (a CSV in a `file` field;
optional `?model_id=` and `?threshold=`) and `POST /smote-demo` (SMOTE step by step on a toy
dataset); shapes and error rules in `docs/contracts.md`. The prediction endpoints need the
exported models, so with example artifacts they answer 503; the SMOTE demo always works.

```bash
curl -F "file=@data/sample/demo_batch.csv" http://localhost:8000/api/v1/predict/batch   # after the export
```
Without `USE_EXAMPLE_ARTIFACTS=true` it reads the real artifacts, and until they all exist
`/health` reports `"degraded"` with one message per missing file and how to produce it.

| variable | default | purpose |
|----------|---------|---------|
| `USE_EXAMPLE_ARTIFACTS` | `false` | serve `docs/contracts/examples/` (fake data, for frontend work); no model file is loaded |
| `ARTIFACTS_DIR` | `artifacts` | `results/*.json` and `demo/presets.json` |
| `MODELS_DIR` | `artifacts/models` | `manifest.json` and the model files it lists |
| `EXAMPLES_DIR` | `docs/contracts/examples` | the example artifacts |
| `ALLOWED_ORIGINS` | empty | comma-separated CORS origins (`http://localhost:5173` is added in development) |
| `ENVIRONMENT` | `development` | `production` refuses `*`, an empty origin list and example artifacts |
| `DEFAULT_MODEL_ID` | manifest's | override the default model (a manifest id) |
| `MAX_UPLOAD_MB` / `MAX_BATCH_ROWS` | `5` / `10000` | upload limits of `POST /predict/batch` (413 above them) |
| `LOG_LEVEL` | `INFO` | log level of the `backend` loggers |
| `RATE_LIMIT_ENABLED` | `true` | slowapi limits per client: 60/min `/predict`, 10/min `/predict/batch`, 30/min `/smote-demo` |

Relative paths resolve from the repository root. Variables may also go in a git-ignored `.env`
file there. Logs never contain request bodies or feature values (uvicorn's own access log can be
turned off with `--no-access-log`; the backend writes its own with the request id).

Full setup (experiment, backend, dashboard) will be added as those parts land.

---

## Notebook, `model_build/` and project plan (from the `main` branch)

How to run the from-scratch SMOTE notebook in `model_build/`, followed by the `main` branch
README as written (headings two levels lower), which describes the notebook, `model_build/`
and the original project plan. Where its setup commands differ from the Quick start above,
the Quick start matches the current code.

### Running the notebook

The notebook's packages are kept out of the backend and CI requirements:

```bash
pip install -r requirements-notebook.txt   # from the repo root: project libraries, jupyter, -e ./model_build
jupyter notebook model_build/src/smote_from_scratch/SMOTE_algo.ipynb
```

It reads the full dataset from `data/raw/creditcard.csv` (`DATA_PATH` in the notebook).

### An Empirical Study of Class-Imbalance Handling Techniques for Credit Card Fraud Detection

College group project: a **controlled ML experiment** on how class-imbalance treatments affect fraud detection, plus a **small demo web app** for the final presentation.

This repository is the implementation of the architecture in [`docs/ML project discussion.pdf`](docs/ML%20project%20discussion.pdf). Planning notes also live in [`docs/ML_RESEARCH_BLUEPRINT.md`](docs/ML_RESEARCH_BLUEPRINT.md).

---

#### Research question

> How do data-level resampling techniques (random oversampling, random undersampling, SMOTE, and hybrid SMOTE + undersampling) and algorithm-level cost-sensitive learning compare against an unmodified baseline for highly imbalanced credit-card fraud detection?

We do **not** optimize for accuracy. Under ~0.17% fraud, a dummy “always legitimate” classifier is already ~99.8% accurate and catches **zero** fraud.

**Primary metric:** PR-AUC (Average Precision).  
**Secondary metrics:** Precision, Recall, F1, F2, ROC-AUC, FPR, FNR, confusion matrix, ROC and PR curves.

The study does **not** claim a winner in advance. The comparison table from a locked test set is the result.

---

#### What we compare

**Models**

| Model | Role |
|---|---|
| Logistic Regression | Interpretable linear baseline |
| Random Forest | Nonlinear bagged trees |
| LightGBM | Gradient boosting (tabular fraud default) |

**Imbalance treatments**

| ID | Technique | Type |
|---|---|---|
| A | None (raw imbalance) | Baseline |
| B | Random Oversampling (ROS) | Data-level |
| C | Random Undersampling (RUS) | Data-level |
| D | SMOTE | Data-level |
| E | SMOTE + Random Undersampling | Hybrid (Chawla et al.) |
| F | Class-weighted / `scale_pos_weight` | Algorithm-level |

**Core matrix:** 3 models × 6 treatments = **18 experiments**, same split, same preprocessor, same seeds. The only intended factor that changes per cell is the imbalance method.

Optional stretch: ADASYN, Borderline-SMOTE.

---

#### Methodology (non-negotiable)

```
raw data → validate → EDA → train/test split
                ↓
         TRAIN ONLY
         preprocess (fit on train fold)
         → resample (fit on train fold only)
         → model
         → stratified CV / tuning
         → threshold from out-of-fold train probabilities
                ↓
         TEST (untouched real distribution, once)
```

- Resampling is **never** applied before the split, and **never** applied to validation or test data.
- Training uses `imblearn.pipeline.Pipeline` (`preprocess → sampler → clf`).
- Inference uses a sklearn pipeline with **preprocessor + model only**. SMOTE is a training-time operation.
- Features are scaled **before** SMOTE so Euclidean *k*-NN is not dominated by `Amount` or `Time`.
- Default SMOTE/ROS strategy is **moderate** (not 1:1). Full balance on ~228k majority rows is avoided.

Inspiration: N. V. Chawla, K. W. Bowyer, L. O. Hall, W. P. Kegelmeyer, *SMOTE: Synthetic Minority Over-sampling Technique*, JAIR 16, 2002. Evaluation is modernized (PR-AUC primary, boosting models, leak-free pipelines).

---

#### Dataset

Working assumption until verified: [ULB / Kaggle European credit-card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Column | Meaning |
|---|---|
| `Time` | Seconds from the first transaction in the dump |
| `V1`–`V28` | PCA components (anonymized) |
| `Amount` | Transaction amount |
| `Class` | `0` legitimate, `1` fraud |

Typical size: 284,807 rows, ~492 frauds (~0.172%).

Place the CSV at `data/raw/creditcard.csv` (gitignored). Record SHA-256 in `data/README.md` when the file is obtained.

A **synthetic schema-identical sample** (Prompt 01) is generated so the team can develop without the full file.

**Do not commit** the full raw dataset or any real cardholder PII. The demo must never collect PAN, CVV, or names.

---

#### Repository layout (target)

```text
ML_Project2026/
├── configs/                 # dataset.yaml, experiment.yaml
├── data/
│   ├── raw/                 # gitignored full CSV
│   └── sample/              # small synthetic/dev CSV
├── src/                     # reusable ML code
│   ├── data/                # load, validate, split
│   ├── preprocessing/
│   ├── sampling/            # ROS, RUS, SMOTE, SMOTE+RUS
│   ├── models/
│   ├── evaluation/
│   └── utils/
├── scripts/                 # CLI: synthetic data, EDA, experiments
├── notebooks/               # thin exploration only
├── artifacts/               # metrics, figures, serialized models
├── backend/                 # FastAPI demo API
├── frontend/                # React + Vite dashboard
├── tests/                   # including anti-leakage tests
├── docs/
├── model_build/             # existing from-scratch SMOTE (to be migrated)
├── pyproject.toml
└── README.md
```

Reusable logic lives in `src/`. Notebooks must not contain the training loop. The web app consumes **saved artifacts**; it does not retrain models.

---

#### Status

**Early / Week 1.** Scaffolding, data contracts, synthetic generator, validation, split, and EDA come first. The 18-run matrix is Week 4. FastAPI + React start after model artifacts exist (Week 5).

Current tree still includes `model_build/` (from-scratch SMOTE). That sampler currently balances to 1:1 and must gain `sampling_strategy` before matrix runs.

---

#### Setup (once the package exists)

```bash
git clone https://github.com/RavenKing45/ML_Project2026.git
cd ML_Project2026

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -e ".[dev]"
```

Generate a small development dataset (unblocks the whole team):

```bash
python scripts/generate_synthetic_data.py
```

Validate and split:

```bash
python scripts/run_eda.py
pytest tests/test_data_and_split.py
```

Run the experiment matrix (after Waves 1–2):

```bash
python scripts/run_experiments.py --quick
```

Full 18-run study (CPU, overnight-class):

```bash
python scripts/run_experiments.py
```

Tests:

```bash
pytest
```

Commands above match the planned CLI. They will fail until the corresponding prompts are implemented — that is expected.

---

#### Demo application (later)

After models are serialized:

1. **Overview** — problem, imbalance, headline metrics  
2. **Comparison** — 18-run table, ROC / PR charts  
3. **SMOTE lab** — 2D interpolation illustration  
4. **Predict** — sample transactions or CSV batch (anonymized features only)

Stack: FastAPI + React / Vite / Tailwind. Frontend contains **no** ML logic.

---

#### Team (10 students, 5 squads)

| Squad | Owns | Branch prefix |
|---|---|---|
| 1 | Data, schema, EDA, synthetic generator |
| 2 | Preprocessing, ROS/RUS/SMOTE | 
| 3 | Models, CV, experiments | 
| 4 | Metrics, FastAPI | 
| 5 | React UI, Docker, deploy | 

Workflow: feature branches → `develop` (PR + review) → `main`. Do not push to `main` directly.

Implementation order is in the discussion PDF (Prompts 01–25) and summarized as:

1. Repo + configs + synthetic data  
2. Load / validate / split  
3. EDA  
4. Preprocessor  
5. Baselines  
6–9. ROS → RUS → SMOTE → SMOTE+RUS  
10. CV + leakage tests  
11–13. Metrics, 18-run runner, serialize  
14–21. API + UI  
22–25. Docker, tests, deploy, docs  

---

#### Tech stack

| Layer | Choice |
|---|---|
| ML | Python, pandas, scikit-learn, imbalanced-learn, LightGBM |
| Tracking | JSON + CSV under `artifacts/` (no MLflow server) |
| API | FastAPI, Pydantic |
| UI | React, TypeScript, Vite, Tailwind, Recharts |
| Deploy (demo) | Docker; frontend Vercel; API Render / Hugging Face Spaces |

---

#### Limitations (we will state these in the report)

- SMOTE samples are interpolations in PCA space, not real frauds.
- Public, anonymized, ~48-hour 2013 European dump; no merchant/velocity features.
- Offline metrics are not a bank production cost matrix.
- Concept drift and adversarial fraud are out of scope.

---

#### License and data
MIT License

Copyright (c) 2026 Pragna Jyoti Mandal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Code in this repo is for academic use. The credit-card CSV is **not** redistributed here; obtain it from the dataset provider and respect their terms.
