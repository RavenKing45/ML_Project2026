# An Empirical Study of Class-Imbalance Handling Techniques for Credit Card Fraud Detection

College group project: a **controlled ML experiment** on how class-imbalance treatments affect fraud detection, plus a **small demo web app** for the final presentation.

This repository is the implementation of the architecture in [`docs/ML project discussion.pdf`](docs/ML%20project%20discussion.pdf). Planning notes also live in [`docs/ML_RESEARCH_BLUEPRINT.md`](docs/ML_RESEARCH_BLUEPRINT.md).

---

## Research question

> How do data-level resampling techniques (random oversampling, random undersampling, SMOTE, and hybrid SMOTE + undersampling) and algorithm-level cost-sensitive learning compare against an unmodified baseline for highly imbalanced credit-card fraud detection?

We do **not** optimize for accuracy. Under ~0.17% fraud, a dummy “always legitimate” classifier is already ~99.8% accurate and catches **zero** fraud.

**Primary metric:** PR-AUC (Average Precision).  
**Secondary metrics:** Precision, Recall, F1, F2, ROC-AUC, FPR, FNR, confusion matrix, ROC and PR curves.

The study does **not** claim a winner in advance. The comparison table from a locked test set is the result.

---

## What we compare

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

## Methodology (non-negotiable)

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

## Dataset

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

## Repository layout (target)

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

## Status

**Early / Week 1.** Scaffolding, data contracts, synthetic generator, validation, split, and EDA come first. The 18-run matrix is Week 4. FastAPI + React start after model artifacts exist (Week 5).

Current tree still includes `model_build/` (from-scratch SMOTE). That sampler currently balances to 1:1 and must gain `sampling_strategy` before matrix runs.

---

## Setup (once the package exists)

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

## Demo application (later)

After models are serialized:

1. **Overview** — problem, imbalance, headline metrics  
2. **Comparison** — 18-run table, ROC / PR charts  
3. **SMOTE lab** — 2D interpolation illustration  
4. **Predict** — sample transactions or CSV batch (anonymized features only)

Stack: FastAPI + React / Vite / Tailwind. Frontend contains **no** ML logic.

---

## Team (10 students, 5 squads)

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

## Tech stack

| Layer | Choice |
|---|---|
| ML | Python, pandas, scikit-learn, imbalanced-learn, LightGBM |
| Tracking | JSON + CSV under `artifacts/` (no MLflow server) |
| API | FastAPI, Pydantic |
| UI | React, TypeScript, Vite, Tailwind, Recharts |
| Deploy (demo) | Docker; frontend Vercel; API Render / Hugging Face Spaces |

---

## Limitations (we will state these in the report)

- SMOTE samples are interpolations in PCA space, not real frauds.
- Public, anonymized, ~48-hour 2013 European dump; no merchant/velocity features.
- Offline metrics are not a bank production cost matrix.
- Concept drift and adversarial fraud are out of scope.

---

## License and data
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
