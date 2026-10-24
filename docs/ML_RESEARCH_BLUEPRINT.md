# An Empirical Study of Class-Imbalance Handling Techniques for Credit Card Fraud Detection

**Document type:** ML research and implementation blueprint  
**Audience:** project team and coding agents  
**Status:** planning only — do not treat this as experimental results

The ML/research component is the main focus. A small web application is a demonstration interface for the final presentation only.

We already understand SMOTE theoretically. This document does **not** teach SMOTE history or intuition. It specifies implementation planning, experimental methodology, pipeline architecture, and coding-agent prompts.

**Final question the experiments must answer (do not answer in advance):**

> Under a controlled experimental setup, which class-imbalance handling strategy provides the most useful trade-off between fraud detection performance and false-positive cost for our credit-card fraud detection models, and why?

---

# PART 1 — Final ML Research Blueprint

**Research question (operationalized):**  
Under a fixed dataset, fixed models, fixed split, and equal tuning budget, how do class-imbalance treatments change **PR-AUC**, **recall/precision at selected operating points**, and **false-positive burden** for credit-card fraud detection?

**What this project is:** a controlled empirical comparison of imbalance treatments.  
**What this project is not:** a Kaggle accuracy contest, a SMOTE tutorial, or a web-product build.

## Locked scientific decisions

| Decision | Choice | Why |
|---|---|---|
| Models | **Logistic Regression + XGBoost** | Linear vs nonlinear; enough diversity; keeps the matrix tractable |
| Core treatments | **None, class weights, ROS, RUS, SMOTE, SMOTE+RUS** | Covers “do nothing”, algorithm-level correction, naive sampling, informed oversampling, and hybrid |
| Deferred | ADASYN, Borderline-SMOTE, Random Forest | Incremental SMOTE variants / redundant tree model; stretch only |
| Outer protocol | Stratified 80/20 holdout + 5-fold Stratified CV **on train only** | Enough fraud cases in test (~100 if ~492 total); test never used for fit/resample/tune/threshold |
| Robustness add-on | Chronological split for **finalists only** | Detects temporal leakage that random splits hide |
| Primary metric | **Average Precision (PR-AUC)** | Threshold-free, positive-class focused; ROC-AUC is too optimistic here |
| Fair comparison | **Stage A defaults → Stage B equal-budget tuning** | Prevents “better sampler” being “more tuning luck” |
| Samplers in production | **Training-time only** | Deploy scaler + model, never SMOTE |
| Tracking | **JSONL + summary CSV** | Enough for a college project; no MLflow server |
| Existing SMOTE code | Keep, wrap, **concordance-check vs imblearn** | Course-owned implementation without making an unvalidated sampler the only engine |

## How the final question gets answered

We will **not** declare a winner from a single F1. We will report, for each model × treatment:

1. CV mean ± std PR-AUC
2. Untouched test PR-AUC + bootstrap CI
3. Test precision/recall/F1 at a threshold chosen on **out-of-fold train** predictions
4. FPR (false-positive cost proxy) and FNR (missed-fraud cost proxy) at that threshold
5. Stability across folds
6. Chronological-split delta for the top two pipelines

The “most useful trade-off” is the pipeline that is **best or tied on PR-AUC**, **stable**, and **acceptable on FPR at a recall-oriented operating point**. If those conflict, we report a Pareto set, not a fake unique winner.

---

# PART 2 — Experimental Methodology

## Canonical data flow (non-negotiable)

```
raw CSV
  → schema/quality validation
  → EDA on FULL data for description only (no model decisions from test labels beyond describing the file)
  → freeze split indices with seed
        TRAIN (fit/resample/tune/threshold)
        TEST  (locked until final evaluation)

TRAIN only:
  StratifiedKFold (k=5)
    for each fold:
      fit preprocessor on fold-train
      transform fold-train and fold-val
      fit sampler on fold-train only          # NEVER on fold-val
      fit model on resampled fold-train
      score fold-val in original (imbalanced) distribution
  hyperparameter search uses this nested structure
  threshold selected from out-of-fold train probabilities

TEST (once per completed pipeline):
  transform with preprocessor fitted on full TRAIN
  predict with model fitted on full TRAIN (sampler used in that fit only)
  apply frozen threshold
  compute metrics
```

## Where every operation lives

| Operation | Fit on | Apply to | Forbidden |
|---|---|---|---|
| Duplicate audit | full file, then again inside train | report only | dropping test dups using train-derived rules that change labels inconsistently without logging |
| Missing-value imputation | train | train/val/test | fitting imputer on full data |
| Scaler / log1p | train | train/val/test | scaling before split |
| Sampler (ROS/RUS/SMOTE/…) | each CV train fold; then full train for final fit | **never** val/test | resampling before split; resampling the whole dataset; resampling validation |
| Class weights | computed from current training labels | model `fit` | using test prevalence |
| Hyperparameter selection | CV on train | frozen for test | peeking at test |
| Threshold | OOF train probabilities | frozen for test | tuning threshold on test |
| Final metrics | — | test | iterating after seeing test |

EDA may plot the full dataset for a data-description chapter. **Model choices that depend on distributions (e.g. log-Amount)** must be justified on **train** and confirmed, not re-tuned, on test.

## Required libraries / abstractions

- `sklearn.model_selection.train_test_split(..., stratify=y, random_state=SEED)`
- `sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)`
- `sklearn.pipeline.Pipeline` for **inference** (preprocess + clf)
- `imblearn.pipeline.Pipeline` for **training** (`preprocess → sampler → clf`)
- `RandomizedSearchCV` wrapping the **imblearn** pipeline, with `scoring='average_precision'`
- Do **not** hand-resample a global `X_train_smote` and then run vanilla `GridSearchCV` on it

`GridSearchCV` is allowed only for tiny LR grids. Default: `RandomizedSearchCV` with a **fixed `n_iter` per cell**.

## Two-stage experiment protocol

**Stage A — fair default comparison (do this first)**  
All 2 × 6 cells, published/default hyperparameters, same seed, same split. This answers the research question with minimal confounding.

**Stage B — equal-budget tuning**  
Same 12 cells, `RandomizedSearchCV`, same `n_iter` (recommend **12**), same inner 5-fold. Sampler hyperparameters (`k_neighbors`, `sampling_strategy`, RUS ratio) are searched **inside** the pipeline.

**Stage C — finalists only**  
Top 2–3 pipelines: threshold analysis, bootstrap CIs, McNemar pairwise, chronological robustness split, serialize inference artifact.

Do not aggressively tune the baseline before Stage A exists.

## Leakage tests (must be automated)

1. Test row indices never appear in any sampler `fit`.
2. `hash`/`checksum` of `X_test, y_test` unchanged after every experiment.
3. For NN samplers: scaler fitted before SMOTE, and SMOTE not in the serialized inference pipeline.

---

# PART 3 — Dataset + EDA Plan

## Dataset contract (replaceable)

Do **not** hardcode “this is the Kaggle `creditcard.csv`” in the trainer. Use a **dataset adapter**:

- `configs/data.yaml`: path, filename, target column, ID columns, feature allowlist/denylist
- `src/fraud_ml/data/loader.py`: load table
- `src/fraud_ml/data/schema.py`: expected columns + dtypes + target domain `{0,1}`
- `src/fraud_ml/data/profile.py`: row count, class counts, imbalance ratio

If the file matches the common schema (`Time`, `V1`–`V28`, `Amount`, `Class`), use the **creditcard profile**. If not, validation should fail loudly with a schema diff, not silently rename columns.

**Until the file is in `data/raw/` and validated, treat the 31-column Kaggle fraud set as a working assumption, not a fact.**

## Working assumption (Kaggle-style)

- ~284,807 rows, 30 features + `Class`
- `Class`: 0 legitimate, 1 fraud; ~492 frauds, **≈0.172%**, ratio ≈ **578:1**
- `V1`–`V28`: PCA scores; not human-interpretable; already roughly centered
- `Time`: seconds from first transaction in the dump (two-day window in the public set)
- `Amount`: raw transaction amount, right-skewed

## Validation checks (blocking)

- Required columns present; no extra surprise target
- `Class` in `{0,1}`; both classes present
- No NaN in features/target (this public set is usually complete; still check)
- Duplicate rows counted
- Duplicate policy: drop exact duplicates on the **full raw file before split**, because they are data errors, not time-evolving labels — **log the count**. If duplicates include both classes for identical features, fail validation.
- `Amount >= 0`
- `Time >= 0` and monotonic-ish (not required to be unique)
- Feature matrix numeric

## EDA — only plots that change beliefs

Store under `reports/eda/` (`figures/` + `eda_summary.json`). Notebook: `notebooks/01_eda.ipynb` as a thin caller of `src/fraud_ml/eda/`.

| Plot / table | Why | Insight to extract | Presentation? |
|---|---|---|---|
| Class counts + pie/bar | Defines the problem | Fraud %, ratio | **Essential** |
| Amount histogram (log x) by class | Whether Amount separates classes | Overlap vs shift; justifies log1p for LR | **Essential** |
| Amount box/violin by class | Outliers and typical fraud size | Fraud not simply “large amounts” | Essential |
| Time vs fraud rate (hourly bins of `Time/3600`) | Temporal structure | If fraud clusters in time, random split may leak | Essential |
| Correlation heatmap of V’s + Amount (train) | Multicollinearity for LR | PCA V’s should be weakly correlated; Amount/Time vs V’s | Optional (1 slide) |
| Boxplots of a few V’s by class (e.g. top \|AUC\| univariate) | Signal exists in anonymized features | Which directions LR might use | Optional |
| Duplicate / missing table | Data trust | 0 missing expected | Appendix |
| Pair of 2D projections (optional subsample): class overlay | SMOTE interpolation assumption | Minority points not a single blob | Optional stretch |

**Do not:** pairplots of 30 features, 28 V histograms as a gallery, 3D rotating PCA movies.

**Univariate screening (train only):** ROC-AUC of each feature vs `Class` to pick “example V features” for plots — not for dropping features in v1. **v1 keeps all features.** Feature dropping is a later ablation, not a prerequisite.

## Time-related analysis (required, even if we later drop Time)

Bin `Time` into hours from start. Plot fraud **rate** and fraud **count** per bin.  
Extract: is prevalence non-stationary? If yes, chronological robustness is mandatory.

---

# PART 4 — Preprocessing Architecture

## Principles

Transformations exist to make **the model and the sampler valid**, not because blog posts say “always scale.”

### `V1`–`V28`

Already PCA features. Keep all. Do not PCA again.  
Scaling again is still **required** for LR and for **Euclidean SMOTE**, because `Amount`/`Time` live on different scales. Fit **one numeric scaler on all modeled numeric columns** so SMOTE distance is not dominated by `Amount` or `Time`.

### `Amount`

- **Log1p:** justified if train skew is high (expected). Helps LR; largely irrelevant to XGBoost.
- **Decision rule:** if train `|skewness| > 1`, apply `log1p` inside the pipeline for **all models** so sampler + LR share one feature space. Trees can tolerate it.
- Do not winsorize unless EDA shows data errors, not heavy tails.

### `Time`

- `Time` is **dataset-elapsed seconds**, not a production feature.
- **Research default (v1):** keep `Time` as a numeric column, scaled with the rest, because dropping it changes the feature set relative to most published notebooks and confounds comparisons.
- **Inference note:** the demo API must document that `Time` means “seconds from dataset origin,” or accept `Time` as an input the user supplies.
- **Open until EDA:** if hourly fraud rate is flat and univariate AUC ≈ 0.5, add a **sensitivity run** without `Time` on finalists.
- Do **not** convert to cyclic hour-of-day unless we know the clock origin. The public dataset’s t=0 is not guaranteed to be midnight.

### Scaler choice

| Scaler | Use when |
|---|---|
| **RobustScaler** | Default for this project: `Amount` outliers, fraud in tails, SMOTE uses neighbors |
| StandardScaler | Sensitivity run if RobustScaler + LR underperforms badly |
| No scaler | Never for LR or SMOTE; XGBoost-only ablation is optional |

**Default preprocessor:** `Amount -> log1p` (FunctionTransformer) → `RobustScaler` on all features used in the model.

### Missing values

If none: no imputer. If any appear in a replacement dataset: `SimpleImputer(median)` fitted on train, inside the pipeline, **before** sampling.

### Order (training)

```
split
→ [ColumnTransformer / log1p Amount]
→ scaler.fit(train_fold)
→ sampler.fit_resample(scaled train_fold, y_fold)
→ model.fit(resampled)
```

SMOTE **after** scaling is mandatory. Unscaled `Amount` would dominate k-NN.

### Order (inference)

```
new transaction
→ log1p Amount
→ scaler.transform
→ model.predict_proba
→ threshold
```

No sampler.

---

# PART 5 — Experiment Matrix

## Final recommended matrix (core)

| | None (imb.) | Class weight | ROS | RUS | SMOTE | SMOTE+RUS |
|---|---|---|---|---|---|---|
| Logistic Regression | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| XGBoost | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

**12 cells.** This is the scientific study.

### Why this set, and not more

- **None** = reference.
- **Class weight / `scale_pos_weight`** = the important alternative to resampling (no synthetic points, no discarded majority).
- **ROS** = naive oversampling control; if SMOTE ≈ ROS, interpolation is not adding value.
- **RUS** = speed/majority-noise control; expected high variance, possible recall↑ precision↓.
- **SMOTE** = informed oversampling.
- **SMOTE+RUS** = practical hybrid so we do not synthesize hundreds of thousands of frauds.

**Out of core:** ADASYN, Borderline-SMOTE (SMOTE family variants), RF (overlaps XGBoost for this question), LightGBM (swap for XGBoost only if runtime forces it).

**Stretch (Phase 12b, only if Stage A done):** ADASYN and Borderline-SMOTE on **XGBoost only** (2 extra cells), to see if neighborhood-adaptive oversampling beats vanilla SMOTE.

## Compute budget (approximate, 284k rows, laptop/CPU)

Assume train ≈ 228k, fraud ≈ 394.

| Item | Estimate |
|---|---|
| Stage A | 12 fits + 5-fold scores ≈ 12 × 5 = **60** train-eval cycles |
| Stage A wall time | LR ~1–3 min/cell; XGBoost ~5–20 min/cell; **~2–6 hours** |
| Stage B | 12 cells × 12 draws × 5 folds = **720** fits |
| Stage B wall time | **overnight CPU** (≈ 8–20 h). Use `n_jobs` carefully with RAM |
| SMOTE `sampling_strategy=1.0` | **avoid as default** — ~227k synthetic rows, RAM/time spike |

**Mitigations:** default SMOTE/ROS strategy **0.1** (minority/majority), not 1.0; RUS to a defined majority reduction; XGBoost `n_estimators` capped in Stage A (e.g. 200); optional `n_jobs=1` for SMOTE+XGB to avoid RAM thrash.

## Sampling ratios (not arbitrary)

Prevalence ≈ 0.0017. Full balance (1.0) is an extreme intervention.

**Stage A frozen settings (same for both models):**

| Treatment | Setting | Rationale |
|---|---|---|
| ROS | `sampling_strategy=0.1` | ~10% fraud in train; large lift without cloning to 50/50 |
| RUS | `sampling_strategy=0.1` | majority downsampled so minority/majority = 0.1; keeps most frauds, drops ~90%+ of majority |
| SMOTE | `sampling_strategy=0.1`, `k_neighbors=5` | same target prevalence as ROS so SMOTE vs ROS is isolated |
| SMOTE+RUS | SMOTE to `0.05`, then RUS to `0.1` | synthesize a moderate minority, then reduce majority; avoids 1:1 fantasy data |
| Class weight | LR `class_weight="balanced"`; XGB `scale_pos_weight = n_neg/n_pos` on **current train fold** | standard inverse-frequency |

**Why 0.1:** it is about **60×** more balanced than 0.0017, but not 50/50. 50/50 would let the model ignore base rate and typically explode FPs. Stage B may search `{0.05, 0.1, 0.25}` with **equal budget**.

**SMOTE+RUS ratio justification method:** treat `(smote_strategy, rus_strategy)` as hyperparameters in Stage B over the grid:

- SMOTE ∈ `{0.05, 0.1, 0.2}`
- RUS ∈ `{0.1, 0.25}` with constraint `rus_strategy >= smote_strategy`

Pick by inner CV PR-AUC. Stage A uses the frozen pair above so the paper has a pre-registered default.

---

# PART 6 — Model Architecture

## Why two models are enough

The research factor is **imbalance treatment**, not algorithm sweep. Two estimators that **react differently** to resampling are enough:

1. **Logistic Regression (L2)** — linear, scaled-feature dependent, interpretable coefficients, cheap, sensitive to class balance and to synthetic density around linear boundaries.
2. **XGBoost** — nonlinear, interaction-capable, strong tabular default, `scale_pos_weight` native, less in need of scaling, still affected by duplicated/synthetic minorities via leaf statistics.

Random Forest is a weaker, slower cousin of boosting here. Skip it in core. If XGBoost environment pain appears, LightGBM is the substitute, not a third model.

| Criterion | LR | XGBoost |
|---|---|---|
| Interpretability | Coefficients (on scaled/log features) | SHAP later optional; not required for v1 |
| Cost | Low | Medium |
| Imbalance sensitivity | High | Medium (weights help) |
| Scaling | Required | Optional but shared pipeline for SMOTE |
| SMOTE compatibility | Yes, after scale | Yes, after scale |
| Nonlinear | No | Yes |
| Fraud suitability | Strong baseline / audit | Strong detector |

**Stage A hyperparameters (frozen):**

- LR: `max_iter=1000`, `solver="lbfgs"`, `C=1.0`
- XGBoost: `n_estimators=200`, `max_depth=4`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, `eval_metric="logloss"`, `n_jobs=4`, `random_state=SEED`
- `tree_method="hist"`

---

# PART 7 — Imbalance Handling Architecture

## Code shape

```text
src/fraud_ml/sampling/
  base.py          # Protocol: fit_resample(X, y)
  none.py          # identity
  ros.py
  rus.py
  smote_imblearn.py
  smote_project.py # wraps existing from-scratch SMOTE
  smote_rus.py     # Pipeline of SMOTE then RUS
  weights.py       # helpers for class_weight / scale_pos_weight
  registry.py      # name → sampler factory from config
```

Training pipeline: `imblearn.pipeline.Pipeline([("pre", preprocessor), ("sampler", sampler), ("clf", clf)])`.

For **class_weight** and **none**, sampler is `passthrough`.

### Per-technique spec

**1. None**

- Entry: no sampler.
- Risk: model collapses to majority.
- Expected: high accuracy, high ROC-AUC, **low recall**, modest PR-AUC.
- Eval: original fold-val distribution.

**2. Class weights**

- Entry: `clf` param, not a sampler.
- LR: `class_weight="balanced"`.
- XGB: `scale_pos_weight` from fold counts.
- Risk: over-predict fraud, FPR↑.
- Expected: recall↑ vs none, no synthetic artifacts.
- Preprocess: same scaler (LR).

**3. ROS**

- `RandomOverSampler(sampling_strategy=0.1, random_state=SEED)`
- After scaling (order consistency; scaling is affine so ROS-before-scale is equivalent for RobustScaler, but **keep one order**: scale then sample).
- Risk: overfitting exact minority copies, especially LR less than trees.
- Expected: trees may memorize duplicates.

**4. RUS**

- `RandomUnderSampler(sampling_strategy=0.1, random_state=SEED)`
- Risk: discarded majority = lost information, high variance across seeds.
- Expected: fast, recall↑, precision↓.
- Report fold std carefully.

**5. SMOTE**

- Default engine for the paper: **imblearn `SMOTE`** (standard, fast).
- Project engine: existing `SMOTE` class, **extended** with `sampling_strategy` (currently it always fills to 1:1 — that must change before experiments).
- Concordance experiment: on a **stratified 20k subsample of train**, compare imblearn vs from-scratch with same `k`, seed; report PR-AUC of LR on that subsample (not the main table).
- Params Stage A: `k_neighbors=5`, `sampling_strategy=0.1`, `random_state=SEED`
- Stage B search: `k_neighbors ∈ {3,5,7}`, strategy ∈ `{0.05,0.1,0.25}`
- Risk: interpolation in PCA space ≠ real fraud; can bridge class boundaries; k-NN meaningless without scaling.
- Guard: `k_neighbors < n_minority` in the fold; if not, skip/fail the fold with a logged error.

**6. SMOTE + RUS**

- Use explicit `SMOTE` then `RandomUnderSampler` for interpretability. Do **not** silently substitute SMOTEENN/SMOTETomek.
- Stage A: SMOTE `0.05` → RUS `0.1`.
- Risk: two sources of randomness; more knobs.
- Expected: less RAM than SMOTE-to-balance; between SMOTE and RUS.

All samplers: `random_state=SEED`.

---

# PART 8 — Cross-Validation + Tuning Strategy

- **Split:** `test_size=0.2`, `stratify=y`, `random_state=42` (config). Persist index files: `data/processed/split_indices.json`.
- **CV:** `StratifiedKFold(5, shuffle=True, random_state=42)`.
- **Repeated CV:** not in core. If Stage A fold std is large (especially RUS), run `RepeatedStratifiedKFold(n_splits=5, n_repeats=2)` **only for the two most unstable cells**.
- **Scorer:** `average_precision` (PR-AUC).
- **Search:** `RandomizedSearchCV` on imblearn Pipeline, `n_iter=12`, `refit=True` (refit on full train with sampler).
- **Do not** use accuracy.
- **Inner vs outer:** college-scope **is not** full nested CV. Unbiased-ish design = holdout test + CV on train. Acknowledge that HP selection is slightly optimistic **on train CV**, not on test.

**Stage B search spaces (small):**

- LR: `C` log-uniform `1e-2..1e2`; `penalty=l2` only.
- XGB: `max_depth` 3–6; `learning_rate` 0.03–0.2; `n_estimators` 100–400; `min_child_weight` 1–10; `subsample` 0.7–1.0.
- Sampler params as in Part 7, only for cells that use them.

**Primary optimization metric: PR-AUC.**  
Recall-only optimization would pick “predict almost everyone fraud.” F1 is threshold-dependent and must not drive RandomizedSearchCV (which uses probabilities via AP). ROC-AUC will be reported, never the `refit` score.

---

# PART 9 — Evaluation Framework

One function: `src/fraud_ml/evaluation/metrics.py` → `compute_metrics(y_true, y_prob, threshold) -> dict`.

**Primary:** `average_precision`  
**Secondary:** precision, recall, F1, ROC-AUC, FPR, FNR, support, TN/FP/FN/TP  
**Plots:** confusion matrix, ROC, PR curve — `reports/figures/{experiment_id}/`

**Why PR-AUC:** with ~99.8% negatives, ROC-AUC can look excellent while the detector is useless at the top of the ranking. PR-AUC is the area under precision as a function of recall and is dominated by positive-class ranking quality.

**Accuracy:** log it if you want to show it is misleading; never rank experiments by it.

**Standardized experiment record:** every run writes the same metric keys. Missing keys fail CI.

**Threshold component (`src/fraud_ml/evaluation/threshold.py`):**

- Inputs: OOF `y_true`, `y_prob` from train CV
- Sweep thresholds `np.linspace(0.01, 0.99, 99)` (or unique probabilities)
- Curves: precision, recall, F1, FPR vs threshold
- **Selection rule (pre-registered):** choose threshold maximizing **F1 on OOF**, subject to **precision ≥ 0.1** if that constraint is feasible; if not, max F1 unconstrained and document it
- **Also report (not used for selection):** recall at precision=0.2; precision at recall=0.5 and 0.8
- Freeze `threshold.json` with the experiment
- Apply to test **once**

Do not use 0.5 as the operating point for fraud.

---

# PART 10 — Statistical / Empirical Analysis

## What to report for every cell

- CV PR-AUC: mean, std, min, max (5 folds)
- Test: PR-AUC, ROC-AUC, P/R/F1 at frozen threshold, FPR, FNR, confusion counts
- Runtime, peak sampling size

## What statistical tests are worth it

With **k=5**, p-values on fold scores are underpowered. Do not fill the paper with tests.

**Recommended:**

1. **Main:** table of mean CV PR-AUC ± std, plus test PR-AUC.
2. **Bootstrap CI** of test Average Precision (resample **test pairs** `(y, p)`, 1000 times). This is a CI for this frozen test set, not a super-population of datasets.
3. **McNemar** on test predictions for **finalist pairs** (same test rows, binary decisions at each pipeline’s own frozen threshold). Use this when arguing “pipeline A errs differently from B.”
4. **Paired fold comparison** (Wilcoxon signed-rank) **only** if you later add RepeatedStratifiedKFold (n≥10 paired scores). Skip with n=5.

**Skip DeLong** unless a reviewer insists on ROC-AUC differences; ROC is not the primary claim.

**Interpretation language:** “On this dataset and protocol, treatment X improved mean CV PR-AUC from a to b for model M; test CIs overlap / do not overlap.” Never “SMOTE is always better.”

---

# PART 11 — Experiment Tracking + Reproducibility

## Tracking: JSONL + CSV

Not MLflow (server/UI overhead). Not SQLite unless JSONL becomes painful.

- `experiments/results/runs.jsonl` — one JSON object per run
- `experiments/results/summary.csv` — flattened table for the paper
- `experiments/results/{run_id}/` — metrics.json, cv_folds.json, plots, config snapshot, threshold.json

**Record:** `run_id`, git SHA, timestamp, model, imbalance, preprocess config, HP, sampling params, seed, CV spec, split hash, train/test checksums, metrics, runtime, artifact paths, notes.

## Reproducibility

- `SEED=42` in `configs/default.yaml`
- `numpy`, `random`, `PYTHONHASHSEED`, sklearn, xgboost seeds
- Dataset: `data/raw/README.md` with source URL, license, SHA256 of the CSV
- `pyproject.toml` **preferred** over `requirements.txt` as the single dependency source; export `requirements.txt` only if a host needs it
- Config per experiment: `configs/experiments/*.yaml`
- Save `environment` via `pip freeze` snapshot in each run folder (optional but cheap)
- XGBoost: `deterministic` is not fully guaranteed on all hardware; document that

Existing `model_build/pyproject.toml` should be **replaced/merged** into a root project package (`fraud-ml` / `fraud_ml`).

---

# PART 12 — ML Repository Architecture

```text
ML_Project2026/
├── data/
│   ├── raw/                    # gitignored CSV
│   ├── processed/              # split indices, checksums
│   └── README.md               # how to obtain data + SHA256
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_analysis_results.ipynb   # reads experiments/, no training
│   └── 03_presentation_figures.ipynb
├── src/
│   └── fraud_ml/               # installable package
│       ├── data/
│       ├── eda/
│       ├── preprocessing/
│       ├── sampling/
│       ├── models/
│       ├── evaluation/
│       ├── experiments/        # run_one, runner, tracking
│       ├── inference/          # load artifact, predict
│       └── utils/
├── configs/
│   ├── default.yaml
│   ├── data.yaml
│   └── experiments/
│       ├── lr_none.yaml
│       ├── lr_weights.yaml
│       ├── ...
├── experiments/results/
├── models/                     # inference joblib only
├── reports/
│   ├── eda/
│   ├── figures/
│   └── tables/
├── tests/
├── scripts/
│   ├── run_eda.py
│   ├── run_experiment.py
│   ├── run_matrix.py
│   └── train_final.py
├── app/                        # secondary
│   ├── api/                    # FastAPI
│   └── web/                    # Vite+React OR Streamlit
├── model_build/                # migrate then delete
├── docs/
│   └── ML_RESEARCH_BLUEPRINT.md
├── pyproject.toml
├── .gitignore
└── README.md
```

**Notebook vs `src/`:** notebooks call library functions and display figures. If a notebook contains a training loop, that is a bug.

**Web app** imports `fraud_ml.inference` only, never `sampling`.

**Migrate** `model_build/src/smote_from_scratch/` → `src/fraud_ml/sampling/smote_from_scratch.py`.

---

# PART 13 — Group Work Division (~10 students, 5 pairs)

Everyone reads this blueprint and runs Stage A at least once.

| Group | People | Owns | Files | Deliverable | Depends on | Learns | Time |
|---|---|---|---|---|---|---|---|
| **G1 Data/EDA** | 2 | loader, schema, EDA, split indices | `src/fraud_ml/data`, `eda`, `notebooks/01_eda`, `data/README` | validation report, EDA figures, frozen split | repo skeleton | pandas, leakage of EDA vs modeling | Week 1–2 |
| **G2 Preprocess** | 2 | transformers, scaler choice, column contract | `src/fraud_ml/preprocessing` | sklearn preprocessor + tests | G1 schema | sklearn Pipeline, leakage | Week 2–3 |
| **G3 Sampling** | 2 | all samplers, imblearn pipe, from-scratch SMOTE `sampling_strategy`, concordance | `src/fraud_ml/sampling` | samplers + “SMOTE not on test” tests | G2 | imblearn, k-NN scaling | Week 3–4 |
| **G4 Models/tuning** | 2 | LR, XGB, Stage A/B runner, HP spaces | `src/fraud_ml/models`, `configs/experiments` | 12 Stage A runs | G2–G3, eval API | XGBoost, RandomizedSearchCV | Week 4–6 |
| **G5 Eval/analysis/demo** | 2 | metrics, threshold, tracking, tables, API+UI, final train | `evaluation`, `experiments`, `app/`, `scripts/train_final.py` | comparison tables, API, slides figures | G4 results | PR-AUC, FastAPI | Week 5–8 |

**Integration points:** G3 sampler must drop into G4 `imblearn.pipeline`; G5 `compute_metrics` is the only scoring API G4 may use.

**Cross-training:** one joint session after Stage A: every student explains one cell of the matrix.

---

# PART 14 — Implementation Roadmap

| Phase | Objective | Prereq | Deliverable | Complexity | DoD |
|---|---|---|---|---|---|
| **0** Repo + env | package, configs, gitignore, seeds | none | installable `fraud_ml` | S | `pip install -e .` works |
| **1** Data + validation | load + schema + checksum | 0, CSV obtained | `validate_data` CLI | S | tests pass on fixture CSV |
| **2** Split freeze | stratified 80/20 indices | 1 | `split_indices.json` | S | checksums saved |
| **3** EDA | planned plots on train (+ descriptive full) | 2 | `reports/eda` | M | notebook runs top-bottom |
| **4** Preprocess | log1p + RobustScaler pipeline | 3 (Amount skew confirmed) | `build_preprocessor()` | M | no-fit on test tests |
| **5** Metrics + tracking | `compute_metrics`, jsonl writer | 0 | eval module | S | unit tests vs sklearn |
| **6** Baseline LR none | Stage A first cell | 4–5 | run folder + metrics | M | test checksum unchanged |
| **7** Baseline XGB none | second cell | 6 | same | M | same |
| **8** Weights | class_weight cells | 6–7 | 2 runs | S | no sampler in pipe |
| **9** ROS / RUS | 4 cells | 8 | 4 runs | M | sampler train-only test |
| **10** SMOTE + SMOTE-RUS | 4 cells + extend from-scratch strategy | 9 | 4 runs + concordance note | L | k_neighbors guard |
| **11** Matrix runner | Stage A table | 10 | `summary.csv` | M | 12 rows |
| **12** Stage B tuning | equal n_iter | 11 | tuned table | L | equal budget logged |
| **13** Threshold | OOF sweep | 12 | threshold plots | M | threshold not from test |
| **14** Stats + finalists | CI, McNemar, chrono split | 13 | analysis notebook | M | limitations written |
| **15** Serialize inference | joblib sklearn pipe | 14 | `models/final_inference.joblib` | S | no SMOTE step inside |
| **16** FastAPI | `/predict`, `/metrics` | 15 | OpenAPI | M | integration test |
| **17** Minimal UI | 4 pages | 16 | demo | M | no ML in frontend |
| **18** Docker (optional) | compose api+ui | 17 | Dockerfile | S | local `compose up` |
| **19** Tests hardening | leakage suite | always | CI optional | M | critical tests green |
| **20** Docs + slides artifacts | export figures | 14–17 | `reports/presentation/` | S | list in Part 19 |

Tuning is **after** the default matrix, not before ROS/SMOTE. Evaluation framework comes **before** extra models. All treatments at default HP before any GridSearch.

---

# PART 15 — Claude Coding-Agent Prompts

## Shared preamble (paste at the top of every prompt)

```text
You are implementing a college empirical-ML project on imbalance handling for fraud detection.
Rules:
1. Inspect the repository before changing anything.
2. Reuse existing components; do not rewrite working modules.
3. Never change the research methodology silently (PR-AUC primary, samplers train-only, stratified split).
4. Never apply ROS/RUS/SMOTE/ADASYN to validation or test data.
5. Never resample or scale-fit before the train/test split.
6. Training uses imblearn.pipeline.Pipeline; inference uses sklearn Pipeline without samplers.
7. Reproducible seeds from configs/default.yaml.
8. Type hints, modular functions, no absolute paths.
9. Put reusable logic in src/fraud_ml/; notebooks stay thin.
10. Add/run tests for what you touched.
11. Do not add MLflow, auth, databases, or extra models.
12. Do not expand the experiment matrix.
13. After finishing: report files changed, tests run, unresolved issues.
14. Ask a question only if blocked by an architectural fork not already decided in METHODOLOGY.
```

Follow `docs/ML_RESEARCH_BLUEPRINT.md` as the methodology source of truth.

---

### PROMPT 01 — Initialize ML repository and environment

- **Context:** Empty-ish repo; leftover `model_build/` SMOTE package.
- **Objective:** Root installable package, configs, folders, gitignore.
- **Prerequisites:** None.
- **Inspect:** `.gitignore`, `model_build/pyproject.toml`, `model_build/src/smote_from_scratch/`.
- **Create:** `pyproject.toml` (pandas, numpy, scikit-learn, imbalanced-learn, xgboost, matplotlib, pyyaml, joblib, fastapi, uvicorn, pydantic, pytest); `src/fraud_ml/__init__.py`; `configs/default.yaml` (`seed: 42`, `test_size: 0.2`, `n_splits: 5`, `primary_metric: average_precision`); `data/README.md`; directory placeholders; methodology pointer in README.
- **Modify:** `.gitignore` (keep data/raw, models artifacts, experiments/results large files).
- **Requirements:** `pip install -e ".[dev]"`; do not delete from-scratch SMOTE yet.
- **Constraints:** ML-first; `app/` empty dirs only.
- **Tests:** `tests/test_imports.py`.
- **Acceptance:** package imports; seed config loads.
- **Do not change:** SMOTE algorithm math.
- **Report:** tree, how to install.

---

### PROMPT 02 — Dataset loading and validation

- **Objective:** Adapter-style loader + schema validation + checksum.
- **Inspect:** `configs/`, `data/README.md`.
- **Create:** `configs/data.yaml`; `src/fraud_ml/data/loader.py`, `schema.py`, `validate.py`; `tests/fixtures/tiny_fraud.csv` (tiny synthetic 0/1 data with Time, V1–V3, Amount, Class — not 28 V’s required in fixture, schema must support `feature_prefix: V` + extras); `tests/test_validate.py`.
- **Requirements:** detect target `Class`; fail on missing columns, invalid labels, negative Amount; compute class ratio; SHA256.
- **Leakage:** no split yet.
- **Acceptance:** CLI `python -m fraud_ml.data.validate --help` or `scripts/validate_data.py`.
- **Do not:** download the dataset from the internet if missing; error with instructions.

---

### PROMPT 03 — Freeze split + EDA pipeline

- **Objective:** Stratified split indices; EDA functions + notebook.
- **Prerequisites:** Prompt 02.
- **Create:** `src/fraud_ml/data/split.py`; `src/fraud_ml/eda/*.py`; `notebooks/01_eda.ipynb`; `scripts/run_eda.py`.
- **Plots:** class bar, Amount-by-class (log), time-bin fraud rate, missing/dup table, univariate AUC table on **train**.
- **Leakage:** univariate AUC and scaler-free stats for modeling decisions use train indices only; a clearly labeled “full-data descriptive” section is allowed for row counts.
- **Tests:** split is stratified, disjoint, reproducible.
- **Do not:** train models.

---

### PROMPT 04 — Preprocessing

- **Objective:** `build_preprocessor(config) -> sklearn Transformer`.
- **Create:** `src/fraud_ml/preprocessing/pipeline.py`; tests that fit on train slice only.
- **Requirements:** log1p Amount (configurable); RobustScaler all numeric; passthrough consistent feature names (`get_feature_names_out` if possible).
- **Leakage:** test `transform` does not change scaler center when called twice; fitting on train+test must not happen in any script.
- **Do not:** drop Time unless config `drop_time: false` default.

---

### PROMPT 05 — Evaluation framework (before more models)

- **Objective:** metrics + plots + threshold sweep on synthetic arrays.
- **Create:** `src/fraud_ml/evaluation/metrics.py`, `plots.py`, `threshold.py`; tests vs sklearn `average_precision_score`.
- **Requirements:** standardized dict keys listed in Part 9; never compute accuracy as primary.
- **Do not:** read the real test set.

---

### PROMPT 06 — Experiment tracking

- **Objective:** JSONL writer + run directory layout.
- **Create:** `src/fraud_ml/experiments/tracking.py`.
- **Record fields:** Part 11.
- **Tests:** write/read roundtrip.
- **Do not:** add MLflow.

---

### PROMPT 07 — Training engine (imblearn pipeline factory)

- **Objective:** `build_train_pipeline(model, sampler, preprocessor)` and `build_inference_pipeline(preprocessor, model)`.
- **Create:** `src/fraud_ml/experiments/pipelines.py`; `src/fraud_ml/models/logistic.py`, `xgboost.py`.
- **Tests:** inference pipeline steps contain no sampler; train pipeline does for SMOTE name.
- **Do not:** run the full dataset if too heavy; use fixture.

---

### PROMPT 08 — Baseline Logistic Regression (none)

- **Objective:** first real Stage A run script.
- **Create:** `configs/experiments/lr_none.yaml`; `scripts/run_experiment.py`.
- **Requirements:** CV on train, fit on train, evaluate test **once**, save run.
- **Leakage tests:** test checksum before/after.
- **Do not:** tune C.

---

### PROMPT 09 — Baseline XGBoost (none)

- **Same as 08** with `xgb_none.yaml`, Stage A defaults from Part 6.
- **Do not:** add RF/LightGBM.

---

### PROMPT 10 — Class weights

- **Create:** `lr_weights.yaml`, `xgb_weights.yaml`; `src/fraud_ml/sampling/weights.py` computing `scale_pos_weight` from y_train inside a custom estimator wrapper or pipeline callback.
- **Requirement:** weights from **current training labels**, not full dataset.
- **Do not:** resample.

---

### PROMPT 11 — Random Oversampling

- **Sampler:** `RandomOverSampler`, strategy 0.1, after preprocess in imblearn Pipeline.
- **Tests:** resampled fold has higher minority count; val fold length unchanged if you unit-test a CV loop helper.
- **Do not:** apply ROS globally.

---

### PROMPT 12 — Random Undersampling

- **Analogous to 11**, `RandomUnderSampler`, strategy 0.1.
- **Tests:** majority reduced; all minority retained when strategy defined as minority/majority ratio.

---

### PROMPT 13 — SMOTE (imblearn) + extend from-scratch sampler

- **Inspect:** `model_build/src/smote_from_scratch/sampler.py`.
- **Migrate** into `src/fraud_ml/sampling/` and add `sampling_strategy` (float, minority/majority after resample, sklearn/imblearn semantics). Default 0.1 in configs; do not keep implicit 1:1 as default.
- **Main experiments use imblearn SMOTE** in Stage A.
- **Add** `scripts/smote_concordance.py` on max 20k train subsample.
- **Tests:** k too large raises; strategy respected; from-scratch still interpolates on a 2D toy set.
- **Do not:** SMOTE the test set in concordance script.

---

### PROMPT 14 — SMOTE + Random Undersampling

- **Implement** as two steps in imblearn Pipeline: SMOTE(0.05) → RUS(0.1).
- **Tests:** counts between pure SMOTE(0.05) and RUS-only.
- **Do not:** use SMOTEENN/SMOTETomek.

---

### PROMPT 15 — Cross-validation helper + matrix runner Stage A

- **Create:** `src/fraud_ml/experiments/cv.py`, `scripts/run_matrix.py` reading all `configs/experiments/*.yaml`.
- **Output:** 12-row `summary.csv`.
- **Constraints:** skip missing configs with error; sequential runs; checkpoint if a cell exists.
- **Do not:** start Stage B.

---

### PROMPT 16 — Hyperparameter tuning Stage B

- **RandomizedSearchCV**, `n_iter=12`, `scoring=average_precision`, same seed, search spaces Part 8.
- **Sampler params only in cells that have samplers.**
- **Log** `n_iter` and scored params.
- **Do not:** different n_iter per cell.

---

### PROMPT 17 — Threshold analysis

- **OOF probabilities** from cross_val_predict or manual loop on **train**.
- **Save** plots + chosen threshold.
- **Test set** used only to apply frozen threshold in a separate function.
- **Tests:** chosen threshold not computed from y_test in unit tests (pass disjoint arrays).

---

### PROMPT 18 — Comparison, bootstrap CI, McNemar, chronological check

- **Create:** `src/fraud_ml/evaluation/compare.py`; `notebooks/02_analysis_results.ipynb`.
- **Chronological:** sort by Time, first 80% train / last 20% test; **retrain finalists only**.
- **Do not:** rerun all 12 on chrono in this prompt.

---

### PROMPT 19 — Final model train + serialization

- **Select** via config `configs/final_model.yaml` filled **by humans after looking at tables** (agent must not auto-pick on test F1).
- **Fit** train pipeline on full train; export **inference** joblib: preprocess+clf only.
- **Save** metadata: threshold, metrics, git SHA, feature names.
- **Tests:** loaded model predicts 1-row DataFrame; pipeline `named_steps` has no SMOTE.

---

### PROMPT 20 — FastAPI

- **Create:** `app/api/main.py` — `POST /predict` (single), `POST /predict/batch`, `GET /health`, `GET /metrics` (read `summary.csv` / frozen json).
- **No** training endpoints.
- **Tests:** `TestClient` with dummy model fixture.

---

### PROMPT 21 — Minimal frontend

- **Vite+React** in `app/web`: pages (1) predict form Time/Amount/V1–V28 (2) metrics table (3) class imbalance chart from static json (4) CSV upload calling batch.
- **No ML libraries in the frontend.**
- **If React is too heavy:** Streamlit in `app/web/streamlit_app.py` is acceptable; pick one.

---

### PROMPT 22 — Connect frontend/backend

- CORS, env `VITE_API_URL`, example `.env.example`.
- **Do not:** add auth.

---

### PROMPT 23 — Docker (optional)

- **Dockerfile** for API bundling `models/final_inference.joblib`.
- **compose:** api 8000, web 5173.
- **Skip** if no final model yet.

---

### PROMPT 24 — Testing and leakage suite

- **Create:** `tests/test_leakage.py`: test checksum invariant; sampler not in inference; feature alignment; SMOTE `fit_resample` monkeypatch asserts not called during `inference.predict`.
- **Run:** full `pytest`.

---

### PROMPT 25 — Documentation

- **README:** how to get data, install, run Stage A, train final, start API.
- **Keep this blueprint as methodology.** Do not invent experimental results.

---

# PART 16 — Minimal Web/API Architecture

```text
Browser (Vite/React or Streamlit)
    → HTTP
FastAPI
    → fraud_ml.inference.load_model()
    → joblib sklearn Pipeline
    → {fraud_probability, label, threshold}
Static JSON for experiment tables/plots (copied from experiments/results)
```

Pages: Predict | Performance | Imbalance viz | Batch CSV.  
No login, no DB, no retraining in the API.

The frontend must not contain ML logic. The web app consumes the trained model through the API.

---

# PART 17 — Deployment Plan

**Demo-grade:**

- **Model:** inside the API image / folder
- **API:** FastAPI on Render/Railway/Fly **or** a laptop + ngrok for the oral exam
- **Frontend:** Netlify/Vercel static, or Streamlit Community Cloud if Streamlit chosen
- **Env:** `MODEL_PATH`, `THRESHOLD_PATH`, `METRICS_PATH`

Do not use Kubernetes, S3 model registries, or GPU instances. Docker is optional but useful so the instructor can run one command.

---

# PART 18 — Testing Strategy

**Unit:** schema, log1p, scaler fit/transform shapes, metric keys, threshold argmax, sampler counts, `sampling_strategy` math.

**Integration:** `run_experiment` on fixture (200 rows); API predict.

**Data:** columns, `{0,1}` target, no NaN in fixture and in real validate script.

**Critical:**

- SMOTE not on test (monkeypatch / pipeline steps)
- test checksum
- train/test feature names equal
- inference accepts a single unseen row with the same columns as training features

---

# PART 19 — Presentation Artifacts

**Essential**

- Imbalance bar (0.172% fraud)
- Pipeline diagram: split → CV → sample inside fold → locked test
- Stage A table (PR-AUC, P, R, F1, FPR)
- PR curves overlay (treatments, one plot per model)
- One confusion matrix at the **operating threshold** for the selected model
- Threshold vs precision/recall plot
- “SMOTE is train-only” inference diagram

**Optional**

- ROC overlay (to teach why ROC looks too good)
- Before/after SMOTE 2D sketch on 2 PCA axes (subsample; synthetic vs real)
- Chronological vs random split delta
- Runtime bar

**Skip:** 28 V histograms, tool-architecture fireworks, MLflow screenshots.

---

# PART 20 — Risks + Limitations

- SMOTE points are interpolations in PCA+amount space, **not** real frauds.
- Public, anonymized, two-day, single-issuer-like dump; no merchants, no geo, no raw MCC.
- `Time` is not a deployable clock without a defined origin.
- Random split can leak temporal correlation; chrono check is partial.
- Offline metrics ignore analyst workload and dollar costs; we **proxy** FP cost with FPR, not a true cost matrix (unknown).
- Concept/distribution drift: a 2013-era dataset will not match 2026 traffic.
- k-NN interpolation can produce samples across class boundaries.
- RUS variance and seed sensitivity.
- XGBoost nondeterminism on some builds.
- Equal-budget tuning still does not try all HP.
- We cannot claim “SMOTE is best” — only what **this matrix** shows.

The final report must not make unsupported claims such as “SMOTE is always better.” Conclusions must follow the experimental tables.

---

# End matter

## A. Complete dependency graph

```text
data file → validate → split
                ↓
              EDA (train)
                ↓
           preprocessor
                ↓
        sampling registry
                ↓
     train pipeline factory → Stage A matrix → summary
                ↓
           Stage B (equal tune)
                ↓
     OOF threshold → test eval → comparison/stats
                ↓
     final_model.yaml (human) → inference joblib → API → UI
```

G1 → G2 → G3 → G4 → G5, with G5 metrics module delivered early (after Phase 5).

## B. Recommended technology stack

Python 3.11+, pandas, numpy, scikit-learn, imbalanced-learn, XGBoost, matplotlib, PyYAML, joblib, pytest, FastAPI, uvicorn, Vite+React **or** Streamlit, optional Docker.

**Not:** MLflow, Spark, Airflow, Postgres, Redis, Kubernetes.

## C. Minimum Viable ML Project

1. Validated split
2. LR + XGB on **none** and **SMOTE** only (4 cells)
3. PR-AUC + confusion at OOF threshold
4. One joblib + one FastAPI `/predict`
5. README

## D. Full Research-Grade Project

All 12 Stage A cells, Stage B equal tuning, threshold analysis, bootstrap CI, McNemar on finalists, chronological sensitivity, from-scratch SMOTE concordance, leakage tests, comparison notebook, demo UI.

## E. Final Recommended Experiment Matrix

**2 models × 6 treatments = 12 core experiments** (None, class weights, ROS, RUS, SMOTE, SMOTE+RUS). Stretch: ADASYN + Borderline-SMOTE × XGBoost only.

## F. Final Recommended Repository Structure

See Part 12. ML package `src/fraud_ml` is the core; `app/` is a consumer.

## G. All assumptions made

- Working dataset is the public European card PCA dataset unless validation says otherwise.
- Binary `Class` with fraud = 1.
- ~0.17% prevalence.
- 10 students, CPU-only, multi-week course timeline.
- From-scratch SMOTE is a **pedagogical/concordance** component; **imblearn SMOTE** is the main experimental engine.
- No real FP/FN dollar costs.
- Instructor demo does not need auth or cloud MLOps.

## H. Decisions that should remain open until EDA/results are available

1. Keep vs drop `Time` (univariate AUC + hourly rate).
2. log1p Amount if skew is unexpectedly mild.
3. RobustScaler vs StandardScaler sensitivity if LR Stage A is pathological.
4. Stage A SMOTE strategy 0.1 vs a one-off diagnostic at 0.25 if recall is near zero.
5. Repeated CV only if fold std is large.
6. Final model identity (human after tables).
7. Operating constraint (precision floor) if F1-optimal threshold is operationally silly.
8. React vs Streamlit based on frontend skill.
9. Docker yes/no based on instructor environment.
10. Whether stretch SMOTE-variants run at all.

---

## Existing code note

`model_build/src/smote_from_scratch/sampler.py` currently always synthesizes until classes are **equal**. That is incompatible with this protocol. Prompt 13 must add `sampling_strategy` before any SMOTE matrix run.

Do not start coding all 12 cells until Phases 0–7 exist: package, validation, frozen split, preprocessor, metrics, tracking, and the LR/XGB **none** baselines. That baseline is the reference every later number must beat or lose to honestly.
