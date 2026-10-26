# An Empirical Study of Class-Imbalance Handling Techniques for Credit Card Fraud Detection

This is our student group project. Only about 0.17% of the payments in the ULB credit-card
dataset are frauds, so a model that always says "legit" is 99.8% accurate and useless. We
compare six ways of dealing with that imbalance (A no treatment, B random oversampling, C random
undersampling, D SMOTE, E SMOTE then random undersampling, F class weighting) on three models
(logistic regression, random forest, LightGBM). Every technique gets the same tuning budget, all
resampling happens inside the training folds only, and the test set is opened once at the end.
The main metric is average precision (PR-AUC). A Streamlit app shows the results.

**Research question:** how do data-level resampling (random oversampling, random undersampling,
SMOTE and SMOTE + undersampling) and algorithm-level class weighting compare with an untreated
baseline for detecting fraud in highly imbalanced credit-card data?

## Folder map

```text
data/raw/creditcard.csv          the real dataset (Git-ignored; download it from Kaggle)
data/sample/                     small fake copy with the same columns, for tests and trial runs
model_build/                     our from-scratch SMOTE (a teammate's code and notebook)
src/fraudstudy/__init__.py       marks fraudstudy as a package
src/fraudstudy/data.py           load, check, remove duplicates, split 70/30
src/fraudstudy/experiments.py    preprocessing, the 6 techniques, the 3 models, tuning, cross-validation
src/fraudstudy/evaluation.py     the one-time test-set evaluation, metrics, confidence intervals, significance tests, figures
notebooks/01_eda.ipynb           exploration and the EDA figures
artifacts/figures/               saved figures (eda_*.png from the notebook, the rest from evaluation.py)
artifacts/metrics/               saved results (eda_summary.json, cells/*.json, test_results.json, stats.json, ...)
artifacts/models/                trained pipelines (only lightgbm__*.joblib are committed, for the app)
artifacts/sample/                results of trial runs on the fake sample (Git-ignored)
tests/test_no_leakage.py         proves no validation or test row reaches training
tests/test_smote.py              checks our from-scratch SMOTE against imblearn
app.py                           the Streamlit presentation app
requirements.txt                 every library, pinned to an exact version
pyproject.toml                   makes src/fraudstudy installable and holds the pytest settings
CLAUDE.md                        rules for coding sessions in this repository
LICENSE                          MIT licence
```

## How to run

Run everything from the repository root, in this order.

```bash
# 1. Create a virtual environment and install everything (Python 3.11 or newer)
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt        # also installs src/fraudstudy and model_build/ (the "-e" lines)
pip install notebook                   # only if you want to open the notebooks

# 2. Download creditcard.csv from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud,
#    unzip it and put it at data/raw/creditcard.csv

# 3. Check the data: prints rows, frauds, duplicates removed and the train/test sizes
python -m fraudstudy.data

# 4. EDA: open notebooks/01_eda.ipynb and run all cells (Kernel > Restart & Run All).
#    It writes artifacts/figures/eda_*.png and artifacts/metrics/eda_summary.json.
jupyter notebook notebooks/01_eda.ipynb

# 5. The experiment: tunes and cross-validates all 18 cells on the TRAINING rows only.
#    Finished cells are skipped, so you can stop and restart; --force re-runs them.
python -m fraudstudy.experiments

# 6. The one-time test-set evaluation. The team picks the default cell (the one the app shows
#    first) from the CV PR-AUC printed in step 5, never from test results.
python -m fraudstudy.evaluation --test --default-cell <model>__<technique>   # e.g. lightgbm__smote

# 7. Significance tests and the report figures
python -m fraudstudy.evaluation --stats --figures

# 8. Tests (they use the fake sample only, under a minute)
pytest -q

# 9. The app
streamlit run app.py
```

Step 6 writes a guard file (`artifacts/metrics/TEST_SET_OPENED.json`). Running `--test` a second
time is refused unless you add `--rerun-reason "<why>"`, and that reason is logged. Never change a
tuned setting because of test results.

**Splitting the experiment across laptops.** The 18 cells are independent, so each person can run
some of them, e.g. one model each:

```bash
python -m fraudstudy.experiments --models logreg          # laptop 1
python -m fraudstudy.experiments --models random_forest   # laptop 2
python -m fraudstudy.experiments --models lightgbm        # laptop 3
```

Use the same code version and `requirements.txt` on every laptop (and either all or none with
`--quick`). Then copy each laptop's `artifacts/metrics/cells/*.json` and `artifacts/models/*.joblib`
into the same folders on one machine and run steps 6 and 7 there. The splits and folds come only
from the seed, so the cells line up fold by fold.

### Trial run without the real data

The same steps on the fake sample (`--source sample --quick`). Everything goes to
`artifacts/sample/`, which is Git-ignored, so a trial never touches the real results. It takes
about 3 minutes; random forest is the slow part, so add `--models logreg lightgbm` to step 5 for a
1-minute run.

```bash
python -m fraudstudy.data --source sample
# notebook: set SOURCE = "sample" in its first code cell, then run all cells
python -m fraudstudy.experiments --source sample --quick
python -m fraudstudy.evaluation --source sample --test --default-cell lightgbm__baseline
python -m fraudstudy.evaluation --source sample --stats --figures
pytest -q
```

**About `data/sample/`:** `creditcard_synthetic.csv` is fake data with the same 31 columns as the
real file (6,000 rows, 1.5% fraud). It was made by an earlier version of our code, so that the
tests and trial runs work without the real dataset. Its numbers mean nothing; never report them.

## Deploy

Live app: <!-- paste the share.streamlit.io link here after deploying -->

The app only reads committed files (`artifacts/metrics/*.json`, `artifacts/figures/*.png` and
`artifacts/models/lightgbm__*.joblib`), so it runs online without the dataset. Deploy only after
the real results are committed; otherwise the app shows "No results".

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app** and choose to deploy from a GitHub repository.
3. Repository `RavenKing45/ML_Project2026`, branch `shaurya`, main file path `app.py`.
4. Under **Advanced settings**, choose Python **3.11**.
5. Click **Deploy**. The first build installs `requirements.txt` and takes a few minutes.

- Whoever deploys needs access to the repository from their own GitHub account (and must let
  Streamlit read it if the repository is private).
- Keep the two `-e` lines in `requirements.txt`: `data.py` finds `artifacts/` relative to its own
  file, which only works when the package is installed from the repository folder.
- A free app goes to sleep when nobody uses it. Open the link about 5 minutes before the demo
  and click the wake-up button if it shows one.
- Fallback if the online app fails: on a laptop with the repository, run `streamlit run app.py`.

## Design decisions

| Decision | Why |
|---|---|
| Every resampler runs inside an `imblearn.pipeline.Pipeline`, only during `fit`, after the split | Resampling before the split or outside the CV folds leaks copied or synthetic frauds into validation data and inflates the scores |
| Primary metric is average precision (PR-AUC); never accuracy | With 0.17% frauds, accuracy and ROC-AUC are dominated by the legit class; PR-AUC measures how well frauds are found |
| 70/30 stratified train/test split with one seed (`SEED` in `data.py`) | Keeps about 140 frauds in the test set and the same fraud rate in both parts; one seed makes every run repeatable |
| Exact duplicate rows are removed before the split | A copy in train and one in test would test the model on a row it learned |
| Raw `Time` is not a model feature | It counts seconds inside one two-day file, so it describes when the data was collected, not new payments |
| Outliers are kept | Frauds are often the extreme values; removing outliers would remove the signal |
| log1p(Amount), then `RobustScaler` on every column, inside the pipeline | Amount is very skewed (skewness 17); the median/IQR scaler is not pulled around by extreme values and learns from training rows only |
| The same ratio grid and the same tuning budget (`N_ITER`) for every technique | A fair comparison: no technique gets more search effort or a better ratio range |
| Class weighting maps each ratio r to the fraud weight r / base ratio | Then F reaches the same class balance a sampler at ratio r would, so the grids are comparable |
| One F1 threshold per cell, chosen on out-of-fold training scores | The test set never picks a threshold; one threshold is enough to explain and compare |
| The test set is opened once, behind a guard file | Every extra look at the test set biases the final estimate |
| Models: logistic regression, random forest, LightGBM | A linear model, a bagged and a boosted tree ensemble react differently to resampling and weights |
| Repeated stratified CV with the same seed for every cell; Wilcoxon vs baseline with Holm, plus a Friedman test per model | Every cell gets the same folds, so techniques can be compared fold by fold; Holm corrects for 5 tests per model |
| PR-AUC 95% confidence intervals by stratified bootstrap (1,000 resamples) | Shows how uncertain a test score is with only about 140 test frauds |
| Settings live in a SETTINGS block at the top of each file, not in YAML configs | One place per job that everyone can read; the YAML loading was a layer nobody needed |
| All research code in three files: `data.py`, `experiments.py`, `evaluation.py` | Each step is one `python -m fraudstudy.<file>` command and every member can explain every line |
| One Streamlit app (`app.py`) instead of a FastAPI backend and a React dashboard | One small Python file is enough to present saved results; no API contracts to maintain |
| `statsmodels` added | Its `multipletests` does the Holm correction, instead of our own code |
| `streamlit` added | Runs the presentation app |
| One `requirements.txt` with exact versions, including pytest and the two `-e` installs | Everyone gets the same versions with one command; Jupyter is installed separately because only notebook users need it |
| Only the LightGBM pipelines are committed (`artifacts/models/lightgbm__*.joblib`) | The app needs them and each is about 2.4 MB; the other models stay local |
| The EDA is a notebook (`notebooks/01_eda.ipynb`) committed with its outputs | GitHub shows the plots, and each step is explained next to its code |
| Cut: probability calibration (isotonic calibrators) | Only the old backend used them; PR-AUC and the F1 threshold do not depend on calibration |
| Cut: the F2 threshold (F2 is still reported at the F1 threshold) | One tuned threshold per cell is enough |
| Cut: the temporal-split sensitivity check | A side analysis outside the main question, and it re-read test rows outside the single evaluation |
| Cut: the Nemenyi critical difference, CD diagrams, the pooled Friedman test and `findings.md` | Wilcoxon with Holm plus one Friedman test per model answer the question |
| Cut: the export manifest, the deploy folder and every SHA-256 check | They protected the FastAPI deployment; the split is fixed by the seed |
| Cut: batch CSV scoring and the demo batch | Only the FastAPI backend served them |
| Cut: `backend/` (FastAPI), `frontend/` (React), the JSON contracts and pyyaml, pydantic, fastapi, uvicorn, slowapi, python-multipart | The Streamlit app replaces them |
| Cut: `scripts/`, the resampling-audit and baseline scripts, `artifacts/results/` and `artifacts/demo/` | Each step is now `python -m fraudstudy.<file>`; the training counts before and after resampling are saved in each cell file |
| Cut: the synthetic-data generator; `data/sample/` is kept as it is | The sample only needs to exist, not to be regenerated |
| Cut: ROC-AUC confidence interval, MCC, metrics at threshold 0.5 and the confusion-matrix, ROC, threshold-effect and fold-boxplot figures | The report needs PR-AUC with its CI, metrics at the tuned threshold, PR curves and the training counts |
| Cut: the optional hour-of-day feature | Never part of the main experiment, and it assumed Time = 0 is midnight |
| Cut: `eda.py` and `run_eda.py` | Replaced by the EDA notebook |
| Cut: `requirements-dev.txt`, `requirements-notebook.txt`, ruff and its settings | One requirements file is simpler; the linter only ran in the CI, which is cut too |
| Cut: `.github/` (CI), `.gitattributes`, `docs/` and the empty `.gitkeep` files | The tests run locally with `pytest -q`; the decisions are in this table and the master guide lives in the team document |

## Where to find things

Questions an examiner may ask, and where the answer is in the code. Line numbers are for the
commit that added this table; update them whenever one of these files changes.
`data.py`, `experiments.py` and `evaluation.py` are in `src/fraudstudy/`; `sampler.py` is in
`model_build/src/smote_from_scratch/`.

| Question | File | Function | Lines |
|---|---|---|---|
| Where is the data split into train and test? | `data.py` | `split_data()` | 54-67 (`train_test_split` with `stratify=y` at 64-66) |
| Where are the split size and the one seed set? | `data.py` | SETTINGS | 17 (`SEED`), 18 (`TEST_SIZE`) |
| Where are duplicate rows removed? | `data.py` | `remove_duplicates()` | 43-51 |
| How do we know duplicates go before the split? | `experiments.py`, `evaluation.py` | `load_training_data()`, `evaluate_test_set()` | experiments 247-248, evaluation 178-179 (dedupe, then split) |
| Why is `Time` dropped, and where? | `data.py` | `split_data()` | 60-62 (the comment gives the reason; `df.drop(columns=["Time", "Class"])` at 62) |
| Where is the preprocessing (imputer, log1p, RobustScaler), and why is it inside the pipeline? | `experiments.py` | `build_preprocessor()` | 76-94 (docstring 77-81 says why) |
| Where does SMOTE run in the experiment? | `experiments.py` | `build_pipeline()` | 106-123 (SMOTE made at 113, placed at 118-119 between preprocess and model); it runs only inside `Pipeline.fit`, called by `search.fit` in `run_cell()` at 197 |
| Where is our from-scratch SMOTE? | `sampler.py` | `SMOTE.fit_resample()`, `SMOTE._find_k_neighbors()` | 49-148 and 29-47; the interpolation `x_new = x_i + gap * (x_nn - x_i)` is at 133-135 |
| How is the from-scratch SMOTE checked, and where is it used? | `tests/test_smote.py`, `app.py` | `test_scratch_smote_*`, `page_smote()` | tests 22-42 and 45-59 (neighbours checked with scikit-learn's `NearestNeighbors`); app 131 |
| How do we prove a sampler only sees training rows? | `tests/test_no_leakage.py` | `SpySampler`, `test_sampler_sees_only_the_training_part_of_each_fold()` | 22-33, 50-59 |
| How do we prove predicting never resamples? | `tests/test_no_leakage.py` | `test_predicting_never_calls_the_sampler()` | 62-68 |
| How do we prove the scaler learns from training rows only? | `tests/test_no_leakage.py` | `test_scaler_ignores_extreme_test_values()` | 71-78 |
| How do we prove train and test share no row? | `tests/test_no_leakage.py` | `test_train_and_test_share_no_row_and_keep_the_fraud_rate()` | 81-86 |
| Where are the ratio grid and the tuning budget set? | `experiments.py` | SETTINGS | 47 (`N_ITER`), 49 (`RATIO_GRID`), 50-51 (hybrid ratios), 55-63 (model search spaces) |
| How does every cell get the same budget and the same ratios? | `experiments.py` | `run_cell()`, `search_space()` | 193-196 (`n_iter` and `RandomizedSearchCV`), 134-135 (ratios taken from `RATIO_GRID`) |
| How does class weighting match the ratios? | `experiments.py` | `search_space()` | 145-149 (fraud weight = r / base, so weight x frauds = r x legit); explained in the docstring at 127-133 |
| Where is the threshold chosen? | `experiments.py` | `best_f1_threshold()`, `run_cell()` | 153-162; called at 214 on the out-of-fold training scores from `cross_val_predict` at 201 |
| Why is the threshold not chosen on test data? | `evaluation.py` | `evaluate_test_set()` | 189-190 read the saved threshold and never recompute it; a threshold tuned on the test set would make the test score too optimistic |
| Where is the test set opened? | `evaluation.py` | `evaluate_test_set()` | 163-206; 179 is the only line that keeps `X_test` (`experiments.py` throws it away at 248) |
| What stops a second opening of the test set? | `evaluation.py` | `open_guard()` | 112-125: if `TEST_SET_OPENED.json` (43) exists and there is no `--rerun-reason`, 115-120 raise an error; called at 176 before any data is loaded; tested in `tests/test_no_leakage.py` 89-96 |
| Where is PR-AUC computed? | `experiments.py`, `evaluation.py` | `run_cell()`, `compute_metrics()` | CV: `scoring="average_precision"` at 195 (tuning) and 207 (fold scores); test: `average_precision_score` at evaluation 65 |
| Where do the fold scores come from, and why are they comparable across cells? | `experiments.py` | `run_cell()` | 204-207 (`RepeatedStratifiedKFold` with `SEED`: the same folds for every cell) |
| How does the bootstrap confidence interval work? | `evaluation.py` | `bootstrap_ci()` | 72-93: redraw frauds and legit rows separately (87-90), PR-AUC of each resample (91), 2.5% and 97.5% quantiles (92); `N_BOOTSTRAP` at 41 |
| Where are the Wilcoxon, Holm and Friedman tests? | `evaluation.py` | `significance_tests()` | 209-248: Wilcoxon 233, Holm 236, Friedman 242 |
| Where are the training class counts before and after resampling recorded? | `experiments.py` | `counts_after_resampling()`, `run_cell()` | 171-182; saved at 215-216 |
| How does the app explain a prediction? | `app.py` | `page_try()` | 212-250: only the preprocess step runs (241), LightGBM's `pred_contrib=True` gives each feature's push in log-odds (242), the 6 largest are drawn (243-249) |
| Where do the app's example payments come from? | `evaluation.py` | `pick_presets()` | 128-160; called at 200, inside the single test-set opening |
| Which results does the app read? | `app.py` | (module level) | 285-296: the real `artifacts/metrics/`, or `artifacts/sample/` with `--sample` or when the real results are missing |

We spot-checked 10 of these line numbers against the code when writing the table: `data.py` 62,
`experiments.py` 113, 146, 197, 201, 214, `evaluation.py` 115, 233, 236 and `app.py` 242.

## Results

Filled in after the test-set evaluation on the real data.

<!-- RESULTS:START -->
| Model | Technique | CV PR-AUC | Test PR-AUC (95% CI) | Precision | Recall | F1 threshold |
|---|---|---|---|---|---|---|
<!-- RESULTS:END -->

## Limitations

- The data is one public, anonymised two-day dump of European card payments from 2013. V1..V28
  are PCA components, so we cannot explain them in business terms or add features such as the
  merchant or the customer's history.
- SMOTE's new frauds are interpolations in PCA space, not real frauds.
- With about 140 frauds in the test set, the test scores are uncertain; the confidence intervals
  show how much.
- Fold scores share training rows, so the p-values of the significance tests are approximate.
- Offline metrics are not a bank's real costs, and concept drift and fraudsters adapting to the
  model are out of scope.

## Citations

- A. Dal Pozzolo, O. Caelen, R. A. Johnson and G. Bontempi, "Calibrating Probability with
  Undersampling for Unbalanced Classification", IEEE Symposium Series on Computational
  Intelligence and Data Mining (CIDM), 2015. (The dataset.)
- N. V. Chawla, K. W. Bowyer, L. O. Hall and W. P. Kegelmeyer, "SMOTE: Synthetic Minority
  Over-sampling Technique", Journal of Artificial Intelligence Research 16, 321-357, 2002.

## Licence

The code is under the MIT licence (see `LICENSE`). The credit-card dataset is not included in
this repository; download it from Kaggle and follow its terms.

## Team

Shaurya Agarwal, Pragna Jyoti Mandal, Utkarsh Gupta, Ayush, Aditi.
<!-- TODO: complete the list with every member's full name and part of the project. -->
