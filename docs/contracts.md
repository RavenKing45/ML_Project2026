# Data contracts

The contracts between research code, backend and frontend. API shapes follow Part 9 of the
master guide. **`src/fraudstudy/contracts.py` is
the single source of truth**: this page explains it, and every example in
[`docs/contracts/examples/`](contracts/examples/) validates against it (`tests/test_contracts.py`).
If this page and the code disagree, the code wins and this page is a bug.

Changing a contract (renaming, removing or retyping a field) needs team agreement, a row in
`docs/decisions.md`, and a bump of `CONTRACT_VERSION`. Adding an optional field is a minor bump.

## Conventions

* Field names are `snake_case`, except `Transaction`, which keeps the dataset's column names
  (`Time`, `V1`..`V28`, `Amount`) so CSV rows map 1:1.
* Every model rejects unknown fields (`extra="forbid"`), so a typo fails loudly, and rejects
  NaN / infinity.
* Every top-level file or payload has `contract_version` (currently `"1.0"`) and `is_example`.
  `is_example` is `true` only in the fixture files; the dashboard should show an
  "example data" banner when it sees `true`.
* Probabilities, rates and areas are floats in [0, 1]; counts are non-negative integers;
  intervals are `[low, high]` with `low <= high`.
* Model ids: `logreg`, `random_forest`, `lightgbm`. Technique ids: `baseline`, `ros`, `rus`,
  `smote`, `smote_rus`, `class_weight` (they must equal `models` and `techniques` in
  `configs/experiment.yaml`; a test enforces it). Display labels: `MODEL_LABELS`,
  `TECHNIQUE_LABELS`.
* Writers serialise with `model.model_dump(mode="json")` and `fraudstudy.io.save_json`; readers
  call `Model.model_validate(load_json(path))`. `EXAMPLE_MODELS` maps file stem to model.

## Where files live

| file | model | written to |
|------|-------|-----------|
| comparison.json | `ComparisonTable` | `artifacts/results/` |
| curves.json | `CurvesFile` | `artifacts/results/` |
| stats.json | `StatsResult` | `artifacts/results/` |
| eda.json | `EdaSummary` | `artifacts/results/` |
| resampling_audit.json | `ResamplingAudit` | `artifacts/results/` |
| manifest.json | `Manifest` | `artifacts/models/` |
| presets.json | `PresetsFile` | `artifacts/demo/` |
| — (API) | `PredictRequest` / `PredictResponse` | `POST /api/v1/predict` |
| — (API) | `BatchResponse` | `POST /api/v1/predict/batch` (CSV upload) |
| — (API) | `SmoteDemoRequest` / `SmoteDemoResponse` | `POST /api/v1/smote-demo` |
| — (API) | `ErrorResponse` | body of every non-2xx response |

`GET /api/v1/results` serves comparison + curves + stats; `GET /api/v1/eda` serves eda.json;
`GET /api/v1/presets` serves presets.json. The `/health` and `/models` response shapes are left
to the backend prompt.

Shared building blocks:

* **`Metrics`** — `precision, recall, f1, f2, fpr, fnr, mcc, accuracy, tp, fp, tn, fn, pr_auc,
  roc_auc`. `f2` = F-beta with beta = 2; `fpr = fp/(fp+tn)`; `fnr = fn/(fn+tp)`; `pr_auc` is
  average precision; `mcc` is in [-1, 1].
* **`ClassCounts`** — `{legit, fraud}`.
* **`CellRef`** — `{model, technique}`: one cell of the 6 x 3 matrix.

---

## comparison.json — `ComparisonTable`

* **Written by:** research squad, test-evaluation step (`scripts/evaluate_test.py`, the only
  code that reads the test set).
* **Read by:** backend (results endpoints), dashboard comparison table and charts, report.
* **Example:** [comparison.json](contracts/examples/comparison.json) (all 18 cells).

| field | type | meaning |
|-------|------|---------|
| primary_metric | `"average_precision"` | always PR-AUC |
| dataset_source | `"real"` \| `"synthetic"` | which data produced it |
| records | `ComparisonRecord[]` | one per cell, `(model, technique)` unique |

`ComparisonRecord`:

| field | type | meaning |
|-------|------|---------|
| model, technique | id | the cell |
| model_label, technique_label | str | display names |
| best_params | {str: scalar} | tuned hyper-parameters (and sampler params) |
| chosen_ratio | float \| null | tuned minority:majority ratio; for `smote_rus` the RUS ratio (SMOTE ratio in `best_params`); for `class_weight` the ratio the weights were derived from; `null` for `baseline` |
| cv_pr_auc_mean, cv_pr_auc_std | float | repeated-CV PR-AUC on the training set |
| cv_roc_auc_mean | float | repeated-CV ROC-AUC |
| threshold_f1, threshold_f2 | float | thresholds maximising F1 / F2 on out-of-fold predictions |
| test_at_0_5 | `Metrics` | test metrics at threshold 0.5 |
| test_at_tuned | `Metrics` | test metrics at `threshold_f1` |
| pr_auc_ci, roc_auc_ci | [low, high] | bootstrap CI on the test set |
| fit_seconds | float | wall-clock time of the final fit |
| n_train_after_resampling | `ClassCounts` | training rows after the sampler (= before, for baseline/class_weight) |

## curves.json — `CurvesFile`

* **Written by:** research squad, test-evaluation step.
* **Read by:** dashboard ROC / PR charts and the threshold slider.
* **Example:** [curves.json](contracts/examples/curves.json).

| field | type | meaning |
|-------|------|---------|
| cells | `Curves[]` | one per cell, `(model, technique)` unique |
| cells[].model, .technique | id | the cell |
| cells[].roc | `{fpr[], tpr[]}` | test-set ROC curve |
| cells[].pr | `{recall[], precision[]}` | test-set PR curve |
| cells[].threshold_table | `{threshold[], precision[], recall[], fpr[]}` | metrics on a threshold grid |

Arrays inside one object have equal length, between 1 and **101** points (`MAX_CURVE_POINTS`);
writers must downsample.

## stats.json — `StatsResult`

* **Written by:** research squad, statistics script (after test evaluation; reads fold scores).
* **Read by:** backend (`/results`), dashboard significance markers and critical-difference
  diagram, report.
* **Example:** [stats.json](contracts/examples/stats.json).

| field | type | meaning |
|-------|------|---------|
| alpha | float | significance level |
| correction | `"holm"` \| `"bonferroni"` \| `"none"` | multiple-comparison correction for `pairwise` |
| n_bootstrap, confidence, seed | int, float, int | from `configs/experiment.yaml` `bootstrap` / `seed` |
| friedman | `FriedmanTest[]` | see below |
| pairwise | `PairwiseTest[]` | see below |

`FriedmanTest`: `scope` (a model id, or `"all"`), `metric`, `n_blocks` (folds, or model x fold),
`statistic`, `p_value`, `average_ranks` ({technique: rank}, 1 = best), `critical_difference`
(Nemenyi CD at `alpha`, or null).

`PairwiseTest`: `model`, `technique_a`, `technique_b`, `test` (`wilcoxon` \| `paired_bootstrap` \|
`mcnemar`), `metric` (str), `difference` (metric(a) − metric(b)), `difference_ci` ([low, high] or
null), `statistic`, `p_value`, `p_value_adjusted`, `significant` (bool, from the adjusted p).

## eda.json — `EdaSummary`

* **Written by:** EDA squad (EDA script/notebook using `src/fraudstudy`).
* **Read by:** backend (`/eda`), dashboard Overview and Imbalance pages, report.
* **Example:** [eda.json](contracts/examples/eda.json).

| field | type | meaning |
|-------|------|---------|
| dataset_source, dataset_sha256 | str | provenance |
| n_rows, n_columns | int | raw shape (before dedup) |
| class_counts, fraud_rate | `ClassCounts`, float | class balance |
| n_duplicate_rows, n_duplicate_frauds | int | exact duplicates beyond the first occurrence |
| missing_values | {column: int} | missing count per column |
| amount | `{legit, fraud}` of `Distribution` | `count, mean, std, min, q1, median, q3, max, n_zero` |
| amount_histogram, time_histogram | `Histogram` | `bin_edges[]` (n+1), `legit_counts[]`, `fraud_counts[]` (n) |
| feature_separation | `FeatureSeparation[]` | per feature: class means/stds and KS statistic, sorted by KS desc |
| training_counts | `TrainingCounts[]` | per technique: `technique`, `ratio`, `before`, `after` (`ClassCounts`) on the full training set at the tuned ratio |

## resampling_audit.json — `ResamplingAudit`

* **Written by:** research squad, CV runner (records what each sampler did inside each fold).
* **Read by:** leakage tests, dashboard "how resampling works" page, report.
* **Example:** [resampling_audit.json](contracts/examples/resampling_audit.json).

| field | type | meaning |
|-------|------|---------|
| seed, cv_folds | int | protocol |
| test_rows_resampled | int | must be 0 |
| all_checks_passed | bool | every entry has `validation_untouched` and `test_rows_resampled == 0` |
| entries[] | `ResamplingAuditEntry` | `model, technique, fold, ratio, train_before, train_after, validation, n_synthetic, validation_untouched` |

## artifacts/models/manifest.json — `Manifest`

* **Written by:** research squad, model exporter.
* **Read by:** backend at start-up: loads each listed model, checks file checksum and library
  versions, builds input tables in `features` order. If anything is off, `/health` reports
  `"degraded"`.
* **Example:** [manifest.json](contracts/examples/manifest.json).

| field | type | meaning |
|-------|------|---------|
| created_at | str | ISO 8601 UTC |
| git_commit | str \| null | `fraudstudy.io.git_commit_hash()` |
| seed, quick_mode | int, bool | run settings |
| dataset | `{name, source, sha256, n_rows}` | training data |
| default_model_id | str \| null | must be one of `models[].id` |
| models[] | `ManifestEntry` | see below; ids unique |

`ManifestEntry`: `id` (must be `"<model>__<technique>"`), `model`, `technique`, `file`
(relative to `artifacts/models/`), `sha256` (64 hex, of the file), `threshold_f1`,
`threshold_f2`, `has_calibrator`, `features` (exact column order the pipeline expects),
`data_sha256`, `library_versions` ({package: version}).

Only the deployed subset of the 18 pipelines is listed (Part 9: not all 18 fit on a small server).

## presets.json — `PresetsFile`

* **Written by:** research squad, exporter: 5-6 real test-set payments (everyday legitimate,
  large legitimate, typical fraud, small-amount fraud, borderline), picked after the single
  test evaluation.
* **Read by:** backend (`/presets`), dashboard "Try a transaction" page, which posts
  `preset.features` to `/predict`.
* **Example:** [presets.json](contracts/examples/presets.json).

| field | type | meaning |
|-------|------|---------|
| presets[].id, .label, .description | str | identity and text |
| presets[].expected_class | 0 \| 1 | true label |
| presets[].features | `Transaction` | `Time` (optional, ignored), `V1`..`V28`, `Amount` |

`Transaction` fields are all finite numbers; `Amount` is in [0, 1 000 000] (`AMOUNT_MAX`);
`V1`..`V28` and `Amount` are required; any other key is rejected.

## POST /api/v1/predict — `PredictRequest` → `PredictResponse`

* **Written by:** backend. **Read by:** dashboard.
* **Example:** [predict_response.json](contracts/examples/predict_response.json).

Request: `features` (`Transaction`), `model_id` (str, or null for the default model),
`threshold` (float in [0, 1], or null for the model's tuned threshold). Unknown `model_id` →
404 with `ErrorResponse` listing valid ids.

| field | type | meaning |
|-------|------|---------|
| label | `"FRAUD"` \| `"LEGITIMATE"` | `"FRAUD"` exactly when `risk_score >= threshold` (validated) |
| risk_score | float | model score in [0, 1]; a ranking score, not a probability |
| calibrated_probability | float \| null | calibrated fraud probability, null if no calibrator |
| threshold | float | threshold applied |
| threshold_source | `"tuned_on_validation_F1"` \| `"tuned_on_validation_F2"` \| `"user_supplied"` \| `"default_0_5"` | where it came from |
| model | `{id, model, technique}` | which model answered (ids; labels via `MODEL_LABELS` / `TECHNIQUE_LABELS`) |
| explanation | `{method, units, top_features[]}` \| null | e.g. `"TreeSHAP"`, `"log-odds"`; `top_features[]`: `feature, value, contribution, legit_percentile` (0-100) |
| warnings | str[] | e.g. "edited transaction may not resemble real payments" |
| disclaimer | str | research-only notice |
| latency_ms | float | server time |

The backend must never log feature values.

## POST /api/v1/predict/batch — `BatchResponse`

* **Written by:** backend (CSV upload, field `file`, ≤ 5 MB, ≤ 10 000 rows, parsed in memory).
  **Read by:** dashboard batch page (builds the downloadable CSV from `results`).
* **Example:** [batch_response.json](contracts/examples/batch_response.json).

| field | type | meaning |
|-------|------|---------|
| model, threshold, threshold_source | as in `PredictResponse` | |
| n_rows, n_scored, n_flagged | int | counts |
| flagged_fraction | float | `n_flagged / n_scored` |
| flagged_amount_total | float | sum of `Amount` over flagged rows |
| metrics | `Metrics` \| null | only if the CSV had a `Class` column |
| results[] | `{row, amount, risk_score, calibrated_probability, label}` | `row` is 0-based, header excluded |
| errors[] | `{row, message}` | rows that could not be scored |
| warnings, disclaimer, latency_ms | as in `PredictResponse` | |

The backend must never log or store uploaded file contents.

## Errors — `ErrorResponse`

Every non-2xx response body is `{"error": {"code": "<snake_case>", "message": "<human text>"}}`.

## POST /api/v1/smote-demo — `SmoteDemoRequest` → `SmoteDemoResponse`

* **Written by:** backend (runs real `imblearn` SMOTE on a 2-D toy dataset). **Read by:**
  dashboard SMOTE explainer (whose built-in JS fallback should produce the same shape).
* **Example:** [smote_demo.json](contracts/examples/smote_demo.json).

Request (all optional, default in brackets): `scenario` (`single_cluster` \| `two_clusters`
[default] \| `overlap`), `n_legit` (2-2000 [300]), `n_fraud` (2-200 [20]), `k_neighbors`
(1-15 [5], must be < `n_fraud`), `sampling_strategy` ((0, 1] [0.5]), `seed` ([42]).

| field | type | meaning |
|-------|------|---------|
| request | `SmoteDemoRequest` | echo of the request |
| original[] | `{x, y, label}` | toy data |
| synthetic[] | `{x, y, parent, neighbor, gap, in_legit_region}` | x_new = x + λ(x̂ − x) with x = `original[parent]`, x̂ = `original[neighbor]`, λ = `gap` |
| n_synthetic_in_legit_region | int | synthetic frauds that landed among legit points (SMOTE failure cases) |
| warnings | str[] | |

---

## Synthetic sample dataset

`data/sample/creditcard_synthetic.csv` (committed, ~1.5 MB) is produced by
`python scripts/generate_synthetic_data.py` from `fraudstudy.synthetic.make_synthetic` with the
`experiment.yaml` seed. It has the real schema (`Time, V1..V28, Amount, Class`), 6000 rows,
90 frauds (1.5%) and 20 exact duplicate rows (2 of them fraud). The fraud rate is deliberately
~9x the real 0.17%: at 0.17% the sample would have ~10 frauds, too few for stratified 5-fold CV
plus SMOTE with `k_neighbors=5` (which needs ≥ 6 frauds in every training fold). Frauds come
from two clusters that overlap the legit class (logistic-regression CV PR-AUC ≈ 0.8, so it is
not trivially separable). It is for tests and development only; no result from it means anything.
