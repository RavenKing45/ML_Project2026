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
* Every top-level file or payload has `contract_version` (currently `"1.6"`) and `is_example`.
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
| — (API) | `HealthResponse` | `GET /api/v1/health` |
| — (API) | `ModelsResponse` | `GET /api/v1/models` |
| — (API) | `ResultsResponse` | `GET /api/v1/results` |
| — (API) | `EdaResponse` | `GET /api/v1/eda` |
| — (API) | `PresetsFile` | `GET /api/v1/presets` |
| — (API) | `PredictRequest` / `PredictResponse` | `POST /api/v1/predict` |
| — (API) | `BatchResponse` | `POST /api/v1/predict/batch` (CSV upload) |
| — (API) | `SmoteDemoRequest` / `SmoteDemoResponse` | `POST /api/v1/smote-demo` |
| — (API) | `ErrorResponse` | body of every non-2xx response |

`GET /api/v1/results` serves comparison + curves + stats (+ the temporal check);
`GET /api/v1/eda` serves eda.json (+ the resampling audit); `GET /api/v1/presets` serves
presets.json unchanged. Their response shapes, and those of `/health` and `/models`, are in
[GET endpoints](#get-endpoints-of-the-backend-15) below.

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
For the Wilcoxon test on fold scores, `difference` is the mean per-fold difference.

Optional fields *(1.3)*: `PairwiseTest` gains `n_pairs`, `median_difference` (median per-fold
difference) and `wins` / `ties` / `losses` (folds where a scored higher / equal / lower than b;
they sum to `n_pairs`). `StatsResult` gains `dataset_source` (`real` \| `synthetic`) and `notes`
(caveats to show next to the tests, e.g. that fold scores are not independent, so p-values are
approximate).

## eda.json — `EdaSummary`

* **Written by:** EDA squad, `python scripts/run_eda.py --source raw|sample`
  (`fraudstudy.eda.compute_eda_summary`).
* **Read by:** backend (`/eda`), dashboard Overview and Imbalance pages, report.
* **Example:** [eda.json](contracts/examples/eda.json).

Raw-file facts (`n_rows`, `n_columns`, `class_counts`, `fraud_rate`, `imbalance_ratio`,
`missing_values`) describe the file as loaded; every distributional field describes the
**deduplicated** data. EDA is descriptive only; nothing in it feeds model choices.

| field | type | meaning |
|-------|------|---------|
| dataset_source, dataset_sha256 | str | provenance |
| n_rows, n_columns | int | raw shape (before dedup) |
| class_counts, fraud_rate | `ClassCounts`, float | class balance of the raw file |
| n_duplicate_rows, n_duplicate_frauds | int | exact duplicates beyond the first occurrence |
| missing_values | {column: int} | missing count per column |
| amount | `{legit, fraud}` of `Distribution` | `count, mean, std, min, q1, median, q3, max, n_zero` |
| amount_histogram | `Histogram` | `bin_edges[]` (n+1), `legit_counts[]`, `fraud_counts[]` (n); the 20 `log1p(Amount)` bins mapped back to raw units with `expm1` |
| time_histogram | `Histogram` | one bin per hour of `Time` |
| feature_separation | `FeatureSeparation[]` | per feature (`V1..V28, Amount`): class means/stds and KS statistic, sorted by KS desc (validated) |
| training_counts | `TrainingCounts[]` | per technique: `technique`, `ratio`, `before`, `after` (`ClassCounts`) on the full training set at the tuned ratio. Filled by the experiment code; `[]` from the EDA script (default since 1.1) |
| imbalance_ratio *(1.1)* | str \| null | `"1 : N"`, `N = round(legit / fraud)` on the raw file |
| class_counts_dedup *(1.1)* | `ClassCounts` \| null | class counts after deduplication |
| amount_summary *(1.1)* | `{raw, log1p}` of `AmountStats` \| null | `min, median, mean, max, skewness` (bias-corrected G1), both classes pooled |
| time_span_hours *(1.1)* | float \| null | `(max Time - min Time) / 3600` |
| effect_sizes *(1.1)* | `EffectSize[]` | per feature: `feature, cohens_d, legit_mean, fraud_mean`; `d = (mean_fraud - mean_legit) / pooled SD`; sorted by `|d|` desc (validated) |
| feature_histograms *(1.1)* | `FeatureHistogram[]` | 20 shared bins for the 6 PCA features with largest `|d|`, then `log1p(Amount)`: `Histogram` fields plus `feature`, `legit_fraction[]`, `fraud_fraction[]` (each class sums to 1) and `bin_range_quantiles` (the bins span the union of each class's quantile range; values beyond it count in the outer bins) |
| hour_of_day *(1.1)* | `HourOfDay` \| null | `fraud_counts[24]`, `legit_counts[24]` with `hour = floor(Time/3600) mod 24`, and `hour_assumption`: `"assumes Time=0 is midnight; not documented by the data provider"` |

Fields marked *(1.1)* are optional, so a 1.0 file still validates.

## resampling_audit.json — `ResamplingAudit`

* **Written by:** `scripts/run_resampling_audit.py` (every technique x every ratio of its search
  space x every CV fold of the TRAINING split; the model step is removed, so `model` is the
  pipeline's model, `logreg`). Later, the CV runner may write the same shape.
* **Read by:** leakage tests, dashboard "how resampling works" page, report.
* **Example:** [resampling_audit.json](contracts/examples/resampling_audit.json).

| field | type | meaning |
|-------|------|---------|
| seed, cv_folds | int | protocol |
| test_rows_resampled | int | must be 0 |
| all_checks_passed | bool | every entry has `validation_untouched` and `test_rows_resampled == 0` |
| entries[] | `ResamplingAuditEntry` | `model, technique, fold, ratio, train_before, train_after, validation, n_synthetic, validation_untouched` |
| base_ratio *(1.2)* | float \| null | `frauds / legit` of the training data the ratios are checked against |
| dropped_ratios *(1.2)* | float[] | `ratio_grid` values not above `base_ratio x 1.05` (imblearn rejects them), so not audited or tuned |

Optional entry fields *(1.2)*: `smote_ratio` (technique E's SMOTE ratio a; `ratio` is then the
final RUS ratio b), `fraud_weight` (technique F's weight `ratio / base_ratio`; data unchanged),
`fraud_unique_after` and `fraud_duplicates_after` (distinct vs exactly-copied fraud rows after
resampling, compared in the preprocessed feature space). `n_synthetic` counts fraud rows after
resampling that equal no row given to the samplers.

## artifacts/models/manifest.json — `Manifest`

* **Written by:** research squad, model exporter (`python scripts/export_artifacts.py --source raw
  --default-cell <model>__<technique>`; `fraudstudy.export`). Written last, after the model files.
* **Read by:** backend at start-up: loads each listed model, checks file checksum and library
  versions, builds input tables in `features` order. If anything is off, `/health` reports
  `"degraded"`.
* **Example:** [manifest.json](contracts/examples/manifest.json).

| field | type | meaning |
|-------|------|---------|
| created_at | str | ISO 8601 UTC |
| git_commit | str \| null | `fraudstudy.io.git_hash()` |
| seed, quick_mode | int, bool | run settings |
| dataset | `{name, source, sha256, n_rows}` | training data |
| default_model_id | str \| null | must be one of `models[].id` |
| models[] | `ManifestEntry` | see below; ids unique |

`ManifestEntry`: `id` (must be `"<model>__<technique>"`), `model`, `technique`, `file`
(relative to `artifacts/models/`, e.g. `deploy/lightgbm__smote_rus.joblib`), `sha256` (64 hex, of
the file), `threshold_f1`, `threshold_f2`, `has_calibrator`, `features` (exact column order the
pipeline expects: `V1..V28, Amount`), `data_sha256`, `library_versions` ({package: version}).

Optional fields *(1.4)*: `model_label`, `technique_label` (display names, as `MODEL_LABELS` /
`TECHNIQUE_LABELS`); `default_threshold` (`"f1"` \| `"f2"`, default `"f1"`: the tuned threshold
applied when a request gives none, reported as `tuned_on_validation_F1` / `_F2`);
`calibrator_file` (relative to `artifacts/models/`) and `calibrator_sha256`, set together and only
when `has_calibrator`. The calibrator is an isotonic regression fitted on the cell's out-of-fold
TRAINING scores; apply it with `fraudstudy.metrics.apply_calibrator` to get
`calibrated_probability`.

Only the deployed subset of the 18 pipelines is listed (Part 9: not all 18 fit on a small server):
`deploy.cells` in `configs/experiment.yaml`, i.e. the six LightGBM pipelines plus logistic
regression for the default cell's technique. The default cell is chosen by the team from the CV
results, never automatically. Each `file` is the cell's pipeline exactly as the test evaluation
scored it (its sampler steps do nothing at prediction time).

**`artifacts/models/feature_reference.json`** (`fraudstudy.explain.FeatureReference`; an internal
file read only through `fraudstudy.explain`, not a contract): per model feature (`V1..V28,
log1p_Amount`), its `raw_column`, 101 quantiles (0th..100th percentile) of that raw column over the
LEGITIMATE training rows, and the feature's training mean in scaled units. Since Prompt 13 it also
holds `range_percentiles` (`deploy.input_range_percentiles`, `[0.1, 99.9]`) and per feature
`train_range`: those percentiles of the raw column over ALL training rows, behind `/predict`'s
"outside the range seen in training" warnings. Training rows only. The backend loads it at
start-up; a missing file is not a problem (predictions then carry no explanation), an invalid one
or one without ranges is.

## presets.json — `PresetsFile`

* **Written by:** research squad, exporter: up to 6 real test-set payments (`legit_everyday`,
  `legit_large`, `fraud_typical`, `fraud_small_amount`, `borderline`, `missed_fraud`), picked
  deterministically with the default model's scores after the single test evaluation (rules in
  `fraudstudy.export.pick_presets` and `deploy.presets`; a preset with no matching row is left
  out). `missed_fraud` is a false negative, kept on purpose. On the raw source the export is
  recorded in the test-set guard. The batch-demo file `data/sample/demo_batch.csv` (500 test rows
  with `Class`, 25 of them fraud: optimistic precision) is written alongside.
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
`threshold` (float in [0, 1], or null for the model's tuned threshold), and *(1.6)*
`input_source` (`"preset"` \| `"edited"` \| `"manual"`, or null). The backend builds the input row
in the manifest entry's `features` order; `Time` is accepted but not used by the main models.

Errors: an invalid body (missing or extra feature, NaN / infinity, a non-numeric string, `Amount`
outside [0, 1 000 000], `threshold` outside [0, 1], unknown `input_source`) → 422 with the field
errors in `details`; unknown `model_id` → 404 with `details = {"valid_ids": [...]}` (the ids that
can score); no manifest, no default model, or a listed model that is not loaded (always the case
with `USE_EXAMPLE_ARTIFACTS=true`, which has no model files) → 503 `artifact_unavailable`; over
60 requests a minute from one client → 429.

| field | type | meaning |
|-------|------|---------|
| label | `"FRAUD"` \| `"LEGITIMATE"` | `"FRAUD"` exactly when `risk_score >= threshold` (validated) |
| risk_score | float | model score in [0, 1]; a ranking score, not a probability |
| calibrated_probability | float \| null | calibrated fraud probability, null if no calibrator |
| threshold | float | threshold applied |
| threshold_source | `"tuned_on_validation_F1"` \| `"tuned_on_validation_F2"` \| `"user_supplied"` \| `"default_0_5"` | where it came from: the request's threshold is `user_supplied`; without one the entry's `default_threshold` gives `tuned_on_validation_F1` / `_F2` (`default_0_5` is not used by the backend) |
| model | `{id, model, technique}` | which model answered (ids; labels via `MODEL_LABELS` / `TECHNIQUE_LABELS`) |
| explanation | `{method, units, top_features[]}` \| null | `fraudstudy.explain.explain`, top 8: `"TreeSHAP"` (LightGBM) or `"linear"` (logreg), units `"log-odds"`; `top_features[]` (largest \|contribution\| first): `feature` (model feature, e.g. `log1p_Amount`), `value` (its RAW input value, e.g. the Amount), `contribution`, `legit_percentile` (0-100, among legitimate training rows). Null, with a warning saying why, when `feature_reference.json` is missing or does not fit the model |
| warnings | str[] | in order: `"Edited transaction may not resemble real payments; the model is extrapolating."` (`input_source` edited / manual); `"<column> is outside the range seen in training (below its 0.1th percentile)."` (or above its 99.9th) per input outside its `train_range`; no explanation; no calibrated probability because the calibrator failed to load |
| disclaimer | str | research-only notice |
| latency_ms | float | server time |
| technique_scores *(1.6)* | `TechniqueScore[]` | the same payment scored by every loaded LightGBM model (manifest order): `id, model, technique, risk_score, threshold, label`. Each uses its OWN tuned default threshold, also when the request gives one (resampling shifts score scales, so one cut-off would not compare techniques fairly); `label` is validated against `risk_score >= threshold`. Empty when no LightGBM model is loaded |

The backend must never log feature values.

## POST /api/v1/predict/batch — `BatchResponse`

* **Written by:** backend (CSV upload, field `file`, ≤ 5 MB, ≤ 10 000 rows, parsed in memory).
  **Read by:** dashboard batch page (builds the downloadable CSV from `results`).
* **Example:** [batch_response.json](contracts/examples/batch_response.json).

Request: `multipart/form-data` with one field `file` whose filename ends in `.csv` (UTF-8, comma-
separated, header row), and optional query parameters `model_id` and `threshold` (as in
`/predict`). The body is streamed and kept in memory, never written to disk; reading stops once the
file passes `MAX_UPLOAD_MB`. Columns: the model's `features` (`V1..V28, Amount`) are required;
`Time` (unless the model uses it) and unknown columns are ignored (unknown ones are named in
`warnings`); an optional `Class` (0/1) adds `metrics` (`fraudstudy.metrics.compute_metrics` at
the threshold used; null, with a warning, if the scored rows do not hold both classes). An EMPTY
cell is a missing value, filled by the pipeline's median imputer and counted in `warnings`, except
an empty `Amount`: that row goes to `errors` and is not scored (its amount could not be shown or
summed). Every row is scored in one vectorised call; `results` keep the input order.

Errors: a filename not ending in `.csv`, no `file` field or a body that is not multipart → 400;
missing columns → 400 with `details = {"missing_columns": [...]}`; a cell that is not a finite
number (text such as `abc` or `NaN`, `inf`), an `Amount` outside [0, 1 000 000] or a `Class` other
than 0/1 → 400 with `details = {"rows": [first 5], "n_rows", "columns"}` (0-based data rows, never
the cell values); a row with more fields than the header, duplicate column names, non-UTF-8 or an
empty file → 400; a file over `MAX_UPLOAD_MB` → 413 (`details.max_bytes`); more than
`MAX_BATCH_ROWS` data rows → 413 (`details.max_rows`); `model_id` / threshold as in `/predict`;
over 10 uploads a minute from one client → 429.

Prompt 13 described the summary as `{rows, flagged, flagged_pct, flagged_amount_total, metrics,
warnings}` and the rows as `{row, risk_score, label, amount, true_class}`; those are this
contract's `n_rows`, `n_flagged`, `flagged_fraction` (a fraction, not a percent),
`flagged_amount_total`, `metrics`, `warnings` and `results[]` (plus `true_class`, added in 1.6).
The names were kept to avoid a breaking change.

| field | type | meaning |
|-------|------|---------|
| model, threshold, threshold_source | as in `PredictResponse` | |
| n_rows, n_scored, n_flagged | int | counts |
| flagged_fraction | float | `n_flagged / n_scored` |
| flagged_amount_total | float | sum of `Amount` over flagged rows |
| metrics | `Metrics` \| null | only if the CSV had a `Class` column |
| results[] | `{row, amount, risk_score, calibrated_probability, label, true_class}` | `row` is 0-based, header excluded; `true_class` *(1.6, optional)* is the row's `Class`, or null without that column |
| errors[] | `{row, message}` | rows that could not be scored (an empty `Amount`) |
| warnings, disclaimer, latency_ms | as in `PredictResponse` | |

The backend must never log or store uploaded file contents.

## Errors — `ErrorResponse`

Every non-2xx response body is
`{"error": {"code": "<snake_case>", "message": "<human text>", "details": <any JSON or null>}}`.

`details` *(1.5, optional)* carries context: the field errors of a 422 (`[{loc, msg, type}]`,
never the submitted values), `{"artifacts": [...], "problems": [...]}` for a 503, the request id
for a 500. The backend always sends it (`null` when there is nothing to add); a body without it
(1.4) still validates. Codes the backend uses so far: `bad_request` (400), `not_found` (404),
`method_not_allowed` (405), `payload_too_large` (413), `validation_error` (422), `rate_limited`
(429), `internal_error` (500), `artifact_unavailable` and `service_unavailable` (503).

## GET endpoints of the backend *(1.5)*

* **Written by:** backend (`backend/app/routers/`), from the artifacts it loaded and validated
  once at start-up. **Read by:** dashboard.
* No example files: run the backend with `USE_EXAMPLE_ARTIFACTS=true` to serve these shapes
  built from the examples above.

`GET /api/v1/health` — `HealthResponse`. Always 200, also when degraded.

| field | type | meaning |
|-------|------|---------|
| status | `"ok"` \| `"degraded"` | `"degraded"` exactly when `problems` is non-empty (validated) |
| problems | str[] | what is wrong and how to fix it (missing / invalid artifact, model checksum or library-version mismatch, unknown default model, unsafe production settings) |
| models_loaded | int | model files loaded from the manifest (0 with example artifacts) |
| default_model | str \| null | manifest id used when a request names none (`DEFAULT_MODEL_ID` overrides the manifest) |
| data_sha256 | str \| null | training data hash from the manifest |
| git_hash | str \| null | the backend's commit (null outside a git checkout) |
| version | str | backend version |
| uptime_s | float | seconds since start-up |

`GET /api/v1/models` — `ModelsResponse`: `default_model_id` (one of the ids, or null) and
`models[]` (`ModelInfo`): `id`, `model`, `technique`, `model_label`, `technique_label`,
`threshold_f1`, `threshold_f2`, `default_threshold` (`"f1"` \| `"f2"`), `has_calibrator`,
`features` (input column order) and `loaded` (its file loaded at start-up, so it can score;
always false with example artifacts). 503 if `manifest.json` is unavailable.

`GET /api/v1/results` — `ResultsResponse`: `comparison` (`ComparisonTable`), `curves`
(`CurvesFile`), `stats` (`StatsResult`) and `temporal_check` (`temporal_check.json` as written by
`scripts/run_temporal_check.py`, a labelled sensitivity analysis whose inner shape is internal,
or null). 503 if any of the first three is unavailable.

`GET /api/v1/eda` — `EdaResponse`: `eda` (`EdaSummary`) and `resampling_audit`
(`ResamplingAudit` or null). 503 if `eda.json` is unavailable.

`GET /api/v1/presets` — `PresetsFile`, as written. 503 if `presets.json` is unavailable.

The wrappers carry their own `contract_version`; their `is_example` is true if any part is an
example.

## POST /api/v1/smote-demo — `SmoteDemoRequest` → `SmoteDemoResponse`

* **Written by:** backend (`backend/app/services/smote_demo.py`), on a fixed 2-D toy dataset. It
  runs a transparent re-implementation of imblearn's `SMOTE` that follows the same procedure
  (`NearestNeighbors(k + 1)` over the fraud points; `RandomState(seed)` draws `randint` picks of
  parent and neighbour, then `uniform` gaps), so for the same points, `k`, count and seed it gives
  exactly imblearn's synthetic points (a test checks it); it exists to expose each point's parent,
  neighbour and gap. The experiment itself uses imblearn. **Read by:** dashboard SMOTE explainer
  (whose built-in JS fallback should produce the same shape).
* **Example:** [smote_demo.json](contracts/examples/smote_demo.json).

Request (all optional, default in brackets): `scenario` (`single_cluster` \| `two_clusters`
[default] \| `overlap`), `n_legit` (2-2000 [300]), `n_fraud` (2-200 [20]), `k_neighbors`
(1-15 [5], must be < `n_fraud`), `sampling_strategy` ((0, 1] [0.5]), `seed` ([42]; 0 to
2^32 - 1, else 422), and *(1.6)* `oversample_percent` (100-500 in steps of 100, or null). The
number of synthetic points is `n_fraud * oversample_percent / 100` when that is given (then
`sampling_strategy` is ignored, with a warning), else `int(n_legit * sampling_strategy -
n_fraud)` (imblearn's rule; 0 with a warning when not positive). The toy data depends only on
`scenario`, `n_legit` and `n_fraud` (never on `seed`); the default has 300 legitimate points and
two separate fraud clusters of 10. Legitimate points come first in `original`, then the fraud
points. Over 30 requests a minute from one client → 429.

| field | type | meaning |
|-------|------|---------|
| request | `SmoteDemoRequest` | echo of the request |
| original[] | `{x, y, label}` | toy data |
| synthetic[] | `{x, y, parent, neighbor, gap, in_legit_region}` | x_new = x + λ(x̂ − x) with x = `original[parent]`, x̂ = `original[neighbor]`, λ = `gap` in [0, 1); `in_legit_region` when most of its 5 nearest original points are legitimate |
| n_synthetic_in_legit_region | int | synthetic frauds that landed among legit points (SMOTE failure cases) |
| warnings | str[] | `sampling_strategy` ignored; nothing generated; "large k can connect distant fraud clusters, creating synthetic points in legitimate territory" (some neighbour lists join two clusters); how many synthetic points landed among legitimate ones |
| fraud_neighbors *(1.6)* | `{point, neighbors[]}[]` | per fraud point (index into `original`), its `k_neighbors` nearest other fraud points, nearest first (indices into `original`) |
| decision_grid *(1.6)* | `{model, x[], y[], before[][], after[][]}` \| null | `DecisionTreeClassifier(max_depth=4, random_state=seed)` labels (0/1) on a 40 x 40 grid over the data, trained on `original` (`before`) and on `original` + `synthetic` (`after`); `before[i][j]` is the label at `(x[j], y[i])` (shape validated) |

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
