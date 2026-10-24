# CLAUDE.md — rules for every coding session in this repository

## The project

"An Empirical Study of Class-Imbalance Handling Techniques for Credit Card Fraud Detection".
Two halves:

1. A controlled, leakage-free experiment: 6 techniques (A baseline, B RandomOverSampler,
   C RandomUnderSampler, D SMOTE, E SMOTE then RandomUnderSampler, F class weighting)
   x 3 models (logistic regression, random forest, LightGBM) on the ULB credit-card dataset.
2. A FastAPI backend and a React + TypeScript dashboard that only READ saved artifacts.

The master guide (docs/master_guide.md or the team doc) explains every design decision.

## Before you change anything

1. Read README.md, docs/contracts.md, docs/decisions.md and every file you will touch.
2. Print the relevant part of the repo tree.
3. Restate the task in 3-5 bullets and list the files you will create or modify.
4. If the task conflicts with docs/decisions.md or docs/contracts.md, STOP and report the
   conflict. Never silently change an architectural decision or a data contract.

## Scientific rules (never break these)

* Never apply any resampler (SMOTE, RandomOverSampler, RandomUnderSampler, ADASYN,
  BorderlineSMOTE) outside an imblearn.pipeline.Pipeline, and never before the train/test split.
* Never resample validation or test data. Samplers run only inside Pipeline.fit.
* Every preprocessing step (imputer, log transform, scaler) lives inside the pipeline so it is
  fitted on training folds only.
* Never use the test set for tuning, threshold choice, model choice, calibration or early
  stopping. Only scripts/evaluate_test.py reads the test set, once.
* One seed from configs/experiment.yaml (`seed`) for every split, sampler and model.
* Raw `Time` is not a model feature in the main experiment. Do not remove outliers.
* Primary metric: average precision (PR-AUC). Never optimise or select on accuracy.
* Every technique gets the same ratio grid and the same tuning budget (configs/experiment.yaml).

## Engineering rules

* Python 3.11. Package code lives in src/fraudstudy/. Type hints on every function.
* Docstrings on public functions: purpose, args, returns, and the maths if any.
* No hard-coded paths or magic numbers: read configs/*.yaml via fraudstudy.config, resolve
  paths from the project root (fraudstudy.paths).
* Small modular functions. Notebooks may import from src; src never imports notebooks.
* Preserve working code: do not rewrite, rename or reformat files you were not asked to change.
  Never delete or weaken a test to make it pass.
* Clear errors: raise exceptions from fraudstudy.errors with messages that say what to fix.
* Never commit data/raw/, .env files, or any file over 10 MB. Tests use data/sample/ only.
* Every new function gets a pytest test. The whole Python test suite must run in under 90 s.
* Frontend: TypeScript strict mode; no `any` without a comment explaining why.
* Never log transaction feature values or uploaded file contents.

## When you finish

1. Run: ruff check . ; pytest -q ; and for frontend work: npm run build ; npm test.
2. Fix any failure you caused. Do not mark the task done with failing checks.
3. Report in this format:
   * Files created / changed (one line each on what and why)
   * Commands run and their results
   * Anything not done, and why
   * Assumptions you made
   * Open questions for the team
