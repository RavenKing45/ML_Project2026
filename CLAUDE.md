# CLAUDE.md — rules for every coding session in this repository

## The project

"An Empirical Study of Class-Imbalance Handling Techniques for Credit Card Fraud Detection".
Two halves:

1. A controlled, leakage-free experiment: 6 techniques (A baseline, B RandomOverSampler,
   C RandomUnderSampler, D SMOTE, E SMOTE then RandomUnderSampler, F class weighting)
   x 3 models (logistic regression, random forest, LightGBM) on the ULB credit-card dataset.
2. A FastAPI backend and a React + TypeScript dashboard that only READ saved artifacts.

The master guide is docs/master_guide.md. Where it disagrees with docs/decisions.md,
decisions.md wins: it records later changes, including the October 2026 simplification.

## Simple enough to explain (the rule above all the others)

Our examiner opens any file and asks any of us to explain any line. So:

1. Write the most obvious code that is correct. A first-year student who knows basic Python,
   pandas and scikit-learn must be able to follow every file.
2. Plain functions and plain dicts. No classes unless a library needs one (a scikit-learn
   transformer, a Pydantic request body, a React component). No dataclasses, enums, Protocols,
   generics, custom decorators, registries, factories or "manager" layers.
3. One job per function. Aim for functions under 40 lines and files under 250 lines. One
   straight-line function is better than five tiny ones that call each other.
4. Check only what really goes wrong (missing file, wrong column, bad user input), each with one
   clear message saying what to fix. No defensive code for things that cannot happen here.
5. Plain names: fraud_rate, train_rows, best_threshold. No abbreviations except X, y, df, cv.
6. Every file starts with 2-4 plain sentences: what it does, who calls it, what it reads and
   writes.
7. Comments explain WHY, at the places a student would get stuck (for example, why the sampler
   sits inside the pipeline). Do not comment obvious lines. Docstrings are one or two sentences;
   Args/Returns sections only for functions with more than 3 parameters.
8. A number used once can be written inline with a short comment. Only values the team may change,
   or that must be the same everywhere (seed, ratio grid, CV folds, n_iter, upload limits), go in
   configs/*.yaml.
9. Use the library instead of writing your own (scikit-learn metrics, imblearn samplers,
   scipy.stats, pandas). The one exception is the from-scratch SMOTE in backend/app/smote_demo.py,
   which exists to teach.
10. No new dependency without a one-line reason in docs/decisions.md.
11. Simple type hints on function signatures only (int, float, str, list, dict, pd.DataFrame).
12. Before creating a file, check whether an existing file is the natural home for the code.

## Before you change anything

1. Read README.md, docs/decisions.md, docs/contracts.md and every file you will touch.
2. Print the relevant part of the repo tree.
3. Restate the task in 3-5 bullets and list the files you will create or change.
4. If the task conflicts with docs/decisions.md or docs/contracts.md, STOP and report the
   conflict. Never silently change an architectural decision or the shape of a JSON file.

## Scientific rules (never break these)

- Never apply any resampler (SMOTE, RandomOverSampler, RandomUnderSampler, ADASYN,
  BorderlineSMOTE) outside an imblearn.pipeline.Pipeline, and never before the train/test split.
- Never resample validation or test data. Samplers run only inside Pipeline.fit.
- Every preprocessing step (imputer, log transform, scaler) lives inside the pipeline so it is
  fitted on training folds only.
- Never use the test set for tuning, threshold choice, model choice, calibration or early
  stopping. Only scripts/evaluate_test.py reads the test set, once.
- One seed from configs/experiment.yaml (`seed`) for every split, sampler and model.
- Raw `Time` is not a model feature. Do not remove outliers.
- Primary metric: average precision (PR-AUC). Never optimise or select on accuracy.
- Every technique gets the same ratio grid and the same tuning budget (configs/experiment.yaml).

## Engineering rules

- Python 3.11. Package code lives in src/fraudstudy/.
- Read settings from configs/*.yaml through fraudstudy.config; build paths from the project root.
- Do not rewrite, rename or reformat files you were not asked to change.
- Every scientific behaviour has a test (leakage, ratios, metrics, splits, the one-time test-set
  guard). Never delete, skip or loosen those tests. Code that was removed may lose its tests.
- The whole Python test suite runs in under 90 s. Tests use data/sample/ only.
- Never commit data/raw/, .env files, or any file over 10 MB.
- Frontend: TypeScript, plain React, plain CSS, Recharts. No `any`.
- Never log transaction feature values or uploaded file contents.

## Git

- Work on a branch made from shaurya and merge it back into shaurya. Never commit to, merge
  into or push to main.
- Before the first commit, run git config user.name and git config user.email and check they
  are the team member's own name and email. If not, stop and ask.
- Commits and pull requests are written as that team member. Never add Co-Authored-By,
  Claude-Session or "Generated with" lines, and never use Claude as the author.

## When you finish

1. Run: ruff check . ; pytest -q ; for frontend work also: cd frontend && npm run build && npm test.
2. Fix any failure you caused. Do not call the task done with failing checks.
3. Report:
   - Files created / changed, one line each on what and why
   - Line counts of the files you touched: before -> after
   - Commands run and their results
   - Anything not done, and why
   - Assumptions you made
   - Open questions for the team
