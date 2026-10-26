# CLAUDE.md — rules for every coding session in this repository

## The project
"An Empirical Study of Class-Imbalance Handling Techniques for Credit Card Fraud Detection",
a student group project. We compare 6 ways of handling class imbalance (A baseline,
B RandomOverSampler, C RandomUnderSampler, D SMOTE, E SMOTE then RandomUnderSampler,
F class weighting) on 3 models (logistic regression, random forest, LightGBM) using the ULB
credit-card dataset, without data leakage, and show the results in one Streamlit app.

## Where things are (the final layout: do not add folders or files unless the task says so)
data/raw/creditcard.csv          the real dataset (Git-ignored; download from Kaggle)
data/sample/creditcard_synthetic.csv   small fake copy with the same columns, for tests and quick runs
model_build/                     our from-scratch SMOTE (a teammate's code: change only when asked)
src/fraudstudy/__init__.py       marks fraudstudy as a package
src/fraudstudy/data.py           load, check, remove duplicates, split 70/30
src/fraudstudy/experiments.py    preprocessing, the 6 techniques, the 3 models, tuning, cross-validation
src/fraudstudy/evaluation.py     the one-time test-set evaluation, metrics, confidence intervals,
                                 significance tests, figures
notebooks/01_eda.ipynb           exploration; writes artifacts/figures/eda_*.png and
                                 artifacts/metrics/eda_summary.json
artifacts/figures/               saved figures
artifacts/metrics/               saved results (cells/*.json, test_results.json, stats.json, ...)
artifacts/models/                trained pipelines; only lightgbm__*.joblib are committed (each under 10 MB)
artifacts/sample/                output of runs on the fake sample (Git-ignored)
tests/test_no_leakage.py, tests/test_smote.py
app.py                           the Streamlit presentation app (comes in P3)
README.md                        front page: how to run, design decisions, results
requirements.txt                 every library pinned, plus -e . and -e ./model_build
pyproject.toml, CLAUDE.md, .gitignore, LICENSE

## Simple enough to explain (the rule above all the others)
Our examiner opens any file and asks any of us to explain any line. So:
1. Write the most obvious correct code. A first-year student who knows basic Python, pandas
   and scikit-learn must be able to follow every file.
2. Plain functions and plain dicts. No classes unless a library needs one. No dataclasses,
   enums, custom decorators, registries, factories or "manager" layers.
3. One job per function, aim for under 40 lines. A straight-line function beats five tiny ones.
4. Check only what really goes wrong (missing file, wrong columns), each with one clear message.
5. Plain names: fraud_rate, train_rows, best_threshold. Abbreviations only X, y, df, cv.
6. Each file starts with 2-4 plain sentences: what it does, who uses it, what it reads and writes.
7. Comments explain WHY at the places a student would get stuck. Docstrings: one or two sentences.
8. Settings live in ONE block called SETTINGS at the top of the file that uses them. Numbers used
   once can be inline with a short comment.
9. Use the library instead of writing your own (scikit-learn, imblearn, scipy, statsmodels,
   pandas, matplotlib). The only exception is the teaching SMOTE in model_build/.
10. No new dependency without a one-line reason in the README's "Design decisions" table.

## Scientific rules (never break these)
- Never apply any resampler outside an imblearn.pipeline.Pipeline, and never before the
  train/test split. Samplers run only inside Pipeline.fit, on training rows only.
- Every preprocessing step (imputer, log transform, scaler) lives inside the pipeline.
- Never use the test set for tuning, threshold choice, model choice or early stopping. Only
  evaluate_test_set() in evaluation.py reads the test set, once, behind its guard file.
- One seed (SEED in data.py) for every split, sampler and model.
- Raw Time is not a model feature. Do not remove outliers.
- Primary metric: average precision (PR-AUC). Never optimise or select on accuracy.
- Every technique gets the same ratio grid and the same tuning budget (SETTINGS in
  experiments.py).

## Git
- Work directly on branch shaurya. Commit locally; push only with: git push origin shaurya.
  Never check out, commit to, merge into or push to main, and never create other branches.
- Do not push until the team member says the step passed review.
- Before the first commit, run git config user.name and git config user.email and check they
  are the team member's own. If not, stop and ask.
- Commits are written as that team member. Never add Co-Authored-By, Claude-Session or
  "Generated with" lines, and never use Claude as the author.
- Never commit data/raw/, .env files, or any file over 10 MB.

## When you finish
1. Run pytest -q (under 60 s). Fix any failure you caused.
2. Never delete, skip or loosen a test in tests/ to make it pass.
3. Report: files created / changed / deleted (one line each, with line counts), commands run
   and their results, anything not done and why, assumptions, open questions for the team.
