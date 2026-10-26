"""The experiment: 6 imbalance techniques x 3 models, each tuned and cross-validated on the
TRAINING split only.

For every (model, technique) cell, run_cell() tunes the pipeline, picks the F1 threshold on
out-of-fold scores and scores the same repeated-CV folds as every other cell. Reads the data
through data.py; writes artifacts/models/<cell>.joblib and artifacts/metrics/cells/<cell>.json
(artifacts/sample/... for the fake sample). The test split is never used here.

Run it: python -m fraudstudy.experiments [--source sample] [--quick] [--models ...]
    [--techniques ...] [--force]
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import (
    ParameterGrid,
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.preprocessing import FunctionTransformer, RobustScaler

from fraudstudy.data import PROJECT_ROOT, SEED, load_data, remove_duplicates, split_data

# ---- SETTINGS ------------------------------------------------------------------------------------
CV_FOLDS = 5  # folds of the tuning search and of the out-of-fold scores
CV_REPEATS = 3  # the repeated CV gives CV_FOLDS x CV_REPEATS fold scores per cell for the tests
N_ITER = 20  # tuning budget: the same number of tries for every cell
# A ratio r means "after resampling, frauds = r x legit". Every technique uses this same grid.
RATIO_GRID = [0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
HYBRID_SMOTE_RATIOS = [0.005, 0.01, 0.05]  # E: SMOTE up to a, then ...
HYBRID_RUS_RATIOS = [0.1, 0.5, 1.0]  # ... undersample the legit rows down to b (only a < b)
SMOTE_K = 5  # neighbours SMOTE picks from
TECHNIQUES = ["baseline", "ros", "rus", "smote", "smote_rus", "class_weight"]  # A to F
MODELS = ["logreg", "random_forest", "lightgbm"]
MODEL_PARAM_SPACES = {
    "logreg": {"C": [0.001, 0.01, 0.1, 1, 10], "max_iter": [2000]},
    "random_forest": {"n_estimators": [200], "max_depth": [8, 12, 16, None],
                      "min_samples_leaf": [1, 5, 20], "max_features": ["sqrt", 0.5],
                      "max_samples": [0.5, None]},
    "lightgbm": {"n_estimators": [200, 400], "learning_rate": [0.03, 0.1],
                 "num_leaves": [15, 31, 63], "min_child_samples": [20, 50],
                 "subsample": [0.8, 1.0], "subsample_freq": [1], "colsample_bytree": [0.8, 1.0]},
}
# --quick: fewer folds and tries, and (on the real data only) 10% of the training rows.
QUICK = {"cv_folds": 3, "cv_repeats": 1, "n_iter": 3, "train_fraction": 0.10}
N_JOBS = -1  # random forest and LightGBM use every core; the search fits one candidate at a time
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
# --------------------------------------------------------------------------------------------------


def artifacts_dir(source: str) -> Path:
    """artifacts/ for the real data, artifacts/sample/ (Git-ignored) for the fake sample."""
    return ARTIFACTS_DIR if source == "raw" else ARTIFACTS_DIR / "sample"


def build_preprocessor() -> ColumnTransformer:
    """Median imputer on every column, then log1p(Amount) + RobustScaler, RobustScaler on V1..V28.

    RobustScaler computes (x - median) / IQR, so the extreme amounts of frauds do not squash the
    other values. It sits inside the pipeline, so it learns its medians from training rows only.
    """
    v_columns = Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", RobustScaler())])
    # Amount is very skewed (most payments are small, a few are huge); log1p evens that out.
    amount = Pipeline([("impute", SimpleImputer(strategy="median")),
                       ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                       ("scale", RobustScaler())])
    preprocessor = ColumnTransformer(
        [("v_columns", v_columns, [f"V{i}" for i in range(1, 29)]),
         ("amount", amount, ["Amount"])],
        verbose_feature_names_out=False,
    )
    # pandas output keeps the row index, which the leakage tests use to see which rows a step got.
    return preprocessor.set_output(transform="pandas")


def build_model(model: str) -> object:
    """An untrained classifier with random_state=SEED; its settings come from the search."""
    if model == "logreg":
        return LogisticRegression(solver="lbfgs", random_state=SEED)
    if model == "random_forest":
        return RandomForestClassifier(random_state=SEED, n_jobs=N_JOBS)
    return LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=N_JOBS)


def build_pipeline(model: str, technique: str) -> Pipeline:
    """imblearn Pipeline [preprocess, the technique's sampler step(s), model].

    An imblearn Pipeline runs its samplers only inside fit, on the rows fit gets; predict and
    predict_proba skip them, so validation and test rows are never resampled. The ratios (and
    technique F's class weight) are set later by the search, through search_space().
    """
    smote = SMOTE(k_neighbors=SMOTE_K, random_state=SEED)
    samplers = {
        "baseline": [],
        "ros": [("ros", RandomOverSampler(random_state=SEED))],
        "rus": [("rus", RandomUnderSampler(random_state=SEED))],
        "smote": [("smote", smote)],
        "smote_rus": [("smote", smote), ("rus", RandomUnderSampler(random_state=SEED))],
        "class_weight": [],  # F changes the model's class weight, not the rows
    }
    return Pipeline([("preprocess", build_preprocessor()), *samplers[technique],
                     ("model", build_model(model))])


def search_space(model: str, technique: str, y_train: pd.Series) -> dict | list:
    """The model's parameters plus the technique's ratios, all taken from the same RATIO_GRID.

    imblearn refuses a ratio at or below the current frauds / legit ratio, so those are left out
    (with 5% headroom for CV folds that hold slightly more frauds). Class weighting maps each
    ratio r to the fraud weight w = r / base: then w x frauds = r x legit, the same balance a
    sampler at ratio r would create.
    """
    base = (y_train == 1).sum() / (y_train == 0).sum()
    ratios = [r for r in RATIO_GRID if r > base * 1.05]
    space = {f"model__{name}": values for name, values in MODEL_PARAM_SPACES[model].items()}
    if technique in ["ros", "rus", "smote"]:
        space[f"{technique}__sampling_strategy"] = ratios
    elif technique == "smote_rus":
        pairs = [(a, b) for a in HYBRID_SMOTE_RATIOS if a > base * 1.05
                 for b in HYBRID_RUS_RATIOS if b > base * 1.05 and a < b]
        # One small grid per (a, b) pair, so a is always below b.
        return [{**space, "smote__sampling_strategy": [a], "rus__sampling_strategy": [b]}
                for a, b in pairs]
    elif technique == "class_weight" and model == "lightgbm":
        space["model__scale_pos_weight"] = [float(r) / float(base) for r in ratios]
    elif technique == "class_weight":
        space["model__class_weight"] = [{0: 1.0, 1: float(r) / float(base)} for r in ratios]
        space["model__class_weight"].append("balanced")
    return space


def best_f1_threshold(y_true: pd.Series, scores: np.ndarray) -> float:
    """The score cut-off with the highest F1 = 2PR / (P + R); ties go to the higher cut-off.

    Only ever given out-of-fold TRAINING scores: the test set never chooses the threshold.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    precision, recall = precision[:-1], recall[:-1]  # the last point has no threshold
    top, bottom = 2 * precision * recall, precision + recall
    f1 = np.divide(top, bottom, out=np.zeros_like(top), where=bottom > 0)
    return float(thresholds[f1 == f1.max()].max())


def class_counts(y: pd.Series) -> dict:
    """{"legit": number of 0 labels, "fraud": number of 1 labels}."""
    y = np.asarray(y)
    return {"legit": int((y == 0).sum()), "fraud": int((y == 1).sum())}


def counts_after_resampling(pipeline: Pipeline, X_train: pd.DataFrame,
                            y_train: pd.Series) -> dict:
    """The training class counts that the model inside `pipeline` is fitted on.

    Pipeline.fit keeps the resampled rows to itself, so the preprocess step and the sampler
    steps are run here by hand, on fresh copies, on TRAINING rows only.
    """
    X_seen = clone(pipeline.named_steps["preprocess"]).fit_transform(X_train)
    y_seen = y_train
    for _, sampler in pipeline.steps[1:-1]:  # every step between preprocess and model
        X_seen, y_seen = clone(sampler).fit_resample(X_seen, y_seen)
    return class_counts(y_seen)


def run_cell(model: str, technique: str, X_train: pd.DataFrame, y_train: pd.Series,
             quick: bool) -> tuple[dict, Pipeline]:
    """Tune and score one cell on TRAINING rows; return (results dict, best fitted pipeline)."""
    settings = QUICK if quick else {"cv_folds": CV_FOLDS, "cv_repeats": CV_REPEATS,
                                    "n_iter": N_ITER}
    cv = StratifiedKFold(n_splits=settings["cv_folds"], shuffle=True, random_state=SEED)
    space = search_space(model, technique, y_train)
    # Same budget for every cell; a space with fewer settings than the budget is tried in full.
    n_iter = min(settings["n_iter"], len(ParameterGrid(space)))
    search = RandomizedSearchCV(build_pipeline(model, technique), space, n_iter=n_iter,
                                scoring="average_precision", cv=cv, random_state=SEED,
                                n_jobs=1, error_score="raise")
    search.fit(X_train, y_train)
    best = search.best_estimator_

    # Out-of-fold scores: every training row scored by a model that did not see it.
    oof = cross_val_predict(clone(best), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    # The repeated folds depend only on SEED and the labels, so every cell gets the same folds
    # and evaluation.py can compare techniques fold by fold.
    repeated = RepeatedStratifiedKFold(n_splits=settings["cv_folds"],
                                       n_repeats=settings["cv_repeats"], random_state=SEED)
    fold_scores = cross_validate(clone(best), X_train, y_train, cv=repeated,
                                 scoring="average_precision", error_score="raise")["test_score"]
    results = {
        "model": model, "technique": technique, "quick": quick,
        "best_params": search.best_params_,
        "cv_pr_auc_mean": float(np.mean(fold_scores)),
        "cv_pr_auc_std": float(np.std(fold_scores)),
        "fold_pr_auc": [float(score) for score in fold_scores],
        "threshold_f1": best_f1_threshold(y_train, oof),
        "train_counts_before": class_counts(y_train),
        "train_counts_after": counts_after_resampling(best, X_train, y_train),
        "fit_seconds": float(search.refit_time_),
        "n_iter": n_iter,
    }
    return results, best


def save_cell(results: dict, pipeline: Pipeline, source: str) -> None:
    """Write the pipeline (.joblib) and then the results (.json) of one cell.

    The .json is written last, so its presence means the cell finished. Laptops that split the
    work merge their results by copying their metrics/cells/*.json and models/*.joblib files into
    the same folders on one machine.
    """
    name = f"{results['model']}__{results['technique']}"
    models_dir = artifacts_dir(source) / "models"
    cells_dir = artifacts_dir(source) / "metrics" / "cells"
    models_dir.mkdir(parents=True, exist_ok=True)
    cells_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, models_dir / f"{name}.joblib")
    # default=str turns the few values JSON cannot write (e.g. class-weight dict keys) into text.
    text = json.dumps(results, indent=2, default=str)
    (cells_dir / f"{name}.json").write_text(text, encoding="utf-8")


def load_training_data(source: str, quick: bool) -> tuple[pd.DataFrame, pd.Series]:
    """Load, deduplicate and split the data; return ONLY the training part.

    In quick mode on the real data, a stratified 10% of the training rows is kept (the fake sample
    is already small, so it is used whole).
    """
    df, _, _ = remove_duplicates(load_data(source))
    X_train, _, y_train, _ = split_data(df)
    if quick and source == "raw":
        X_train, _, y_train, _ = train_test_split(X_train, y_train,
                                                  train_size=QUICK["train_fraction"],
                                                  stratify=y_train, random_state=SEED)
    return X_train, y_train


def main() -> None:
    """Run every requested cell that is not finished yet (or every one with --force)."""
    parser = argparse.ArgumentParser(description="Tune and cross-validate every cell.")
    parser.add_argument("--source", choices=["raw", "sample"], default="raw")
    parser.add_argument("--quick", action="store_true", help="fewer folds and tries")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--techniques", nargs="+", choices=TECHNIQUES, default=TECHNIQUES)
    parser.add_argument("--force", action="store_true", help="re-run finished cells")
    args = parser.parse_args()

    X_train, y_train = load_training_data(args.source, args.quick)
    print(f"{len(y_train):,} training rows, {int(y_train.sum())} frauds (test set not loaded)")
    for model in args.models:
        for technique in args.techniques:
            name = f"{model}__{technique}"
            cell_file = artifacts_dir(args.source) / "metrics" / "cells" / f"{name}.json"
            done = cell_file.is_file() and json.loads(cell_file.read_text())["quick"] == args.quick
            if done and not args.force:
                print(f"{name:<28} already done (use --force to run it again)")
                continue
            start = time.perf_counter()
            results, pipeline = run_cell(model, technique, X_train, y_train, args.quick)
            save_cell(results, pipeline, args.source)
            print(f"{name:<28} CV PR-AUC {results['cv_pr_auc_mean']:.4f} "
                  f"+- {results['cv_pr_auc_std']:.4f}  F1 threshold "
                  f"{results['threshold_f1']:.3f}  {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
