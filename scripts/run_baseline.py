"""Cross-validate technique A (baseline) with logistic regression on the TRAINING split only.

Usage::

    python scripts/run_baseline.py --source raw|sample [--quick] [--no-verify-hash]
                                   [--results-dir DIR]

Steps: load -> deduplicate -> stratified split -> stratified k-fold cross_validate on the
training part, scored by PR-AUC (average precision) and ROC-AUC. The test part is never used.
--quick keeps a stratified quick.train_fraction of the training rows and uses quick.cv_folds.
With --results-dir the fold scores are also written to DIR/baseline_cv.json (used by
scripts/run_all_sample.py). Prints only scores, never feature values.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from fraudstudy.config import get_dataset_config, get_experiment_config, get_models_config
from fraudstudy.data import deduplicate, load_dataset, split_dataset
from fraudstudy.io import save_json
from fraudstudy.pipeline import build_pipeline

MODEL = "logreg"
TECHNIQUE = "baseline"
SCORING = {"PR-AUC": "average_precision", "ROC-AUC": "roc_auc"}


def default_model_params(models_cfg: dict) -> dict:
    """Untuned logistic regression: only max_iter from models.yaml; C stays at sklearn's 1.0."""
    return {"max_iter": models_cfg[MODEL]["max_iter"][0]}


def quick_subsample(X: pd.DataFrame, y: pd.Series, fraction: float,
                    seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """Keep a stratified `fraction` of the training rows for --quick (all rows if >= 1)."""
    if fraction >= 1.0:
        return X, y
    X_sub, _, y_sub, _ = train_test_split(X, y, train_size=fraction, stratify=y, random_state=seed)
    return X_sub, y_sub


def run_cv(X: pd.DataFrame, y: pd.Series, cfg: dict, exp_cfg: dict, model_params: dict,
           n_folds: int) -> dict:
    """Stratified k-fold cross_validate of the baseline pipeline; returns {metric: fold scores}.

    The whole pipeline (imputer, log, scalers, model) is refitted inside each fold, so a
    validation fold never influences any fitted statistic.
    """
    n_frauds = int((y == 1).sum())
    if n_frauds < n_folds:
        raise ValueError(
            f"Only {n_frauds} frauds in the training data, fewer than {n_folds} CV folds. Use "
            "more data (drop --quick or use --source raw)."
        )
    pipe = build_pipeline(MODEL, TECHNIQUE, model_params, cfg=cfg, exp_cfg=exp_cfg)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=exp_cfg["seed"])
    result = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, error_score="raise")
    return {label: np.asarray(result[f"test_{label}"]) for label in SCORING}


def main(argv: list[str] | None = None) -> int:
    """Run the baseline cross-validation and print mean +- std per metric; 1 on a data error."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=("raw", "sample"), required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-verify-hash", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg, exp_cfg, models_cfg = get_dataset_config(), get_experiment_config(), get_models_config()
    seed = int(exp_cfg["seed"])
    n_folds = int(exp_cfg["quick"]["cv_folds"] if args.quick else exp_cfg["cv_folds"])
    model_params = default_model_params(models_cfg)

    start = time.perf_counter()
    try:
        df = load_dataset(args.source, verify_hash=not args.no_verify_hash, cfg=cfg)
        deduped, _ = deduplicate(df, target=cfg["target"])
        split = split_dataset(deduped, cfg, exp_cfg, strategy="stratified")
        X, y = split["X_train"], split["y_train"]
        if args.quick:
            X, y = quick_subsample(X, y, float(exp_cfg["quick"]["train_fraction"]), seed)
        scores = run_cv(X, y, cfg, exp_cfg, model_params, n_folds)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - start

    params_text = ", ".join(f"{k}={v}" for k, v in model_params.items())
    print(f"source     {args.source}{' (quick)' if args.quick else ''}")
    print(f"pipeline   technique={TECHNIQUE}, model={MODEL} ({params_text})")
    print(f"train      {len(y):,} rows, {int((y == 1).sum()):,} frauds "
          "(test split held out, not used)")
    print(f"cv         StratifiedKFold(n_splits={n_folds}, shuffle=True, random_state={seed})")
    print()
    for label, values in scores.items():
        folds = " ".join(f"{v:.4f}" for v in values)
        print(f"{label:<8} {values.mean():.4f} +- {values.std():.4f}   folds: {folds}")
    if args.results_dir is not None:
        result = {
            "source": args.source,
            "quick": args.quick,
            "model": MODEL,
            "technique": TECHNIQUE,
            "model_params": model_params,
            "cv_folds": n_folds,
            "seed": seed,
            "train_rows": len(y),
            "train_frauds": int((y == 1).sum()),
            "fold_scores": {label: values.tolist() for label, values in scores.items()},
        }
        print(f"\nwrote      {save_json(result, args.results_dir / 'baseline_cv.json')}")
    print(f"\nelapsed    {elapsed:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
