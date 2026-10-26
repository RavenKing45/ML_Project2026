"""Cross-validate technique A (baseline) with logistic regression on the TRAINING split only.

Usage::

    python scripts/run_baseline.py --source raw|sample [--quick] [--no-verify-hash]
                                   [--results-dir DIR]

Steps: load -> deduplicate -> stratified train/test split -> StratifiedKFold(cv_folds,
shuffle=True, random_state=seed) ``cross_validate`` on the training part with scoring
``average_precision`` (PR-AUC) and ``roc_auc``. The test part is never used here.

``--quick`` keeps a stratified ``quick.train_fraction`` of the training rows and uses
``quick.cv_folds`` folds (configs/experiment.yaml). Only scores are printed, never feature
values. With ``--results-dir`` the per-fold scores are also written to
``<DIR>/baseline_cv.json`` (used by scripts/run_all_sample.py and the golden check).
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from fraudstudy.config import get_dataset_config, get_experiment_config, get_models_config
from fraudstudy.data import deduplicate, load_dataset, split_dataset
from fraudstudy.errors import DataValidationError, FraudStudyError
from fraudstudy.io import save_json
from fraudstudy.pipeline import build_pipeline

MODEL = "logreg"
TECHNIQUE = "baseline"
SCORING: dict[str, str] = {"PR-AUC": "average_precision", "ROC-AUC": "roc_auc"}


def default_model_params(models_cfg: dict[str, Any]) -> dict[str, Any]:
    """Untuned logistic-regression settings: only ``max_iter`` from models.yaml.

    ``C`` stays at sklearn's default (1.0); tuning over models.yaml's ``C`` grid comes later.

    Args:
        models_cfg: Contents of ``configs/models.yaml``.

    Returns:
        ``{"max_iter": <first value of logreg.max_iter>}``.
    """
    return {"max_iter": models_cfg[MODEL]["max_iter"][0]}


def quick_subsample(
    X: pd.DataFrame, y: pd.Series, fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Keep a stratified ``fraction`` of the training rows (for ``--quick``).

    Args:
        X: Training features.
        y: Training labels.
        fraction: Share of rows to keep, in (0, 1].
        seed: Random seed.

    Returns:
        ``(X_sub, y_sub)``; the inputs unchanged if ``fraction >= 1``.
    """
    if fraction >= 1.0:
        return X, y
    X_sub, _, y_sub, _ = train_test_split(X, y, train_size=fraction, stratify=y, random_state=seed)
    return X_sub, y_sub


def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict[str, Any],
    exp_cfg: dict[str, Any],
    model_params: dict[str, Any],
    n_folds: int,
) -> dict[str, np.ndarray]:
    """Stratified k-fold ``cross_validate`` of the baseline logistic-regression pipeline.

    The whole pipeline (imputer, log transform, scalers, model) is refitted inside each fold, so
    validation folds never influence any fitted statistic.

    Args:
        X: Training features only.
        y: Training labels only.
        cfg: Dataset config.
        exp_cfg: Experiment config (``seed``).
        model_params: Logistic-regression keyword arguments.
        n_folds: Number of CV folds.

    Returns:
        ``{metric label: per-fold scores}`` for each entry of :data:`SCORING`.

    Raises:
        DataValidationError: If there are fewer frauds than folds.
    """
    n_frauds = int((y == 1).sum())
    if n_frauds < n_folds:
        raise DataValidationError(
            f"Only {n_frauds} frauds in the training data, fewer than {n_folds} CV folds. Use "
            "more data (drop --quick or use --source raw)."
        )
    pipe = build_pipeline(MODEL, TECHNIQUE, model_params, cfg=cfg, exp_cfg=exp_cfg)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=exp_cfg["seed"])
    result = cross_validate(pipe, X, y, cv=cv, scoring=dict(SCORING), error_score="raise")
    return {label: np.asarray(result[f"test_{label}"]) for label in SCORING}


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the baseline cross-validation and print mean +- std per metric.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success, 1 on a data or config error).
    """
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
        X, y = split.X_train, split.y_train
        if args.quick:
            X, y = quick_subsample(X, y, float(exp_cfg["quick"]["train_fraction"]), seed)
        scores = run_cv(X, y, cfg, exp_cfg, model_params, n_folds)
    except FraudStudyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - start

    params_text = ", ".join(f"{k}={v}" for k, v in model_params.items())
    print(f"source     {args.source}{' (quick)' if args.quick else ''}")
    print(f"pipeline   technique={TECHNIQUE}, model={MODEL} ({params_text})")
    print(
        f"train      {len(y):,} rows, {int((y == 1).sum()):,} frauds "
        f"(test split held out, not used)"
    )
    print(f"cv         StratifiedKFold(n_splits={n_folds}, shuffle=True, random_state={seed})")
    print()
    for label, values in scores.items():
        print(
            f"{label:<8} {values.mean():.4f} +- {values.std():.4f}   folds: "
            + " ".join(f"{v:.4f}" for v in values)
        )
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
