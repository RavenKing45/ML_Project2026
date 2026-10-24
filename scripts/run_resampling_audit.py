"""Audit what every technique's sampler steps do to the TRAINING data, at every shared ratio.

Usage::

    python scripts/run_resampling_audit.py --source raw|sample [--quick] [--no-verify-hash]

Steps: load -> deduplicate -> stratified train/test split -> StratifiedKFold(cv_folds,
shuffle=True, random_state=seed) on the training part. For each technique, each ratio of its
search space (:func:`fraudstudy.sampling.technique_search_space`, so the same grid the tuning
uses) and each fold, the pipeline without its model is run on the fold's training rows and the
class counts before/after are recorded (``fraudstudy.audit.resampling_audit``). The validation
fold is only counted and fingerprinted, never resampled; the test part is never used.

Writes ``artifacts/results/resampling_audit.json`` (``ResamplingAudit``). ``--quick`` keeps a
stratified ``quick.train_fraction`` of the training rows and uses ``quick.cv_folds`` folds.
Only counts are printed, never feature values.
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from fraudstudy.audit import ResamplingAudit as FoldAudit
from fraudstudy.audit import resampling_audit, row_hashes
from fraudstudy.config import get_dataset_config, get_experiment_config
from fraudstudy.contracts import ClassCounts, ResamplingAudit, ResamplingAuditEntry
from fraudstudy.data import deduplicate, load_dataset, split_dataset
from fraudstudy.errors import DataValidationError, FraudStudyError
from fraudstudy.io import save_json
from fraudstudy.paths import resolve
from fraudstudy.pipeline import build_pipeline, without_model
from fraudstudy.sampling import (
    FRAUD_WEIGHT_KEY,
    RATIO_MARGIN,
    SMOTE_RUS_KEYS,
    STRATEGY_KEY,
    fraud_weight,
    technique_search_space,
    valid_ratios,
)
from fraudstudy.sampling import base_ratio as compute_base_ratio

# The audit is model-independent (the model step is removed); entries carry the pipeline's model.
MODEL = "logreg"
RESULTS_DIR = "artifacts/results"
AUDIT_FILE = "resampling_audit.json"


@dataclass(frozen=True)
class Setting:
    """One technique configuration to audit."""

    technique: str
    params: dict[str, Any]
    ratio: float | None
    smote_ratio: float | None = None
    fraud_weight: float | None = None


def audit_settings(exp_cfg: dict[str, Any], base: float) -> list[Setting]:
    """Every (technique, ratio) the tuning will try, from the shared search spaces.

    ``class_weight``'s ``"balanced"`` option is skipped (it is not a grid ratio, and data is
    unchanged for every weight anyway).

    Args:
        exp_cfg: Experiment config (``techniques``, ``ratio_grid``, ``hybrid``).
        base: ``frauds / legit`` of the training set.

    Returns:
        The settings, in ``techniques`` order.
    """
    settings: list[Setting] = []
    for technique in exp_cfg["techniques"]:
        if technique == "baseline":
            settings.append(Setting(technique, {}, None))
        elif technique == "smote_rus":
            space = technique_search_space(technique, exp_cfg, base)
            assert isinstance(space, list)
            for pair in space:
                a, b = pair[f"smote__{STRATEGY_KEY}"][0], pair[f"rus__{STRATEGY_KEY}"][0]
                params = dict(zip(SMOTE_RUS_KEYS, (a, b), strict=True))
                settings.append(Setting(technique, params, b, smote_ratio=a))
        elif technique == "class_weight":
            for r in valid_ratios(list(exp_cfg["ratio_grid"]), base):
                w = fraud_weight(r, base)
                settings.append(Setting(technique, {FRAUD_WEIGHT_KEY: w}, r, fraud_weight=w))
        else:
            space = technique_search_space(technique, exp_cfg, base)
            assert isinstance(space, dict)
            for r in space[f"{technique}__{STRATEGY_KEY}"]:
                settings.append(Setting(technique, {STRATEGY_KEY: r}, r))
    return settings


def _fingerprint(X: pd.DataFrame, y: pd.Series) -> tuple[bytes, bytes, bytes]:
    """Hash of a fold's index, values and labels (to prove it was not modified)."""
    return (
        np.asarray(X.index).tobytes(),
        row_hashes(X).tobytes(),
        np.asarray(y).tobytes(),
    )


def _counts(y: pd.Series) -> ClassCounts:
    """Class counts of a label vector."""
    return ClassCounts(legit=int((y == 0).sum()), fraud=int((y == 1).sum()))


def run_audit(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict[str, Any],
    exp_cfg: dict[str, Any],
    n_folds: int,
) -> ResamplingAudit:
    """Audit every setting on every CV training fold of ``(X, y)`` (training data only).

    ``validation_untouched`` is true when the validation fold is identical before and after the
    audit and the audit saw exactly the fold's training rows (no validation row).

    Args:
        X: Training features only.
        y: Training labels only.
        cfg: Dataset config.
        exp_cfg: Experiment config.
        n_folds: Number of CV folds.

    Returns:
        The ``ResamplingAudit`` contract object.

    Raises:
        DataValidationError: If there are fewer frauds than folds, or a training fold has no
            more frauds than SMOTE's ``k_neighbors``.
    """
    n_frauds = int((y == 1).sum())
    if n_frauds < n_folds:
        raise DataValidationError(
            f"Only {n_frauds} frauds in the training data, fewer than {n_folds} CV folds. Use "
            "more data (drop --quick or use --source raw)."
        )
    seed = int(exp_cfg["seed"])
    base = compute_base_ratio(y)
    grid = [float(r) for r in exp_cfg["ratio_grid"]]
    kept = set(valid_ratios(grid, base))
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(cv.split(X, y))
    k = int(exp_cfg["smote_k_neighbors"])
    fewest = min(int((y.iloc[train_idx] == 1).sum()) for train_idx, _ in folds)
    if fewest <= k:
        raise DataValidationError(
            f"A CV training fold has only {fewest} frauds; SMOTE needs more than "
            f"smote_k_neighbors={k}. Use more data (drop --quick or use --source raw)."
        )

    entries: list[ResamplingAuditEntry] = []
    for setting in audit_settings(exp_cfg, base):
        pipe = without_model(
            build_pipeline(MODEL, setting.technique, None, setting.params, cfg, exp_cfg)
        )
        for fold, (train_idx, val_idx) in enumerate(folds):
            X_fit, y_fit = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            before = _fingerprint(X_val, y_val)
            result: FoldAudit = resampling_audit(pipe, X_fit, y_fit)
            untouched = (
                _fingerprint(X_val, y_val) == before
                and result.legit_before + result.fraud_before == len(train_idx)
                and X_fit.index.intersection(X_val.index).empty
            )
            entries.append(
                ResamplingAuditEntry(
                    model=MODEL,
                    technique=setting.technique,
                    fold=fold,
                    ratio=setting.ratio,
                    train_before=ClassCounts(legit=result.legit_before, fraud=result.fraud_before),
                    train_after=ClassCounts(legit=result.legit_after, fraud=result.fraud_after),
                    validation=_counts(y_val),
                    n_synthetic=result.n_synthetic,
                    validation_untouched=untouched,
                    smote_ratio=setting.smote_ratio,
                    fraud_weight=setting.fraud_weight,
                    fraud_unique_after=result.fraud_unique_after,
                    fraud_duplicates_after=result.fraud_duplicates_after,
                )
            )
    # Only training rows are ever passed above; the test split is not referenced at all.
    test_rows_resampled = 0
    return ResamplingAudit(
        seed=seed,
        cv_folds=n_folds,
        test_rows_resampled=test_rows_resampled,
        all_checks_passed=test_rows_resampled == 0 and all(e.validation_untouched for e in entries),
        entries=entries,
        base_ratio=base,
        dropped_ratios=[r for r in grid if r not in kept],
    )


def summary_lines(audit: ResamplingAudit) -> list[str]:
    """Fold-averaged counts per (technique, ratio), one line each."""
    rows: dict[tuple[str, Any, Any], list[ResamplingAuditEntry]] = {}
    for e in audit.entries:
        rows.setdefault((e.technique, e.smote_ratio, e.ratio), []).append(e)
    lines = [
        f"{'technique':<13}{'ratio':>12}{'legit':>10}{'fraud':>10}{'unique':>10}"
        f"{'dupes':>10}{'synth':>10}"
    ]
    for (technique, a, r), group in rows.items():
        ratio = "-" if r is None else (f"{a}->{r}" if a is not None else f"{r}")

        def mean(values: list[int]) -> float:
            return float(np.mean(values))

        lines.append(
            f"{technique:<13}{ratio:>12}"
            f"{mean([e.train_after.legit for e in group]):>10.0f}"
            f"{mean([e.train_after.fraud for e in group]):>10.0f}"
            f"{mean([e.fraud_unique_after or 0 for e in group]):>10.0f}"
            f"{mean([e.fraud_duplicates_after or 0 for e in group]):>10.0f}"
            f"{mean([e.n_synthetic for e in group]):>10.0f}"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the audit, write the JSON and print a fold-averaged summary.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success, 1 on a data or config error or a failed check).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=("raw", "sample"), required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-verify-hash", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=resolve(RESULTS_DIR))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg, exp_cfg = get_dataset_config(), get_experiment_config()
    seed = int(exp_cfg["seed"])
    n_folds = int(exp_cfg["quick"]["cv_folds"] if args.quick else exp_cfg["cv_folds"])

    start = time.perf_counter()
    try:
        df = load_dataset(args.source, verify_hash=not args.no_verify_hash, cfg=cfg)
        deduped, _ = deduplicate(df, target=cfg["target"])
        split = split_dataset(deduped, cfg, exp_cfg, strategy="stratified")
        X, y = split.X_train, split.y_train
        if args.quick:
            fraction = float(exp_cfg["quick"]["train_fraction"])
            X, _, y, _ = train_test_split(X, y, train_size=fraction, stratify=y, random_state=seed)
        audit = run_audit(X, y, cfg, exp_cfg, n_folds)
    except FraudStudyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    path = save_json(audit.model_dump(mode="json"), args.results_dir / AUDIT_FILE)
    elapsed = time.perf_counter() - start

    print(f"source     {args.source}{' (quick)' if args.quick else ''}")
    print(
        f"train      {len(y):,} rows, {int((y == 1).sum()):,} frauds, base ratio "
        f"{audit.base_ratio:.5f} (test split held out, not used)"
    )
    print(f"dropped    {audit.dropped_ratios or 'none'} (not above base ratio x {RATIO_MARGIN})")
    print(f"cv         StratifiedKFold(n_splits={n_folds}, shuffle=True, random_state={seed})")
    print("\nfold-averaged training counts after resampling:")
    print("\n".join(summary_lines(audit)))
    print(f"\nchecks     {'all passed' if audit.all_checks_passed else 'FAILED'}")
    print(f"wrote      {path}")
    print(f"elapsed    {elapsed:.1f} s")
    return 0 if audit.all_checks_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
