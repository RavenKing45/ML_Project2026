"""Audit what every technique's sampler steps do to the TRAINING data, at every shared ratio.

Usage::

    python scripts/run_resampling_audit.py --source raw|sample [--quick] [--no-verify-hash]
                                           [--results-dir DIR]

Steps: load -> deduplicate -> stratified split -> stratified k-fold on the training part. For
each technique, each ratio of its search space (the same grid the tuning uses) and each fold,
the pipeline minus its model runs on the fold's training rows and the class counts are recorded.
The validation fold is only counted and fingerprinted, never resampled; the test part is never
used. Writes DIR/resampling_audit.json (default artifacts/results). Prints only counts.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from fraudstudy.config import get_dataset_config, get_experiment_config, resolve
from fraudstudy.contracts import ClassCounts, ResamplingAudit, ResamplingAuditEntry
from fraudstudy.data import deduplicate, load_dataset, split_dataset
from fraudstudy.io import save_json
from fraudstudy.pipeline import build_pipeline
from fraudstudy.sampling import (
    RATIO_MARGIN,
    base_ratio,
    fraud_weight,
    resampling_audit,
    row_hashes,
    technique_search_space,
    valid_ratios,
)

# The audit removes the model step, so the model does not matter; entries record it anyway.
MODEL = "logreg"


def audit_settings(exp_cfg: dict, base: float) -> list[dict]:
    """Every (technique, ratio) the tuning will try, taken from the shared search spaces.

    Each setting is a dict with technique, params, ratio, smote_ratio and fraud_weight.
    class_weight's "balanced" option is skipped: it is not a grid ratio, and weights never
    change the data anyway.
    """
    settings = []
    for technique in exp_cfg["techniques"]:
        setting = {"technique": technique, "smote_ratio": None, "fraud_weight": None}
        if technique == "baseline":
            settings.append({**setting, "params": {}, "ratio": None})
        elif technique == "smote_rus":
            for pair in technique_search_space(technique, exp_cfg, base):
                a, b = pair["smote__sampling_strategy"][0], pair["rus__sampling_strategy"][0]
                params = {"smote_sampling_strategy": a, "rus_sampling_strategy": b}
                settings.append({**setting, "params": params, "ratio": b, "smote_ratio": a})
        elif technique == "class_weight":
            for r in valid_ratios(exp_cfg["ratio_grid"], base):
                w = fraud_weight(r, base)
                settings.append({**setting, "params": {"fraud_weight": w}, "ratio": r,
                                 "fraud_weight": w})
        else:
            space = technique_search_space(technique, exp_cfg, base)
            for r in space[f"{technique}__sampling_strategy"]:
                settings.append({**setting, "params": {"sampling_strategy": r}, "ratio": r})
    return settings


def fingerprint(X: pd.DataFrame, y: pd.Series) -> tuple[bytes, bytes, bytes]:
    """Bytes of a fold's index, row values and labels, to prove the fold was not changed."""
    return np.asarray(X.index).tobytes(), row_hashes(X).tobytes(), np.asarray(y).tobytes()


def run_audit(X: pd.DataFrame, y: pd.Series, cfg: dict, exp_cfg: dict,
              n_folds: int) -> ResamplingAudit:
    """Audit every setting on every CV training fold of (X, y), which must be training rows.

    validation_untouched is true when the validation fold is identical before and after, and
    the audit saw exactly the fold's training rows.
    """
    n_frauds = int((y == 1).sum())
    if n_frauds < n_folds:
        raise ValueError(
            f"Only {n_frauds} frauds in the training data, fewer than {n_folds} CV folds. Use "
            "more data (drop --quick or use --source raw)."
        )
    seed, k = int(exp_cfg["seed"]), int(exp_cfg["smote_k_neighbors"])
    base = base_ratio(y)
    grid = [float(r) for r in exp_cfg["ratio_grid"]]
    kept = valid_ratios(grid, base)
    folds = list(StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X, y))
    fewest = min(int((y.iloc[train_idx] == 1).sum()) for train_idx, _ in folds)
    if fewest <= k:
        raise ValueError(
            f"A CV training fold has only {fewest} frauds; SMOTE needs more than "
            f"smote_k_neighbors={k}. Use more data (drop --quick or use --source raw)."
        )

    entries = []
    for setting in audit_settings(exp_cfg, base):
        pipe = build_pipeline(MODEL, setting["technique"], None, setting["params"], cfg, exp_cfg)
        for fold, (train_idx, val_idx) in enumerate(folds):
            X_fit, y_fit = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            before = fingerprint(X_val, y_val)
            result = resampling_audit(pipe, X_fit, y_fit)
            untouched = (
                fingerprint(X_val, y_val) == before
                and result["legit_before"] + result["fraud_before"] == len(train_idx)
                and X_fit.index.intersection(X_val.index).empty
            )
            entries.append(ResamplingAuditEntry(
                model=MODEL,
                technique=setting["technique"],
                fold=fold,
                ratio=setting["ratio"],
                train_before=ClassCounts(legit=result["legit_before"],
                                         fraud=result["fraud_before"]),
                train_after=ClassCounts(legit=result["legit_after"], fraud=result["fraud_after"]),
                validation=ClassCounts(legit=int((y_val == 0).sum()),
                                       fraud=int((y_val == 1).sum())),
                n_synthetic=result["n_synthetic"],
                validation_untouched=untouched,
                smote_ratio=setting["smote_ratio"],
                fraud_weight=setting["fraud_weight"],
                fraud_unique_after=result["fraud_unique_after"],
                fraud_duplicates_after=result["fraud_duplicates_after"],
            ))
    # Only training rows are passed above; the test split is not referenced at all.
    return ResamplingAudit(
        seed=seed,
        cv_folds=n_folds,
        test_rows_resampled=0,
        all_checks_passed=all(e.validation_untouched for e in entries),
        entries=entries,
        base_ratio=base,
        dropped_ratios=[r for r in grid if r not in kept],
    )


def summary_lines(audit: ResamplingAudit) -> list[str]:
    """Fold-averaged training counts after resampling, one line per (technique, ratio)."""
    groups = {}
    for e in audit.entries:
        groups.setdefault((e.technique, e.smote_ratio, e.ratio), []).append(e)
    lines = [f"{'technique':<13}{'ratio':>12}{'legit':>10}{'fraud':>10}{'unique':>10}"
             f"{'dupes':>10}{'synth':>10}"]
    for (technique, a, r), group in groups.items():
        ratio = "-" if r is None else (f"{a}->{r}" if a is not None else f"{r}")
        means = [np.mean([e.train_after.legit for e in group]),
                 np.mean([e.train_after.fraud for e in group]),
                 np.mean([e.fraud_unique_after or 0 for e in group]),
                 np.mean([e.fraud_duplicates_after or 0 for e in group]),
                 np.mean([e.n_synthetic for e in group])]
        lines.append(f"{technique:<13}{ratio:>12}" + "".join(f"{m:>10.0f}" for m in means))
    return lines


def main(argv: list[str] | None = None) -> int:
    """Run the audit, write the JSON and print a summary; 1 on a data error or a failed check."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=("raw", "sample"), required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-verify-hash", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=resolve("artifacts/results"))
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
        X, y = split["X_train"], split["y_train"]
        if args.quick:
            fraction = float(exp_cfg["quick"]["train_fraction"])
            X, _, y, _ = train_test_split(X, y, train_size=fraction, stratify=y, random_state=seed)
        audit = run_audit(X, y, cfg, exp_cfg, n_folds)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    path = save_json(audit.model_dump(mode="json"), args.results_dir / "resampling_audit.json")
    elapsed = time.perf_counter() - start

    print(f"source     {args.source}{' (quick)' if args.quick else ''}")
    print(f"train      {len(y):,} rows, {int((y == 1).sum()):,} frauds, base ratio "
          f"{audit.base_ratio:.5f} (test split held out, not used)")
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
