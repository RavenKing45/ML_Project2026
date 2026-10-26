"""Tune and cross-validate every (model, technique) cell on the TRAINING split; resumable.

Usage::

    python scripts/run_experiments.py --source raw|sample [--quick] [--models M ...]
        [--techniques T ...] [--force] [--no-verify-hash]

Each cell: equal-budget ``RandomizedSearchCV`` scored by average precision, out-of-fold scores
and F1 / F2 thresholds, repeated-CV fold scores (same folds for every cell), resampling audit
(``fraudstudy.experiment.run_cell``). Finished cells are skipped unless ``--force``.
``--quick`` uses the ``quick`` block of configs/experiment.yaml; its ``train_fraction`` applies
to the raw source only. The test split is never used.

Writes ``artifacts/results/cells/<model>__<technique>.json`` and ``_oof.npy``,
``artifacts/results/fold_scores.csv``, ``artifacts/results/run_meta.json`` and
``artifacts/models/cells/<model>__<technique>.joblib`` (git-ignored). Only scores and counts
are printed, never feature values.
"""

import argparse
import logging
import sys
from pathlib import Path

from fraudstudy.config import load_configs, resolve
from fraudstudy.experiment import run_experiments

RESULTS_DIR = "artifacts/results"
MODELS_DIR = "artifacts/models"


def main(argv: list[str] | None = None) -> int:
    """Run the requested cells and print a summary table; 1 on a data or config error."""
    cfgs = load_configs()
    exp_cfg = cfgs["experiment"]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=("raw", "sample"), required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--models", nargs="+", choices=exp_cfg["models"])
    parser.add_argument("--techniques", nargs="+", choices=exp_cfg["techniques"])
    parser.add_argument("--force", action="store_true", help="re-run finished cells")
    parser.add_argument("--no-verify-hash", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=resolve(RESULTS_DIR))
    parser.add_argument("--models-dir", type=Path, default=resolve(MODELS_DIR))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("fraudstudy.experiment").setLevel(logging.INFO)

    try:
        out = run_experiments(
            args.source,
            args.quick,
            args.results_dir,
            args.models_dir,
            models=args.models,
            techniques=args.techniques,
            force=args.force,
            verify_hash=not args.no_verify_hash,
            cfgs=cfgs,
            log=lambda line: print(line, flush=True),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"\n{'model':<14}{'technique':<14}{'ratio':>8}{'CV PR-AUC':>18}{'ROC-AUC':>9}"
        f"{'thr F1':>8}{'thr F2':>8}{'tries':>6}{'fit s':>7}  status"
    )
    for r, status in out:
        ratio = "-" if r["chosen_ratio"] is None else f"{r['chosen_ratio']:g}"
        print(
            f"{r['model']:<14}{r['technique']:<14}{ratio:>8}"
            f"{r['cv_pr_auc_mean']:>11.4f} +- {r['cv_pr_auc_std']:.3f}{r['cv_roc_auc_mean']:>9.4f}"
            f"{r['thresholds']['f1']:>8.3f}{r['thresholds']['f2']:>8.3f}"
            f"{r['n_candidates_evaluated']:>6}{r['fit_seconds']:>7.1f}  {status}"
        )
    print(f"\nwrote      {args.results_dir} (cells, fold_scores.csv, run_meta.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
