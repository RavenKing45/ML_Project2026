"""Evaluate finished cells on the locked test set, ONCE, and write comparison.json / curves.json.

Usage::

    python scripts/evaluate_test.py --source raw|sample [--cells model__technique ...]
        [--rerun-reason "<why>"] [--no-verify-hash]

The only code that reads the test set. It rebuilds the training run's split (same load, dedup,
stratified split, seed) and stops unless the training-set hash equals the one in run_meta.json
and in every cell. Each fitted pipeline scores the test set once; metrics are computed at 0.5,
at the cell's F1 threshold and at its F2 threshold (both tuned on out-of-fold TRAINING scores,
read from the cell files), with bootstrap CIs for PR-AUC and ROC-AUC.

Raw source: the opening is recorded in ``artifacts/results/TEST_SET_OPENED.json``; if that file
exists the script refuses to run unless ``--rerun-reason`` is given, which is then appended to
``docs/decisions.md``. Sample source: exempt from the guard, writes to
``artifacts/results/sample_test_eval/``. Never change a tuned value because of these results.
"""

import argparse
import sys
from pathlib import Path

from fraudstudy.config import resolve
from fraudstudy.final_test import evaluate_test

RESULTS_DIR = "artifacts/results"
MODELS_DIR = "artifacts/models"


def main(argv: list[str] | None = None) -> int:
    """Evaluate and print the comparison table; 1 on an error or a guard refusal."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=("raw", "sample"), required=True)
    parser.add_argument("--cells", nargs="+", metavar="MODEL__TECHNIQUE")
    parser.add_argument("--rerun-reason", help="required if the test set was already opened")
    parser.add_argument("--no-verify-hash", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=resolve(RESULTS_DIR))
    parser.add_argument("--models-dir", type=Path, default=resolve(MODELS_DIR))
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="default: results dir (raw) or results dir/sample_test_eval (sample)",
    )
    parser.add_argument("--decisions", type=Path, help="default: docs/decisions.md")
    args = parser.parse_args(argv)

    try:
        report = evaluate_test(
            args.source,
            args.results_dir,
            args.models_dir,
            cells=args.cells,
            rerun_reason=args.rerun_reason,
            decisions_path=args.decisions,
            output_dir=args.output_dir,
            verify_hash=not args.no_verify_hash,
            log=lambda line: print(line, flush=True),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"\n{'model':<14}{'technique':<14}{'CV PR-AUC':>10}{'test PR-AUC (95% CI)':>25}"
        f"{'ROC-AUC':>9}{'thr F1':>8}{'P@F1':>7}{'R@F1':>7}{'R@0.5':>7}"
    )
    for r in report["comparison"]["records"]:
        if (r["model"], r["technique"]) not in report["evaluated"]:
            continue
        t = r["test_at_tuned"]
        ci = f"{t['pr_auc']:.4f} ({r['pr_auc_ci'][0]:.3f}-{r['pr_auc_ci'][1]:.3f})"
        print(
            f"{r['model']:<14}{r['technique']:<14}{r['cv_pr_auc_mean']:>10.4f}{ci:>25}"
            f"{t['roc_auc']:>9.4f}{r['threshold_f1']:>8.3f}{t['precision']:>7.3f}"
            f"{t['recall']:>7.3f}{r['test_at_0_5']['recall']:>7.3f}"
        )
    for name, path in report["paths"].items():
        print(f"wrote      {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
