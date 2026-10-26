"""SENSITIVITY ANALYSIS: refit CV-chosen configurations on a temporal split, score the later part.

Usage::

    python scripts/run_temporal_check.py --source raw|sample [--model lightgbm]
        [--techniques T ...] [--rerun-reason "<why>"] [--no-verify-hash]

For each technique of ``--model`` (default lightgbm), the cell's fitted pipeline is cloned (same
hyper-parameters, ratio and weight), refitted on the earliest ``1 - test_size`` of the
deduplicated data by ``Time`` and scored on the rest, at 0.5 and at the cell's stored F1
threshold. Nothing is tuned. Writes ``temporal_check.json`` (raw: ``artifacts/results``; sample:
``artifacts/results/sample_test_eval``).

The temporal parts contain rows of the locked test set, so on the raw source this runs only
after ``scripts/evaluate_test.py`` and is logged in ``TEST_SET_OPENED.json``. It is NOT the main
result and must not change any choice made from cross-validation.
"""

import argparse
import sys
from pathlib import Path

from fraudstudy.config import get_experiment_config, resolve
from fraudstudy.final_test import temporal_check

RESULTS_DIR = "artifacts/results"
MODELS_DIR = "artifacts/models"
DEFAULT_MODEL = "lightgbm"


def main(argv: list[str] | None = None) -> int:
    """Run the check and print one line per technique; 1 on an error or a guard refusal."""
    exp_cfg = get_experiment_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=("raw", "sample"), required=True)
    parser.add_argument("--model", choices=exp_cfg["models"], default=DEFAULT_MODEL)
    parser.add_argument("--techniques", nargs="+", choices=exp_cfg["techniques"])
    parser.add_argument("--rerun-reason", help="required if the check already ran (raw)")
    parser.add_argument("--no-verify-hash", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=resolve(RESULTS_DIR))
    parser.add_argument("--models-dir", type=Path, default=resolve(MODELS_DIR))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--decisions", type=Path, help="default: docs/decisions.md")
    args = parser.parse_args(argv)

    print("SENSITIVITY ANALYSIS (temporal split) - not the main result\n", flush=True)
    try:
        result = temporal_check(
            args.source,
            args.results_dir,
            args.models_dir,
            model=args.model,
            techniques=args.techniques,
            rerun_reason=args.rerun_reason,
            decisions_path=args.decisions,
            output_dir=args.output_dir,
            verify_hash=not args.no_verify_hash,
            log=lambda line: print(line, flush=True),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for item in result["skipped"]:
        print(f"skipped    {item['cell']}: {item['reason']}")
    print(f"\nwrote      {len(result['cells'])} cells to temporal_check.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
