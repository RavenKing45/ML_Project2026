"""Merge experiment cells from several machines into cv_summary.json and fold_scores.csv.

Usage::

    python scripts/merge_results.py --inputs DIR [DIR ...] [--output artifacts/results]

Each input is a results folder copied from a machine (with a ``cells/`` sub-folder written by
``scripts/run_experiments.py``). Cells must share the seed, mode, data and CV folds, else the
merge stops. Writes ``<output>/cv_summary.json``, a de-duplicated ``<output>/fold_scores.csv``
and copies the cell JSON / OOF files into ``<output>/cells``; prints the cells still missing.
"""

import argparse
import logging
import sys
from pathlib import Path

from fraudstudy.config import get_experiment_config, resolve
from fraudstudy.experiment import cell_id, merge_results

RESULTS_DIR = "artifacts/results"


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, merge and report.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        0 when all expected cells are present, 2 when some are missing, 1 on an error.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=resolve(RESULTS_DIR))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    exp_cfg = get_experiment_config()
    try:
        report = merge_results(
            args.inputs, args.output, list(exp_cfg["models"]), list(exp_cfg["techniques"])
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    expected = len(exp_cfg["models"]) * len(exp_cfg["techniques"])
    print(f"merged     {len(report.cells)} of {expected} cells")
    if report.duplicates:
        print(f"duplicates {', '.join(cell_id(*k) for k in report.duplicates)} (identical)")
    if report.conflicts:
        print(
            f"conflicts  {', '.join(cell_id(*k) for k in report.conflicts)} (differ; kept the "
            "latest finished_at)"
        )
    print(f"missing    {', '.join(cell_id(*k) for k in report.missing) or 'none'}")
    for name, path in report.paths.items():
        print(f"wrote      {name}: {path}")
    return 2 if report.missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
