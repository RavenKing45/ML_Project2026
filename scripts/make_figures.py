"""Draw every report figure from the saved result artifacts (nothing is refitted or rescored).

Usage::

    python scripts/make_figures.py --source raw|sample [--results-dir DIR] [--output-dir DIR]
        [--audit PATH] [--figures-dir DIR]

Reads ``comparison.json``, ``curves.json`` and ``stats.json`` from the evaluation folder
(``artifacts/results`` for raw, ``artifacts/results/sample_test_eval`` for the sample; run
``scripts/evaluate_test.py`` and ``scripts/run_stats.py`` first), ``<results>/fold_scores.csv``
and ``<results>/resampling_audit.json`` (skipped with a warning if absent). Writes 300-dpi PNGs to
``artifacts/figures`` (raw) or ``artifacts/figures/sample`` (sample): pr_curves_by_model,
roc_curves_by_model, pr_auc_with_ci, confusion_matrices_tuned, threshold_effect,
resampling_counts, fold_score_boxplots and cd_diagram_<model> (and cd_diagram_all).
Re-run ``scripts/run_stats.py`` afterwards to link the figures from the findings file.
"""

import argparse
import sys
from pathlib import Path

from fraudstudy.contracts import ComparisonTable, CurvesFile, ResamplingAudit, StatsResult
from fraudstudy.errors import DataValidationError, FraudStudyError
from fraudstudy.evaluation.figures import figures_dir_for, make_all_figures
from fraudstudy.experiment.results import source_fold_scores
from fraudstudy.experiment.test_eval import COMPARISON_FILE, CURVES_FILE, output_dir_for
from fraudstudy.io import load_json
from fraudstudy.paths import resolve

RESULTS_DIR = "artifacts/results"
AUDIT_FILE = "resampling_audit.json"
STATS_FILE = "stats.json"


def _load(path: Path, what: str, hint: str) -> object:
    """Load a JSON artifact or raise a clear error naming the script that writes it."""
    if not path.is_file():
        raise DataValidationError(f"{what} not found at {path}; run {hint} first.")
    return load_json(path)


def run(
    source: str,
    results_dir: Path,
    output_dir: Path | None = None,
    audit_path: Path | None = None,
    figures_dir: Path | None = None,
) -> dict[str, Path]:
    """Load the artifacts and write every figure.

    Args:
        source: ``"raw"`` or ``"sample"``.
        results_dir: Folder with ``fold_scores.csv`` (and ``resampling_audit.json``).
        output_dir: Evaluation folder with comparison / curves / stats
            (default ``output_dir_for(source, results_dir)``).
        audit_path: Resampling audit (default ``results_dir/resampling_audit.json``).
        figures_dir: Output folder (default ``artifacts/figures`` or ``artifacts/figures/sample``).

    Returns:
        Figure name -> path written.
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else output_dir_for(source, results_dir)
    figures_dir = Path(figures_dir) if figures_dir else figures_dir_for(source)
    evaluate = f"scripts/evaluate_test.py --source {source}"
    comparison = ComparisonTable.model_validate(
        _load(output_dir / COMPARISON_FILE, "comparison.json", evaluate)
    )
    curves = CurvesFile.model_validate(_load(output_dir / CURVES_FILE, "curves.json", evaluate))
    stats = StatsResult.model_validate(
        _load(output_dir / STATS_FILE, "stats.json", f"scripts/run_stats.py --source {source}")
    )
    audit_path = Path(audit_path) if audit_path else results_dir / AUDIT_FILE
    audit = None
    if audit_path.is_file():
        audit = ResamplingAudit.model_validate(load_json(audit_path))
    else:
        print(f"warning: {audit_path} not found; skipping resampling_counts.png", file=sys.stderr)
    return make_all_figures(
        comparison, curves, stats, source_fold_scores(results_dir, source), audit, figures_dir
    )


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and draw the figures.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        0 on success, 1 on a data or config error.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=("raw", "sample"), required=True)
    parser.add_argument("--results-dir", type=Path, default=resolve(RESULTS_DIR))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--figures-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        written = run(args.source, args.results_dir, args.output_dir, args.audit, args.figures_dir)
    except FraudStudyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for name, path in written.items():
        print(f"wrote {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
