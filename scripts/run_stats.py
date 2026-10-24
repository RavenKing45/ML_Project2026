"""Statistical tests on the repeated-CV fold scores -> stats.json and findings.md.

Usage::

    python scripts/run_stats.py --source raw|sample [--results-dir DIR] [--output-dir DIR]
        [--findings PATH] [--figures-dir DIR]

Reads ``<results>/fold_scores.csv`` (restricted to the finished cells of ``--source``) and, if
present, ``comparison.json`` from the evaluation folder. Per model: Wilcoxon signed-rank test of
every technique vs the baseline with Holm correction, Friedman test with average ranks and the
Nemenyi critical difference (``fraudstudy.evaluation.stats``); plus a Friedman test pooling all
models. Alpha comes from ``stats.alpha`` in configs/experiment.yaml.

Writes ``stats.json`` (``StatsResult``) to the evaluation folder (``artifacts/results`` for raw,
``artifacts/results/sample_test_eval`` for the sample) and the findings file
(``docs/findings.md`` for raw, ``<evaluation folder>/findings.md`` for the sample): auto-generated
tables and an empty "Interpretation (written by the team)" section, whose text is kept on
re-runs. The script never writes conclusions.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from fraudstudy.config import get_experiment_config
from fraudstudy.contracts import ComparisonTable, StatsResult
from fraudstudy.errors import FraudStudyError
from fraudstudy.evaluation.figures import FIGURE_NAMES, figures_dir_for
from fraudstudy.evaluation.stats import compute_stats, existing_interpretation, findings_markdown
from fraudstudy.experiment.results import source_fold_scores
from fraudstudy.experiment.test_eval import COMPARISON_FILE, DATASET_SOURCE, output_dir_for
from fraudstudy.io import load_json, save_json
from fraudstudy.paths import resolve

RESULTS_DIR = "artifacts/results"
FINDINGS_FILE = "docs/findings.md"
STATS_FILE = "stats.json"
DEFAULT_ALPHA = 0.05


@dataclass(frozen=True)
class StatsRun:
    """What :func:`run` wrote.

    Attributes:
        result: The validated statistics.
        stats_path: ``stats.json`` path.
        findings_path: Findings file path.
    """

    result: StatsResult
    stats_path: Path
    findings_path: Path


def run(
    source: str,
    results_dir: Path,
    output_dir: Path | None = None,
    findings_path: Path | None = None,
    figures_dir: Path | None = None,
) -> StatsRun:
    """Compute the tests and write ``stats.json`` and the findings file.

    Args:
        source: ``"raw"`` or ``"sample"``.
        results_dir: Folder with ``cells/`` and ``fold_scores.csv``.
        output_dir: Where ``stats.json`` goes and ``comparison.json`` is read from
            (default ``output_dir_for(source, results_dir)``).
        findings_path: Default ``docs/findings.md`` (raw) or ``<output_dir>/findings.md``.
        figures_dir: Folder whose existing figures are linked from the findings.

    Returns:
        A :class:`StatsRun`.
    """
    cfg = get_experiment_config()
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else output_dir_for(source, results_dir)
    if findings_path is None:
        findings_path = resolve(FINDINGS_FILE) if source == "raw" else output_dir / "findings.md"
    figures_dir = Path(figures_dir) if figures_dir else figures_dir_for(source)

    result = compute_stats(
        source_fold_scores(results_dir, source),
        models=cfg["models"],
        techniques=cfg["techniques"],
        alpha=float(cfg.get("stats", {}).get("alpha", DEFAULT_ALPHA)),
        n_bootstrap=int(cfg["bootstrap"]["n_resamples"]),
        confidence=float(cfg["bootstrap"]["confidence"]),
        seed=int(cfg["seed"]),
        dataset_source=DATASET_SOURCE[source],
    )
    stats_path = save_json(result.model_dump(mode="json"), output_dir / STATS_FILE)

    comparison_path = output_dir / COMPARISON_FILE
    comparison = (
        ComparisonTable.model_validate(load_json(comparison_path))
        if comparison_path.is_file()
        else None
    )
    names = [*FIGURE_NAMES, *(f"cd_diagram_{f.scope}.png" for f in result.friedman)]
    figures = [figures_dir / n for n in names if (figures_dir / n).is_file()]
    text = findings_markdown(
        result,
        comparison,
        figures,
        findings_path,
        source,
        interpretation=existing_interpretation(findings_path),
    )
    findings_path.parent.mkdir(parents=True, exist_ok=True)
    findings_path.write_text(text, encoding="utf-8")
    return StatsRun(result, stats_path, findings_path)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the tests and print a short summary.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        0 on success, 1 on a data or config error.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=("raw", "sample"), required=True)
    parser.add_argument("--results-dir", type=Path, default=resolve(RESULTS_DIR))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--findings", type=Path)
    parser.add_argument("--figures-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        out = run(args.source, args.results_dir, args.output_dir, args.findings, args.figures_dir)
    except FraudStudyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for f in out.result.friedman:
        print(
            f"Friedman {f.scope:<14} N={f.n_blocks:<3} chi2={f.statistic:8.3f} "
            f"p={f.p_value:.3g} CD={f.critical_difference:.3f}"
        )
    for t in out.result.pairwise:
        print(
            f"Wilcoxon {t.model:<14}{t.technique_a:<13} vs baseline  median diff "
            f"{t.median_difference:+.4f}  W/T/L {t.wins}/{t.ties}/{t.losses}  "
            f"p={t.p_value:.3g}  p_holm={t.p_value_adjusted:.3g}"
        )
    print(f"wrote stats: {out.stats_path}")
    print(f"wrote findings: {out.findings_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
