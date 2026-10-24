"""Load, deduplicate and split the dataset, then print a summary table.

Usage::

    python scripts/data_summary.py [--source auto|raw|sample] [--strategy stratified|temporal]
                                   [--no-verify-hash]

``auto`` (the default) uses the real CSV at ``raw_path`` if it exists, otherwise the synthetic
sample. Only counts are printed, never feature values.
"""

import argparse
import sys

from fraudstudy.config import get_dataset_config, get_experiment_config
from fraudstudy.data import deduplicate, load_dataset, split_dataset
from fraudstudy.data.split import STRATEGIES
from fraudstudy.errors import FraudStudyError
from fraudstudy.paths import resolve


def format_table(summary: dict[str, dict[str, float]]) -> str:
    """Render a split summary as a fixed-width text table.

    Args:
        summary: ``Split.summary``.

    Returns:
        The table, one line per part (``all``, ``train``, ``test``).
    """
    lines = [f"{'part':<6} {'rows':>9} {'frauds':>7} {'fraud rate':>11}"]
    for part, stats in summary.items():
        lines.append(
            f"{part:<6} {int(stats['rows']):>9,} {int(stats['frauds']):>7,} "
            f"{stats['fraud_rate']:>11.4%}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run load -> dedup -> split and print the results.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success, 1 if the data could not be loaded or split).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=("auto", "raw", "sample"), default="auto")
    parser.add_argument("--strategy", choices=STRATEGIES, default="stratified")
    parser.add_argument("--no-verify-hash", action="store_true")
    args = parser.parse_args(argv)

    cfg, exp_cfg = get_dataset_config(), get_experiment_config()
    source = args.source
    if source == "auto":
        source = "raw" if resolve(cfg["raw_path"]).exists() else "sample"

    try:
        df = load_dataset(source, verify_hash=not args.no_verify_hash, cfg=cfg)
        deduped, report = deduplicate(df, target=cfg["target"])
        split = split_dataset(deduped, cfg, exp_cfg, strategy=args.strategy)
    except FraudStudyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"source     {source} ({cfg['raw_path' if source == 'raw' else 'sample_path']})")
    print(f"loaded     {len(df):,} rows x {df.shape[1]} columns")
    print(f"duplicates {report.n_removed:,} removed ({report.n_frauds_removed} fraud)")
    print(f"strategy   {split.strategy} (test_size={exp_cfg['test_size']}, seed={exp_cfg['seed']})")
    print(f"features   {split.X_train.shape[1]}")
    print()
    print(format_table(split.summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
