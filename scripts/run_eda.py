"""Compute the EDA summary (eda.json) and the EDA figures, then print the headline numbers.

Usage::

    python scripts/run_eda.py [--source raw|sample] [--no-verify-hash]
                              [--results-dir DIR] [--figures-dir DIR]

Writes ``artifacts/results/eda.json`` (``EdaSummary``) and ``artifacts/figures/eda_*.png`` by
default. Only aggregate statistics are printed, never individual transactions.
"""

import argparse
import sys
from pathlib import Path

from fraudstudy.config import get_dataset_config, resolve
from fraudstudy.contracts import EdaSummary
from fraudstudy.data import deduplicate, load_dataset
from fraudstudy.eda import compute_eda_summary, make_eda_figures
from fraudstudy.io import save_json, sha256_file

RESULTS_DIR = "artifacts/results"
FIGURES_DIR = "artifacts/figures"
EDA_FILE = "eda.json"
N_PRINTED_EFFECTS = 6


def headline(summary: EdaSummary) -> str:
    """Render the headline numbers of an EDA summary as text lines.

    Args:
        summary: The computed summary.

    Returns:
        A multi-line string.
    """
    dedup = summary.class_counts_dedup
    amount, log_amount = summary.amount_summary.raw, summary.amount_summary.log1p
    top = ", ".join(
        f"{e.feature} {e.cohens_d:+.2f}" for e in summary.effect_sizes[:N_PRINTED_EFFECTS]
    )
    hours = summary.hour_of_day
    peak = max(range(len(hours.fraud_counts)), key=hours.fraud_counts.__getitem__)
    missing = sum(summary.missing_values.values())
    return "\n".join(
        [
            f"rows              {summary.n_rows:,} x {summary.n_columns} columns",
            f"frauds            {summary.class_counts.fraud:,} "
            f"(fraud rate {summary.fraud_rate:.4%}, imbalance {summary.imbalance_ratio})",
            f"missing values    {missing}",
            f"duplicates        {summary.n_duplicate_rows:,} removed "
            f"({summary.n_duplicate_frauds} fraud)",
            f"after dedup       {dedup.legit + dedup.fraud:,} rows, {dedup.fraud:,} frauds",
            f"Amount raw        min {amount.min:,.2f}  median {amount.median:,.2f}  "
            f"mean {amount.mean:,.2f}  max {amount.max:,.2f}  skew {amount.skewness:.2f}",
            f"Amount log1p      min {log_amount.min:.2f}  median {log_amount.median:.2f}  "
            f"mean {log_amount.mean:.2f}  max {log_amount.max:.2f}  skew {log_amount.skewness:.2f}",
            f"Time span         {summary.time_span_hours:.2f} hours",
            f"top |Cohen's d|   {top}",
            f"top KS            {summary.feature_separation[0].feature} "
            f"{summary.feature_separation[0].ks_statistic:.3f}",
            f"peak fraud hour   {peak:02d}:00 ({hours.fraud_counts[peak]} frauds; "
            f"{hours.hour_assumption})",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    """Load and deduplicate the data, write eda.json and the figures, print the headline.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success, 1 if the data could not be loaded).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=("raw", "sample"), default="raw")
    parser.add_argument("--no-verify-hash", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=resolve(RESULTS_DIR))
    parser.add_argument("--figures-dir", type=Path, default=resolve(FIGURES_DIR))
    args = parser.parse_args(argv)

    cfg = get_dataset_config()
    try:
        df = load_dataset(args.source, verify_hash=not args.no_verify_hash, cfg=cfg)
        deduped, _ = deduplicate(df, target=cfg["target"])
        summary = compute_eda_summary(
            df,
            deduped,
            cfg,
            dataset_source="real" if args.source == "raw" else "synthetic",
            dataset_sha256=sha256_file(
                resolve(cfg["raw_path" if args.source == "raw" else "sample_path"])
            ),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    json_path = save_json(summary.model_dump(mode="json"), args.results_dir / EDA_FILE)
    figures = make_eda_figures(summary, deduped, cfg, args.figures_dir)

    print(f"source            {args.source}")
    print(headline(summary))
    print()
    print(f"wrote {json_path}")
    for path in figures:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
