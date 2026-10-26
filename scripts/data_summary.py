"""Load, deduplicate and split the dataset, then print how many rows and frauds each part has.

Usage::

    python scripts/data_summary.py [--source auto|raw|sample] [--strategy stratified|temporal]
                                   [--no-verify-hash]

"auto" (the default) uses the real CSV if it is there, otherwise the synthetic sample. Writes
nothing; prints only counts, never feature values.
"""

import argparse
import sys

from fraudstudy.config import get_dataset_config, get_experiment_config, resolve
from fraudstudy.data import deduplicate, load_dataset, split_dataset


def format_table(summary: dict) -> str:
    """The split summary (rows, frauds, fraud rate per part) as a fixed-width text table."""
    lines = [f"{'part':<6} {'rows':>9} {'frauds':>7} {'fraud rate':>11}"]
    for part, counts in summary.items():
        lines.append(f"{part:<6} {int(counts['rows']):>9,} {int(counts['frauds']):>7,} "
                     f"{counts['fraud_rate']:>11.4%}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run load -> deduplicate -> split and print the counts; return 1 if the data is unusable."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=("auto", "raw", "sample"), default="auto")
    parser.add_argument("--strategy", choices=("stratified", "temporal"), default="stratified")
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
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"source     {source} ({cfg['raw_path' if source == 'raw' else 'sample_path']})")
    print(f"loaded     {len(df):,} rows x {df.shape[1]} columns")
    print(f"duplicates {report['rows_removed']:,} removed ({report['frauds_removed']} fraud)")
    print(f"strategy   {split['strategy']} (test_size={exp_cfg['test_size']}, "
          f"seed={exp_cfg['seed']})")
    print(f"features   {split['X_train'].shape[1]}")
    print()
    print(format_table(split["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
