"""Write the synthetic sample dataset to sample_path from configs/dataset.yaml.

Usage::

    python scripts/generate_synthetic_data.py [--n-rows 6000] [--fraud-rate 0.015] [--seed N]
                                              [--out PATH]

The seed defaults to `seed` in configs/experiment.yaml. Prints the row, fraud and duplicate
counts and the file's SHA-256 (tests/test_synthetic.py checks the committed file matches).
"""

import argparse
from pathlib import Path

import pandas as pd

from fraudstudy.config import get_dataset_config, get_experiment_config, resolve
from fraudstudy.io import sha256_file
from fraudstudy.synthetic import make_synthetic


def write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Write `df` as CSV with \\n line endings, so the bytes (and the hash) match on every OS."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")
    return path


def main(argv: list[str] | None = None) -> int:
    """Generate the dataset, write it and print a summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-rows", type=int, default=6000)
    parser.add_argument("--fraud-rate", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=None, help="default: experiment.yaml seed")
    parser.add_argument("--out", type=str, default=None, help="default: dataset.yaml sample_path")
    args = parser.parse_args(argv)

    seed = args.seed if args.seed is not None else get_experiment_config()["seed"]
    out = resolve(args.out or get_dataset_config()["sample_path"])

    df = make_synthetic(n_rows=args.n_rows, fraud_rate=args.fraud_rate, seed=seed)
    write_csv(df, out)

    print(f"wrote      {out}")
    print(f"rows       {len(df)}")
    print(f"frauds     {int(df['Class'].sum())} ({df['Class'].mean():.4%})")
    print(f"duplicates {int(df.duplicated().sum())}")
    print(f"size       {out.stat().st_size:,} bytes")
    print(f"sha256     {sha256_file(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
