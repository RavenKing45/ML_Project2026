"""Load the credit-card dataset, check its columns, remove duplicate rows and split it 70/30.

experiments.py and evaluation.py call load_data(), remove_duplicates() and split_data(), in
that order. Reads data/raw/creditcard.csv (the real data) or data/sample/creditcard_synthetic.csv
(a small fake copy for tests and quick runs); writes nothing.

Run it on its own to see the numbers: python -m fraudstudy.data [--source sample]
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ---- SETTINGS ------------------------------------------------------------------------------------
SEED = 42  # the one seed for every split, sampler and model in the project
TEST_SIZE = 0.30  # 30% of the rows go to the locked test set
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # this file is src/fraudstudy/data.py
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
SAMPLE_PATH = PROJECT_ROOT / "data" / "sample" / "creditcard_synthetic.csv"
# --------------------------------------------------------------------------------------------------

EXPECTED_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]  # 31 columns


def load_data(source: str = "raw") -> pd.DataFrame:
    """Read the real ("raw") or the fake ("sample") CSV and check that it has the 31 columns."""
    path = RAW_PATH if source == "raw" else SAMPLE_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Download creditcard.csv from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud, unzip it and put it in "
            "data/raw/, or use --source sample for the small fake copy."
        )
    df = pd.read_csv(path)
    missing = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing the columns {missing}; is it the Kaggle file?")
    return df


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Drop exact duplicate rows, keeping the first copy; return (df, rows removed, frauds removed).

    A duplicate could land in both train and test, and the model would then be tested on a row
    it was trained on.
    """
    is_copy = df.duplicated(keep="first")
    frauds_removed = int((df.loc[is_copy, "Class"] == 1).sum())
    return df[~is_copy].copy(), int(is_copy.sum()), frauds_removed


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 70/30 split with SEED: (X_train, X_test, y_train, y_test).

    Stratified means both parts keep the same fraud rate. The row index is kept, so every row
    can be traced back to the original file.
    """
    # Raw Time is dropped: it counts seconds from the first transaction of a two-day window, so it
    # says when the data was collected, not something that carries over to new payments.
    X = df.drop(columns=["Time", "Class"])
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    return X_train, X_test, y_train, y_test


def main() -> None:
    """Print rows, frauds, fraud rate, duplicates removed and the train/test sizes."""
    parser = argparse.ArgumentParser(description="Summarise the dataset and the 70/30 split.")
    parser.add_argument("--source", choices=["raw", "sample"], default="raw")
    args = parser.parse_args()

    df = load_data(args.source)
    frauds = int(df["Class"].sum())
    print(f"rows:        {len(df):,}")
    print(f"frauds:      {frauds:,}")
    print(f"fraud rate:  {frauds / len(df):.4%}")
    df, removed, frauds_removed = remove_duplicates(df)
    print(f"duplicates:  {removed:,} rows removed ({frauds_removed} of them frauds)")
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"train:       {len(y_train):,} rows, {int(y_train.sum())} frauds")
    print(f"test:        {len(y_test):,} rows, {int(y_test.sum())} frauds")


if __name__ == "__main__":
    main()
