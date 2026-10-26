"""Load, check, deduplicate and split the credit-card dataset.

Every script that touches the data calls load_dataset(), then deduplicate(), then
split_dataset(). Reads the real CSV (data/raw/) or the synthetic sample (data/sample/) named in
configs/dataset.yaml; writes nothing. No scaling or resampling happens here.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fraudstudy.config import get_dataset_config, resolve
from fraudstudy.io import sha256_file

KAGGLE_URL = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"


def load_dataset(source: str = "raw", verify_hash: bool = True,
                 cfg: dict | None = None) -> pd.DataFrame:
    """Read the real ("raw") or synthetic ("sample") CSV and check it with check_dataset().

    The real file's SHA-256 is compared with configs/dataset.yaml, so a corrupted or different
    download is caught before any result is computed from it.
    """
    cfg = cfg or get_dataset_config()
    path_text = cfg["raw_path"] if source == "raw" else cfg["sample_path"]
    path = resolve(path_text)

    if source == "sample" and not path.is_file():
        raise FileNotFoundError(
            f"Synthetic sample not found at {path_text}. "
            "Run: python scripts/generate_synthetic_data.py"
        )
    if path.is_dir():
        # Unzipping the Kaggle download often creates creditcard.csv/creditcard.csv.
        raise FileNotFoundError(
            f"{path_text} is a folder, not a file. Move the CSV inside it up one level so that "
            f"{path_text} is the file itself."
        )
    if not path.is_file():
        raise FileNotFoundError(
            f"Raw dataset not found at {path_text}. Download creditcard.csv from {KAGGLE_URL}, "
            f"unzip it and put it at {path_text}, or use --source sample for the synthetic data."
        )
    if source == "raw" and verify_hash:
        actual = sha256_file(path)
        if actual != str(cfg["sha256"]).lower():
            raise ValueError(
                f"SHA-256 mismatch for {path_text}: expected {cfg['sha256']}, got {actual}. The "
                f"file is corrupted or a different version; download it again from {KAGGLE_URL}."
            )

    df = pd.read_csv(path)
    check_dataset(df, cfg)
    return df


def check_dataset(df: pd.DataFrame, cfg: dict) -> None:
    """Raise one ValueError listing every problem: missing or non-numeric columns, infinite
    values, a target that is not 0/1, or a negative Amount. Missing feature values are allowed
    (the pipeline imputes them)."""
    target, amount = cfg["target"], cfg["amount_column"]
    required = [cfg["time_column"], *cfg["pca_features"], amount, target]
    problems = []

    missing = [col for col in required if col not in df.columns]
    if missing:
        problems.append(f"missing required columns: {', '.join(missing)}")
    present = [col for col in required if col in df.columns]
    # A bool column counts as numeric in pandas, but True/False is not a valid feature value.
    text = [col for col in present
            if not pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col])]
    if text:
        problems.append(f"non-numeric columns (expected numbers): {', '.join(text)}")
    numeric = [col for col in present if col not in text]

    infinite = {col: int(np.isinf(df[col]).sum()) for col in numeric}
    infinite = {col: count for col, count in infinite.items() if count}
    if infinite:
        problems.append("infinite values in: "
                        + ", ".join(f"{col} ({count} rows)" for col, count in infinite.items()))
    if target in numeric:
        bad = ~df[target].isin([0, 1])  # a missing label is not 0 or 1 either
        if bad.any():
            seen = ", ".join(sorted(map(str, df.loc[bad, target].unique())))
            problems.append(f"target '{target}' must be 0 or 1; {int(bad.sum())} rows have {seen}")
    if amount in numeric and (df[amount] < 0).any():
        problems.append(f"'{amount}' must be >= 0; {int((df[amount] < 0).sum())} rows are negative")

    if problems:
        raise ValueError("Dataset failed validation:\n- " + "\n- ".join(problems) + "\nFix "
                         "the CSV (or the column names in configs/dataset.yaml) and reload.")


def deduplicate(df: pd.DataFrame, target: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Remove exact duplicate rows, keeping the first copy and its original index label.

    Returns the new frame and a dict with rows_before, rows_after, rows_removed, frauds_removed.
    """
    target = target or get_dataset_config()["target"]
    is_copy = df.duplicated(keep="first")
    report = {
        "rows_before": len(df),
        "rows_after": int((~is_copy).sum()),
        "rows_removed": int(is_copy.sum()),
        "frauds_removed": int((df.loc[is_copy, target] == 1).sum()),
    }
    return df.loc[~is_copy].copy(), report


def feature_columns(cfg: dict) -> list[str]:
    """The model input columns: V1..V28 and Amount. Raw Time is left out (docs/decisions.md)."""
    return [*cfg["pca_features"], cfg["amount_column"]]


def class_summary(y: pd.Series) -> dict:
    """Rows, frauds and fraud rate of one label column."""
    rows, frauds = len(y), int((y == 1).sum())
    return {"rows": rows, "frauds": frauds, "fraud_rate": frauds / rows if rows else 0.0}


def split_dataset(df: pd.DataFrame, cfg: dict, exp_cfg: dict, strategy: str = "stratified") -> dict:
    """Split `df` into train and test parts (test_size and seed from experiment.yaml).

    "stratified" keeps the fraud rate the same in both parts. "temporal" sorts by Time and puts
    the earliest 70% in train, so every test transaction happens after every training one.
    Returns a dict with X_train, X_test, y_train, y_test, strategy and summary (class_summary
    of all, train and test). Index labels are those of `df`, so rows stay traceable.
    """
    if strategy not in ("stratified", "temporal"):
        raise ValueError(f"strategy must be 'stratified' or 'temporal'; got {strategy!r}.")
    target, columns, test_size = cfg["target"], feature_columns(cfg), float(exp_cfg["test_size"])

    if strategy == "stratified":
        X_train, X_test, y_train, y_test = train_test_split(
            df[columns], df[target], test_size=test_size, stratify=df[target],
            random_state=exp_cfg["seed"],
        )
    else:
        # kind="stable" keeps rows with the same Time in file order, so the split is repeatable.
        ordered = df.sort_values(cfg["time_column"], kind="stable")
        train_rows = int(round(len(ordered) * (1.0 - test_size)))
        train, test = ordered.iloc[:train_rows], ordered.iloc[train_rows:]
        X_train, X_test = train[columns].copy(), test[columns].copy()
        y_train, y_test = train[target].copy(), test[target].copy()

    summary = {"all": class_summary(df[target]), "train": class_summary(y_train),
               "test": class_summary(y_test)}
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
            "strategy": strategy, "summary": summary}
