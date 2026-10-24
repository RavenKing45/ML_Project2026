"""Schema and content checks for the credit-card dataset (see configs/dataset.yaml)."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fraudstudy.errors import DataValidationError

VALID_LABELS = frozenset({0, 1})


@dataclass(frozen=True)
class ValidationReport:
    """Summary of a dataset that passed :func:`validate_dataframe`.

    Attributes:
        n_rows: Number of rows.
        n_columns: Number of columns.
        n_frauds: Rows with ``target == 1``.
        n_legit: Rows with ``target == 0``.
        missing_values: Missing-value count for every column (zeros included), in column order.
        extra_columns: Columns present in the frame but not named in the config.
    """

    n_rows: int
    n_columns: int
    n_frauds: int
    n_legit: int
    missing_values: dict[str, int]
    extra_columns: list[str]

    @property
    def fraud_rate(self) -> float:
        """Fraction of rows that are fraud (0.0 for an empty frame)."""
        return self.n_frauds / self.n_rows if self.n_rows else 0.0


def required_columns(cfg: dict[str, Any]) -> list[str]:
    """Return every column the dataset must contain, in file order.

    Args:
        cfg: Contents of ``configs/dataset.yaml``.

    Returns:
        ``[time_column, *pca_features, amount_column, target]``.
    """
    return [cfg["time_column"], *cfg["pca_features"], cfg["amount_column"], cfg["target"]]


def validate_dataframe(df: pd.DataFrame, cfg: dict[str, Any]) -> ValidationReport:
    """Check ``df`` against the dataset config and collect every problem before raising.

    Checks: required columns exist; required columns are numeric; no +-inf in them; the target
    contains only 0 and 1 (missing targets count as invalid); ``Amount >= 0`` (missing amounts
    are counted, not rejected). Missing values in feature columns are reported, not rejected.
    ``df`` is never modified.

    Args:
        df: The dataset to check.
        cfg: Contents of ``configs/dataset.yaml``.

    Returns:
        A :class:`ValidationReport` describing the frame.

    Raises:
        DataValidationError: Listing every problem found, one per line.
    """
    problems: list[str] = []
    required = required_columns(cfg)
    target, amount = cfg["target"], cfg["amount_column"]

    missing_cols = [col for col in required if col not in df.columns]
    if missing_cols:
        problems.append(f"missing required columns: {', '.join(missing_cols)}")
    present = [col for col in required if col in df.columns]

    non_numeric = [
        col
        for col in present
        if not pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col])
    ]
    if non_numeric:
        problems.append(f"non-numeric columns (expected numbers): {', '.join(non_numeric)}")
    numeric = [col for col in present if col not in non_numeric]

    inf_counts = {col: int(np.isinf(df[col].to_numpy(dtype=float)).sum()) for col in numeric}
    inf_cols = {col: n for col, n in inf_counts.items() if n}
    if inf_cols:
        detail = ", ".join(f"{col} ({n} rows)" for col, n in inf_cols.items())
        problems.append(f"infinite values in: {detail}")

    if target in numeric:
        labels = df[target]
        bad = labels.isna() | ~labels.isin(list(VALID_LABELS))
        if bad.any():
            seen = sorted(map(str, labels[bad].unique()))
            problems.append(
                f"target '{target}' must be 0 or 1; {int(bad.sum())} rows have {', '.join(seen)}"
            )

    if amount in numeric:
        negative = int((df[amount] < 0).sum())
        if negative:
            problems.append(f"'{amount}' must be >= 0; {negative} rows are negative")

    if problems:
        raise DataValidationError(
            "Dataset failed validation:\n- " + "\n- ".join(problems) + "\nFix the CSV (or "
            "configs/dataset.yaml if the column names are wrong) and reload."
        )

    labels = df[target]
    return ValidationReport(
        n_rows=len(df),
        n_columns=df.shape[1],
        n_frauds=int((labels == 1).sum()),
        n_legit=int((labels == 0).sum()),
        missing_values={str(col): int(n) for col, n in df.isna().sum().items()},
        extra_columns=[str(col) for col in df.columns if col not in required],
    )
