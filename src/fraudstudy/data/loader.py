"""Loading the real or synthetic CSV, and exact-duplicate removal."""

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from fraudstudy.config import get_dataset_config
from fraudstudy.data.validate import validate_dataframe
from fraudstudy.errors import DataValidationError
from fraudstudy.io import sha256_file
from fraudstudy.paths import resolve

Source = Literal["raw", "sample"]
KAGGLE_URL = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"


@dataclass(frozen=True)
class DedupReport:
    """What :func:`deduplicate` removed.

    Attributes:
        n_rows_before: Rows in the input.
        n_rows_after: Rows in the output.
        n_removed: Exact duplicates removed (every occurrence after the first).
        n_frauds_removed: How many of the removed rows were fraud.
    """

    n_rows_before: int
    n_rows_after: int
    n_removed: int
    n_frauds_removed: int


def _raw_missing_message(path_text: str, path_is_dir: bool) -> str:
    """Explain how to put the Kaggle CSV where ``raw_path`` expects it."""
    if path_is_dir:
        return (
            f"{path_text} is a folder, not a file (unzipping often creates "
            f"creditcard.csv/creditcard.csv). Move the CSV inside it up one level so that "
            f"{path_text} is the file itself, then retry."
        )
    return (
        f"Raw dataset not found at {path_text}. Download 'creditcard.csv' from {KAGGLE_URL} "
        f"(Kaggle account needed; or `kaggle datasets download -d mlg-ulb/creditcardfraud`), "
        f"unzip it and place the CSV at {path_text}. Use source='sample' to work with the "
        f"synthetic data instead."
    )


def load_dataset(
    source: Source = "raw",
    verify_hash: bool = True,
    *,
    cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Read the dataset CSV and validate it against configs/dataset.yaml.

    For ``source="raw"`` the file's SHA-256 is compared with ``sha256`` in the config when
    ``verify_hash`` is true. If the config value is null, the computed hash is printed with a
    hint to paste it into the config. The sample file is never hash-checked here (its bytes are
    pinned by tests/test_synthetic.py).

    Args:
        source: ``"raw"`` for the Kaggle file at ``raw_path``, ``"sample"`` for the synthetic
            file at ``sample_path``.
        verify_hash: Whether to check the raw file's SHA-256 against the config.
        cfg: Dataset config; defaults to ``configs/dataset.yaml`` (pass one in tests).

    Returns:
        The validated dataset, with a fresh ``RangeIndex``.

    Raises:
        DataValidationError: If ``source`` is unknown, the file is missing, the hash does not
            match, or the contents fail :func:`validate_dataframe`.
    """
    cfg = cfg if cfg is not None else get_dataset_config()
    if source not in ("raw", "sample"):
        raise DataValidationError(f"source must be 'raw' or 'sample'; got {source!r}.")
    path_text = cfg["raw_path"] if source == "raw" else cfg["sample_path"]
    path = resolve(path_text)

    if not path.is_file():
        if source == "raw":
            raise DataValidationError(_raw_missing_message(str(path_text), path.is_dir()))
        raise DataValidationError(
            f"Synthetic sample not found at {path_text}. "
            "Run: python scripts/generate_synthetic_data.py"
        )

    if source == "raw" and verify_hash:
        actual = sha256_file(path)
        expected = cfg.get("sha256")
        if expected is None:
            print(f"sha256 of {path_text}: {actual}")
            print("Hint: paste this value into `sha256:` in configs/dataset.yaml to pin the file.")
        elif actual != str(expected).lower():
            raise DataValidationError(
                f"SHA-256 mismatch for {path_text}: expected {expected}, got {actual}. The file "
                f"is corrupted or a different version; re-download it from {KAGGLE_URL}."
            )

    df = pd.read_csv(path)
    validate_dataframe(df, cfg)
    return df


def deduplicate(df: pd.DataFrame, *, target: str | None = None) -> tuple[pd.DataFrame, DedupReport]:
    """Remove exact duplicate rows (all columns equal), keeping the first occurrence.

    The original index labels of the kept rows are preserved, so rows stay traceable.

    Args:
        df: Dataset containing the target column. Not modified.
        target: Target column name; defaults to ``target`` in configs/dataset.yaml.

    Returns:
        ``(new_frame, report)``.
    """
    target = target or get_dataset_config()["target"]
    dup_mask = df.duplicated(keep="first")
    out = df.loc[~dup_mask].copy()
    report = DedupReport(
        n_rows_before=len(df),
        n_rows_after=len(out),
        n_removed=int(dup_mask.sum()),
        n_frauds_removed=int((df.loc[dup_mask, target] == 1).sum()),
    )
    return out, report
