"""Tests for the synthetic dataset generator and its script."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraudstudy.config import get_dataset_config
from fraudstudy.errors import DataValidationError
from fraudstudy.io import sha256_file
from fraudstudy.paths import PROJECT_ROOT, resolve
from fraudstudy.synthetic import (
    COLUMNS,
    DUPLICATE_FRAUDS,
    DUPLICATE_ROWS,
    TIME_MAX_SECONDS,
    feature_scales,
    make_synthetic,
)

MAX_SAMPLE_BYTES = 2 * 1024 * 1024

# scripts/ is not a package, so load the script module from its path.
_spec = importlib.util.spec_from_file_location(
    "generate_synthetic_data", PROJECT_ROOT / "scripts" / "generate_synthetic_data.py"
)
assert _spec is not None and _spec.loader is not None
_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_script)
main, write_csv = _script.main, _script.write_csv


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return make_synthetic()


def test_columns_exact(df: pd.DataFrame) -> None:
    assert list(df.columns) == ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]
    assert list(df.columns) == COLUMNS
    assert df["Class"].dtype == np.int64
    assert set(df["Class"].unique()) == {0, 1}
    assert not df.isna().any().any()


def test_row_count_and_fraud_rate(df: pd.DataFrame) -> None:
    assert len(df) == 6000
    assert abs(df["Class"].mean() - 0.015) <= 0.002


@pytest.mark.parametrize("n_rows,rate", [(2000, 0.01), (10_000, 0.0017), (3001, 0.05)])
def test_fraud_rate_within_tolerance_for_other_settings(n_rows: int, rate: float) -> None:
    small = make_synthetic(n_rows=n_rows, fraud_rate=rate, seed=1)
    assert len(small) == n_rows
    assert abs(small["Class"].mean() - rate) <= 0.002


def test_duplicates(df: pd.DataFrame) -> None:
    dup_mask = df.duplicated(keep="first")
    assert dup_mask.sum() == DUPLICATE_ROWS == 20
    assert df.loc[dup_mask, "Class"].sum() == DUPLICATE_FRAUDS == 2
    assert len(df.drop_duplicates()) == len(df) - DUPLICATE_ROWS


def test_duplicates_survive_csv_round_trip(df: pd.DataFrame, tmp_path: Path) -> None:
    path = write_csv(df, tmp_path / "s.csv")
    assert pd.read_csv(path).duplicated().sum() == DUPLICATE_ROWS


def test_deterministic_same_seed(tmp_path: Path) -> None:
    a = write_csv(make_synthetic(seed=7), tmp_path / "a.csv")
    b = write_csv(make_synthetic(seed=7), tmp_path / "b.csv")
    assert sha256_file(a) == sha256_file(b)


def test_different_seed_differs(tmp_path: Path) -> None:
    a = write_csv(make_synthetic(seed=1), tmp_path / "a.csv")
    b = write_csv(make_synthetic(seed=2), tmp_path / "b.csv")
    assert sha256_file(a) != sha256_file(b)


def test_time_and_amount_ranges(df: pd.DataFrame) -> None:
    assert df["Time"].min() >= 0 and df["Time"].max() <= TIME_MAX_SECONDS
    assert (df["Amount"] >= 0).all()
    assert (df["Amount"] == 0).sum() > 0
    assert df["Amount"].skew() > 2  # heavy right tail, as for a lognormal


def test_legit_feature_scales_decrease(df: pd.DataFrame) -> None:
    scales = feature_scales()
    assert scales[0] == pytest.approx(2.0) and scales[-1] == pytest.approx(0.3)
    stds = df.loc[df["Class"] == 0, [f"V{i}" for i in range(1, 29)]].std().to_numpy()
    np.testing.assert_allclose(stds, scales, rtol=0.1)


def test_fraud_shift_directions(df: pd.DataFrame) -> None:
    fraud_means = df.loc[df["Class"] == 1].mean()
    for col in ("V14", "V12", "V10", "V17"):
        assert fraud_means[col] < -0.5, col
    for col in ("V4", "V11"):
        assert fraud_means[col] > 0.5, col


def test_classes_overlap(df: pd.DataFrame) -> None:
    """Not trivially separable: some frauds fall inside the legit V14 interquartile range."""
    legit_v14 = df.loc[df["Class"] == 0, "V14"]
    q1, q3 = legit_v14.quantile([0.25, 0.75])
    fraud_v14 = df.loc[df["Class"] == 1, "V14"]
    assert fraud_v14.between(q1, q3).sum() >= 3


def test_two_fraud_clusters(df: pd.DataFrame) -> None:
    """Frauds split into two groups on opposite sides of V3."""
    fraud = df.loc[df["Class"] == 1]
    assert (fraud["V3"] > 0.5).sum() >= 10
    assert (fraud["V3"] < -0.5).sum() >= 10


def test_enough_frauds_for_cv_and_smote(df: pd.DataFrame) -> None:
    """70% train split, 5-fold CV: each training fold needs > k_neighbors=5 frauds."""
    train_frauds = df["Class"].sum() * 0.7
    assert train_frauds * 4 / 5 > 5


@pytest.mark.parametrize(
    "kwargs", [{"fraud_rate": 0.0}, {"fraud_rate": 1.0}, {"n_rows": 50, "fraud_rate": 0.01}]
)
def test_invalid_arguments_raise(kwargs: dict[str, float]) -> None:
    with pytest.raises(DataValidationError):
        make_synthetic(**kwargs)


def test_committed_sample_matches_generator(tmp_path: Path) -> None:
    """The committed CSV is exactly what the script produces with the default settings."""
    committed = resolve(get_dataset_config()["sample_path"])
    assert committed.is_file(), "Run: python scripts/generate_synthetic_data.py"
    assert committed.stat().st_size < MAX_SAMPLE_BYTES
    out = tmp_path / "regen.csv"
    assert main(["--out", str(out)]) == 0
    assert sha256_file(out) == sha256_file(committed)


def test_script_prints_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "x.csv"
    main(["--out", str(out), "--n-rows", "1000", "--fraud-rate", "0.02", "--seed", "3"])
    text = capsys.readouterr().out
    assert "rows       1000" in text
    assert "frauds     20" in text
    assert "duplicates 20" in text
    assert sha256_file(out) in text
