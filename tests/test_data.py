"""Tests for the data layer (load, validate, deduplicate, split) and scripts/data_summary.py."""

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from fraudstudy.config import get_dataset_config, get_experiment_config
from fraudstudy.data import (
    DedupReport,
    ValidationReport,
    deduplicate,
    feature_columns,
    load_dataset,
    required_columns,
    split_dataset,
    validate_dataframe,
)
from fraudstudy.errors import ConfigError, DataValidationError
from fraudstudy.io import sha256_file
from fraudstudy.paths import PROJECT_ROOT, resolve

PCA = [f"V{i}" for i in range(1, 29)]

_spec = importlib.util.spec_from_file_location(
    "data_summary", PROJECT_ROOT / "scripts" / "data_summary.py"
)
assert _spec is not None and _spec.loader is not None
_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_script)


@pytest.fixture(scope="module")
def cfg() -> dict[str, Any]:
    return get_dataset_config()


@pytest.fixture(scope="module")
def exp_cfg() -> dict[str, Any]:
    return get_experiment_config()


@pytest.fixture(scope="module")
def sample(cfg: dict[str, Any]) -> pd.DataFrame:
    return load_dataset("sample", cfg=cfg)


@pytest.fixture(scope="module")
def deduped(sample: pd.DataFrame) -> pd.DataFrame:
    return deduplicate(sample)[0]


def _cfg_with(cfg: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    return {**cfg, **overrides}


# --- validate -----------------------------------------------------------------------------------


def test_required_columns_order(cfg: dict[str, Any]) -> None:
    assert required_columns(cfg) == ["Time", *PCA, "Amount", "Class"]


def test_validate_sample_report(sample: pd.DataFrame, cfg: dict[str, Any]) -> None:
    report = validate_dataframe(sample, cfg)
    assert isinstance(report, ValidationReport)
    assert (report.n_rows, report.n_columns) == (6000, 31)
    assert report.n_frauds == 90 and report.n_legit == 5910
    assert report.fraud_rate == pytest.approx(0.015)
    assert list(report.missing_values) == list(sample.columns)
    assert sum(report.missing_values.values()) == 0
    assert report.extra_columns == []


def test_validate_counts_missing_and_extra_columns(
    sample: pd.DataFrame, cfg: dict[str, Any]
) -> None:
    df = sample.head(50).copy()
    df.loc[[0, 1, 2], "V3"] = np.nan
    df.loc[4, "Amount"] = np.nan
    df["note"] = 1.0
    report = validate_dataframe(df, cfg)
    assert report.missing_values["V3"] == 3
    assert report.missing_values["Amount"] == 1
    assert report.extra_columns == ["note"]


def test_validate_missing_column_raises(sample: pd.DataFrame, cfg: dict[str, Any]) -> None:
    with pytest.raises(DataValidationError, match="missing required columns: V7"):
        validate_dataframe(sample.drop(columns=["V7"]), cfg)


def test_validate_inf_raises(sample: pd.DataFrame, cfg: dict[str, Any]) -> None:
    df = sample.head(20).copy()
    df.loc[3, "V2"] = np.inf
    df.loc[5, "V2"] = -np.inf
    with pytest.raises(DataValidationError, match=r"infinite values in: V2 \(2 rows\)"):
        validate_dataframe(df, cfg)


def test_validate_bad_class_raises(sample: pd.DataFrame, cfg: dict[str, Any]) -> None:
    df = sample.head(20).copy()
    df.loc[0, "Class"] = 2
    with pytest.raises(DataValidationError, match="must be 0 or 1; 1 rows have 2"):
        validate_dataframe(df, cfg)


def test_validate_missing_class_raises(sample: pd.DataFrame, cfg: dict[str, Any]) -> None:
    df = sample.head(20).astype({"Class": float})
    df.loc[0, "Class"] = np.nan
    with pytest.raises(DataValidationError, match="must be 0 or 1"):
        validate_dataframe(df, cfg)


def test_validate_reports_every_problem_at_once(sample: pd.DataFrame, cfg: dict[str, Any]) -> None:
    df = sample.head(20).drop(columns=["V9"])
    df.loc[1, "V1"] = np.inf
    df.loc[2, "Class"] = 2
    df.loc[3, "Amount"] = -5.0
    df["V5"] = "x"
    with pytest.raises(DataValidationError) as excinfo:
        validate_dataframe(df, cfg)
    message = str(excinfo.value)
    for fragment in (
        "missing required columns: V9",
        "non-numeric columns",
        "V5",
        "infinite values in: V1",
        "must be 0 or 1",
        "'Amount' must be >= 0",
    ):
        assert fragment in message


def test_validate_does_not_mutate(sample: pd.DataFrame, cfg: dict[str, Any]) -> None:
    before = sample.copy()
    validate_dataframe(sample, cfg)
    pd.testing.assert_frame_equal(sample, before)


# --- load ---------------------------------------------------------------------------------------


def test_load_sample(sample: pd.DataFrame) -> None:
    assert sample.shape == (6000, 31)
    assert list(sample.columns) == ["Time", *PCA, "Amount", "Class"]
    assert isinstance(sample.index, pd.RangeIndex)


def test_load_unknown_source_raises(cfg: dict[str, Any]) -> None:
    with pytest.raises(DataValidationError, match="source must be"):
        load_dataset("processed", cfg=cfg)  # type: ignore[arg-type]


def test_load_raw_missing_explains_download(cfg: dict[str, Any], tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="kaggle.com/datasets/mlg-ulb/creditcardfraud"):
        load_dataset("raw", cfg=_cfg_with(cfg, raw_path=str(tmp_path / "nope.csv")))


def test_load_raw_path_is_folder_explains_fix(cfg: dict[str, Any], tmp_path: Path) -> None:
    folder = tmp_path / "creditcard.csv"
    folder.mkdir()
    with pytest.raises(DataValidationError, match="is a folder, not a file"):
        load_dataset("raw", cfg=_cfg_with(cfg, raw_path=str(folder)))


def test_load_sample_missing_raises(cfg: dict[str, Any], tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="generate_synthetic_data"):
        load_dataset("sample", cfg=_cfg_with(cfg, sample_path=str(tmp_path / "x.csv")))


@pytest.fixture
def raw_copy(cfg: dict[str, Any], tmp_path: Path) -> Path:
    """A small valid CSV standing in for the raw file."""
    path = tmp_path / "raw.csv"
    path.write_bytes(resolve(cfg["sample_path"]).read_bytes())
    return path


def test_load_raw_hash_match(cfg: dict[str, Any], raw_copy: Path) -> None:
    df = load_dataset(
        "raw", cfg=_cfg_with(cfg, raw_path=str(raw_copy), sha256=sha256_file(raw_copy))
    )
    assert len(df) == 6000


def test_load_raw_hash_mismatch_raises(cfg: dict[str, Any], raw_copy: Path) -> None:
    bad = _cfg_with(cfg, raw_path=str(raw_copy), sha256="0" * 64)
    with pytest.raises(DataValidationError, match="SHA-256 mismatch"):
        load_dataset("raw", cfg=bad)
    assert len(load_dataset("raw", verify_hash=False, cfg=bad)) == 6000


def test_load_raw_null_hash_prints_hint(
    cfg: dict[str, Any], raw_copy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    load_dataset("raw", cfg=_cfg_with(cfg, raw_path=str(raw_copy), sha256=None))
    out = capsys.readouterr().out
    assert sha256_file(raw_copy) in out
    assert "configs/dataset.yaml" in out


def test_load_raw_invalid_contents_raise(cfg: dict[str, Any], tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"Time": [0.0], "Amount": [1.0], "Class": [0]}).to_csv(path, index=False)
    with pytest.raises(DataValidationError, match="missing required columns"):
        load_dataset("raw", verify_hash=False, cfg=_cfg_with(cfg, raw_path=str(path)))


# --- deduplicate --------------------------------------------------------------------------------


def test_dedup_counts(sample: pd.DataFrame) -> None:
    out, report = deduplicate(sample)
    assert report == DedupReport(
        n_rows_before=6000, n_rows_after=5980, n_removed=20, n_frauds_removed=2
    )
    assert len(out) == 5980
    assert not out.duplicated().any()
    assert int(out["Class"].sum()) == 88


def test_dedup_keeps_first_and_preserves_index(sample: pd.DataFrame) -> None:
    out, _ = deduplicate(sample)
    first = ~sample.duplicated(keep="first")
    assert out.index.equals(sample.index[first])


def test_dedup_returns_new_frame_without_mutating(sample: pd.DataFrame) -> None:
    before = sample.copy()
    out, _ = deduplicate(sample)
    out.iloc[0, 0] = -1.0
    pd.testing.assert_frame_equal(sample, before)


def test_dedup_no_duplicates() -> None:
    df = pd.DataFrame({"a": [1, 2], "Class": [0, 1]})
    out, report = deduplicate(df, target="Class")
    assert report.n_removed == 0 and len(out) == 2


# --- feature_columns ----------------------------------------------------------------------------


def test_feature_columns_default(cfg: dict[str, Any]) -> None:
    assert feature_columns(cfg, use_hour_feature=False) == [*PCA, "Amount"]


def test_feature_columns_with_hour(cfg: dict[str, Any]) -> None:
    cols = feature_columns(cfg, use_hour_feature=True)
    assert cols == [*PCA, "Amount", "Time"]
    assert feature_columns(cfg, use_hour_feature=True) == cols  # stable


# --- split --------------------------------------------------------------------------------------


def test_stratified_split(
    deduped: pd.DataFrame, cfg: dict[str, Any], exp_cfg: dict[str, Any]
) -> None:
    split = split_dataset(deduped, cfg, exp_cfg, strategy="stratified")
    s = split.summary
    assert split.strategy == "stratified"
    assert abs(s["train"]["fraud_rate"] - s["test"]["fraud_rate"]) <= 0.003
    assert set(split.X_train.index).isdisjoint(split.X_test.index)
    assert s["train"]["rows"] + s["test"]["rows"] == len(deduped) == s["all"]["rows"]
    assert s["train"]["frauds"] + s["test"]["frauds"] == 88
    assert s["test"]["rows"] == pytest.approx(len(deduped) * exp_cfg["test_size"], abs=1)
    assert list(split.X_train.columns) == [*PCA, "Amount"]
    assert split.X_train.index.equals(split.y_train.index)
    assert split.X_test.index.equals(split.y_test.index)


def test_stratified_split_is_deterministic(
    deduped: pd.DataFrame, cfg: dict[str, Any], exp_cfg: dict[str, Any]
) -> None:
    a = split_dataset(deduped, cfg, exp_cfg)
    b = split_dataset(deduped, cfg, exp_cfg)
    assert a.X_test.index.equals(b.X_test.index)
    c = split_dataset(deduped, cfg, {**exp_cfg, "seed": exp_cfg["seed"] + 1})
    assert not a.X_test.index.equals(c.X_test.index)


def test_temporal_split(
    deduped: pd.DataFrame, cfg: dict[str, Any], exp_cfg: dict[str, Any]
) -> None:
    split = split_dataset(deduped, cfg, exp_cfg, strategy="temporal")
    time = deduped["Time"]
    assert time.loc[split.X_train.index].max() <= time.loc[split.X_test.index].min()
    assert set(split.X_train.index).isdisjoint(split.X_test.index)
    assert len(split.X_train) == round(len(deduped) * (1 - exp_cfg["test_size"]))
    assert len(split.X_train) + len(split.X_test) == len(deduped)
    assert "Time" not in split.X_train.columns


def test_hour_feature_keeps_time(
    deduped: pd.DataFrame, cfg: dict[str, Any], exp_cfg: dict[str, Any]
) -> None:
    split = split_dataset(deduped, cfg, {**exp_cfg, "use_hour_feature": True})
    assert list(split.X_train.columns)[-1] == "Time"


def test_split_does_not_mutate(
    deduped: pd.DataFrame, cfg: dict[str, Any], exp_cfg: dict[str, Any]
) -> None:
    before = deduped.copy()
    for strategy in ("stratified", "temporal"):
        split_dataset(deduped, cfg, exp_cfg, strategy=strategy)  # type: ignore[arg-type]
    pd.testing.assert_frame_equal(deduped, before)


def test_split_bad_arguments_raise(
    deduped: pd.DataFrame, cfg: dict[str, Any], exp_cfg: dict[str, Any]
) -> None:
    with pytest.raises(ConfigError, match="strategy"):
        split_dataset(deduped, cfg, exp_cfg, strategy="random")  # type: ignore[arg-type]
    with pytest.raises(ConfigError, match="test_size"):
        split_dataset(deduped, cfg, {**exp_cfg, "test_size": 1.5})


def test_temporal_split_too_small_raises(cfg: dict[str, Any], exp_cfg: dict[str, Any]) -> None:
    tiny = pd.DataFrame({c: [0.0] for c in required_columns(cfg)})
    with pytest.raises(DataValidationError, match="empty part"):
        split_dataset(tiny, cfg, exp_cfg, strategy="temporal")


# --- scripts/data_summary.py --------------------------------------------------------------------


def test_format_table() -> None:
    table = _script.format_table({"train": {"rows": 1000, "frauds": 15, "fraud_rate": 0.015}})
    assert "train" in table and "1,000" in table and "1.5000%" in table


@pytest.mark.parametrize("strategy", ["stratified", "temporal"])
def test_data_summary_script(strategy: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert _script.main(["--source", "sample", "--strategy", strategy]) == 0
    out = capsys.readouterr().out
    assert "duplicates 20 removed (2 fraud)" in out
    assert f"strategy   {strategy}" in out
    for part in ("all", "train", "test"):
        assert part in out
    assert "5,980" in out


def test_data_summary_script_reports_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    real = _script.get_dataset_config()
    monkeypatch.setattr(
        _script, "get_dataset_config", lambda: {**real, "raw_path": str(tmp_path / "x.csv")}
    )
    assert _script.main(["--source", "raw"]) == 1
    assert "error: Raw dataset not found" in capsys.readouterr().err
