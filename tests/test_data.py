"""Tests for fraudstudy.data (load, check, deduplicate, split) and scripts/data_summary.py."""

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from fraudstudy.config import PROJECT_ROOT, get_dataset_config, get_experiment_config, resolve
from fraudstudy.data import check_dataset, deduplicate, feature_columns, load_dataset, split_dataset
from fraudstudy.io import sha256_file

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


def test_check_accepts_sample_and_missing_feature_values(
    sample: pd.DataFrame, cfg: dict[str, Any]
) -> None:
    check_dataset(sample, cfg)
    df = sample.head(50).copy()
    df.loc[[0, 1, 2], "V3"] = np.nan  # blank cells are imputed later, not rejected
    df.loc[4, "Amount"] = np.nan
    df["note"] = 1.0  # extra columns are ignored
    check_dataset(df, cfg)


def test_check_missing_column_raises(sample: pd.DataFrame, cfg: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="missing required columns: V7"):
        check_dataset(sample.drop(columns=["V7"]), cfg)


def test_check_inf_raises(sample: pd.DataFrame, cfg: dict[str, Any]) -> None:
    df = sample.head(20).copy()
    df.loc[3, "V2"] = np.inf
    df.loc[5, "V2"] = -np.inf
    with pytest.raises(ValueError, match=r"infinite values in: V2 \(2 rows\)"):
        check_dataset(df, cfg)


def test_check_bad_class_raises(sample: pd.DataFrame, cfg: dict[str, Any]) -> None:
    df = sample.head(20).copy()
    df.loc[0, "Class"] = 2
    with pytest.raises(ValueError, match="must be 0 or 1; 1 rows have 2"):
        check_dataset(df, cfg)


def test_check_missing_class_raises(sample: pd.DataFrame, cfg: dict[str, Any]) -> None:
    df = sample.head(20).astype({"Class": float})
    df.loc[0, "Class"] = np.nan
    with pytest.raises(ValueError, match="must be 0 or 1"):
        check_dataset(df, cfg)


def test_check_reports_every_problem_at_once(sample: pd.DataFrame, cfg: dict[str, Any]) -> None:
    df = sample.head(20).drop(columns=["V9"])
    df.loc[1, "V1"] = np.inf
    df.loc[2, "Class"] = 2
    df.loc[3, "Amount"] = -5.0
    df["V5"] = "x"
    with pytest.raises(ValueError) as excinfo:
        check_dataset(df, cfg)
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


def test_check_does_not_mutate(sample: pd.DataFrame, cfg: dict[str, Any]) -> None:
    before = sample.copy()
    check_dataset(sample, cfg)
    pd.testing.assert_frame_equal(sample, before)


# --- load ---------------------------------------------------------------------------------------


def test_load_sample(sample: pd.DataFrame) -> None:
    assert sample.shape == (6000, 31)
    assert list(sample.columns) == ["Time", *PCA, "Amount", "Class"]
    assert isinstance(sample.index, pd.RangeIndex)


def test_load_raw_missing_explains_download(cfg: dict[str, Any], tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="kaggle.com/datasets/mlg-ulb/creditcardfraud"):
        load_dataset("raw", cfg=_cfg_with(cfg, raw_path=str(tmp_path / "nope.csv")))


def test_load_raw_path_is_folder_explains_fix(cfg: dict[str, Any], tmp_path: Path) -> None:
    folder = tmp_path / "creditcard.csv"
    folder.mkdir()
    with pytest.raises(FileNotFoundError, match="is a folder, not a file"):
        load_dataset("raw", cfg=_cfg_with(cfg, raw_path=str(folder)))


def test_load_sample_missing_raises(cfg: dict[str, Any], tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="generate_synthetic_data"):
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
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_dataset("raw", cfg=bad)
    assert len(load_dataset("raw", verify_hash=False, cfg=bad)) == 6000


def test_load_raw_invalid_contents_raise(cfg: dict[str, Any], tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"Time": [0.0], "Amount": [1.0], "Class": [0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset("raw", verify_hash=False, cfg=_cfg_with(cfg, raw_path=str(path)))


# --- deduplicate --------------------------------------------------------------------------------


def test_dedup_counts(sample: pd.DataFrame) -> None:
    out, report = deduplicate(sample)
    assert report == {
        "rows_before": 6000, "rows_after": 5980, "rows_removed": 20, "frauds_removed": 2
    }
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
    assert report["rows_removed"] == 0 and len(out) == 2


# --- feature_columns ----------------------------------------------------------------------------


def test_feature_columns_leave_out_time(cfg: dict[str, Any]) -> None:
    assert feature_columns(cfg) == [*PCA, "Amount"]


# --- split --------------------------------------------------------------------------------------


def test_stratified_split(
    deduped: pd.DataFrame, cfg: dict[str, Any], exp_cfg: dict[str, Any]
) -> None:
    split = split_dataset(deduped, cfg, exp_cfg, strategy="stratified")
    s = split["summary"]
    assert split["strategy"] == "stratified"
    assert abs(s["train"]["fraud_rate"] - s["test"]["fraud_rate"]) <= 0.003
    assert set(split["X_train"].index).isdisjoint(split["X_test"].index)
    assert s["train"]["rows"] + s["test"]["rows"] == len(deduped) == s["all"]["rows"]
    assert s["train"]["frauds"] + s["test"]["frauds"] == 88
    assert s["test"]["rows"] == pytest.approx(len(deduped) * exp_cfg["test_size"], abs=1)
    assert list(split["X_train"].columns) == [*PCA, "Amount"]
    assert split["X_train"].index.equals(split["y_train"].index)
    assert split["X_test"].index.equals(split["y_test"].index)


def test_stratified_split_is_deterministic(
    deduped: pd.DataFrame, cfg: dict[str, Any], exp_cfg: dict[str, Any]
) -> None:
    a = split_dataset(deduped, cfg, exp_cfg)
    b = split_dataset(deduped, cfg, exp_cfg)
    assert a["X_test"].index.equals(b["X_test"].index)
    c = split_dataset(deduped, cfg, {**exp_cfg, "seed": exp_cfg["seed"] + 1})
    assert not a["X_test"].index.equals(c["X_test"].index)


def test_temporal_split(
    deduped: pd.DataFrame, cfg: dict[str, Any], exp_cfg: dict[str, Any]
) -> None:
    split = split_dataset(deduped, cfg, exp_cfg, strategy="temporal")
    time = deduped["Time"]
    assert time.loc[split["X_train"].index].max() <= time.loc[split["X_test"].index].min()
    assert set(split["X_train"].index).isdisjoint(split["X_test"].index)
    assert len(split["X_train"]) == round(len(deduped) * (1 - exp_cfg["test_size"]))
    assert len(split["X_train"]) + len(split["X_test"]) == len(deduped)
    assert "Time" not in split["X_train"].columns


def test_split_does_not_mutate(
    deduped: pd.DataFrame, cfg: dict[str, Any], exp_cfg: dict[str, Any]
) -> None:
    before = deduped.copy()
    for strategy in ("stratified", "temporal"):
        split_dataset(deduped, cfg, exp_cfg, strategy=strategy)
    pd.testing.assert_frame_equal(deduped, before)


def test_split_unknown_strategy_raises(
    deduped: pd.DataFrame, cfg: dict[str, Any], exp_cfg: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="strategy"):
        split_dataset(deduped, cfg, exp_cfg, strategy="random")


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
