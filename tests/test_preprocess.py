"""Tests for fraudstudy.preprocess, fraudstudy.pipeline and scripts/run_baseline.py (sample)."""

import importlib.util
import time
from typing import Any

import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import RobustScaler

from fraudstudy.config import (
    PROJECT_ROOT,
    get_dataset_config,
    get_experiment_config,
    get_models_config,
)
from fraudstudy.data import deduplicate, load_dataset, split_dataset
from fraudstudy.pipeline import build_pipeline
from fraudstudy.preprocess import build_preprocessor, log1p_clipped

PCA = [f"V{i}" for i in range(1, 29)]
SAMPLE_RUNTIME_LIMIT_S = 30.0

_spec = importlib.util.spec_from_file_location(
    "run_baseline", PROJECT_ROOT / "scripts" / "run_baseline.py"
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
def deduped(cfg: dict[str, Any]) -> pd.DataFrame:
    return deduplicate(load_dataset("sample", cfg=cfg))[0]


@pytest.fixture(scope="module")
def split(deduped: pd.DataFrame, cfg: dict[str, Any], exp_cfg: dict[str, Any]) -> dict:
    return split_dataset(deduped, cfg, exp_cfg)


def _logreg_params() -> dict[str, Any]:
    return _script.default_model_params(get_models_config())


def _scalers(pipe: ImbPipeline) -> dict[str, RobustScaler]:
    pre = pipe.named_steps["preprocess"]
    return {name: pre.named_transformers_[name].named_steps["scale"] for name in ("pca", "amount")}


# --- log1p_clipped ------------------------------------------------------------------------------


def test_log1p_maps_zero_to_zero_and_clips_negatives() -> None:
    X = pd.DataFrame({"Amount": [0.0, -5.0, -1e9, np.e - 1.0, 100.0]})
    out = np.asarray(log1p_clipped(X))
    np.testing.assert_allclose(out[:, 0], [0.0, 0.0, 0.0, 1.0, np.log1p(100.0)])
    assert np.all(np.isfinite(out))


# --- build_preprocessor -------------------------------------------------------------------------


def test_preprocessor_default_columns(split: dict, cfg: dict[str, Any]) -> None:
    out = build_preprocessor(cfg).fit_transform(split["X_train"])
    assert isinstance(out, pd.DataFrame)
    assert out.shape[1] == 29
    assert list(out.columns) == [*PCA, "log1p_Amount"]
    assert "Time" not in out.columns
    assert out.index.equals(split["X_train"].index)


def test_preprocessor_drops_time_even_if_present(
    deduped: pd.DataFrame, cfg: dict[str, Any]
) -> None:
    out = build_preprocessor(cfg).fit_transform(deduped)  # has Time and Class columns
    assert list(out.columns) == [*PCA, "log1p_Amount"]


def test_preprocessor_imputes_missing_values(split: dict, cfg: dict[str, Any]) -> None:
    pre = build_preprocessor(cfg).fit(split["X_train"])
    X = split["X_test"].head(5).copy()
    X.iloc[0, 0] = np.nan
    X.iloc[1, X.columns.get_loc("Amount")] = np.nan
    out = pre.transform(X)
    assert not out.isna().any().any()


def test_amount_scaler_learns_from_log_amount(split: dict, cfg: dict[str, Any]) -> None:
    pre = build_preprocessor(cfg).fit(split["X_train"])
    scaler = pre.named_transformers_["amount"].named_steps["scale"]
    expected = RobustScaler().fit(np.log1p(split["X_train"][["Amount"]].to_numpy()))
    np.testing.assert_allclose(scaler.center_, expected.center_)
    np.testing.assert_allclose(scaler.scale_, expected.scale_)


# --- build_pipeline -----------------------------------------------------------------------------


def test_pipeline_structure(cfg: dict[str, Any], exp_cfg: dict[str, Any]) -> None:
    pipe = build_pipeline("logreg", "baseline", _logreg_params(), cfg=cfg, exp_cfg=exp_cfg)
    assert isinstance(pipe, ImbPipeline)
    assert [name for name, _ in pipe.steps] == ["preprocess", "model"]
    assert pipe.named_steps["model"].random_state == exp_cfg["seed"]
    assert pipe.named_steps["model"].max_iter == _logreg_params()["max_iter"]


def test_pipeline_rejects_unknown_or_unimplemented(
    cfg: dict[str, Any], exp_cfg: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="model 'svm'"):
        build_pipeline("svm", "baseline", cfg=cfg, exp_cfg=exp_cfg)
    with pytest.raises(ValueError, match="technique 'adasyn'"):
        build_pipeline("logreg", "adasyn", cfg=cfg, exp_cfg=exp_cfg)
    with pytest.raises(ValueError, match="no parameters"):
        build_pipeline(
            "logreg", "baseline", technique_params={"ratio": 0.1}, cfg=cfg, exp_cfg=exp_cfg
        )


def test_scaler_isolation_from_test_data(
    split: dict, cfg: dict[str, Any], exp_cfg: dict[str, Any]
) -> None:
    """Fitted scaler statistics come from X_train only and never move when test data flows."""
    pipe = build_pipeline("logreg", "baseline", _logreg_params(), cfg=cfg, exp_cfg=exp_cfg)
    pipe.fit(split["X_train"], split["y_train"])
    before = {n: (s.center_.copy(), s.scale_.copy()) for n, s in _scalers(pipe).items()}

    X_extreme = split["X_test"].copy()
    X_extreme["Amount"] = 1e7
    X_extreme["V14"] = -100.0
    transformed = pipe.named_steps["preprocess"].transform(X_extreme)
    scores = pipe.predict_proba(X_extreme)

    for name, scaler in _scalers(pipe).items():
        np.testing.assert_array_equal(scaler.center_, before[name][0])
        np.testing.assert_array_equal(scaler.scale_, before[name][1])
    # The statistics equal a scaler fitted on the training rows alone.
    train_only = RobustScaler().fit(split["X_train"][PCA])
    np.testing.assert_allclose(before["pca"][0], train_only.center_)
    np.testing.assert_allclose(before["pca"][1], train_only.scale_)
    assert transformed.shape == (len(X_extreme), 29)
    assert np.all(np.isfinite(scores))


# --- scripts/run_baseline.py --------------------------------------------------------------------


def test_quick_subsample_is_stratified(split: dict) -> None:
    X, y = _script.quick_subsample(split["X_train"], split["y_train"], 0.5, 0)
    assert len(y) == pytest.approx(len(split["y_train"]) * 0.5, abs=1)
    assert y.mean() == pytest.approx(split["y_train"].mean(), abs=0.005)
    X_all, _ = _script.quick_subsample(split["X_train"], split["y_train"], 1.0, 0)
    assert X_all is split["X_train"]


def test_run_cv_needs_enough_frauds(
    split: dict, cfg: dict[str, Any], exp_cfg: dict[str, Any]
) -> None:
    y = split["y_train"].copy()
    y[:] = 0
    y.iloc[:2] = 1
    with pytest.raises(ValueError, match="fewer than 5 CV folds"):
        _script.run_cv(split["X_train"], y, cfg, exp_cfg, _logreg_params(), 5)


def test_run_baseline_sample(capsys: pytest.CaptureFixture[str]) -> None:
    start = time.perf_counter()
    assert _script.main(["--source", "sample"]) == 0
    assert time.perf_counter() - start < SAMPLE_RUNTIME_LIMIT_S
    out = capsys.readouterr().out
    assert "PR-AUC" in out and "ROC-AUC" in out and "+-" in out
    pr_line = next(line for line in out.splitlines() if line.startswith("PR-AUC"))
    mean = float(pr_line.split()[1])
    assert 0.0 < mean <= 1.0


def test_run_baseline_quick(capsys: pytest.CaptureFixture[str]) -> None:
    assert _script.main(["--source", "sample", "--quick"]) == 0
    assert "(quick)" in capsys.readouterr().out
