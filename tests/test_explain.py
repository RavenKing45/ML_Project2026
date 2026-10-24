"""Tests for per-prediction explanations and the feature reference (synthetic sample only).

Small pipelines are fitted directly (no tuning) so the module stays fast.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from fraudstudy.contracts import Explanation
from fraudstudy.data import deduplicate, load_dataset, split_dataset
from fraudstudy.errors import ConfigError, DataValidationError
from fraudstudy.experiment import Configs
from fraudstudy.explain import (
    METHOD_LINEAR,
    METHOD_TREESHAP,
    N_QUANTILES,
    UNITS,
    FeatureReference,
    build_feature_reference,
    compute_contributions,
    explain,
    feature_map,
    legit_percentile,
)
from fraudstudy.pipeline import build_pipeline

TOL = 1e-6
N_ROWS = 100
LGBM_PARAMS = {"n_estimators": 40, "num_leaves": 15}


@dataclass(frozen=True)
class Fitted:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    logreg: Any
    lightgbm: Any
    reference: FeatureReference


@pytest.fixture(scope="module")
def fitted(test_cfgs: Configs) -> Fitted:
    """Logreg + SMOTE and LightGBM + SMOTE/RUS pipelines fitted on the sample's training split."""
    cfg, exp = test_cfgs.dataset, test_cfgs.experiment
    df, _ = deduplicate(load_dataset("sample", cfg=cfg), target=cfg["target"])
    split = split_dataset(df, cfg, exp)
    X, y = split.X_train, split.y_train
    logreg = build_pipeline(
        "logreg", "smote", {"max_iter": 2000}, {"sampling_strategy": 0.1}, cfg, exp
    ).fit(X, y)
    lightgbm = build_pipeline(
        "lightgbm",
        "smote_rus",
        {**LGBM_PARAMS, "n_jobs": exp["parallel"]["model_n_jobs"]},
        {"smote_sampling_strategy": 0.05, "rus_sampling_strategy": 0.5},
        cfg,
        exp,
    ).fit(X, y)
    reference = build_feature_reference(
        logreg.named_steps["preprocess"], X, y, source="sample", data_sha256=None, train_sha256="x"
    )
    return Fitted(X, y, split.X_test, logreg, lightgbm, reference)


# --- additivity -----------------------------------------------------------------------------------


def test_logreg_contributions_add_up_to_decision_function(fitted: Fitted) -> None:
    rows = fitted.X_test.iloc[:N_ROWS]
    c = compute_contributions(fitted.logreg, rows, fitted.reference)
    assert c.method == METHOD_LINEAR and c.units == UNITS
    np.testing.assert_allclose(
        c.base + c.contributions.sum(axis=1), fitted.logreg.decision_function(rows), atol=TOL
    )
    model = fitted.logreg.named_steps["model"]
    means = np.array([f.train_mean_scaled for f in fitted.reference.features])
    expected_base = model.intercept_[0] + model.coef_.ravel() @ means
    np.testing.assert_allclose(c.base, expected_base, atol=TOL)


def test_lightgbm_contributions_add_up_to_raw_score(fitted: Fitted) -> None:
    rows = fitted.X_test.iloc[:N_ROWS]
    c = compute_contributions(fitted.lightgbm, rows, fitted.reference)
    assert c.method == METHOD_TREESHAP and c.units == UNITS
    model = fitted.lightgbm.named_steps["model"]
    raw = model.predict(fitted.lightgbm.named_steps["preprocess"].transform(rows), raw_score=True)
    np.testing.assert_allclose(c.base + c.contributions.sum(axis=1), raw, atol=TOL)
    # the raw score is the log-odds of the risk score the backend reports
    proba = fitted.lightgbm.predict_proba(rows)[:, 1]
    np.testing.assert_allclose(1.0 / (1.0 + np.exp(-c.score)), proba, atol=TOL)


def test_samplers_never_run(fitted: Fitted, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_: Any, **__: Any) -> Any:
        raise AssertionError("explanations must not resample")

    monkeypatch.setattr(SMOTE, "fit_resample", forbidden)
    monkeypatch.setattr(RandomUnderSampler, "fit_resample", forbidden)
    for pipe in (fitted.logreg, fitted.lightgbm):
        explain(pipe, fitted.X_test.iloc[[0]], fitted.reference)


# --- the explanation ------------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["logreg", "lightgbm"])
def test_explanation_is_sorted_deterministic_and_contract_valid(fitted: Fitted, which: str) -> None:
    pipe = getattr(fitted, which)
    row = fitted.X_test.iloc[[3]]
    first = explain(pipe, row, fitted.reference, top_k=8)
    Explanation.model_validate(first.model_dump(mode="json"))
    assert first.units == "log-odds" and len(first.top_features) == 8
    sizes = [abs(f.contribution) for f in first.top_features]
    assert sizes == sorted(sizes, reverse=True)
    assert explain(pipe, row, fitted.reference, top_k=8) == first
    full = compute_contributions(pipe, row, fitted.reference)
    assert sizes[0] == pytest.approx(np.abs(full.contributions[0]).max())


def test_values_are_raw_inputs_and_percentiles_come_from_the_reference(fitted: Fitted) -> None:
    row = fitted.X_test.iloc[[5]]
    items = explain(fitted.logreg, row, fitted.reference, top_k=29).top_features
    assert len(items) == 29
    ref = {f.feature: f for f in fitted.reference.features}
    for item in items:
        column = ref[item.feature].raw_column
        assert item.value == float(row[column].iloc[0])
        quantiles = ref[item.feature].legit_quantiles
        assert item.legit_percentile == legit_percentile(item.value, quantiles)
    amount = next(i for i in items if i.feature == "log1p_Amount")
    assert amount.value == float(row["Amount"].iloc[0])  # raw Amount, not log1p(Amount)


def test_top_k_larger_than_feature_count_returns_all(fitted: Fitted) -> None:
    items = explain(fitted.lightgbm, fitted.X_test.iloc[[0]], fitted.reference, top_k=100)
    assert len(items.top_features) == len(fitted.reference.features)


# --- feature reference ----------------------------------------------------------------------------


def test_reference_uses_legit_training_rows_in_raw_units(fitted: Fitted) -> None:
    ref = fitted.reference
    legit = fitted.X_train[fitted.y_train == 0]
    assert ref.n_legit == len(legit) and ref.n_train == len(fitted.X_train)
    assert [f.feature for f in ref.features] == [f"V{i}" for i in range(1, 29)] + ["log1p_Amount"]
    for f in ref.features:
        assert len(f.legit_quantiles) == N_QUANTILES
        np.testing.assert_allclose(
            f.legit_quantiles, np.quantile(legit[f.raw_column], np.linspace(0, 1, N_QUANTILES))
        )
    amount = ref.features[-1]
    assert amount.raw_column == "Amount" and amount.legit_quantiles[-1] == legit["Amount"].max()
    scaled = fitted.logreg.named_steps["preprocess"].transform(fitted.X_train)
    np.testing.assert_allclose(
        [f.train_mean_scaled for f in ref.features], np.asarray(scaled).mean(axis=0)
    )
    assert FeatureReference.model_validate(ref.model_dump(mode="json")) == ref


def test_feature_map_pairs_outputs_with_raw_columns(fitted: Fitted) -> None:
    pairs = feature_map(fitted.lightgbm.named_steps["preprocess"])
    assert pairs[0] == ("V1", "V1") and pairs[-1] == ("log1p_Amount", "Amount")


def test_legit_percentile_edges_ties_and_interpolation() -> None:
    q = list(np.linspace(0.0, 100.0, N_QUANTILES))  # q_k = k
    assert legit_percentile(-5.0, q) == 0.0
    assert legit_percentile(500.0, q) == 100.0
    assert legit_percentile(42.0, q) == 42.0
    assert legit_percentile(42.25, q) == pytest.approx(42.25)
    tied = [0.0] * 11 + list(np.linspace(1.0, 90.0, N_QUANTILES - 11))  # q_0..q_10 are all 0
    assert legit_percentile(0.0, tied) == 5.0


def test_reference_rejects_bad_quantiles(fitted: Fitted) -> None:
    data = fitted.reference.model_dump(mode="json")
    data["features"][0]["legit_quantiles"] = data["features"][0]["legit_quantiles"][::-1]
    with pytest.raises(ValueError, match="non-decreasing"):
        FeatureReference.model_validate(data)
    data = fitted.reference.model_dump(mode="json")
    data["features"][0]["legit_quantiles"] = [0.0] * 10
    with pytest.raises(ValueError, match="101 quantiles"):
        FeatureReference.model_validate(data)


# --- errors ---------------------------------------------------------------------------------------


def test_random_forest_is_not_explained(fitted: Fitted, test_cfgs: Configs) -> None:
    rf = build_pipeline(
        "random_forest",
        "baseline",
        {"n_estimators": 5, "n_jobs": 1},
        cfg=test_cfgs.dataset,
        exp_cfg=test_cfgs.experiment,
    ).fit(fitted.X_train, fitted.y_train)
    with pytest.raises(ConfigError, match="shap"):
        explain(rf, fitted.X_test.iloc[[0]], fitted.reference)


def test_explain_input_errors(fitted: Fitted) -> None:
    ref = fitted.reference
    with pytest.raises(DataValidationError, match="exactly one row"):
        explain(fitted.logreg, fitted.X_test.iloc[:2], ref)
    with pytest.raises(DataValidationError, match="V14"):
        explain(fitted.logreg, fitted.X_test.iloc[[0]].drop(columns="V14"), ref)
    nan_row = fitted.X_test.iloc[[0]].copy()
    nan_row["V3"] = np.nan
    with pytest.raises(DataValidationError, match="finite"):
        explain(fitted.logreg, nan_row, ref)
    with pytest.raises(ConfigError, match="top_k"):
        explain(fitted.logreg, fitted.X_test.iloc[[0]], ref, top_k=0)
    short = ref.model_copy(update={"features": ref.features[:-1]})
    with pytest.raises(DataValidationError, match="feature_reference.json"):
        explain(fitted.logreg, fitted.X_test.iloc[[0]], short)
