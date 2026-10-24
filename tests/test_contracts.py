"""Tests for the data contracts and the example JSON fixtures."""

import copy
import itertools
from typing import Any, get_args

import pytest
from pydantic import ValidationError

from fraudstudy import contracts as c
from fraudstudy.config import get_experiment_config
from fraudstudy.io import load_json
from fraudstudy.paths import PROJECT_ROOT

EXAMPLES_DIR = PROJECT_ROOT / "docs" / "contracts" / "examples"
EXPECTED_EXAMPLES = {
    "comparison",
    "curves",
    "stats",
    "eda",
    "resampling_audit",
    "manifest",
    "presets",
    "predict_response",
    "batch_response",
    "smote_demo",
}


def _example(name: str) -> dict[str, Any]:
    return load_json(EXAMPLES_DIR / f"{name}.json")


def test_every_example_file_exists_and_is_registered() -> None:
    on_disk = {p.stem for p in EXAMPLES_DIR.glob("*.json")}
    assert on_disk == EXPECTED_EXAMPLES == set(c.EXAMPLE_MODELS)


@pytest.mark.parametrize("name", sorted(EXPECTED_EXAMPLES))
def test_example_validates_and_is_flagged(name: str) -> None:
    data = _example(name)
    assert data["is_example"] is True
    parsed = c.EXAMPLE_MODELS[name].model_validate(data)
    assert parsed.is_example is True
    assert parsed.contract_version == c.CONTRACT_VERSION


@pytest.mark.parametrize("name", sorted(EXPECTED_EXAMPLES))
def test_example_round_trips(name: str) -> None:
    model = c.EXAMPLE_MODELS[name]
    parsed = model.model_validate(_example(name))
    assert model.model_validate(parsed.model_dump(mode="json")) == parsed


def test_comparison_example_has_all_18_cells() -> None:
    table = c.ComparisonTable.model_validate(_example("comparison"))
    cells = {(r.model, r.technique) for r in table.records}
    assert len(table.records) == 18
    assert cells == set(itertools.product(get_args(c.ModelId), get_args(c.TechniqueId)))


def test_ids_match_experiment_config() -> None:
    experiment = get_experiment_config()
    assert list(get_args(c.ModelId)) == experiment["models"]
    assert list(get_args(c.TechniqueId)) == experiment["techniques"]
    assert set(c.MODEL_LABELS) == set(experiment["models"])
    assert set(c.TECHNIQUE_LABELS) == set(experiment["techniques"])


def test_transaction_fields_match_dataset_columns() -> None:
    assert list(c.Transaction.model_fields) == ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]


def test_predict_response_has_master_guide_fields() -> None:
    fields = set(c.PredictResponse.model_fields)
    assert {
        "label",
        "risk_score",
        "calibrated_probability",
        "threshold",
        "threshold_source",
        "model",
        "explanation",
        "warnings",
        "disclaimer",
        "latency_ms",
    } <= fields
    assert set(c.ModelRef.model_fields) == {"id", "model", "technique"}
    assert set(c.Explanation.model_fields) == {"method", "units", "top_features"}
    assert set(c.FeatureContribution.model_fields) == {
        "feature",
        "value",
        "contribution",
        "legit_percentile",
    }


def test_comparison_record_fields() -> None:
    assert set(c.ComparisonRecord.model_fields) == {
        "model",
        "technique",
        "model_label",
        "technique_label",
        "best_params",
        "chosen_ratio",
        "cv_pr_auc_mean",
        "cv_pr_auc_std",
        "cv_roc_auc_mean",
        "threshold_f1",
        "threshold_f2",
        "test_at_0_5",
        "test_at_tuned",
        "pr_auc_ci",
        "roc_auc_ci",
        "fit_seconds",
        "n_train_after_resampling",
    }
    assert len(c.Metrics.model_fields) == 14


def _features() -> dict[str, Any]:
    return copy.deepcopy(_example("presets")["presets"][0]["features"])


def test_predict_request_defaults() -> None:
    req = c.PredictRequest.model_validate({"features": _features()})
    assert req.model_id is None and req.threshold is None
    assert c.SmoteDemoRequest().k_neighbors == 5


def test_error_response_shape() -> None:
    body = {"error": {"code": "unknown_model", "message": "Valid ids: lightgbm__smote."}}
    assert c.ErrorResponse.model_validate(body).model_dump() == body


# --- negative cases: the contracts must reject bad data ---------------------------------------


def _record() -> dict[str, Any]:
    return copy.deepcopy(_example("comparison")["records"][0])


def test_unknown_field_rejected() -> None:
    bad = _record() | {"pr_auc_typo": 0.5}
    with pytest.raises(ValidationError, match="pr_auc_typo"):
        c.ComparisonRecord.model_validate(bad)


def test_unknown_model_rejected() -> None:
    with pytest.raises(ValidationError):
        c.ComparisonRecord.model_validate(_record() | {"model": "svm"})


def test_probability_out_of_range_rejected() -> None:
    bad = _record()
    bad["test_at_tuned"]["precision"] = 1.5
    with pytest.raises(ValidationError):
        c.ComparisonRecord.model_validate(bad)


def test_reversed_ci_rejected() -> None:
    with pytest.raises(ValidationError, match="low <= high"):
        c.ComparisonRecord.model_validate(_record() | {"pr_auc_ci": [0.9, 0.1]})


def test_duplicate_cells_rejected() -> None:
    data = _example("comparison")
    data["records"].append(data["records"][0])
    with pytest.raises(ValidationError, match="duplicate"):
        c.ComparisonTable.model_validate(data)


def test_curve_too_many_points_rejected() -> None:
    n = c.MAX_CURVE_POINTS + 1
    with pytest.raises(ValidationError, match="Downsample"):
        c.RocCurve(fpr=[0.0] * n, tpr=[0.0] * n)


def test_curve_unequal_lengths_rejected() -> None:
    with pytest.raises(ValidationError, match="equal length"):
        c.PrCurve(recall=[0.0, 1.0], precision=[1.0])


def test_histogram_shape_rejected() -> None:
    with pytest.raises(ValidationError):
        c.Histogram(bin_edges=[0.0, 1.0], legit_counts=[1, 2], fraud_counts=[0, 0])


def _manifest_entry() -> dict[str, Any]:
    return copy.deepcopy(_example("manifest")["models"][0])


def test_bad_sha256_rejected() -> None:
    with pytest.raises(ValidationError):
        c.ManifestEntry.model_validate(_manifest_entry() | {"sha256": "xyz"})


def test_manifest_id_must_match_cell() -> None:
    with pytest.raises(ValidationError, match="Manifest id"):
        c.ManifestEntry.model_validate(_manifest_entry() | {"id": "logreg__smote"})


def test_manifest_default_model_must_exist() -> None:
    data = _example("manifest") | {"default_model_id": "random_forest__rus"}
    with pytest.raises(ValidationError, match="default_model_id"):
        c.Manifest.model_validate(data)


@pytest.mark.parametrize("amount", [-1.0, c.AMOUNT_MAX + 1])
def test_amount_out_of_range_rejected(amount: float) -> None:
    with pytest.raises(ValidationError):
        c.Transaction.model_validate(_features() | {"Amount": amount})


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_feature_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        c.Transaction.model_validate(_features() | {"V3": value})


def test_unknown_feature_rejected() -> None:
    with pytest.raises(ValidationError, match="V29"):
        c.PredictRequest.model_validate({"features": _features() | {"V29": 0.0}})


def test_missing_feature_rejected() -> None:
    features = _features()
    del features["V7"]
    with pytest.raises(ValidationError, match="V7"):
        c.PredictRequest.model_validate({"features": features})


def test_threshold_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        c.PredictRequest.model_validate({"features": _features(), "threshold": 1.5})


def test_predict_label_must_match_threshold() -> None:
    data = _example("predict_response") | {"label": "LEGITIMATE"}
    with pytest.raises(ValidationError, match="label must be FRAUD"):
        c.PredictResponse.model_validate(data)


def test_smote_demo_k_must_be_below_n_fraud() -> None:
    with pytest.raises(ValidationError, match="k_neighbors"):
        c.SmoteDemoRequest(n_fraud=5, k_neighbors=5)


def test_examples_are_lf_utf8() -> None:
    for path in EXAMPLES_DIR.glob("*.json"):
        assert b"\r\n" not in path.read_bytes(), path.name


# --- 1.5: error details and the backend's GET responses --------------------------------------


@pytest.mark.parametrize("details", [None, ["lightgbm__smote"], {"loc": ["body"]}])
def test_error_details_round_trip_when_given(details: Any) -> None:
    body = {"error": {"code": "unknown_model", "message": "Unknown.", "details": details}}
    assert c.ErrorResponse.model_validate(body).model_dump(mode="json") == body


def _health(**overrides: Any) -> dict[str, Any]:
    return {
        "status": "ok",
        "problems": [],
        "models_loaded": 0,
        "default_model": None,
        "data_sha256": None,
        "git_hash": None,
        "version": "0.1.0",
        "uptime_s": 1.5,
    } | overrides


def test_health_status_must_match_problems() -> None:
    assert c.HealthResponse.model_validate(_health()).status == "ok"
    assert c.HealthResponse.model_validate(_health(status="degraded", problems=["x"]))
    with pytest.raises(ValidationError, match="status must be 'degraded'"):
        c.HealthResponse.model_validate(_health(problems=["x"]))
    with pytest.raises(ValidationError, match="status must be 'ok'"):
        c.HealthResponse.model_validate(_health(status="degraded"))


def _model_info(**overrides: Any) -> dict[str, Any]:
    entry = _manifest_entry()
    keep = ("id", "model", "technique", "threshold_f1", "threshold_f2", "has_calibrator")
    return {key: entry[key] for key in keep} | {
        "model_label": "LightGBM",
        "technique_label": "A. Baseline (no resampling)",
        "default_threshold": "f1",
        "features": entry["features"],
        "loaded": False,
    } | overrides


def test_models_response_checks_ids_and_default() -> None:
    info = _model_info()
    assert c.ModelsResponse(default_model_id=info["id"], models=[info]).models[0].loaded is False
    with pytest.raises(ValidationError, match="Model id"):
        c.ModelInfo.model_validate(_model_info(id="logreg__smote"))
    with pytest.raises(ValidationError, match="default_model_id"):
        c.ModelsResponse(default_model_id="logreg__rus", models=[info])
    with pytest.raises(ValidationError, match="unique"):
        c.ModelsResponse(default_model_id=None, models=[info, info])


def test_results_and_eda_responses_wrap_the_example_files() -> None:
    results = c.ResultsResponse(
        comparison=_example("comparison"),
        curves=_example("curves"),
        stats=_example("stats"),
    )
    assert results.temporal_check is None
    eda = c.EdaResponse(eda=_example("eda"))
    assert eda.resampling_audit is None
    with pytest.raises(ValidationError):
        c.EdaResponse(eda=_example("eda") | {"n_rows": -1})
