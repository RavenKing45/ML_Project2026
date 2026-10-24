"""Tests for ``POST /api/v1/predict`` (Prompt 13), on small real models fitted from the synthetic
sample (``real_artifacts`` in ``tests/backend/conftest.py``)."""

import json
import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from real_models import (
    DEFAULT_ID,
    EXAMPLES,
    LOGREG_ID,
    TECHNIQUE_PARAMS,
    UNCALIBRATED_ID,
    RealArtifacts,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

from backend.app.errors import ApiError
from backend.app.logging import LOGGER_NAME
from backend.app.main import API_PREFIX, create_app
from backend.app.ratelimit import PREDICT_LIMIT, limiter
from backend.app.services import artifacts as art
from backend.app.services import inference as inf
from backend.app.settings import Settings
from fraudstudy.contracts import ErrorResponse, PredictResponse, Transaction
from fraudstudy.evaluation import apply_calibrator
from fraudstudy.explain import DEFAULT_TOP_K, FeatureReference
from fraudstudy.export import FEATURE_REFERENCE_FILE
from fraudstudy.io import load_json, save_json

pytestmark = pytest.mark.usefixtures("rate_limits_off")

PREDICT = f"{API_PREFIX}/predict"
LIGHTGBM_IDS = [f"lightgbm__{technique}" for technique in TECHNIQUE_PARAMS]
SECRET_VALUE = -7.31415926  # a distinctive input that must never reach a log
EXPECTED_LABELS = {
    "legit_everyday": "LEGITIMATE",
    "legit_large": "LEGITIMATE",
    "fraud_typical": "FRAUD",
    "missed_fraud": "LEGITIMATE",
}


def features_of(real: RealArtifacts, preset_id: str = "fraud_typical") -> dict[str, float]:
    """A preset's features as a JSON-ready dict (``Time`` included)."""
    preset = next(p for p in real.presets.presets if p.id == preset_id)
    return preset.features.model_dump(exclude_none=True)


def post(client: TestClient, features: dict[str, Any], **fields: Any) -> PredictResponse:
    """POST a prediction and return the validated 200 response."""
    response = client.post(PREDICT, json={"features": features, **fields})
    assert response.status_code == 200, response.text
    return PredictResponse.model_validate(response.json())


def assert_error(response: Any, status: int, code: str) -> dict[str, Any]:
    """Check the status and the error shape; return ``error``."""
    assert response.status_code == status, response.text
    error = ErrorResponse.model_validate(response.json()).error
    assert error.code == code and error.message
    return response.json()["error"]


def score_of(real: RealArtifacts, model_id: str, features: dict[str, float]) -> float:
    """The pipeline's own score of ``features`` (manifest column order)."""
    entry = next(e for e in real.manifest.models if e.id == model_id)
    row = pd.DataFrame([[features[c] for c in entry.features]], columns=entry.features)
    return float(real.pipelines[model_id].predict_proba(row)[0, 1])


def entry_of(real: RealArtifacts, model_id: str) -> Any:
    """The manifest entry of ``model_id``."""
    return next(e for e in real.manifest.models if e.id == model_id)


# --- acceptance ----------------------------------------------------------------------------------


def test_preset_returns_200_with_score_explanation_and_technique_scores(
    real_client: TestClient, real_artifacts: RealArtifacts
) -> None:
    features = features_of(real_artifacts)
    result = post(real_client, features, input_source="preset")
    entry = entry_of(real_artifacts, DEFAULT_ID)
    assert result.model.id == DEFAULT_ID and result.model.technique == "smote_rus"
    assert result.threshold == entry.threshold_f1
    assert result.threshold_source == "tuned_on_validation_F1"
    assert result.risk_score == pytest.approx(score_of(real_artifacts, DEFAULT_ID, features))
    calibrated = apply_calibrator(real_artifacts.calibrators[DEFAULT_ID], [result.risk_score])
    assert result.calibrated_probability == pytest.approx(float(calibrated[0]))
    assert result.label == ("FRAUD" if result.risk_score >= result.threshold else "LEGITIMATE")
    assert result.disclaimer == inf.DISCLAIMER and result.latency_ms >= 0
    assert inf.EDITED_WARNING not in result.warnings

    explanation = result.explanation
    assert explanation is not None and explanation.method == "TreeSHAP"
    assert explanation.units == "log-odds" and len(explanation.top_features) == DEFAULT_TOP_K
    sizes = [abs(item.contribution) for item in explanation.top_features]
    assert sizes == sorted(sizes, reverse=True)
    for item in explanation.top_features:  # values are the RAW inputs
        assert item.value == pytest.approx(features[item.feature.removeprefix("log1p_")])

    assert [t.id for t in result.technique_scores] == LIGHTGBM_IDS
    for t in result.technique_scores:  # each at its own tuned threshold
        assert t.risk_score == pytest.approx(score_of(real_artifacts, t.id, features))
        assert t.threshold == entry_of(real_artifacts, t.id).threshold_f1
        assert t.label == ("FRAUD" if t.risk_score >= t.threshold else "LEGITIMATE")
    own = next(t for t in result.technique_scores if t.id == DEFAULT_ID)
    assert own.risk_score == pytest.approx(result.risk_score) and own.label == result.label


def test_every_preset_scores_as_it_was_picked(
    real_client: TestClient, real_artifacts: RealArtifacts
) -> None:
    for preset in real_artifacts.presets.presets:
        result = post(real_client, features_of(real_artifacts, preset.id))
        if preset.id in EXPECTED_LABELS:
            assert result.label == EXPECTED_LABELS[preset.id], preset.id


def _with(features: dict[str, Any], **changes: Any) -> dict[str, Any]:
    return features | changes


@pytest.mark.parametrize(
    ("case", "field"),
    [
        ("nan", "V1"),
        ("inf_string", "V2"),
        ("infinity", "V3"),
        ("string", "V4"),
        ("negative_amount", "Amount"),
        ("amount_too_large", "Amount"),
        ("null", "V5"),
        ("missing", "V14"),
        ("extra", "V29"),
    ],
)
def test_invalid_features_give_422_naming_the_field(
    real_client: TestClient, real_artifacts: RealArtifacts, case: str, field: str
) -> None:
    features: dict[str, Any] = features_of(real_artifacts)
    values: dict[str, Any] = {
        "nan": float("nan"),
        "inf_string": "inf",
        "infinity": float("inf"),
        "string": "abc",
        "negative_amount": -0.01,
        "amount_too_large": 1_000_000.01,
        "null": None,
        "extra": 0.0,
    }
    if case == "missing":
        del features[field]
    else:
        features[field] = values[case]
    body = json.dumps({"features": features})  # json.dumps writes NaN / Infinity literally
    response = real_client.post(PREDICT, content=body, headers={"content-type": "application/json"})
    error = assert_error(response, 422, "validation_error")
    assert ["body", "features", field] in [detail["loc"][:3] for detail in error["details"]]


def test_unknown_model_id_gives_404_with_the_valid_ids(
    real_client: TestClient, real_artifacts: RealArtifacts
) -> None:
    response = real_client.post(
        PREDICT, json={"features": features_of(real_artifacts), "model_id": "svm__magic"}
    )
    error = assert_error(response, 404, "not_found")
    valid = [e.id for e in real_artifacts.manifest.models]
    assert error["details"] == {"valid_ids": valid}
    assert all(model_id in error["message"] for model_id in valid)
    assert "svm__magic" not in error["message"]


def test_threshold_override_is_respected(
    real_client: TestClient, real_artifacts: RealArtifacts
) -> None:
    features = features_of(real_artifacts)
    tuned = post(real_client, features)
    at_score = post(real_client, features, threshold=tuned.risk_score)
    assert at_score.threshold == tuned.risk_score and at_score.label == "FRAUD"  # score >= t
    assert at_score.threshold_source == "user_supplied"
    assert at_score.risk_score == tuned.risk_score
    above = post(real_client, features, threshold=min(1.0, tuned.risk_score + 1e-6))
    assert above.label == "LEGITIMATE"
    assert [t.threshold for t in above.technique_scores] == [
        t.threshold for t in tuned.technique_scores
    ]  # the comparison keeps each model's own threshold
    for bad in (-0.1, 1.5):
        response = real_client.post(PREDICT, json={"features": features, "threshold": bad})
        assert_error(response, 422, "validation_error")


def test_model_id_selects_logreg_and_an_uncalibrated_model(
    real_client: TestClient, real_artifacts: RealArtifacts
) -> None:
    features = features_of(real_artifacts, "legit_everyday")
    logreg = post(real_client, features, model_id=LOGREG_ID)
    assert logreg.model.model == "logreg" and logreg.explanation is not None
    assert logreg.explanation.method == "linear"
    assert logreg.risk_score == pytest.approx(score_of(real_artifacts, LOGREG_ID, features))
    plain = post(real_client, features, model_id=UNCALIBRATED_ID)
    assert plain.calibrated_probability is None
    assert inf.NO_CALIBRATOR_WARNING not in plain.warnings  # it never had one


@pytest.mark.parametrize(
    ("source", "warned"), [("preset", False), ("edited", True), ("manual", True)]
)
def test_edited_and_manual_inputs_are_flagged(
    real_client: TestClient, real_artifacts: RealArtifacts, source: str, warned: bool
) -> None:
    result = post(real_client, features_of(real_artifacts, "legit_everyday"), input_source=source)
    assert (inf.EDITED_WARNING in result.warnings) is warned
    assert inf.EDITED_WARNING.startswith("Edited transaction may not resemble real payments")


def test_unknown_input_source_gives_422(
    real_client: TestClient, real_artifacts: RealArtifacts
) -> None:
    body = {"features": features_of(real_artifacts), "input_source": "pasted"}
    assert_error(real_client.post(PREDICT, json=body), 422, "validation_error")


def test_values_outside_the_training_range_are_flagged(
    real_client: TestClient, real_artifacts: RealArtifacts
) -> None:
    ranges = {f.raw_column: f.train_range for f in real_artifacts.reference.features}
    assert ranges["V14"] is not None and ranges["Amount"] is not None
    features = features_of(real_artifacts, "legit_everyday")
    normal = post(real_client, _with(features, V14=float(np.mean(ranges["V14"]))))
    assert not any(w.startswith("V14 ") for w in normal.warnings)
    odd = _with(features, V14=ranges["V14"][0] - 1.0, Amount=ranges["Amount"][1] + 1.0)
    result = post(real_client, odd, input_source="edited")
    assert result.warnings[:3] == [
        inf.EDITED_WARNING,
        "V14 is outside the range seen in training (below its 0.1th percentile).",
        "Amount is outside the range seen in training (above its 99.9th percentile).",
    ]


def test_time_is_ignored_and_key_order_does_not_matter(
    real_client: TestClient, real_artifacts: RealArtifacts
) -> None:
    features = features_of(real_artifacts)
    shuffled = dict(reversed(list(_with(features, Time=12_345.0).items())))
    without_time = {k: v for k, v in features.items() if k != "Time"}
    scores = {post(real_client, f).risk_score for f in (shuffled, without_time)}
    assert len(scores) == 1


def test_logs_never_contain_feature_values(
    real_client: TestClient, real_artifacts: RealArtifacts, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    features = _with(features_of(real_artifacts), V1=SECRET_VALUE, Amount=4321.987)
    result = post(real_client, features, input_source="manual")
    bad = real_client.post(PREDICT, json={"features": _with(features, V2=f"x{SECRET_VALUE}")})
    assert bad.status_code == 422
    assert str(SECRET_VALUE)[1:7] not in caplog.text and "4321.98" not in caplog.text
    records = [r.getMessage() for r in caplog.records if r.name == f"{LOGGER_NAME}.predict"]
    assert any(f"model_id={DEFAULT_ID} label={result.label}" in r for r in records)


def test_predict_is_rate_limited_to_60_per_minute(
    make_real_settings: Callable[..., Settings], real_artifacts: RealArtifacts
) -> None:
    limiter.reset()
    limit = int(PREDICT_LIMIT.split("/")[0])
    body = {"features": features_of(real_artifacts), "model_id": "unknown"}  # fast 404s
    with TestClient(create_app(make_real_settings(rate_limit_enabled=True))) as client:
        assert {client.post(PREDICT, json=body).status_code for _ in range(limit)} == {404}
        response = client.post(PREDICT, json=body)
    error = assert_error(response, 429, "rate_limited")
    assert "Too many requests" in error["message"]


# --- degraded artifacts --------------------------------------------------------------------------


def test_example_artifacts_have_no_model_to_predict_with(real_artifacts: RealArtifacts) -> None:
    settings = Settings(_env_file=None, use_example_artifacts=True, examples_dir=EXAMPLES)
    with TestClient(create_app(settings)) as client:
        response = client.post(PREDICT, json={"features": features_of(real_artifacts)})
    error = assert_error(response, 503, "artifact_unavailable")
    assert "example artifacts have no model files" in error["message"]
    default_id = load_json(EXAMPLES / "manifest.json")["default_model_id"]
    assert error["details"]["artifacts"] == [default_id]


def _models_copy(real: RealArtifacts, target: Path) -> Path:
    shutil.copytree(real.models, target)
    return target


def test_without_a_feature_reference_predictions_have_no_explanation(
    make_real_settings: Callable[..., Settings], real_artifacts: RealArtifacts, tmp_path: Path
) -> None:
    models = _models_copy(real_artifacts, tmp_path / "models")
    (models / FEATURE_REFERENCE_FILE).unlink()
    with TestClient(create_app(make_real_settings(models_dir=models))) as client:
        assert client.get(f"{API_PREFIX}/health").json()["status"] == "ok"  # it is optional
        result = post(client, features_of(real_artifacts))
    assert result.explanation is None and inf.NO_REFERENCE_WARNING in result.warnings
    assert not any("outside the range" in w for w in result.warnings)


def test_load_feature_reference(real_artifacts: RealArtifacts, tmp_path: Path) -> None:
    assert art.load_feature_reference(tmp_path) == (None, [])  # missing: not a problem
    reference, problems = art.load_feature_reference(real_artifacts.models)
    assert reference == real_artifacts.reference and problems == []
    path = tmp_path / FEATURE_REFERENCE_FILE
    path.write_text('{"features": NaN}', encoding="utf-8")
    reference, problems = art.load_feature_reference(tmp_path)
    assert reference is None and "could not be used (ValueError)" in problems[0]
    data = load_json(real_artifacts.models / FEATURE_REFERENCE_FILE)
    data["features"][0]["legit_quantiles"] = [1.0]
    save_json(data, path)
    reference, problems = art.load_feature_reference(tmp_path)
    assert reference is None and "features.0" in problems[0]
    reference = real_artifacts.reference
    features = [f.model_copy(update={"train_range": None}) for f in reference.features]
    old = reference.model_copy(update={"range_percentiles": None, "features": features})
    save_json(old.model_dump(mode="json"), path)
    reference, problems = art.load_feature_reference(tmp_path)
    assert reference == old and "has no training ranges" in problems[0]


# --- service helpers -----------------------------------------------------------------------------


def test_resolve_threshold(real_artifacts: RealArtifacts) -> None:
    entry = entry_of(real_artifacts, DEFAULT_ID)
    assert inf.resolve_threshold(entry, None) == (entry.threshold_f1, "tuned_on_validation_F1")
    f2 = entry.model_copy(update={"default_threshold": "f2"})
    assert inf.resolve_threshold(f2, None) == (entry.threshold_f2, "tuned_on_validation_F2")
    assert inf.resolve_threshold(entry, 0.0) == (0.0, "user_supplied")


def test_input_frame_follows_the_manifest_order(real_artifacts: RealArtifacts) -> None:
    entry = entry_of(real_artifacts, DEFAULT_ID)
    transaction = Transaction.model_validate(features_of(real_artifacts))
    frame = inf.input_frame(entry, transaction)
    assert list(frame.columns) == entry.features and "Time" not in frame.columns
    reordered = entry.model_copy(update={"features": list(reversed(entry.features))})
    assert list(inf.input_frame(reordered, transaction).columns) == reordered.features
    needs_time = entry.model_copy(update={"features": [*entry.features, "Time"]})
    with pytest.raises(ApiError) as raised:
        inf.input_frame(needs_time, transaction.model_copy(update={"Time": None}))
    assert raised.value.status_code == 422
    assert raised.value.details == [
        {"loc": ["body", "features", "Time"], "msg": inf.REQUIRED_MSG, "type": inf.MISSING}
    ]


def test_range_warnings(real_artifacts: RealArtifacts) -> None:
    reference = real_artifacts.reference
    inside = {f.raw_column: float(np.mean(f.train_range or (0, 0))) for f in reference.features}
    assert inf.range_warnings(reference, inside) == []
    low, high = next(f.train_range for f in reference.features if f.raw_column == "V3") or (0, 0)
    assert inf.range_warnings(reference, inside | {"V3": low - 1}) == [
        "V3 is outside the range seen in training (below its 0.1th percentile)."
    ]
    assert inf.range_warnings(reference, {"V3": high + 1, "V4": None}) == [
        "V3 is outside the range seen in training (above its 99.9th percentile)."
    ]
    no_ranges = FeatureReference.model_validate(
        reference.model_dump()
        | {
            "range_percentiles": None,
            "features": [f.model_dump() | {"train_range": None} for f in reference.features],
        }
    )
    assert inf.range_warnings(no_ranges, {"V3": high + 1}) == []


def test_resolve_model_errors(real_client: TestClient) -> None:
    store: art.ArtifactStore = real_client.app.state.store  # the FastAPI app's state
    assert inf.resolve_model(store, None).entry.id == DEFAULT_ID
    no_default = art.ArtifactStore(source="real", manifest=store.manifest, models=store.models)
    with pytest.raises(ApiError) as raised:
        inf.resolve_model(no_default, None)
    assert raised.value.status_code == 503 and raised.value.details["valid_ids"]
    not_loaded = art.ArtifactStore(source="real", manifest=store.manifest, problems=["x"])
    with pytest.raises(ApiError) as raised:
        inf.resolve_model(not_loaded, DEFAULT_ID)
    assert raised.value.status_code == 503
    assert raised.value.details == {"artifacts": [DEFAULT_ID], "problems": ["x"]}
    with pytest.raises(ApiError) as raised:
        inf.resolve_model(art.ArtifactStore(source="real"), None)
    assert raised.value.status_code == 503  # no manifest


def test_label_for_and_elapsed_ms() -> None:
    assert inf.label_for(0.5, 0.5) == "FRAUD" and inf.label_for(0.49, 0.5) == "LEGITIMATE"
    assert inf.elapsed_ms(0.0) > 0


def test_loaded_models_predict_single_threaded(real_client: TestClient) -> None:
    store: art.ArtifactStore = real_client.app.state.store  # the FastAPI app's state
    for loaded in store.models.values():
        assert loaded.pipeline.named_steps["model"].get_params()["n_jobs"] == art.PREDICT_N_JOBS


def test_single_threaded() -> None:
    estimator = LGBMClassifier(n_jobs=4)
    assert art.single_threaded(estimator) is estimator and estimator.n_jobs == art.PREDICT_N_JOBS
    pipeline = Pipeline([("scale", RobustScaler()), ("model", RandomForestClassifier(n_jobs=-1))])
    art.single_threaded(pipeline)
    assert pipeline.named_steps["model"].n_jobs == art.PREDICT_N_JOBS
    plain = object()
    assert art.single_threaded(plain) is plain


def test_explanation_for_a_model_the_explainer_cannot_use(real_artifacts: RealArtifacts) -> None:
    entry = entry_of(real_artifacts, DEFAULT_ID)
    n = len(entry.features)
    bare = LogisticRegression().fit(np.eye(n), np.arange(n) % 2)
    model = art.LoadedModel(entry, bare, None)
    X = inf.input_frame(entry, Transaction.model_validate(features_of(real_artifacts)))
    explanation, warnings = inf.explanation_for(model, X, real_artifacts.reference)
    assert explanation is None
    assert warnings == [
        f"No explanation for {DEFAULT_ID}: Expected a pipeline with 'preprocess' and 'model' steps."
    ]


def test_openapi_documents_the_post_endpoints(real_client: TestClient) -> None:
    paths = real_client.get("/openapi.json").json()["paths"]
    for path in ("/predict", "/predict/batch", "/smote-demo"):
        assert "post" in paths[f"{API_PREFIX}{path}"]
    batch = paths[f"{API_PREFIX}/predict/batch"]["post"]
    schema = batch["requestBody"]["content"]["multipart/form-data"]["schema"]
    assert schema["required"] == ["file"] and schema["properties"]["file"]["format"] == "binary"
    assert {"400", "413", "404", "422", "429", "503"} <= set(batch["responses"])
    assert {p["name"] for p in batch["parameters"]} == {"model_id", "threshold"}
