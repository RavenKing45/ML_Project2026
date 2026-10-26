"""Tests for the FastAPI application core (Prompt 12).

Most tests run the app on the example artifacts (``docs/contracts/examples``). Every app is
built from explicit settings without a ``.env`` file, so a developer's environment cannot change
the outcome. Real-mode model loading uses a tiny logistic regression written to a temp folder.
"""

import logging
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from pydantic import BaseModel
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from starlette.datastructures import Headers

from backend.app import __version__
from backend.app.errors import (
    STATUS_CODES,
    ApiError,
    error_body,
    error_code_for_status,
    validation_details,
)
from backend.app.logging import (
    LOGGER_NAME,
    MAX_LOGGED_PATH,
    access_message,
    configure_logging,
    safe_path,
)
from backend.app.main import API_PREFIX, create_app
from backend.app.main import app as module_app
from backend.app.middleware import REQUEST_ID_HEADER, TIMING_HEADER, request_id_from
from backend.app.ratelimit import limiter
from backend.app.routers.health import build_health
from backend.app.routers.models import model_info
from backend.app.services import artifacts as art
from backend.app.settings import DEV_ORIGINS, Settings, get_settings
from fraudstudy import contracts as c
from fraudstudy.config import PROJECT_ROOT
from fraudstudy.experiment import library_versions
from fraudstudy.io import load_json, save_json, sha256_file

EXAMPLES = PROJECT_ROOT / "docs" / "contracts" / "examples"
DASHBOARD = "https://fraud-study.vercel.app"
DEV_SERVER = DEV_ORIGINS[0]
SECRET = "SECRET-VALUE-7731"  # must never appear in a response body or a log record
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
DATA_SHA = "ab" * 32
REAL_ID = "logreg__smote"
GET_ENDPOINTS: list[tuple[str, type[BaseModel]]] = [
    ("/health", c.HealthResponse),
    ("/models", c.ModelsResponse),
    ("/results", c.ResultsResponse),
    ("/eda", c.EdaResponse),
    ("/presets", c.PresetsFile),
]
ENDPOINT_OF: dict[str, str] = {
    "comparison": "/results",
    "curves": "/results",
    "stats": "/results",
    "eda": "/eda",
    "presets": "/presets",
    "manifest": "/models",
}


def make_settings(**overrides: Any) -> Settings:
    """Settings on the example artifacts, development mode, no ``.env``; ``overrides`` win."""
    values: dict[str, Any] = {
        "use_example_artifacts": True,
        "examples_dir": EXAMPLES,
        "artifacts_dir": PROJECT_ROOT / "artifacts",
        "models_dir": PROJECT_ROOT / "artifacts" / "models",
        "allowed_origins": "",
        "environment": "development",
        "default_model_id": None,
        "log_level": "INFO",
        "rate_limit_enabled": True,
    }
    return Settings(_env_file=None, **(values | overrides))


def api(path: str) -> str:
    """``/api/v1`` + ``path``."""
    return f"{API_PREFIX}{path}"


def assert_error(response: Any, status: int, code: str) -> dict[str, Any]:
    """Check the status and the error shape (``details`` always present); return ``error``."""
    assert response.status_code == status, response.text
    body = response.json()
    c.ErrorResponse.model_validate(body)
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message", "details"}
    assert body["error"]["code"] == code
    assert body["error"]["message"]
    return body["error"]


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A started app on the example artifacts."""
    with TestClient(create_app(make_settings())) as test_client:
        yield test_client


@pytest.fixture
def examples_copy(tmp_path: Path) -> Path:
    """A writable copy of the example artifacts."""
    target = tmp_path / "examples"
    shutil.copytree(EXAMPLES, target)
    return target


def start(settings: Settings) -> TestClient:
    """A TestClient for ``create_app(settings)``; use it as a context manager."""
    return TestClient(create_app(settings))


# --- acceptance: GET endpoints on the example artifacts ------------------------------------------


@pytest.mark.parametrize(("path", "contract"), GET_ENDPOINTS)
def test_get_endpoints_return_200_and_validate_against_contracts(
    client: TestClient, path: str, contract: type[BaseModel]
) -> None:
    response = client.get(api(path))
    assert response.status_code == 200, response.text
    parsed = contract.model_validate(response.json())
    assert parsed.is_example is True


def test_health_is_ok_on_the_example_artifacts(client: TestClient) -> None:
    health = c.HealthResponse.model_validate(client.get(api("/health")).json())
    manifest = c.Manifest.model_validate(load_json(EXAMPLES / "manifest.json"))
    assert health.status == "ok" and health.problems == []
    assert health.models_loaded == 0  # the examples have no model files
    assert health.default_model == manifest.default_model_id
    assert health.data_sha256 == manifest.dataset.sha256
    assert health.version == __version__
    assert health.uptime_s >= 0


def _same(served: Any, name: str) -> bool:
    """``served`` equals the example file, compared as contracts (optional defaults filled)."""
    contract = c.EXAMPLE_MODELS[name]
    return contract.model_validate(served) == contract.model_validate(
        load_json(EXAMPLES / f"{name}.json")
    )


def test_results_eda_and_presets_serve_the_example_files(client: TestClient) -> None:
    results = client.get(api("/results")).json()
    assert all(_same(results[name], name) for name in ("comparison", "curves", "stats"))
    assert len(results["comparison"]["records"]) == 18
    assert results["temporal_check"] is None  # no example temporal check
    eda = client.get(api("/eda")).json()
    assert _same(eda["eda"], "eda") and _same(eda["resampling_audit"], "resampling_audit")
    assert _same(client.get(api("/presets")).json(), "presets")


def test_models_list_the_manifest_unloaded_on_the_example_artifacts(client: TestClient) -> None:
    body = c.ModelsResponse.model_validate(client.get(api("/models")).json())
    manifest = c.Manifest.model_validate(load_json(EXAMPLES / "manifest.json"))
    assert [m.id for m in body.models] == [m.id for m in manifest.models]
    assert body.default_model_id == manifest.default_model_id
    assert not any(m.loaded for m in body.models)
    assert all(m.model_label and m.technique_label for m in body.models)


# --- acceptance: a missing or broken artifact degrades health and gives 503 ----------------------


@pytest.mark.parametrize("name", sorted(ENDPOINT_OF))
def test_missing_required_artifact_degrades_health_and_gives_503(
    examples_copy: Path, name: str
) -> None:
    (examples_copy / f"{name}.json").unlink()
    with start(make_settings(examples_dir=examples_copy)) as test_client:
        health = test_client.get(api("/health"))
        assert health.status_code == 200
        body = c.HealthResponse.model_validate(health.json())
        assert body.status == "degraded"
        assert any(f"{name}.json not found" in p for p in body.problems), body.problems
        error = assert_error(test_client.get(api(ENDPOINT_OF[name])), 503, "artifact_unavailable")
        assert error["details"]["artifacts"] == [name]
        assert f"{name}.json" in error["message"]
        others = {path for path in ENDPOINT_OF.values() if path != ENDPOINT_OF[name]}
        for path in others:
            assert test_client.get(api(path)).status_code == 200, path


def test_invalid_artifact_is_reported_without_its_values(examples_copy: Path) -> None:
    data = load_json(examples_copy / "comparison.json")
    data["records"][0]["model"] = SECRET
    save_json(data, examples_copy / "comparison.json")
    with start(make_settings(examples_dir=examples_copy)) as test_client:
        problems = test_client.get(api("/health")).json()["problems"]
        error = assert_error(test_client.get(api("/results")), 503, "artifact_unavailable")
    assert len(problems) == 1
    assert "ComparisonTable contract" in problems[0] and "records.0.model" in problems[0]
    assert SECRET not in problems[0] and SECRET not in str(error)


def test_non_finite_numbers_are_refused(examples_copy: Path) -> None:
    text = '{"analysis": "temporal_split_sensitivity_check", "pr_auc_ci": [NaN, 1.0]}'
    (examples_copy / "temporal_check.json").write_text(text, encoding="utf-8")
    with start(make_settings(examples_dir=examples_copy)) as test_client:
        problems = test_client.get(api("/health")).json()["problems"]
        assert test_client.get(api("/results")).json()["temporal_check"] is None
    assert len(problems) == 1
    assert "temporal_check.json contains NaN (not finite" in problems[0]


def test_invalid_json_is_reported(examples_copy: Path) -> None:
    (examples_copy / "eda.json").write_text("{not json", encoding="utf-8")
    with start(make_settings(examples_dir=examples_copy)) as test_client:
        problems = test_client.get(api("/health")).json()["problems"]
        assert_error(test_client.get(api("/eda")), 503, "artifact_unavailable")
    assert any("eda.json is not valid JSON" in p for p in problems)


def test_optional_artifacts(examples_copy: Path) -> None:
    (examples_copy / "resampling_audit.json").unlink()
    temporal = {"analysis": art.TEMPORAL_ANALYSIS, "is_sensitivity_analysis": True, "cells": []}
    save_json(temporal, examples_copy / "temporal_check.json")
    with start(make_settings(examples_dir=examples_copy)) as test_client:
        assert test_client.get(api("/health")).json()["status"] == "ok"
        assert test_client.get(api("/eda")).json()["resampling_audit"] is None
        assert test_client.get(api("/results")).json()["temporal_check"] == temporal
    save_json({"analysis": "something else"}, examples_copy / "temporal_check.json")
    with start(make_settings(examples_dir=examples_copy)) as test_client:
        health = test_client.get(api("/health")).json()
        assert health["status"] == "degraded"
        assert "temporal_check.json is not a temporal-check result" in health["problems"][0]
        assert test_client.get(api("/results")).json()["temporal_check"] is None


# --- acceptance: error shape ---------------------------------------------------------------------


def test_unknown_route_returns_404_in_the_error_shape(client: TestClient) -> None:
    response = client.get(api("/does-not-exist"))
    error = assert_error(response, 404, "not_found")
    assert error["details"] == {"path": api("/does-not-exist")}
    assert response.headers[REQUEST_ID_HEADER]


def test_wrong_method_returns_405_in_the_error_shape(client: TestClient) -> None:
    error = assert_error(client.post(api("/health")), 405, "method_not_allowed")
    assert error["details"] is None


# Test-only endpoints, defined once: slowapi adds a limit each time a function is decorated.


def _raise_api_error(status: int) -> None:
    raise ApiError(status, error_code_for_status(status), f"Status {status}.", {"n": status})


def _raise_http_error() -> None:
    raise HTTPException(400, "Upload a .csv file.")


def _crash() -> None:
    raise RuntimeError(SECRET)


def _predict(body: c.PredictRequest) -> dict[str, bool]:
    return {"ok": True}


@limiter.limit("1/minute")
def _limited(request: Request) -> dict[str, bool]:
    return {"ok": True}


def _scored(request: Request) -> dict[str, str]:
    request.state.model_id = "lightgbm__smote"  # what a prediction endpoint will do
    return {"request_id": request.state.request_id}


def with_test_routes(app: FastAPI) -> FastAPI:
    """Add the test-only endpoints above to ``app``."""
    app.add_api_route("/test/api-error/{status}", _raise_api_error, methods=["GET"])
    app.add_api_route("/test/http-error", _raise_http_error, methods=["GET"])
    app.add_api_route("/test/crash", _crash, methods=["GET"])
    app.add_api_route("/test/predict", _predict, methods=["POST"])
    app.add_api_route("/test/limited", _limited, methods=["GET"])
    app.add_api_route("/test/scored", _scored, methods=["GET"])
    return app


@pytest.fixture
def error_client() -> Iterator[TestClient]:
    """The app plus test-only routes that raise each kind of error."""
    with TestClient(with_test_routes(create_app(make_settings()))) as test_client:
        yield test_client


@pytest.mark.parametrize("status", [400, 404, 413, 422, 429, 500, 503])
def test_api_errors_keep_their_status_and_code(error_client: TestClient, status: int) -> None:
    error = assert_error(
        error_client.get(f"/test/api-error/{status}"), status, STATUS_CODES[status]
    )
    assert error["details"] == {"n": status}


def test_http_exception_keeps_its_message(error_client: TestClient) -> None:
    error = assert_error(error_client.get("/test/http-error"), 400, "bad_request")
    assert error["message"] == "Upload a .csv file."


def test_validation_error_is_422_in_the_error_shape_without_input_values(
    error_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    features = {name: 0.0 for name in FEATURES} | {"V1": SECRET}
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    response = error_client.post("/test/predict", json={"features": features})
    error = assert_error(response, 422, "validation_error")
    assert ["body", "features", "V1"] in [d["loc"] for d in error["details"]]
    assert all(set(d) == {"loc", "msg", "type"} for d in error["details"])
    assert SECRET not in response.text
    assert SECRET not in caplog.text


def test_unhandled_exception_is_500_with_request_id_and_no_message_logged(
    error_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    response = error_client.get("/test/crash")
    error = assert_error(response, 500, "internal_error")
    request_id = response.headers[REQUEST_ID_HEADER]
    assert error["details"] == {"request_id": request_id}
    assert request_id in error["message"]
    assert SECRET not in response.text and SECRET not in caplog.text
    assert "unhandled RuntimeError" in caplog.text


def test_rate_limit_gives_429_in_the_error_shape(error_client: TestClient) -> None:
    limiter.reset()
    assert error_client.get("/test/limited").status_code == 200
    error = assert_error(error_client.get("/test/limited"), 429, "rate_limited")
    assert "Too many requests" in error["message"]


def test_rate_limit_can_be_switched_off() -> None:
    limiter.reset()
    app = with_test_routes(create_app(make_settings(rate_limit_enabled=False)))
    try:
        with TestClient(app) as test_client:
            statuses = {test_client.get("/test/limited").status_code for _ in range(3)}
        assert statuses == {200}
    finally:
        limiter.enabled = True


@pytest.mark.parametrize(("status", "code"), [*STATUS_CODES.items(), (418, "http_418")])
def test_error_code_for_status(status: int, code: str) -> None:
    assert error_code_for_status(status) == code


def test_error_body_always_has_details() -> None:
    assert error_body("x", "y") == {"error": {"code": "x", "message": "y", "details": None}}
    body = error_body("x", "y", {"ids": ["a"]})
    assert c.ErrorResponse.model_validate(body).model_dump(mode="json") == body


def test_validation_details_drop_inputs_and_context() -> None:
    raw = [
        {"loc": ("body", "x"), "msg": "bad", "type": "float_parsing", "input": SECRET, "ctx": {}}
    ]
    assert validation_details(raw) == [
        {"loc": ["body", "x"], "msg": "bad", "type": "float_parsing"}
    ]


# --- middleware: request id, timing, access log --------------------------------------------------


def test_request_id_is_generated_or_propagated(client: TestClient) -> None:
    generated = client.get(api("/health")).headers[REQUEST_ID_HEADER]
    assert len(generated) == 32 and int(generated, 16) >= 0
    echoed = client.get(api("/health"), headers={REQUEST_ID_HEADER: "client-id.42"})
    assert echoed.headers[REQUEST_ID_HEADER] == "client-id.42"
    unsafe = client.get(api("/health"), headers={REQUEST_ID_HEADER: "bad id;" + "x" * 80})
    assert unsafe.headers[REQUEST_ID_HEADER] != "bad id;" + "x" * 80


@pytest.mark.parametrize(
    ("incoming", "kept"),
    [("abc-123_X.y", True), ("has space", False), ("x" * 65, False), ("", False)],
)
def test_request_id_from(incoming: str, kept: bool) -> None:
    result = request_id_from(Headers({REQUEST_ID_HEADER: incoming}))
    assert (result == incoming) is kept
    assert 1 <= len(result) <= 64


def test_server_timing_header(client: TestClient) -> None:
    timing = client.get(api("/health")).headers[TIMING_HEADER]
    assert timing.startswith("app;dur=") and float(timing.split("=")[1]) >= 0


def test_access_log_has_one_record_per_request_without_query_or_body(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    response = client.get(api("/health"), params={"card": SECRET})
    records = [r.getMessage() for r in caplog.records if r.name == f"{LOGGER_NAME}.access"]
    assert len(records) == 1
    record = records[0]
    assert f"request_id={response.headers[REQUEST_ID_HEADER]}" in record
    assert f"method=GET path={api('/health')} status=200 latency_ms=" in record
    assert record.endswith("model_id=-")
    assert SECRET not in caplog.text


def test_access_log_names_the_model_a_route_used(
    error_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    response = error_client.get("/test/scored")
    request_id = response.headers[REQUEST_ID_HEADER]
    assert response.json() == {"request_id": request_id}  # routes can read the request id
    records = [r.getMessage() for r in caplog.records if r.name == f"{LOGGER_NAME}.access"]
    assert len(records) == 1
    assert records[0].startswith(f"request_id={request_id} method=GET path=/test/scored status=200")
    assert records[0].endswith(" model_id=lightgbm__smote")


def test_safe_path_and_access_message() -> None:
    assert safe_path("/a\nb c") == "/a?b?c"
    assert len(safe_path("/" + "x" * 500)) == MAX_LOGGED_PATH
    message = access_message("rid", "GET", "/p\n", 200, 1.234, "lightgbm__smote")
    assert (
        message
        == "request_id=rid method=GET path=/p? status=200 latency_ms=1.2 model_id=lightgbm__smote"
    )


def test_configure_logging_is_idempotent() -> None:
    logger = configure_logging("INFO")
    configure_logging("WARNING")
    marked = [h for h in logger.handlers if getattr(h, "_fraudstudy_backend_handler", False)]
    assert len(marked) == 1 and logger.level == logging.WARNING
    configure_logging("INFO")


# --- CORS and settings ---------------------------------------------------------------------------


def _preflight(test_client: TestClient, origin: str) -> Any:
    headers = {"Origin": origin, "Access-Control-Request-Method": "GET"}
    return test_client.options(api("/results"), headers=headers)


def test_cors_allows_the_dev_server_in_development(client: TestClient) -> None:
    allowed = _preflight(client, DEV_SERVER)
    assert allowed.headers.get("access-control-allow-origin") == DEV_SERVER
    assert "access-control-allow-origin" not in _preflight(client, "https://evil.example").headers
    simple = client.get(api("/health"), headers={"Origin": DEV_SERVER})
    assert REQUEST_ID_HEADER in simple.headers.get("access-control-expose-headers", "")


def test_cors_in_production_allows_only_listed_origins() -> None:
    settings = make_settings(
        environment="production", allowed_origins=f"{DASHBOARD},*", use_example_artifacts=True
    )
    with start(settings) as test_client:
        assert (
            _preflight(test_client, DASHBOARD).headers["access-control-allow-origin"] == DASHBOARD
        )
        assert "access-control-allow-origin" not in _preflight(test_client, DEV_SERVER).headers
        health = test_client.get(api("/health")).json()
    assert health["status"] == "degraded"
    assert any("'*'" in p for p in health["problems"])
    assert any("USE_EXAMPLE_ARTIFACTS" in p for p in health["problems"])


def test_settings_origins() -> None:
    dev = make_settings(allowed_origins=f" {DASHBOARD}/ , {DASHBOARD},, ")
    assert dev.listed_origins() == [DASHBOARD]
    assert dev.cors_origins() == [DASHBOARD, *DEV_ORIGINS]
    assert dev.config_problems() == []
    prod = make_settings(environment="production", allowed_origins="*", use_example_artifacts=False)
    assert prod.cors_origins() == []
    assert len(prod.config_problems()) == 2  # wildcard ignored, and nothing left
    ok = make_settings(
        environment="production", allowed_origins=DASHBOARD, use_example_artifacts=False
    )
    assert ok.cors_origins() == [DASHBOARD] and ok.config_problems() == []


def test_settings_resolve_paths_and_normalise_values() -> None:
    settings = make_settings(
        artifacts_dir="artifacts", log_level="debug", environment="Production", default_model_id=" "
    )
    assert settings.artifacts_dir == PROJECT_ROOT / "artifacts"
    assert settings.log_level == "DEBUG" and settings.environment == "production"
    assert settings.default_model_id is None


def test_settings_read_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USE_EXAMPLE_ARTIFACTS", "true")
    monkeypatch.setenv("ALLOWED_ORIGINS", f"{DASHBOARD}/")
    monkeypatch.setenv("MAX_UPLOAD_MB", "2")
    monkeypatch.setenv("MAX_BATCH_ROWS", "500")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("DEFAULT_MODEL_ID", REAL_ID)
    settings = Settings(_env_file=None)
    assert settings.use_example_artifacts is True and settings.listed_origins() == [DASHBOARD]
    assert settings.max_upload_mb == 2 and settings.max_batch_rows == 500
    assert settings.rate_limit_enabled is False and settings.default_model_id == REAL_ID
    assert get_settings() is get_settings()


# --- default model -------------------------------------------------------------------------------


def test_default_model_override() -> None:
    with start(make_settings(default_model_id="lightgbm__rus")) as test_client:
        assert test_client.get(api("/health")).json()["default_model"] == "lightgbm__rus"
        assert test_client.get(api("/models")).json()["default_model_id"] == "lightgbm__rus"
    manifest_default = load_json(EXAMPLES / "manifest.json")["default_model_id"]
    with start(make_settings(default_model_id="svm__magic")) as test_client:
        health = test_client.get(api("/health")).json()
    assert health["status"] == "degraded" and health["default_model"] == manifest_default
    assert "DEFAULT_MODEL_ID='svm__magic' is not in manifest.json" in health["problems"][0]


def test_resolve_default_model() -> None:
    manifest = c.Manifest.model_validate(load_json(EXAMPLES / "manifest.json"))
    default = manifest.default_model_id
    assert art.resolve_default_model(manifest, None, None) == (default, [])
    chosen, problems = art.resolve_default_model(manifest, None, [])
    assert chosen == default and "is not loaded" in problems[0]
    no_default = manifest.model_copy(update={"default_model_id": None})
    chosen, problems = art.resolve_default_model(no_default, None, None)
    assert chosen is None and "No default model" in problems[0]


# --- real artifacts: model loading ---------------------------------------------------------------


@pytest.fixture
def real_tree(tmp_path: Path) -> dict[str, Path]:
    """A real-mode tree: the example JSON files in their real folders, plus a manifest with one
    small, genuinely loadable model and calibrator."""
    artifacts, models = tmp_path / "artifacts", tmp_path / "models"
    (artifacts / "results").mkdir(parents=True)
    (artifacts / "demo").mkdir()
    for name in ("comparison", "curves", "stats", "eda", "resampling_audit"):
        shutil.copy(EXAMPLES / f"{name}.json", artifacts / "results" / f"{name}.json")
    shutil.copy(EXAMPLES / "presets.json", artifacts / "demo" / "presets.json")
    deploy = models / "deploy"
    deploy.mkdir(parents=True)
    rng = np.random.default_rng(0)
    pipeline = LogisticRegression(max_iter=200).fit(rng.normal(size=(40, 29)), [0, 1] * 20)
    joblib.dump(pipeline, deploy / f"{REAL_ID}.joblib")
    joblib.dump(IsotonicRegression().fit([0.0, 1.0], [0.0, 1.0]), deploy / f"{REAL_ID}.cal.joblib")
    tree = {
        "root": tmp_path,
        "artifacts": artifacts,
        "models": models,
        "model": deploy / f"{REAL_ID}.joblib",
        "calibrator": deploy / f"{REAL_ID}.cal.joblib",
    }
    write_manifest(tree)
    return tree


def write_manifest(tree: dict[str, Path], **entry_overrides: Any) -> None:
    """Write ``manifest.json`` with one entry for the tree's model; ``entry_overrides`` win."""
    entry = {
        "id": REAL_ID,
        "model": "logreg",
        "technique": "smote",
        "file": f"deploy/{REAL_ID}.joblib",
        "sha256": sha256_file(tree["model"]),
        "threshold_f1": 0.6,
        "threshold_f2": 0.3,
        "has_calibrator": True,
        "calibrator_file": f"deploy/{REAL_ID}.cal.joblib",
        "calibrator_sha256": sha256_file(tree["calibrator"]),
        "features": FEATURES,
        "data_sha256": DATA_SHA,
        "library_versions": library_versions(),
    } | entry_overrides
    manifest = c.Manifest(
        created_at="2026-10-24T00:00:00+00:00",
        git_commit=None,
        seed=42,
        quick_mode=True,
        dataset=c.DatasetInfo(
            name="ulb_creditcard", source="synthetic", sha256=DATA_SHA, n_rows=60
        ),
        default_model_id=REAL_ID,
        models=[c.ManifestEntry.model_validate(entry)],
    )
    save_json(manifest.model_dump(mode="json"), tree["models"] / "manifest.json")


def real_settings(tree: dict[str, Path], **overrides: Any) -> Settings:
    """Settings that read the real-mode ``tree``."""
    paths = {"artifacts_dir": tree["artifacts"], "models_dir": tree["models"]}
    return make_settings(use_example_artifacts=False, **(paths | overrides))


def test_real_artifacts_load_a_model_with_matching_checksum_and_versions(
    real_tree: dict[str, Path],
) -> None:
    with start(real_settings(real_tree)) as test_client:
        health = c.HealthResponse.model_validate(test_client.get(api("/health")).json())
        models = c.ModelsResponse.model_validate(test_client.get(api("/models")).json())
        store = test_client.app.state.store  # the FastAPI app's state
        for path in ("/results", "/eda", "/presets"):
            assert test_client.get(api(path)).status_code == 200
    assert health.status == "ok", health.problems
    assert health.is_example is False and health.models_loaded == 1
    assert health.default_model == REAL_ID and health.data_sha256 == DATA_SHA
    assert [(m.id, m.loaded) for m in models.models] == [(REAL_ID, True)]
    loaded = store.models[REAL_ID]
    assert callable(loaded.pipeline.predict_proba) and loaded.calibrator is not None


def _break(tree: dict[str, Path], case: str) -> dict[str, Any]:
    """Manifest overrides that break the model in one way."""
    if case == "checksum":
        return {"sha256": "0" * 64}
    if case == "versions":
        return {"library_versions": library_versions() | {"scikit-learn": "0.0.0"}}
    if case == "missing":
        return {"file": "deploy/absent.joblib"}
    if case == "outside":
        outside = tree["root"] / "outside.joblib"
        shutil.copy(tree["model"], outside)
        return {"file": "../outside.joblib", "sha256": sha256_file(outside)}
    joblib.dump({"not": "a model"}, tree["model"])  # case == "not_a_model"
    return {"sha256": sha256_file(tree["model"])}


@pytest.mark.parametrize(
    ("case", "words"),
    [
        ("checksum", "does not match its manifest SHA-256"),
        ("versions", "library versions differ from the export (scikit-learn 0.0.0 at export"),
        ("missing", "not found"),
        ("outside", "points outside MODELS_DIR"),
        ("not_a_model", "has no predict_proba"),
    ],
)
def test_real_artifacts_refuse_a_bad_model(
    real_tree: dict[str, Path], case: str, words: str
) -> None:
    write_manifest(real_tree, **_break(real_tree, case))
    with start(real_settings(real_tree)) as test_client:
        health = test_client.get(api("/health")).json()
        models = test_client.get(api("/models")).json()
    assert health["status"] == "degraded" and health["models_loaded"] == 0
    assert words in health["problems"][0], health["problems"]
    assert "is not loaded" in health["problems"][-1]  # the default model
    assert models["models"][0]["loaded"] is False


def test_real_artifacts_keep_a_model_whose_calibrator_fails(real_tree: dict[str, Path]) -> None:
    write_manifest(real_tree, calibrator_sha256="1" * 64)
    with start(real_settings(real_tree)) as test_client:
        health = test_client.get(api("/health")).json()
        store = test_client.app.state.store  # the FastAPI app's state
    assert health["models_loaded"] == 1 and health["status"] == "degraded"
    assert "served without calibrated probabilities" in health["problems"][0]
    assert store.models[REAL_ID].calibrator is None


def test_real_artifacts_without_manifest(real_tree: dict[str, Path]) -> None:
    (real_tree["models"] / "manifest.json").unlink()
    with start(real_settings(real_tree)) as test_client:
        health = test_client.get(api("/health")).json()
        assert_error(test_client.get(api("/models")), 503, "artifact_unavailable")
        assert test_client.get(api("/results")).status_code == 200
    assert health["models_loaded"] == 0 and health["default_model"] is None
    assert "manifest.json not found" in health["problems"][0]
    assert "scripts/export_artifacts.py" in health["problems"][0]


# --- service helpers -----------------------------------------------------------------------------


def test_artifact_paths(tmp_path: Path) -> None:
    examples = art.artifact_paths(make_settings(examples_dir=tmp_path))
    assert examples == {name: tmp_path / f"{name}.json" for name in art.ARTIFACTS}
    real = art.artifact_paths(
        make_settings(
            use_example_artifacts=False, artifacts_dir=tmp_path, models_dir=tmp_path / "m"
        )
    )
    assert real["comparison"] == tmp_path / "results" / "comparison.json"
    assert real["temporal_check"] == tmp_path / "results" / "temporal_check.json"
    assert real["presets"] == tmp_path / "demo" / "presets.json"
    assert real["manifest"] == tmp_path / "m" / "manifest.json"
    assert set(real) == set(art.ARTIFACTS)


def test_read_artifact_raises_for_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        art.read_artifact("comparison", tmp_path / "comparison.json")


def test_version_mismatches() -> None:
    current = {"numpy": "2.0", "scikit-learn": "1.9", "imbalanced-learn": "0.1", "lightgbm": "4"}
    assert art.version_mismatches(dict(current), current) == []
    recorded = current | {"lightgbm": "3"}
    del recorded["numpy"]
    assert art.version_mismatches(recorded, current) == [
        "numpy not recorded at export, 2.0 here",
        "lightgbm 3 at export, 4 here",
    ]


def test_display_path_hides_the_project_location(tmp_path: Path) -> None:
    assert art.display_path(PROJECT_ROOT / "artifacts" / "x.json") == "artifacts/x.json"
    assert art.display_path(tmp_path) == tmp_path.as_posix()


def test_load_artifacts_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_: Any) -> None:
        raise RuntimeError(SECRET)

    monkeypatch.setattr(art, "load_json_artifacts", boom)
    store = art.load_artifacts(make_settings())
    assert store.status == "degraded"
    assert store.problems == [
        "Unexpected RuntimeError while loading artifacts; see the server log."
    ]


def test_requests_before_start_up_get_503() -> None:
    test_client = TestClient(create_app(make_settings()))  # no `with`: start-up does not run
    assert_error(test_client.get(api("/health")), 503, "service_unavailable")


def test_build_health_and_model_info() -> None:
    store = art.load_artifacts(make_settings())
    assert build_health(store).status == "ok"
    entry = store.manifest.models[0] if store.manifest else None
    assert entry is not None
    unlabelled = entry.model_copy(update={"model_label": None, "technique_label": None})
    info = model_info(unlabelled, loaded=True)
    assert info.model_label == c.MODEL_LABELS[entry.model] and info.loaded is True
    assert info.technique_label == c.TECHNIQUE_LABELS[entry.technique]


def test_module_app_and_openapi() -> None:
    assert isinstance(module_app, FastAPI)
    with start(make_settings()) as test_client:
        schema = test_client.get("/openapi.json").json()
    assert {api(path) for path, _ in GET_ENDPOINTS} <= set(schema["paths"])
    schemas = schema["components"]["schemas"]
    detail = schemas["ErrorDetail"]
    if "$ref" in detail:  # the serialised shape is documented by ErrorDetailBody
        detail = schemas[detail["$ref"].rsplit("/", 1)[-1]]
    assert set(detail["properties"]) == {"code", "message", "details"}
    assert set(detail["required"]) == {"code", "message"}
