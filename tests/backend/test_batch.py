"""Tests for ``POST /api/v1/predict/batch`` (Prompt 13), on small real models fitted from the
synthetic sample (``real_artifacts`` in ``tests/backend/conftest.py``)."""

import asyncio
import logging
import time
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from real_models import DEFAULT_ID, LOGREG_ID, RealArtifacts

from backend.app.errors import ApiError
from backend.app.logging import LOGGER_NAME
from backend.app.main import API_PREFIX, create_app
from backend.app.ratelimit import BATCH_LIMIT, limiter
from backend.app.services import batch as bt
from backend.app.settings import Settings
from fraudstudy.contracts import BatchResponse, ErrorResponse
from fraudstudy.evaluation import apply_calibrator, compute_metrics

pytestmark = pytest.mark.usefixtures("rate_limits_off")

BATCH = f"{API_PREFIX}/predict/batch"
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
MAX_ROWS = 10_000
MB = 1024 * 1024
SECRET_VALUE = 9876.54321  # a distinctive cell that must never reach a log
SECRET_NAME = "secret-merchant-file.csv"


def csv_bytes(frame: pd.DataFrame) -> bytes:
    """``frame`` as CSV bytes, without the index."""
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def upload(
    client: TestClient, data: bytes, filename: str = "batch.csv", **params: Any
) -> Any:
    """POST ``data`` as the ``file`` field."""
    return client.post(BATCH, params=params, files={"file": (filename, data, "text/csv")})


def ok(response: Any) -> BatchResponse:
    """The validated 200 body."""
    assert response.status_code == 200, response.text
    return BatchResponse.model_validate(response.json())


def assert_error(response: Any, status: int, code: str) -> dict[str, Any]:
    """Check the status and the error shape; return ``error``."""
    assert response.status_code == status, response.text
    error = ErrorResponse.model_validate(response.json()).error
    assert error.code == code and error.message
    return response.json()["error"]


@pytest.fixture(scope="module")
def demo_rows(real_artifacts: RealArtifacts) -> pd.DataFrame:
    """``demo_batch.csv`` as written (``Time``, ``V1..V28``, ``Amount``, ``Class``)."""
    return pd.read_csv(real_artifacts.demo_batch)


def pipeline_scores(real: RealArtifacts, model_id: str, rows: pd.DataFrame) -> np.ndarray:
    """The pipeline's own scores of ``rows``."""
    return real.pipelines[model_id].predict_proba(rows[FEATURES])[:, 1]


def threshold_of(real: RealArtifacts, model_id: str) -> float:
    return next(e.threshold_f1 for e in real.manifest.models if e.id == model_id)


# --- acceptance ----------------------------------------------------------------------------------


def test_demo_batch_returns_200_with_metrics(
    real_client: TestClient, real_artifacts: RealArtifacts, demo_rows: pd.DataFrame
) -> None:
    result = ok(upload(real_client, real_artifacts.demo_batch.read_bytes(), "demo_batch.csv"))
    threshold = threshold_of(real_artifacts, DEFAULT_ID)
    scores = pipeline_scores(real_artifacts, DEFAULT_ID, demo_rows)
    y = demo_rows["Class"].to_numpy()
    assert result.model.id == DEFAULT_ID and result.threshold == threshold
    assert result.threshold_source == "tuned_on_validation_F1"
    assert result.n_rows == result.n_scored == len(demo_rows) and result.errors == []
    assert [r.row for r in result.results] == list(range(len(demo_rows)))  # input order
    np.testing.assert_allclose([r.risk_score for r in result.results], scores)
    calibrated = apply_calibrator(real_artifacts.calibrators[DEFAULT_ID], scores)
    np.testing.assert_allclose([r.calibrated_probability for r in result.results], calibrated)
    assert [r.true_class for r in result.results] == y.tolist()
    assert [r.amount for r in result.results] == demo_rows["Amount"].tolist()
    flagged = scores >= threshold
    assert [r.label == "FRAUD" for r in result.results] == flagged.tolist()
    assert result.n_flagged == flagged.sum()
    assert result.flagged_fraction == pytest.approx(flagged.mean())
    assert result.flagged_amount_total == pytest.approx(demo_rows["Amount"][flagged].sum())
    assert result.metrics == compute_metrics(y, scores, threshold)
    assert result.warnings == [
        "Metrics describe this file only (5.00% of its scored rows are fraud, 1.50% in the full "
        "dataset); precision depends on the fraud share."
    ]


def test_a_6_mb_file_gives_413(real_client: TestClient) -> None:
    error = assert_error(upload(real_client, b"V1\n" + b"1\n" * (3 * MB)), 413, "payload_too_large")
    assert error["details"] == {"max_bytes": 5 * MB}
    assert "5 MB" in error["message"]


def _multipart(chunks: int) -> list[bytes]:
    """A multipart body whose ``file`` part is ``chunks`` MB, as separate chunks."""
    head = b'--B\r\nContent-Disposition: form-data; name="file"; filename="a.csv"\r\n\r\n'
    return [head, *([b"1" * MB] * chunks), b"\r\n--B--\r\n"]


def test_a_chunked_upload_without_content_length_gives_413(real_client: TestClient) -> None:
    def body() -> Iterator[bytes]:
        yield from _multipart(6)

    response = real_client.post(
        BATCH, content=body(), headers={"content-type": "multipart/form-data; boundary=B"}
    )
    assert "content-length" not in response.request.headers
    assert_error(response, 413, "payload_too_large")


def test_read_upload_stops_reading_past_the_limit() -> None:
    chunks, received = _multipart(12), []

    async def receive() -> dict[str, Any]:
        received.append(1)
        return {"type": "http.request", "body": chunks[len(received) - 1], "more_body": True}

    headers = [(b"content-type", b"multipart/form-data; boundary=B")]
    request = Request({"type": "http", "method": "POST", "headers": headers}, receive)
    with pytest.raises(ApiError) as raised:
        asyncio.run(bt.read_upload(request, 5 * MB))
    assert raised.value.status_code == 413
    assert len(received) == 7  # the head and 6 MB: stopped at the first chunk over 5 MB


def test_10001_rows_give_413(real_client: TestClient, real_artifacts: RealArtifacts) -> None:
    rows = real_artifacts.test_rows.sample(MAX_ROWS + 1, replace=True, random_state=0).round(3)
    data = csv_bytes(rows)
    assert len(data) < 5 * MB  # the row limit, not the size limit, must trigger
    error = assert_error(upload(real_client, data), 413, "payload_too_large")
    assert error["details"] == {"max_rows": MAX_ROWS} and "10,000" in error["message"]


def test_10000_rows_are_scored_in_under_2_seconds(
    real_client: TestClient, real_artifacts: RealArtifacts
) -> None:
    rows = real_artifacts.test_rows.sample(MAX_ROWS, replace=True, random_state=1)
    data = csv_bytes(rows)
    start = time.perf_counter()
    response = upload(real_client, data)
    elapsed = time.perf_counter() - start
    result = ok(response)
    assert result.n_scored == MAX_ROWS and result.metrics is not None
    assert elapsed < 2.0, f"10,000 rows took {elapsed:.2f} s"


def test_a_txt_file_gives_400(real_client: TestClient, demo_rows: pd.DataFrame) -> None:
    error = assert_error(upload(real_client, csv_bytes(demo_rows), "batch.txt"), 400, "bad_request")
    assert ".csv" in error["message"]


def test_a_missing_v14_column_gives_400(real_client: TestClient, demo_rows: pd.DataFrame) -> None:
    response = upload(real_client, csv_bytes(demo_rows.drop(columns=["V14"])))
    error = assert_error(response, 400, "bad_request")
    assert error["details"] == {"missing_columns": ["V14"]}
    assert "Missing column(s): V14." in error["message"]


@pytest.mark.parametrize("bad", ["abc", "inf", "-Infinity", "NaN", "1,5"])
def test_a_non_numeric_cell_gives_400_listing_the_row(
    real_client: TestClient, demo_rows: pd.DataFrame, bad: str
) -> None:
    rows = demo_rows.astype({"V2": object})
    rows.loc[3, "V2"] = bad
    error = assert_error(upload(real_client, csv_bytes(rows)), 400, "bad_request")
    assert error["details"] == {"rows": [3], "n_rows": 1, "columns": ["V2"]}
    assert error["message"] == (  # names the row and column, never the cell
        "Non-numeric or infinite values in column(s) V2 at rows 3 (0-based data rows, header not "
        "counted). Use plain numbers, and leave a cell empty if its value is missing."
    )


def test_at_most_five_bad_rows_are_listed(real_client: TestClient, demo_rows: pd.DataFrame) -> None:
    rows = demo_rows.astype({"V2": object, "Amount": object})
    rows.loc[[10, 11, 12], "V2"] = "x"
    rows.loc[[1, 30, 40, 50], "Amount"] = "y"
    error = assert_error(upload(real_client, csv_bytes(rows)), 400, "bad_request")
    expected = {"rows": [1, 10, 11, 12, 30], "n_rows": 7, "columns": ["V2", "Amount"]}
    assert error["details"] == expected
    assert "1, 10, 11, 12, 30 and 2 more" in error["message"]


# --- empty cells, columns and Class --------------------------------------------------------------


def test_empty_cells_are_imputed_and_counted(
    real_client: TestClient, real_artifacts: RealArtifacts, demo_rows: pd.DataFrame
) -> None:
    rows = demo_rows.copy()
    rows.loc[[1, 2], "V3"] = np.nan  # written as empty cells
    rows.loc[5, "V7"] = np.nan
    rows.loc[8, "Amount"] = np.nan
    result = ok(upload(real_client, csv_bytes(rows)))
    assert result.n_rows == len(rows) and result.n_scored == len(rows) - 1
    assert [e.model_dump() for e in result.errors] == [
        {"row": 8, "message": bt.EMPTY_AMOUNT_MESSAGE}
    ]
    assert 8 not in [r.row for r in result.results]
    scored = rows.drop(index=8)
    np.testing.assert_allclose(  # the pipeline's imputer fills the NaN cells
        [r.risk_score for r in result.results], pipeline_scores(real_artifacts, DEFAULT_ID, scored)
    )
    assert result.warnings[:2] == [
        "3 empty cells (in V3, V7) filled with training medians by the model's preprocessing.",
        "Not scored: 1 row with an empty Amount (see errors).",
    ]


def test_time_and_unknown_columns_are_ignored(
    real_client: TestClient, demo_rows: pd.DataFrame
) -> None:
    plain = ok(upload(real_client, csv_bytes(demo_rows.drop(columns=["Time", "Class"]))))
    extra = demo_rows.assign(Time=0.0, note="x", merchant_id=7).drop(columns=["Class"])
    with_extra = ok(upload(real_client, csv_bytes(extra)))
    assert [r.risk_score for r in plain.results] == [r.risk_score for r in with_extra.results]
    assert with_extra.warnings == ["Ignored columns the model does not use: note, merchant_id."]
    assert plain.warnings == [] and plain.metrics is None
    assert all(r.true_class is None for r in plain.results)


def test_one_class_only_gives_no_metrics(real_client: TestClient, demo_rows: pd.DataFrame) -> None:
    legit = demo_rows[demo_rows["Class"] == 0].head(50)
    result = ok(upload(real_client, csv_bytes(legit)))
    assert result.metrics is None
    assert "No metrics: the Class column needs both 0 and 1 among the scored rows." in (
        result.warnings
    )


@pytest.mark.parametrize(("column", "value"), [("Class", 2), ("Class", np.nan), ("Amount", -1.0)])
def test_invalid_class_or_amount_gives_400(
    real_client: TestClient, demo_rows: pd.DataFrame, column: str, value: float
) -> None:
    rows = demo_rows.copy()
    rows.loc[4, column] = value
    error = assert_error(upload(real_client, csv_bytes(rows)), 400, "bad_request")
    assert error["details"] == {"rows": [4], "n_rows": 1, "columns": [column]}


def test_header_spaces_and_a_byte_order_mark_are_accepted(
    real_client: TestClient, demo_rows: pd.DataFrame
) -> None:
    text = csv_bytes(demo_rows.head(20)).decode()
    header, rest = text.split("\n", 1)
    spaced = ", ".join(header.split(",")) + "\n" + rest
    result = ok(upload(real_client, b"\xef\xbb\xbf" + spaced.encode()))
    assert result.n_scored == 20 and result.metrics is not None


# --- query parameters, request shape -------------------------------------------------------------


def test_model_id_and_threshold_query_parameters(
    real_client: TestClient, real_artifacts: RealArtifacts, demo_rows: pd.DataFrame
) -> None:
    data = csv_bytes(demo_rows.head(40))
    everything = ok(upload(real_client, data, threshold=0.0))
    assert everything.n_flagged == 40 and everything.threshold_source == "user_supplied"
    logreg = ok(upload(real_client, data, model_id=LOGREG_ID))
    assert logreg.model.id == LOGREG_ID
    np.testing.assert_allclose(
        [r.risk_score for r in logreg.results],
        pipeline_scores(real_artifacts, LOGREG_ID, demo_rows.head(40)),
    )
    error = assert_error(upload(real_client, data, model_id="nope"), 404, "not_found")
    assert LOGREG_ID in error["details"]["valid_ids"]
    for bad in ("2", "-0.5", "nan"):
        assert_error(upload(real_client, data, threshold=bad), 422, "validation_error")


def test_requests_that_are_not_one_csv_file_give_400(
    real_client: TestClient, demo_rows: pd.DataFrame
) -> None:
    data = csv_bytes(demo_rows.head(5))
    not_multipart = real_client.post(BATCH, json={"file": "x"})
    assert "multipart/form-data" in assert_error(not_multipart, 400, "bad_request")["message"]
    other_field = real_client.post(BATCH, files={"upload": ("a.csv", data, "text/csv")})
    assert "No 'file' field" in assert_error(other_field, 400, "bad_request")["message"]
    two_files = real_client.post(
        BATCH, files=[("file", ("a.csv", data, "text/csv")), ("file", ("b.csv", data, "text/csv"))]
    )
    assert "exactly one" in assert_error(two_files, 400, "bad_request")["message"]
    multipart = {"content-type": "multipart/form-data; boundary=B"}
    broken = real_client.post(BATCH, content=b"--B\r\nnot a part", headers=multipart)
    assert_error(broken, 400, "bad_request")


def test_logs_never_contain_file_contents_or_names(
    real_client: TestClient, demo_rows: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    rows = demo_rows.head(30).copy()
    rows.loc[0, "Amount"] = SECRET_VALUE
    result = ok(upload(real_client, csv_bytes(rows), SECRET_NAME))
    bad = rows.astype({"V1": object})
    bad.loc[2, "V1"] = f"x{SECRET_VALUE}"
    assert upload(real_client, csv_bytes(bad), SECRET_NAME).status_code == 400
    assert str(SECRET_VALUE) not in caplog.text and SECRET_NAME not in caplog.text
    records = [r.getMessage() for r in caplog.records if r.name == f"{LOGGER_NAME}.predict"]
    assert any(f"rows=30 scored=30 flagged={result.n_flagged}" in r for r in records)


def test_batch_is_rate_limited_to_10_per_minute(
    make_real_settings: Callable[..., Settings],
) -> None:
    limiter.reset()
    limit = int(BATCH_LIMIT.split("/")[0])
    with TestClient(create_app(make_real_settings(rate_limit_enabled=True))) as client:
        statuses = {upload(client, b"V1\n1\n", "a.txt").status_code for _ in range(limit)}
        assert statuses == {400}  # cheap rejections still count
        response = upload(client, b"V1\n1\n", "a.txt")
    assert_error(response, 429, "rate_limited")


# --- the CSV parser on its own -------------------------------------------------------------------


def _parse_error(data: bytes, max_rows: int = MAX_ROWS) -> ApiError:
    with pytest.raises(ApiError) as raised:
        bt.parse_batch_csv(data, FEATURES, max_rows)
    return raised.value


@pytest.mark.parametrize(
    ("data", "status", "words"),
    [
        (b"", 400, "The file is empty"),
        (",".join(FEATURES).encode() + b"\n", 400, "no data rows"),
        ("é".encode("latin-1") + b"\n1\n", 400, "not UTF-8"),
        (b"V1,V2\n1,2,3,4\n", 400, "could not be read as CSV"),
        (b"V1,V2\n1,2,\n3,4,\n", 400, "more comma-separated fields than the header"),
        (b"V1,V1 ,V2\n1,2,3\n", 400, "Duplicate column name(s): V1"),
    ],
)
def test_parser_rejects_unreadable_files(data: bytes, status: int, words: str) -> None:
    error = _parse_error(data)
    assert error.status_code == status and words in error.message


def test_parser_reads_short_rows_as_empty_cells_and_limits_rows() -> None:
    header = ",".join(FEATURES)
    full = ",".join(["1.5"] * len(FEATURES))
    short = ",".join(["2"] * 3)  # the remaining cells are missing
    parsed = bt.parse_batch_csv(f"{header}\n{full}\n{short}\n".encode(), FEATURES, MAX_ROWS)
    assert parsed.X.shape == (2, len(FEATURES)) and parsed.y is None
    assert parsed.X.iloc[0].tolist() == [1.5] * len(FEATURES)
    assert parsed.empty_cells == {column: 1 for column in FEATURES[3:]}
    error = _parse_error(f"{header}\n{full}\n{full}\n{full}\n".encode(), max_rows=2)
    assert error.status_code == 413 and error.details == {"max_rows": 2}


def test_to_numbers() -> None:
    values, empty, invalid = bt.to_numbers(pd.Series([" 1.5 ", "", "  ", "x", "inf", "-2e3"]))
    np.testing.assert_array_equal(values, [1.5, np.nan, np.nan, np.nan, np.nan, -2000.0])
    assert empty.tolist() == [False, True, True, False, False, False]
    assert invalid.tolist() == [False, False, False, True, True, False]


def test_upload_collector_checks_name_and_size() -> None:
    collector = bt.UploadCollector(max_bytes=4)
    collector._disposition = b'form-data; name="file"; filename="x.CSV"'
    collector.on_headers_finished()
    collector.on_part_data(b"abcd", 0, 4)
    assert collector.filename == "x.CSV" and bytes(collector.data) == b"abcd"
    with pytest.raises(ApiError) as raised:
        collector.on_part_data(b"e", 0, 1)
    assert raised.value.status_code == 413
    other = bt.UploadCollector(max_bytes=4)
    other._disposition = b'form-data; name="note"'
    other.on_headers_finished()
    other.on_part_data(b"ignored", 0, 7)
    assert other.filename is None and bytes(other.data) == b""


def test_file_too_large_message() -> None:
    error = bt.file_too_large(int(2.5 * MB))
    assert error.status_code == 413 and "2.5 MB" in error.message
