"""``POST /api/v1/predict`` (one payment) and ``POST /api/v1/predict/batch`` (a CSV upload).

Both are rate-limited per client (:mod:`backend.app.ratelimit`), set ``request.state.model_id``
for the access log and log one line with the request id, model id and outcome; never feature
values, file contents or filenames.
"""

import logging
import time
from typing import Annotated, Any

from fastapi import APIRouter, Query, Request
from starlette.concurrency import run_in_threadpool

from backend.app.errors import ARTIFACT_UNAVAILABLE
from backend.app.logging import LOGGER_NAME
from backend.app.ratelimit import BATCH_LIMIT, PREDICT_LIMIT, limiter
from backend.app.services.artifacts import StoreDep
from backend.app.services.batch import BYTES_PER_MB, FILE_FIELD, read_upload, score_batch
from backend.app.services.inference import predict, resolve_model
from backend.app.settings import Settings
from fraudstudy.contracts import BatchResponse, ErrorResponse, PredictRequest, PredictResponse

logger = logging.getLogger(f"{LOGGER_NAME}.predict")

router = APIRouter(tags=["predict"])


PREDICT_ERRORS: dict[int, str] = {
    404: "unknown model_id (details.valid_ids lists the ids that can score)",
    422: "invalid request (details lists each field, never the values sent)",
    429: "rate limited",
    503: f"{ARTIFACT_UNAVAILABLE}: no manifest, no default model or the model is not loaded",
}
BATCH_ERRORS: dict[int, str] = {
    400: "not a .csv upload in a 'file' field, missing columns or invalid cells",
    413: "file over MAX_UPLOAD_MB or more than MAX_BATCH_ROWS rows",
    **PREDICT_ERRORS,
}
def _responses(errors: dict[int, str]) -> dict[int | str, dict[str, Any]]:
    """OpenAPI ``responses``: each status documented with the ``ErrorResponse`` body."""
    return {code: {"model": ErrorResponse, "description": text} for code, text in errors.items()}


UPLOAD_BODY: dict[str, Any] = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": [FILE_FIELD],
                    "properties": {
                        FILE_FIELD: {
                            "type": "string",
                            "format": "binary",
                            "description": "CSV with V1..V28 and Amount (Time, Class optional)",
                        }
                    },
                }
            }
        },
    }
}


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Score one payment, with an explanation",
    responses=_responses(PREDICT_ERRORS),
)
@limiter.limit(PREDICT_LIMIT)
def post_predict(request: Request, body: PredictRequest, store: StoreDep) -> PredictResponse:
    """Risk score, label at the tuned (or given) threshold, explanation, warnings and the same
    payment scored by every deployed LightGBM technique."""
    response = predict(store, body, time.perf_counter())
    request.state.model_id = response.model.id
    logger.info(
        "predict request_id=%s model_id=%s label=%s",
        request.state.request_id,
        response.model.id,
        response.label,
    )
    return response


@router.post(
    "/predict/batch",
    response_model=BatchResponse,
    summary="Score an uploaded CSV of payments",
    responses=_responses(BATCH_ERRORS),
    openapi_extra=UPLOAD_BODY,
)
@limiter.limit(BATCH_LIMIT)
async def post_predict_batch(
    request: Request,
    store: StoreDep,
    model_id: Annotated[
        str | None, Query(description="A manifest id; the default model when omitted.")
    ] = None,
    threshold: Annotated[
        float | None,
        Query(ge=0.0, le=1.0, allow_inf_nan=False, description="The model's tuned one if omitted."),
    ] = None,
) -> BatchResponse:
    """Summary (rows, flagged, flagged share and amount, metrics if the CSV has ``Class``) and one
    result per row, in input order. The file is read in memory only and never stored."""
    started = time.perf_counter()
    model = resolve_model(store, model_id)  # before reading the upload: a bad id costs nothing
    request.state.model_id = model.entry.id
    settings: Settings = request.app.state.settings
    upload = await read_upload(request, int(settings.max_upload_mb * BYTES_PER_MB))
    dataset_fraud_rate = store.eda.fraud_rate if store.eda is not None else None
    response = await run_in_threadpool(
        score_batch,
        model,
        upload,
        threshold,
        settings.max_batch_rows,
        started,
        dataset_fraud_rate,
    )
    logger.info(
        "batch request_id=%s model_id=%s rows=%d scored=%d flagged=%d",
        request.state.request_id,
        model.entry.id,
        response.n_rows,
        response.n_scored,
        response.n_flagged,
    )
    return response
