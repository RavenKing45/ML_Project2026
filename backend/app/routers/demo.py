"""``POST /api/v1/smote-demo``: SMOTE step by step on a 2-D toy dataset (the teaching page).

It needs no artifact, so it works even when ``/health`` is degraded. Rate-limited per client
(:data:`backend.app.ratelimit.SMOTE_DEMO_LIMIT`).
"""

from typing import Annotated

from fastapi import APIRouter, Body, Request

from backend.app.ratelimit import SMOTE_DEMO_LIMIT, limiter
from backend.app.services.smote_demo import run_smote_demo
from fraudstudy.contracts import ErrorResponse, SmoteDemoRequest, SmoteDemoResponse

router = APIRouter(tags=["smote-demo"])


@router.post(
    "/smote-demo",
    response_model=SmoteDemoResponse,
    summary="SMOTE step by step on a toy dataset",
    responses={
        422: {"model": ErrorResponse, "description": "invalid request"},
        429: {"model": ErrorResponse, "description": "rate limited"},
    },
)
@limiter.limit(SMOTE_DEMO_LIMIT)
def post_smote_demo(
    request: Request, body: Annotated[SmoteDemoRequest | None, Body()] = None
) -> SmoteDemoResponse:
    """The toy data, every synthetic point with its parent, neighbour and gap (lambda), each
    fraud point's neighbours, and a decision-tree boundary before and after oversampling. An
    empty body uses the defaults."""
    return run_smote_demo(body if body is not None else SmoteDemoRequest())
