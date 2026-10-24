"""``GET /api/v1/eda``: the dataset summary and the resampling audit."""

from fastapi import APIRouter

from backend.app.errors import ARTIFACT_UNAVAILABLE
from backend.app.services.artifacts import StoreDep
from fraudstudy.contracts import EdaResponse, ErrorResponse

router = APIRouter(tags=["eda"])


@router.get(
    "/eda",
    response_model=EdaResponse,
    summary="Dataset summary and resampling audit",
    responses={503: {"model": ErrorResponse, "description": ARTIFACT_UNAVAILABLE}},
)
def get_eda(store: StoreDep) -> EdaResponse:
    """``eda.json`` plus ``resampling_audit.json`` (``null`` when it was not produced)."""
    store.require("eda")
    eda, audit = store.eda, store.resampling_audit
    assert eda is not None  # guaranteed by require()
    return EdaResponse(
        is_example=eda.is_example or (audit is not None and audit.is_example),
        eda=eda,
        resampling_audit=audit,
    )
