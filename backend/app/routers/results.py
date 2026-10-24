"""``GET /api/v1/results``: comparison table, curves, statistics and the temporal check."""

from fastapi import APIRouter

from backend.app.errors import ARTIFACT_UNAVAILABLE
from backend.app.services.artifacts import StoreDep
from fraudstudy.contracts import ErrorResponse, ResultsResponse

router = APIRouter(tags=["results"])


@router.get(
    "/results",
    response_model=ResultsResponse,
    summary="Comparison, curves, statistics (and the temporal check, if run)",
    responses={503: {"model": ErrorResponse, "description": ARTIFACT_UNAVAILABLE}},
)
def get_results(store: StoreDep) -> ResultsResponse:
    """``comparison.json``, ``curves.json`` and ``stats.json`` as written, plus
    ``temporal_check.json`` (a sensitivity analysis) or ``null``."""
    store.require("comparison", "curves", "stats")
    comparison, curves, stats = store.comparison, store.curves, store.stats
    assert comparison is not None and curves is not None and stats is not None  # require()
    return ResultsResponse(
        is_example=any(part.is_example for part in (comparison, curves, stats)),
        comparison=comparison,
        curves=curves,
        stats=stats,
        temporal_check=store.temporal_check,
    )
