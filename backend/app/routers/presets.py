"""``GET /api/v1/presets``: real test-set payments for the one-click demo."""

from fastapi import APIRouter

from backend.app.errors import ARTIFACT_UNAVAILABLE
from backend.app.services.artifacts import StoreDep
from fraudstudy.contracts import ErrorResponse, PresetsFile

router = APIRouter(tags=["presets"])


@router.get(
    "/presets",
    response_model=PresetsFile,
    summary="Demo presets",
    responses={503: {"model": ErrorResponse, "description": ARTIFACT_UNAVAILABLE}},
)
def get_presets(store: StoreDep) -> PresetsFile:
    """``presets.json`` as written by the exporter."""
    store.require("presets")
    presets = store.presets
    assert presets is not None  # guaranteed by require()
    return presets
