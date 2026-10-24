"""``GET /api/v1/health``: liveness plus what loaded at start-up.

Always 200, also when degraded, so the dashboard can show the problems and a deploy check can
tell "up but degraded" from "down".
"""

from fastapi import APIRouter

from backend.app import __version__
from backend.app.services.artifacts import ArtifactStore, StoreDep
from fraudstudy.contracts import HealthResponse

router = APIRouter(tags=["health"])


def build_health(store: ArtifactStore) -> HealthResponse:
    """The health payload of ``store``.

    Args:
        store: The artifact store.

    Returns:
        The ``HealthResponse``.
    """
    return HealthResponse(
        is_example=store.is_example,
        status=store.status,
        problems=list(store.problems),
        models_loaded=len(store.models),
        default_model=store.default_model_id,
        data_sha256=store.manifest.dataset.sha256 if store.manifest is not None else None,
        git_hash=store.git_hash,
        version=__version__,
        uptime_s=round(store.uptime_s(), 3),
    )


@router.get("/health", response_model=HealthResponse, summary="Is the API up and fully loaded?")
def get_health(store: StoreDep) -> HealthResponse:
    """Status (``ok`` / ``degraded``), problems with fixes, models loaded and provenance."""
    return build_health(store)
