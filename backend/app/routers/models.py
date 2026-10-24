"""``GET /api/v1/models``: the deployed models from the manifest and the default one."""

from fastapi import APIRouter

from backend.app.errors import ARTIFACT_UNAVAILABLE
from backend.app.services.artifacts import StoreDep
from fraudstudy.contracts import (
    MODEL_LABELS,
    TECHNIQUE_LABELS,
    ErrorResponse,
    ManifestEntry,
    ModelInfo,
    ModelsResponse,
)

router = APIRouter(tags=["models"])


def model_info(entry: ManifestEntry, loaded: bool) -> ModelInfo:
    """The ``/models`` view of a manifest entry (labels default to the contract's).

    Args:
        entry: The manifest entry.
        loaded: Whether its file was loaded at start-up.

    Returns:
        The ``ModelInfo``.
    """
    return ModelInfo(
        id=entry.id,
        model=entry.model,
        technique=entry.technique,
        model_label=entry.model_label or MODEL_LABELS[entry.model],
        technique_label=entry.technique_label or TECHNIQUE_LABELS[entry.technique],
        threshold_f1=entry.threshold_f1,
        threshold_f2=entry.threshold_f2,
        default_threshold=entry.default_threshold,
        has_calibrator=entry.has_calibrator,
        features=list(entry.features),
        loaded=loaded,
    )


@router.get(
    "/models",
    response_model=ModelsResponse,
    summary="Deployed models and the default",
    responses={503: {"model": ErrorResponse, "description": ARTIFACT_UNAVAILABLE}},
)
def list_models(store: StoreDep) -> ModelsResponse:
    """Every manifest model (``loaded`` says whether it can score) and the default model id."""
    store.require("manifest")
    manifest = store.manifest
    assert manifest is not None  # guaranteed by require()
    return ModelsResponse(
        is_example=manifest.is_example,
        default_model_id=store.default_model_id,
        models=[model_info(entry, entry.id in store.models) for entry in manifest.models],
    )
