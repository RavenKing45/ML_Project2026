"""Score one payment with a deployed model (``POST /api/v1/predict``; master guide Parts 7 and 9).

Steps: pick the model (the request's ``model_id``, else the default) and the threshold (the
request's, else the model's tuned default threshold) -> build the one-row input table in the
manifest's ``features`` order -> ``risk_score = predict_proba[:, 1]`` and, with a calibrator,
``calibrated_probability`` -> explanation (the top ``DEFAULT_TOP_K`` contributions,
:func:`fraudstudy.explain.explain`) -> warnings -> the same row scored by every deployed LightGBM
technique, each at its own tuned threshold. The label is ``"FRAUD"`` exactly when
``risk_score >= threshold``.

Feature values are never logged and never put into error messages.
"""

import logging
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from backend.app.errors import ARTIFACT_UNAVAILABLE, STATUS_CODES, ApiError
from backend.app.logging import LOGGER_NAME
from backend.app.services.artifacts import ArtifactStore, LoadedModel
from fraudstudy.contracts import (
    Explanation,
    ManifestEntry,
    ModelRef,
    PredictedLabel,
    PredictRequest,
    PredictResponse,
    TechniqueScore,
    ThresholdSource,
    Transaction,
)
from fraudstudy.errors import FraudStudyError
from fraudstudy.evaluation import apply_calibrator
from fraudstudy.explain import DEFAULT_TOP_K, FeatureReference, explain

logger = logging.getLogger(f"{LOGGER_NAME}.predict")

DISCLAIMER = "Research demo on anonymised public data. Not a real fraud decision."
EDITED_SOURCES: tuple[str, ...] = ("edited", "manual")
EDITED_WARNING = "Edited transaction may not resemble real payments; the model is extrapolating."
NO_REFERENCE_WARNING = (
    "No explanation: feature_reference.json was not exported with the models (re-run "
    "scripts/export_artifacts.py)."
)
NO_CALIBRATOR_WARNING = (
    "No calibrated probability: this model's calibrator did not load (see /api/v1/health)."
)
# The model family whose six techniques are scored side by side (deploy.cells in experiment.yaml).
TECHNIQUE_SCORES_MODEL = "lightgbm"
TUNED_SOURCES: dict[str, ThresholdSource] = {
    "f1": "tuned_on_validation_F1",
    "f2": "tuned_on_validation_F2",
}
USER_THRESHOLD: ThresholdSource = "user_supplied"
FRAUD: PredictedLabel = "FRAUD"
LEGITIMATE: PredictedLabel = "LEGITIMATE"
MS_PER_S = 1000.0
LATENCY_DECIMALS = 3
MISSING = "missing"  # pydantic's error type for an absent field
REQUIRED_MSG = "Required by this model"


def elapsed_ms(started: float) -> float:
    """Milliseconds since ``started`` (a ``time.perf_counter()`` value), rounded."""
    return round((time.perf_counter() - started) * MS_PER_S, LATENCY_DECIMALS)


def label_for(risk_score: float, threshold: float) -> PredictedLabel:
    """``"FRAUD"`` when ``risk_score >= threshold``, else ``"LEGITIMATE"`` (the contract's rule)."""
    return FRAUD if risk_score >= threshold else LEGITIMATE


def resolve_model(store: ArtifactStore, model_id: str | None) -> LoadedModel:
    """The loaded model a request asks for.

    Args:
        store: The artifact store.
        model_id: A manifest id, or ``None`` for the default model.

    Returns:
        The :class:`LoadedModel`.

    Raises:
        ApiError: 404 ``not_found`` for an id that is not in the manifest, with ``details =
            {"valid_ids": [...]}`` (the ids that can score); 503 ``artifact_unavailable`` when
            the manifest is unavailable, no default model is set, or the model is listed but not
            loaded (always the case with the example artifacts, which have no model files).
    """
    store.require("manifest")
    manifest = store.manifest
    assert manifest is not None  # guaranteed by require()
    valid = [entry.id for entry in manifest.models if entry.id in store.models]
    listed = ", ".join(valid) or "none (no model is loaded; see /api/v1/health)"
    chosen = model_id if model_id is not None else store.default_model_id
    if chosen is None:
        raise ApiError(
            503,
            ARTIFACT_UNAVAILABLE,
            f"No default model is set; pass model_id (valid ids: {listed}) or set "
            "DEFAULT_MODEL_ID.",
            {"valid_ids": valid},
        )
    if chosen not in {entry.id for entry in manifest.models}:
        raise ApiError(
            404, STATUS_CODES[404], f"Unknown model_id. Valid ids: {listed}.", {"valid_ids": valid}
        )
    loaded = store.models.get(chosen)
    if loaded is None:
        why = (
            "the example artifacts have no model files; export real ones with "
            "scripts/export_artifacts.py and set USE_EXAMPLE_ARTIFACTS=false"
            if store.is_example
            else "see /api/v1/health for why"
        )
        raise ApiError(
            503,
            ARTIFACT_UNAVAILABLE,
            f"Model {chosen} is listed in manifest.json but not loaded ({why}).",
            {"artifacts": [chosen], "problems": list(store.problems)},
        )
    return loaded


def resolve_threshold(
    entry: ManifestEntry, requested: float | None
) -> tuple[float, ThresholdSource]:
    """The threshold to apply and where it came from.

    Args:
        entry: The model's manifest entry.
        requested: The request's threshold, or ``None``.

    Returns:
        ``(requested, "user_supplied")``, else the entry's ``threshold_f1`` / ``threshold_f2``
        (its ``default_threshold``) with ``"tuned_on_validation_F1"`` / ``"_F2"``.
    """
    if requested is not None:
        return float(requested), USER_THRESHOLD
    tuned = entry.threshold_f1 if entry.default_threshold == "f1" else entry.threshold_f2
    return float(tuned), TUNED_SOURCES[entry.default_threshold]


def input_frame(entry: ManifestEntry, features: Transaction) -> pd.DataFrame:
    """The one-row input table, columns in the manifest's ``features`` order.

    Inputs the model does not use (``Time`` in the main experiment) are left out.

    Args:
        entry: The model's manifest entry.
        features: The validated payment.

    Returns:
        A one-row ``DataFrame`` of floats.

    Raises:
        ApiError: 422 ``validation_error`` if the model needs an input the request left out (only
            ``Time`` is optional in ``Transaction``, so only a model using the hour feature can
            need one).
    """
    values = features.model_dump()
    missing = [name for name in entry.features if values.get(name) is None]
    if missing:
        raise ApiError(
            422,
            STATUS_CODES[422],
            f"Model {entry.id} needs {', '.join(missing)}; add it to features.",
            [
                {"loc": ["body", "features", name], "msg": REQUIRED_MSG, "type": MISSING}
                for name in missing
            ],
        )
    return pd.DataFrame([[float(values[name]) for name in entry.features]], columns=entry.features)


def score_rows(model: LoadedModel, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
    """Score every row of ``X`` in one call.

    Args:
        model: A loaded model.
        X: Rows with the model's input columns.

    Returns:
        ``(risk scores, calibrated probabilities or None)``: ``predict_proba[:, 1]`` clipped to
        [0, 1], and :func:`fraudstudy.evaluation.apply_calibrator` of them when the model has a
        calibrator.
    """
    scores = np.clip(np.asarray(model.pipeline.predict_proba(X), dtype=float)[:, 1], 0.0, 1.0)
    calibrated = None if model.calibrator is None else apply_calibrator(model.calibrator, scores)
    return scores, calibrated


def range_warnings(reference: FeatureReference, values: Mapping[str, Any]) -> list[str]:
    """One warning per raw input column whose value lies outside its training range.

    The range is ``ReferenceFeature.train_range`` (the reference's ``range_percentiles``, e.g. the
    0.1th and 99.9th percentiles of ALL training rows). Columns follow the reference's order and
    are checked once even when several model features come from one column; columns without a
    range or a value are skipped. The messages name the column, never its value.

    Args:
        reference: The export's feature reference.
        values: Raw input values by column name.

    Returns:
        E.g. ``["V14 is outside the range seen in training (below its 0.1th percentile)."]``.
    """
    if reference.range_percentiles is None:
        return []
    low_pct, high_pct = reference.range_percentiles
    warnings: list[str] = []
    seen: set[str] = set()
    for feature in reference.features:
        column, bounds = feature.raw_column, feature.train_range
        if column in seen or bounds is None or values.get(column) is None:
            continue
        seen.add(column)
        value = float(values[column])
        if value < bounds[0]:
            side = f"below its {low_pct:g}th percentile"
        elif value > bounds[1]:
            side = f"above its {high_pct:g}th percentile"
        else:
            continue
        warnings.append(f"{column} is outside the range seen in training ({side}).")
    return warnings


def explanation_for(
    model: LoadedModel, X: pd.DataFrame, reference: FeatureReference | None
) -> tuple[Explanation | None, list[str]]:
    """The explanation of the one row in ``X``, or ``None`` with a warning saying why.

    Args:
        model: The model that scored the row.
        X: The one-row input table.
        reference: The export's feature reference, or ``None`` if it was not loaded.

    Returns:
        ``(explanation or None, warnings)``.
    """
    if reference is None:
        return None, [NO_REFERENCE_WARNING]
    try:
        return explain(model.pipeline, X, reference, top_k=DEFAULT_TOP_K), []
    except FraudStudyError as exc:  # a model or reference the explainer cannot use
        logger.warning("no explanation for model_id=%s (%s)", model.entry.id, type(exc).__name__)
        return None, [f"No explanation for {model.entry.id}: {exc}"]


def technique_scores(store: ArtifactStore, features: Transaction) -> list[TechniqueScore]:
    """The payment scored by every loaded LightGBM model (manifest order), each at its own
    tuned default threshold, so the dashboard can compare the techniques on one payment.

    Args:
        store: The artifact store.
        features: The validated payment.

    Returns:
        One :class:`TechniqueScore` per loaded ``lightgbm`` model.
    """
    manifest = store.manifest
    scores: list[TechniqueScore] = []
    for entry in manifest.models if manifest is not None else []:
        loaded = store.models.get(entry.id)
        if entry.model != TECHNIQUE_SCORES_MODEL or loaded is None:
            continue
        risk = float(score_rows(loaded, input_frame(entry, features))[0][0])
        threshold, _ = resolve_threshold(entry, None)
        scores.append(
            TechniqueScore(
                id=entry.id,
                model=entry.model,
                technique=entry.technique,
                risk_score=risk,
                threshold=threshold,
                label=label_for(risk, threshold),
            )
        )
    return scores


def predict(store: ArtifactStore, body: PredictRequest, started: float) -> PredictResponse:
    """Score, explain and annotate one payment (see the module docstring).

    Warnings, in order: edited or manual input; each input outside its training range; no
    explanation; no calibrated probability although the model has a calibrator.

    Args:
        store: The artifact store.
        body: The validated request.
        started: ``time.perf_counter()`` when the request reached the endpoint.

    Returns:
        The ``PredictResponse``.

    Raises:
        ApiError: From :func:`resolve_model` (404 / 503) and :func:`input_frame` (422).
    """
    model = resolve_model(store, body.model_id)
    entry = model.entry
    threshold, source = resolve_threshold(entry, body.threshold)
    X = input_frame(entry, body.features)
    scores, calibrated = score_rows(model, X)
    risk = float(scores[0])
    explanation, explanation_warnings = explanation_for(model, X, store.feature_reference)
    warnings = [EDITED_WARNING] if body.input_source in EDITED_SOURCES else []
    if store.feature_reference is not None:
        warnings += range_warnings(store.feature_reference, body.features.model_dump())
    warnings += explanation_warnings
    if entry.has_calibrator and model.calibrator is None:
        warnings.append(NO_CALIBRATOR_WARNING)
    return PredictResponse(
        label=label_for(risk, threshold),
        risk_score=risk,
        calibrated_probability=None if calibrated is None else float(calibrated[0]),
        threshold=threshold,
        threshold_source=source,
        model=ModelRef(id=entry.id, model=entry.model, technique=entry.technique),
        explanation=explanation,
        warnings=warnings,
        disclaimer=DISCLAIMER,
        technique_scores=technique_scores(store, body.features),
        latency_ms=elapsed_ms(started),
    )
