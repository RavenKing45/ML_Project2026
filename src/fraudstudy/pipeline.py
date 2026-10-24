"""Builds the full model pipeline: preprocess -> (sampler steps) -> model.

The result is always an ``imblearn.pipeline.Pipeline``, so any sampler steps run only during
``fit`` and are skipped by ``predict`` / ``predict_proba`` (docs/decisions.md). Models are looked
up in the registry below; techniques A-F and their sampler steps live in
:mod:`fraudstudy.sampling`. Later prompts add models without changing :func:`build_pipeline`'s
signature.
"""

from collections.abc import Callable
from typing import Any

from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from fraudstudy.config import get_dataset_config, get_experiment_config
from fraudstudy.errors import ConfigError
from fraudstudy.features import build_preprocessor
from fraudstudy.sampling import (
    BALANCED,
    CLASS_WEIGHT_PARAM,
    FRAUD_WEIGHT_KEY,
    TECHNIQUES,
    build_sampler_steps,
    class_weight_model_params,
)

ModelBuilder = Callable[[dict[str, Any], int], BaseEstimator]


def _build_logreg(params: dict[str, Any], seed: int) -> BaseEstimator:
    """Logistic regression with ``random_state=seed`` and the given hyper-parameters."""
    return LogisticRegression(random_state=seed, **params)


MODEL_BUILDERS: dict[str, ModelBuilder] = {"logreg": _build_logreg}


def _model_params_for(
    model_name: str, technique: str, model_params: dict[str, Any], technique_params: dict[str, Any]
) -> dict[str, Any]:
    """Model keyword arguments, plus the fraud weight for technique F.

    Raises:
        ConfigError: If ``model_params`` already sets the class-weight parameter (the weight is
            technique F's job, set via ``technique_params["fraud_weight"]``).
    """
    weight_param = CLASS_WEIGHT_PARAM.get(model_name)
    if weight_param is not None and weight_param in model_params:
        raise ConfigError(
            f"Do not pass {weight_param!r} in model_params; class weighting is technique "
            f"'class_weight' (use technique_params={{'{FRAUD_WEIGHT_KEY}': ...}})."
        )
    if technique != "class_weight":
        return model_params
    weight = technique_params.get(FRAUD_WEIGHT_KEY, BALANCED)
    return {**model_params, **class_weight_model_params(model_name, weight)}


def build_pipeline(
    model_name: str,
    technique: str,
    model_params: dict[str, Any] | None = None,
    technique_params: dict[str, Any] | None = None,
    cfg: dict[str, Any] | None = None,
    exp_cfg: dict[str, Any] | None = None,
) -> Pipeline:
    """Build an unfitted ``imblearn`` pipeline: preprocess, sampler steps, model.

    Step names: ``"preprocess"``, then the technique's sampler steps (``"ros"``, ``"rus"``,
    ``"smote"``, or ``"smote"`` + ``"rus"``; none for ``baseline`` and ``class_weight``), then
    ``"model"``. For ``class_weight`` the model gets the fraud weight
    ``technique_params["fraud_weight"]`` (default ``"balanced"``).

    Every random component receives ``random_state = exp_cfg["seed"]``; SMOTE gets
    ``k_neighbors = exp_cfg["smote_k_neighbors"]``.

    Args:
        model_name: Key of :data:`MODEL_BUILDERS` (currently ``"logreg"``).
        technique: One of :data:`fraudstudy.sampling.TECHNIQUES`.
        model_params: Keyword arguments for the model (e.g. ``{"C": 0.1, "max_iter": 2000}``).
        technique_params: Technique settings, see :func:`fraudstudy.sampling.build_sampler_steps`
            (e.g. ``{"sampling_strategy": 0.1}``); empty for the baseline.
        cfg: Contents of ``configs/dataset.yaml``; loaded if omitted.
        exp_cfg: Contents of ``configs/experiment.yaml``; loaded if omitted.

    Returns:
        The unfitted pipeline.

    Raises:
        ConfigError: If the model or technique is unknown / not implemented yet, or the
            technique parameters are invalid.
    """
    cfg = cfg if cfg is not None else get_dataset_config()
    exp_cfg = exp_cfg if exp_cfg is not None else get_experiment_config()
    if model_name not in MODEL_BUILDERS:
        raise ConfigError(
            f"Unknown or not yet implemented model {model_name!r}; available: "
            f"{', '.join(MODEL_BUILDERS)}."
        )
    if technique not in TECHNIQUES:
        raise ConfigError(
            f"Unknown or not yet implemented technique {technique!r}; available: "
            f"{', '.join(TECHNIQUES)}."
        )
    seed = int(exp_cfg["seed"])
    params = dict(technique_params or {})
    sampler_steps = build_sampler_steps(technique, params, seed, int(exp_cfg["smote_k_neighbors"]))
    model_kwargs = _model_params_for(model_name, technique, dict(model_params or {}), params)
    model = MODEL_BUILDERS[model_name](model_kwargs, seed)
    return Pipeline(
        [("preprocess", build_preprocessor(cfg, exp_cfg)), *sampler_steps, ("model", model)]
    )


def without_model(pipe: Pipeline) -> Pipeline:
    """Return a new pipeline with the same (unfitted-state) steps except the final ``model``.

    Used by the resampling audit to see what the sampler steps do to the training data.

    Args:
        pipe: A pipeline from :func:`build_pipeline`.

    Returns:
        ``Pipeline(pipe.steps[:-1])`` (step objects are shared, not copied).

    Raises:
        ConfigError: If the last step is not named ``"model"``.
    """
    if pipe.steps[-1][0] != "model":
        raise ConfigError("Expected a pipeline whose last step is 'model' (from build_pipeline).")
    return Pipeline(pipe.steps[:-1])
