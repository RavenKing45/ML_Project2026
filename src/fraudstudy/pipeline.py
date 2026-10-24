"""Builds the full model pipeline: preprocess -> (sampler steps) -> model.

The result is always an ``imblearn.pipeline.Pipeline``, so any sampler steps run only during
``fit`` and are skipped by ``predict`` / ``predict_proba`` (docs/decisions.md). Techniques and
models are looked up in the two registries below; later prompts add entries there without
changing :func:`build_pipeline`'s signature.
"""

from collections.abc import Callable
from typing import Any

from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from fraudstudy.config import get_dataset_config, get_experiment_config
from fraudstudy.errors import ConfigError
from fraudstudy.features import build_preprocessor

ModelBuilder = Callable[[dict[str, Any], int], BaseEstimator]
SamplerBuilder = Callable[[dict[str, Any], int], list[tuple[str, BaseEstimator]]]


def _build_logreg(params: dict[str, Any], seed: int) -> BaseEstimator:
    """Logistic regression with ``random_state=seed`` and the given hyper-parameters."""
    return LogisticRegression(random_state=seed, **params)


def _baseline_steps(params: dict[str, Any], seed: int) -> list[tuple[str, BaseEstimator]]:
    """Technique A: no sampler, so no extra steps.

    Raises:
        ConfigError: If any parameters are given (the baseline has none).
    """
    if params:
        raise ConfigError(
            f"technique 'baseline' takes no parameters; got {', '.join(sorted(params))}."
        )
    return []


MODEL_BUILDERS: dict[str, ModelBuilder] = {"logreg": _build_logreg}
TECHNIQUE_STEPS: dict[str, SamplerBuilder] = {"baseline": _baseline_steps}


def build_pipeline(
    model_name: str,
    technique: str,
    model_params: dict[str, Any] | None = None,
    technique_params: dict[str, Any] | None = None,
    cfg: dict[str, Any] | None = None,
    exp_cfg: dict[str, Any] | None = None,
) -> Pipeline:
    """Build an unfitted ``imblearn`` pipeline: preprocess, sampler steps, model.

    Step names: ``"preprocess"``, then the technique's sampler steps (none for the baseline),
    then ``"model"``.

    Every random component receives ``random_state = exp_cfg["seed"]``.

    Args:
        model_name: Key of :data:`MODEL_BUILDERS` (currently ``"logreg"``).
        technique: Key of :data:`TECHNIQUE_STEPS` (currently ``"baseline"``).
        model_params: Keyword arguments for the model (e.g. ``{"C": 0.1, "max_iter": 2000}``).
        technique_params: Technique settings (e.g. a sampling ratio); empty for the baseline.
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
    if technique not in TECHNIQUE_STEPS:
        raise ConfigError(
            f"Unknown or not yet implemented technique {technique!r}; available: "
            f"{', '.join(TECHNIQUE_STEPS)}."
        )
    seed = int(exp_cfg["seed"])
    sampler_steps = TECHNIQUE_STEPS[technique](dict(technique_params or {}), seed)
    model = MODEL_BUILDERS[model_name](dict(model_params or {}), seed)
    return Pipeline(
        [("preprocess", build_preprocessor(cfg, exp_cfg)), *sampler_steps, ("model", model)]
    )
