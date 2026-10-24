"""Imbalance techniques A-F: sampler steps and the shared ratio search space (master guide Part 6).

Every technique draws its ratio from the same ``ratio_grid`` in ``configs/experiment.yaml``
(SMOTE + RUS draws ``a`` and ``b`` from ``hybrid``, whose values are grid members). A ratio
``r`` is imblearn's float ``sampling_strategy``: after resampling, ``fraud = r * legit``.
Class weighting (F) maps the same ratios to a fraud weight ``w = r / base_ratio``, so that
weighted fraud mass equals ``r`` times the legitimate mass, like a resampler at ratio ``r``.

Samplers built here are only ever placed inside an ``imblearn.pipeline.Pipeline``
(:func:`fraudstudy.pipeline.build_pipeline`), so they run during ``fit`` on training rows only.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator

from fraudstudy.errors import ConfigError, DataValidationError

logger = logging.getLogger(__name__)

TECHNIQUES: tuple[str, ...] = ("baseline", "ros", "rus", "smote", "smote_rus", "class_weight")
SINGLE_SAMPLER: dict[str, str] = {"ros": "ros", "rus": "rus", "smote": "smote"}
STRATEGY_KEY = "sampling_strategy"
SMOTE_RUS_KEYS: tuple[str, str] = ("smote_sampling_strategy", "rus_sampling_strategy")
FRAUD_WEIGHT_KEY = "fraud_weight"
BALANCED = "balanced"
# imblearn rejects a target ratio at or below the current one; 5% headroom also covers CV
# training folds whose fraud share is slightly above the full training set's.
RATIO_MARGIN = 1.05
CLASS_WEIGHT_PARAM: dict[str, str] = {
    "logreg": "class_weight",
    "random_forest": "class_weight",
    "lightgbm": "scale_pos_weight",
}
MODELS_WITH_BALANCED: tuple[str, ...] = ("logreg", "random_forest")

SamplerSteps = list[tuple[str, BaseEstimator]]
SearchSpace = dict[str, list[Any]]


def base_ratio(y: pd.Series | np.ndarray) -> float:
    """Return ``frauds / legit`` of a label vector (the current imblearn ratio).

    Args:
        y: Binary labels (1 = fraud), typically the TRAINING labels.

    Returns:
        The fraud-to-legitimate ratio.

    Raises:
        DataValidationError: If either class is missing.
    """
    labels = np.asarray(y)
    frauds, legit = int((labels == 1).sum()), int((labels == 0).sum())
    if frauds == 0 or legit == 0:
        raise DataValidationError(
            f"Need both classes to compute the base ratio; got {frauds} frauds and {legit} "
            "legitimate rows. Check the training labels."
        )
    return frauds / legit


def valid_ratios(ratios: list[float], base: float) -> list[float]:
    """Keep the ratios strictly greater than ``base * RATIO_MARGIN``; log the dropped ones.

    Args:
        ratios: Candidate target ratios (from ``ratio_grid`` or ``hybrid``).
        base: ``frauds / legit`` of the training set.

    Returns:
        The kept ratios, in their original order.

    Raises:
        ConfigError: If no ratio survives.
    """
    threshold = base * RATIO_MARGIN
    kept = [float(r) for r in ratios if float(r) > threshold]
    dropped = [float(r) for r in ratios if float(r) <= threshold]
    if dropped:
        logger.info(
            "Dropped ratios %s: not above base ratio %.5f x %.2f = %.5f (imblearn rejects them).",
            dropped,
            base,
            RATIO_MARGIN,
            threshold,
        )
    if not kept:
        raise ConfigError(
            f"No ratio in {list(ratios)} exceeds base ratio {base:.5f} x {RATIO_MARGIN}; add "
            "larger values to ratio_grid in configs/experiment.yaml."
        )
    return kept


def fraud_weight(ratio: float, base: float) -> float:
    """Fraud class weight matching resampling ratio ``ratio``: ``w = ratio / base``.

    With ``n_f`` frauds and ``n_l`` legit rows, weighted fraud mass ``w * n_f = ratio * n_l``,
    the same class balance a resampler at ``ratio`` produces.

    Args:
        ratio: Target ratio from the shared grid.
        base: ``frauds / legit`` of the training set.

    Returns:
        The fraud weight (legitimate weight is 1).
    """
    return float(ratio) / float(base)


def class_weight_model_params(model_name: str, weight: float | str) -> dict[str, Any]:
    """Model keyword arguments that give the fraud class weight ``weight`` (technique F).

    Args:
        model_name: ``logreg``, ``random_forest`` or ``lightgbm``.
        weight: A positive fraud weight, or ``"balanced"`` (logreg / random_forest only).

    Returns:
        ``{"class_weight": {0: 1.0, 1: w}}`` (or ``"balanced"``) for logreg / random_forest;
        ``{"scale_pos_weight": w}`` for lightgbm.

    Raises:
        ConfigError: If the model is unknown, or the weight is invalid for it.
    """
    if model_name not in CLASS_WEIGHT_PARAM:
        raise ConfigError(
            f"No class-weight parameter known for model {model_name!r}; available: "
            f"{', '.join(CLASS_WEIGHT_PARAM)}."
        )
    param = CLASS_WEIGHT_PARAM[model_name]
    if weight == BALANCED:
        if model_name not in MODELS_WITH_BALANCED:
            raise ConfigError(f"{model_name} has no 'balanced' option; pass a numeric weight.")
        return {param: BALANCED}
    if isinstance(weight, bool) or not isinstance(weight, int | float) or weight <= 0:
        raise ConfigError(f"fraud_weight must be a positive number or 'balanced'; got {weight!r}.")
    if param == "class_weight":
        return {param: {0: 1.0, 1: float(weight)}}
    return {param: float(weight)}


def _check_keys(technique: str, params: dict[str, Any], allowed: tuple[str, ...]) -> None:
    """Raise ``ConfigError`` if ``params`` has keys outside ``allowed``."""
    unknown = sorted(set(params) - set(allowed))
    if unknown:
        if not allowed:
            raise ConfigError(
                f"technique {technique!r} takes no parameters; got {', '.join(unknown)}."
            )
        raise ConfigError(
            f"technique {technique!r} got unknown parameters {', '.join(unknown)}; allowed: "
            f"{', '.join(allowed)}."
        )


def _check_ratio(name: str, value: Any) -> float:
    """Return ``value`` as a float ratio in (0, 1], else raise ``ConfigError``."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 < value <= 1.0:
        raise ConfigError(f"{name} must be a ratio in (0, 1]; got {value!r}.")
    return float(value)


def _strategy(params: dict[str, Any], key: str) -> Any:
    """``sampling_strategy`` for a sampler: the given ratio, else imblearn's ``"auto"``."""
    return _check_ratio(key, params[key]) if key in params else "auto"


def build_sampler_steps(
    technique: str, params: dict[str, Any], seed: int, k_neighbors: int
) -> SamplerSteps:
    """Return the pipeline steps that resample the training data for ``technique``.

    * ``baseline``: ``[]`` (no parameters allowed).
    * ``ros`` / ``rus`` / ``smote``: one step named after the technique; optional
      ``sampling_strategy`` (ratio in (0, 1], else imblearn's ``"auto"`` = 1.0, usually set later
      by the search via ``<step>__sampling_strategy``).
    * ``smote_rus``: ``[("smote", ...), ("rus", ...)]``; optional ``smote_sampling_strategy`` = a
      and ``rus_sampling_strategy`` = b with ``a < b``.
    * ``class_weight``: ``[]``; its optional ``fraud_weight`` goes to the model
      (:func:`class_weight_model_params`).

    Every sampler gets ``random_state=seed``; SMOTE gets ``k_neighbors``.

    Args:
        technique: One of :data:`TECHNIQUES`.
        params: Technique parameters (see above).
        seed: Random seed from ``configs/experiment.yaml``.
        k_neighbors: SMOTE's ``k_neighbors`` (``smote_k_neighbors``).

    Returns:
        A list of ``(step name, unfitted sampler)``.

    Raises:
        ConfigError: If the technique is unknown or the parameters are invalid.
    """
    if technique not in TECHNIQUES:
        raise ConfigError(
            f"Unknown or not yet implemented technique {technique!r}; available: "
            f"{', '.join(TECHNIQUES)}."
        )

    def smote(strategy: Any) -> SMOTE:
        return SMOTE(sampling_strategy=strategy, k_neighbors=k_neighbors, random_state=seed)

    if technique == "baseline":
        _check_keys(technique, params, ())
        return []
    if technique == "class_weight":
        _check_keys(technique, params, (FRAUD_WEIGHT_KEY,))
        return []
    if technique in SINGLE_SAMPLER:
        _check_keys(technique, params, (STRATEGY_KEY,))
        strategy = _strategy(params, STRATEGY_KEY)
        sampler: BaseEstimator
        if technique == "ros":
            sampler = RandomOverSampler(sampling_strategy=strategy, random_state=seed)
        elif technique == "rus":
            sampler = RandomUnderSampler(sampling_strategy=strategy, random_state=seed)
        else:
            sampler = smote(strategy)
        return [(SINGLE_SAMPLER[technique], sampler)]
    # smote_rus
    _check_keys(technique, params, SMOTE_RUS_KEYS)
    a, b = (_strategy(params, key) for key in SMOTE_RUS_KEYS)
    if isinstance(a, float) and isinstance(b, float) and not a < b:
        raise ConfigError(f"smote_rus needs smote ratio a < rus ratio b; got a={a}, b={b}.")
    return [
        ("smote", smote(a)),
        ("rus", RandomUnderSampler(sampling_strategy=b, random_state=seed)),
    ]


def technique_search_space(
    technique: str,
    exp_cfg: dict[str, Any],
    base_ratio: float,
    model_name: str = "logreg",
) -> SearchSpace | list[SearchSpace]:
    """Pipeline parameter grid for ``technique``, drawn from the shared ratio grid.

    * ``baseline``: ``{}``.
    * ``ros`` / ``rus`` / ``smote``: ``{"<step>__sampling_strategy": valid ratio_grid}``.
    * ``smote_rus``: a LIST of dicts, one per valid pair ``a < b`` (``a`` from
      ``hybrid.smote_ratios``, ``b`` from ``hybrid.rus_ratios``):
      ``{"smote__sampling_strategy": [a], "rus__sampling_strategy": [b]}``.
    * ``class_weight``: ``{"model__class_weight": [{0: 1, 1: r / base_ratio}, ..., "balanced"]}``
      for logreg / random_forest, ``{"model__scale_pos_weight": [r / base_ratio, ...]}`` for
      lightgbm, over the same valid ratios.

    Valid ratios are those ``> base_ratio * 1.05`` (:func:`valid_ratios`); dropped ones are logged.

    Args:
        technique: One of :data:`TECHNIQUES`.
        exp_cfg: Contents of ``configs/experiment.yaml`` (``ratio_grid``, ``hybrid``).
        base_ratio: ``frauds / legit`` of the TRAINING set (:func:`base_ratio`).
        model_name: Model the space is for (only matters for ``class_weight``).

    Returns:
        A dict of pipeline parameter name -> candidate values, or a list of such dicts for
        ``smote_rus`` (``RandomizedSearchCV`` accepts both).

    Raises:
        ConfigError: If the technique is unknown or no ratio / pair is valid.
    """
    if technique not in TECHNIQUES:
        raise ConfigError(f"Unknown technique {technique!r}; available: {', '.join(TECHNIQUES)}.")
    if technique == "baseline":
        return {}
    if technique in SINGLE_SAMPLER:
        ratios = valid_ratios(list(exp_cfg["ratio_grid"]), base_ratio)
        return {f"{SINGLE_SAMPLER[technique]}__{STRATEGY_KEY}": ratios}
    if technique == "smote_rus":
        hybrid = exp_cfg["hybrid"]
        smote_ratios = valid_ratios(list(hybrid["smote_ratios"]), base_ratio)
        rus_ratios = valid_ratios(list(hybrid["rus_ratios"]), base_ratio)
        pairs = [(a, b) for a in smote_ratios for b in rus_ratios if a < b]
        if not pairs:
            raise ConfigError(
                "No hybrid (a, b) pair with a < b survives; check hybrid in "
                "configs/experiment.yaml."
            )
        return [{f"smote__{STRATEGY_KEY}": [a], f"rus__{STRATEGY_KEY}": [b]} for a, b in pairs]
    # class_weight
    ratios = valid_ratios(list(exp_cfg["ratio_grid"]), base_ratio)
    weights: list[Any] = [
        class_weight_model_params(model_name, fraud_weight(r, base_ratio)) for r in ratios
    ]
    param = CLASS_WEIGHT_PARAM[model_name]
    values = [w[param] for w in weights]
    if model_name in MODELS_WITH_BALANCED:
        values.append(BALANCED)
    return {f"model__{param}": values}
