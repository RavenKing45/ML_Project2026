"""Per-prediction explanations for the deployed models (master guide Part 7).

Both methods are exact, additive decompositions of the model's log-odds score, so
``base + sum(contributions) == score`` up to rounding:

* Logistic regression (method ``"linear"``): ``contribution_j = coef_j * (x_scaled_j -
  mean_scaled_j)``, where ``mean_scaled_j`` is the training mean of feature ``j`` in the
  preprocessor's (scaled) units; ``base = intercept + sum_j coef_j * mean_scaled_j`` and the
  score is ``decision_function``.
* LightGBM (method ``"TreeSHAP"``): LightGBM's own ``predict(pred_contrib=True)`` on the
  preprocessed row; the last column is the bias (the expected raw score) and the score is the
  raw score. No dependency on the ``shap`` package.

Only ``pipeline.named_steps["preprocess"]`` transforms the row; the samplers never run.

"How unusual" is the percentile of the raw input value among LEGITIMATE training rows, read
from a :class:`FeatureReference` (101 quantiles per model feature, written by the exporter).
The reference may also hold each raw column's central range over ALL training rows (e.g. the
0.1th-99.9th percentiles), which the backend uses to warn about values the model rarely saw.
Contributions say what the model used, not what caused a fraud.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from pydantic import model_validator
from sklearn.linear_model import LogisticRegression

from fraudstudy.contracts import Explanation, FeatureContribution, StrictModel
from fraudstudy.errors import ConfigError, DataValidationError

N_QUANTILES = 101
PERCENT_MIN = 0.0
PERCENT_MAX = 100.0
DEFAULT_TOP_K = 8
UNITS = "log-odds"
METHOD_LINEAR = "linear"
METHOD_TREESHAP = "TreeSHAP"
EXPLAINABLE_MODELS: tuple[str, ...] = ("logreg", "lightgbm")


class ReferenceFeature(StrictModel):
    """Reference data of one model feature (a preprocessor output column).

    ``raw_column`` is the input column its raw value comes from (``log1p_Amount`` -> ``Amount``).
    ``legit_quantiles[k]`` is the k-th percentile (k = 0..100) of ``raw_column`` over the
    legitimate training rows, in raw units. ``train_mean_scaled`` is the mean of the feature over
    all training rows in scaled units (the centring point of logistic-regression contributions).
    ``train_range`` (optional) is ``[low, high]``: the reference's ``range_percentiles`` of
    ``raw_column`` over ALL training rows, in raw units.
    """

    feature: str
    raw_column: str
    legit_quantiles: list[float]
    train_mean_scaled: float
    train_range: tuple[float, float] | None = None

    @model_validator(mode="after")
    def _quantiles(self) -> "ReferenceFeature":
        q = self.legit_quantiles
        if len(q) != N_QUANTILES:
            raise ValueError(f"{self.feature}: need {N_QUANTILES} quantiles; got {len(q)}.")
        if any(b < a for a, b in zip(q, q[1:], strict=False)):
            raise ValueError(f"{self.feature}: quantiles must be non-decreasing.")
        if self.train_range is not None and self.train_range[0] > self.train_range[1]:
            raise ValueError(f"{self.feature}: train_range must be [low, high] with low <= high.")
        return self


class FeatureReference(StrictModel):
    """``feature_reference.json``: per-feature reference data, from TRAINING rows only.

    Written by :mod:`fraudstudy.export`, read by :func:`explain`. ``features`` follows the
    preprocessor's output order. ``range_percentiles`` (optional) are the percentiles behind every
    feature's ``train_range``; both are set together or not at all.
    """

    created_at: str
    source: str
    data_sha256: str | None
    train_sha256: str
    n_train: int
    n_legit: int
    features: list[ReferenceFeature]
    range_percentiles: tuple[float, float] | None = None

    @model_validator(mode="after")
    def _unique(self) -> "FeatureReference":
        names = [f.feature for f in self.features]
        if len(names) != len(set(names)):
            raise ValueError(f"Reference features must be unique; got {names}.")
        with_range = {f.train_range is not None for f in self.features}
        if with_range - {self.range_percentiles is not None}:
            raise ValueError("Set range_percentiles and every feature's train_range together.")
        return self


def check_range_percentiles(percentiles: Sequence[float]) -> tuple[float, float]:
    """Validate a ``[low, high]`` percentile pair.

    Args:
        percentiles: Two numbers, e.g. ``[0.1, 99.9]``.

    Returns:
        ``(low, high)`` as floats.

    Raises:
        ConfigError: Unless there are two of them with ``0 <= low < high <= 100``.
    """
    values = [float(p) for p in percentiles]
    if len(values) != 2 or not PERCENT_MIN <= values[0] < values[1] <= PERCENT_MAX:
        raise ConfigError(
            f"Range percentiles must be [low, high] with 0 <= low < high <= 100; got {values}."
        )
    return values[0], values[1]


@dataclass(frozen=True)
class Contributions:
    """Full additive decomposition of the scores of ``n`` rows.

    Attributes:
        method: ``"linear"`` (logreg) or ``"TreeSHAP"`` (LightGBM).
        units: Always ``"log-odds"``.
        features: Model features (preprocessor output order), ``k`` of them.
        raw_columns: Input column of each feature.
        values: Raw input values, shape ``(n, k)``.
        contributions: Shape ``(n, k)``.
        base: Shape ``(n,)``; ``base + contributions.sum(axis=1)`` equals ``score``.
        score: The model's own log-odds output, shape ``(n,)`` (``decision_function`` for
            logreg, ``predict(raw_score=True)`` for LightGBM).
    """

    method: str
    units: str
    features: list[str]
    raw_columns: list[str]
    values: np.ndarray
    contributions: np.ndarray
    base: np.ndarray
    score: np.ndarray


def feature_map(preprocess: Any) -> list[tuple[str, str]]:
    """``(model feature, raw input column)`` pairs in the fitted preprocessor's output order.

    A branch with as many outputs as inputs maps them 1:1 (``V1 -> V1``,
    ``log1p_Amount -> Amount``); a one-input branch maps every output to its input
    (``hour_sin, hour_cos -> Time``).

    Args:
        preprocess: The fitted ``ColumnTransformer`` (``pipeline.named_steps["preprocess"]``).

    Returns:
        The pairs, one per output column.

    Raises:
        DataValidationError: If a branch cannot be mapped or the order disagrees with
            ``get_feature_names_out``.
    """
    pairs: list[tuple[str, str]] = []
    for name, transformer, columns in preprocess.transformers_:
        if name == "remainder" or transformer == "drop":
            continue
        inputs = [str(c) for c in columns]
        outputs = [str(c) for c in transformer.get_feature_names_out(inputs)]
        if len(outputs) == len(inputs):
            pairs.extend(zip(outputs, inputs, strict=True))
        elif len(inputs) == 1:
            pairs.extend((out, inputs[0]) for out in outputs)
        else:
            raise DataValidationError(
                f"Preprocessor branch {name!r} maps {inputs} to {outputs}; cannot tell which raw "
                "column each output comes from."
            )
    names = [str(c) for c in preprocess.get_feature_names_out()]
    if [f for f, _ in pairs] != names:
        raise DataValidationError(
            f"Preprocessor output order {names} does not match its branches {pairs}."
        )
    return pairs


def build_feature_reference(
    preprocess: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    source: str,
    data_sha256: str | None,
    train_sha256: str,
    range_percentiles: Sequence[float] | None = None,
) -> FeatureReference:
    """Build the reference from TRAINING rows (never test rows).

    For each model feature: ``np.quantile`` at 0, 0.01, ..., 1 (linear interpolation) of its raw
    column over the legitimate training rows, and the mean over all training rows of
    ``preprocess.transform(X_train)``. With ``range_percentiles = [p_low, p_high]``, also
    ``train_range = np.percentile(raw column over ALL training rows, [p_low, p_high])``.

    Args:
        preprocess: The fitted preprocessor of a deployed pipeline.
        X_train: Training features (raw units).
        y_train: Training labels (1 = fraud).
        source: ``"raw"`` or ``"sample"``.
        data_sha256: SHA-256 of the dataset file.
        train_sha256: Hash of the training rows (``training_sha256``).
        range_percentiles: ``[low, high]`` percentiles for ``train_range`` (e.g. ``deploy.
            input_range_percentiles`` in experiment.yaml), or ``None`` for no ranges.

    Returns:
        The :class:`FeatureReference`.

    Raises:
        ConfigError: If ``range_percentiles`` is not a valid ``[low, high]`` pair.
        DataValidationError: If there are no legitimate rows or the sizes disagree.
    """
    percentiles = None if range_percentiles is None else check_range_percentiles(range_percentiles)
    labels = np.asarray(y_train)
    if len(labels) != len(X_train):
        raise DataValidationError(f"X_train has {len(X_train)} rows but y_train {len(labels)}.")
    legit = X_train.loc[labels == 0]
    if legit.empty:
        raise DataValidationError("No legitimate training rows to build the reference from.")
    pairs = feature_map(preprocess)
    scaled_mean = np.asarray(preprocess.transform(X_train), dtype=float).mean(axis=0)
    levels = np.linspace(0.0, 1.0, N_QUANTILES)
    features = [
        ReferenceFeature(
            feature=feature,
            raw_column=column,
            legit_quantiles=np.quantile(legit[column].to_numpy(dtype=float), levels).tolist(),
            train_mean_scaled=float(scaled_mean[j]),
            train_range=None
            if percentiles is None
            else _percentile_pair(X_train[column].to_numpy(dtype=float), percentiles),
        )
        for j, (feature, column) in enumerate(pairs)
    ]
    return FeatureReference(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        source=source,
        data_sha256=data_sha256,
        train_sha256=train_sha256,
        n_train=len(labels),
        n_legit=len(legit),
        features=features,
        range_percentiles=percentiles,
    )


def _percentile_pair(values: np.ndarray, percentiles: tuple[float, float]) -> tuple[float, float]:
    """``np.percentile(values, percentiles)`` (linear interpolation) as a float pair."""
    low, high = np.percentile(values, percentiles)
    return float(low), float(high)


def legit_percentile(value: float, quantiles: Sequence[float]) -> float:
    """Where ``value`` falls among legitimate training rows, in percent (0-100).

    ``quantiles[k]`` is the k-th percentile. Below the 0th -> 0; above the 100th -> 100; strictly
    between ``q_k`` and ``q_{k+1}`` -> ``k + (value - q_k) / (q_{k+1} - q_k)`` (linear
    interpolation); equal to a run of tied quantiles ``q_lo = ... = q_hi`` -> the middle of the
    run, ``(lo + hi) / 2``.

    Args:
        value: Raw input value.
        quantiles: The feature's ``legit_quantiles`` (non-decreasing).

    Returns:
        The percentile.
    """
    q = np.asarray(quantiles, dtype=float)
    step = PERCENT_MAX / (len(q) - 1)
    lo = int(np.searchsorted(q, value, side="left"))
    hi = int(np.searchsorted(q, value, side="right"))
    if hi > lo:
        return float((lo + hi - 1) / 2 * step)
    if lo == 0:
        return 0.0
    if lo == len(q):
        return PERCENT_MAX
    fraction = (value - q[lo - 1]) / (q[lo] - q[lo - 1])
    return float((lo - 1 + fraction) * step)


def model_kind(pipeline: Any) -> str:
    """``"logreg"`` or ``"lightgbm"`` for the pipeline's final ``model`` step.

    Raises:
        ConfigError: If the pipeline has no ``preprocess`` / ``model`` step or the model cannot
            be explained here (random forest is not deployed; use ``shap`` in a notebook).
    """
    steps = getattr(pipeline, "named_steps", {})
    if "preprocess" not in steps or "model" not in steps:
        raise ConfigError("Expected a pipeline with 'preprocess' and 'model' steps.")
    model = steps["model"]
    if isinstance(model, LogisticRegression):
        return "logreg"
    if isinstance(model, LGBMClassifier):
        return "lightgbm"
    raise ConfigError(
        f"Cannot explain {type(model).__name__}; only {', '.join(EXPLAINABLE_MODELS)} are "
        "deployed. For random forest use the shap package in a notebook."
    )


def _aligned_reference(
    reference: FeatureReference, pairs: list[tuple[str, str]]
) -> list[ReferenceFeature]:
    """The reference entries in the pipeline's feature order; raise if they do not match."""
    expected = [(r.feature, r.raw_column) for r in reference.features]
    if expected != pairs:
        raise DataValidationError(
            f"feature_reference.json lists {expected} but the pipeline produces {pairs}; "
            "re-run scripts/export_artifacts.py so both come from the same export."
        )
    return list(reference.features)


def compute_contributions(
    pipeline: Any, X: pd.DataFrame, reference: FeatureReference
) -> Contributions:
    """Decompose the log-odds score of every row of ``X`` (see the module docstring).

    Args:
        pipeline: A fitted logreg or LightGBM pipeline from ``build_pipeline``.
        X: Rows with the pipeline's input columns (raw units); extra columns are ignored.
        reference: The export's :class:`FeatureReference`.

    Returns:
        A :class:`Contributions`.

    Raises:
        ConfigError: If the model cannot be explained.
        DataValidationError: If ``X`` lacks input columns or the reference does not match.
    """
    kind = model_kind(pipeline)
    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    missing = [str(c) for c in preprocess.feature_names_in_ if c not in X.columns]
    if missing:
        raise DataValidationError(f"Rows lack the model's input columns: {', '.join(missing)}.")
    pairs = feature_map(preprocess)
    ref = _aligned_reference(reference, pairs)
    X_transformed = preprocess.transform(X)
    if kind == "logreg":
        coef = np.asarray(model.coef_, dtype=float).ravel()
        mean = np.array([r.train_mean_scaled for r in ref])
        contrib = coef * (np.asarray(X_transformed, dtype=float) - mean)
        base = np.full(len(X), float(model.intercept_[0]) + float(coef @ mean))
        score = np.asarray(model.decision_function(X_transformed), dtype=float)
        method = METHOD_LINEAR
    else:
        out = np.asarray(model.predict(X_transformed, pred_contrib=True), dtype=float)
        contrib, base = out[:, :-1], out[:, -1]
        score = np.asarray(model.predict(X_transformed, raw_score=True), dtype=float)
        method = METHOD_TREESHAP
    raw_columns = [column for _, column in pairs]
    return Contributions(
        method=method,
        units=UNITS,
        features=[feature for feature, _ in pairs],
        raw_columns=raw_columns,
        values=X[raw_columns].to_numpy(dtype=float),
        contributions=contrib,
        base=base,
        score=score,
    )


def explain(
    pipeline: Any,
    row_df: pd.DataFrame,
    reference: FeatureReference,
    top_k: int = DEFAULT_TOP_K,
) -> Explanation:
    """Explain one payment: its ``top_k`` largest contributions, with how unusual each value is.

    Items are sorted by ``|contribution|`` (largest first; ties keep the feature order), so the
    result is deterministic. ``feature`` is the model feature (e.g. ``log1p_Amount``), ``value``
    the RAW input value of its column (e.g. the ``Amount``), ``contribution`` in log-odds, and
    ``legit_percentile`` from :func:`legit_percentile`.

    Args:
        pipeline: A fitted logreg or LightGBM pipeline.
        row_df: Exactly one row with the pipeline's input columns.
        reference: The export's :class:`FeatureReference`.
        top_k: How many features to return (at least 1).

    Returns:
        The contract ``Explanation``.

    Raises:
        ConfigError: If ``top_k < 1`` or the model cannot be explained.
        DataValidationError: If ``row_df`` is not one row of finite values with the input columns.
    """
    if top_k < 1:
        raise ConfigError(f"top_k must be at least 1; got {top_k}.")
    if len(row_df) != 1:
        raise DataValidationError(f"explain() takes exactly one row; got {len(row_df)}.")
    c = compute_contributions(pipeline, row_df, reference)
    values, contrib = c.values[0], c.contributions[0]
    if not np.all(np.isfinite(values)):
        raise DataValidationError("explain() needs finite values for every input column.")
    order = sorted(range(len(c.features)), key=lambda j: (-abs(contrib[j]), j))[:top_k]
    quantiles = {r.feature: r.legit_quantiles for r in reference.features}
    items = [
        FeatureContribution(
            feature=c.features[j],
            value=float(values[j]),
            contribution=float(contrib[j]),
            legit_percentile=legit_percentile(float(values[j]), quantiles[c.features[j]]),
        )
        for j in order
    ]
    return Explanation(method=c.method, units=c.units, top_features=items)
