"""Stateless, sklearn-compatible feature transformers used inside the preprocessing pipeline.

Both learn nothing from the data in ``fit`` (they only record the input width and names), so
they cannot leak information between folds. They return NumPy arrays; the surrounding
``ColumnTransformer`` turns them into named pandas columns via ``get_feature_names_out``.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from fraudstudy.contracts import HOURS_PER_DAY
from fraudstudy.errors import DataValidationError

SECONDS_PER_HOUR = 3600


def _record_input(est: BaseEstimator, X: Any) -> None:
    """Store ``n_features_in_`` (and ``feature_names_in_`` for DataFrames) on ``est``."""
    est.n_features_in_ = np.shape(X)[1]
    if isinstance(X, pd.DataFrame):
        est.feature_names_in_ = np.asarray(X.columns, dtype=object)
    elif hasattr(est, "feature_names_in_"):
        del est.feature_names_in_


def _input_names(est: BaseEstimator, input_features: Any) -> list[str]:
    """Resolve input feature names the way sklearn transformers do."""
    check_is_fitted(est, "n_features_in_")
    if input_features is not None:
        names = [str(name) for name in input_features]
    elif hasattr(est, "feature_names_in_"):
        names = [str(name) for name in est.feature_names_in_]
    else:
        names = [f"x{i}" for i in range(est.n_features_in_)]
    if len(names) != est.n_features_in_:
        raise DataValidationError(
            f"{type(est).__name__} was fitted on {est.n_features_in_} columns but got "
            f"{len(names)} input feature names."
        )
    return names


def _as_float(est: BaseEstimator, X: Any) -> np.ndarray:
    """Return ``X`` as a 2-D float array, checking its width against ``fit``."""
    check_is_fitted(est, "n_features_in_")
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != est.n_features_in_:
        raise DataValidationError(
            f"{type(est).__name__} expects {est.n_features_in_} columns; got shape {arr.shape}."
        )
    return arr


class Log1pTransformer(TransformerMixin, BaseEstimator):
    """Element-wise ``log(1 + max(x, 0))``.

    Negative inputs (impossible for ``Amount`` but possible in a bad upload) are clipped to 0
    first, so the output is always finite and ``>= 0``, and ``0 -> 0``. Output columns are
    named ``log1p_<input name>``.
    """

    def fit(self, X: Any, y: Any = None) -> "Log1pTransformer":
        """Record the input width/names; nothing is learned.

        Args:
            X: 2-D array-like of shape ``(n_samples, n_features)``.
            y: Ignored.

        Returns:
            ``self``.
        """
        _record_input(self, X)
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Apply ``log1p(clip(X, 0, None))``.

        Args:
            X: 2-D array-like with the width seen in ``fit``.

        Returns:
            Float array of the same shape.
        """
        return np.log1p(np.clip(_as_float(self, X), 0.0, None))

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return ``["log1p_<name>", ...]`` for each input column."""
        return np.asarray([f"log1p_{name}" for name in _input_names(self, input_features)])


class CyclicalHourTransformer(TransformerMixin, BaseEstimator):
    """Turn seconds since the first transaction into a cyclical hour-of-day encoding.

    ``hour = (t / 3600) mod 24`` (fractional), then ``hour_sin = sin(2*pi*hour/24)`` and
    ``hour_cos = cos(2*pi*hour/24)``, so 23:59 and 00:00 are neighbours. This assumes ``t = 0``
    was midnight, which the dataset does not state; it is only used in the optional
    ``use_hour_feature`` side test.

    Expects exactly one input column (the time column). Output: ``hour_sin``, ``hour_cos``.
    """

    def fit(self, X: Any, y: Any = None) -> "CyclicalHourTransformer":
        """Check there is exactly one column and record it; nothing is learned.

        Args:
            X: 2-D array-like of shape ``(n_samples, 1)``.
            y: Ignored.

        Returns:
            ``self``.

        Raises:
            DataValidationError: If ``X`` does not have exactly one column.
        """
        if np.ndim(X) != 2 or np.shape(X)[1] != 1:
            raise DataValidationError(
                "CyclicalHourTransformer expects exactly one (time) column; got shape "
                f"{np.shape(X)}."
            )
        _record_input(self, X)
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Return ``[sin, cos]`` of the hour angle for each row.

        Args:
            X: Array-like of shape ``(n_samples, 1)`` holding seconds.

        Returns:
            Float array of shape ``(n_samples, 2)``.
        """
        seconds = _as_float(self, X)[:, 0]
        hour = np.mod(seconds / SECONDS_PER_HOUR, HOURS_PER_DAY)
        angle = 2.0 * np.pi * hour / HOURS_PER_DAY
        return np.column_stack([np.sin(angle), np.cos(angle)])

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return ``["hour_sin", "hour_cos"]``."""
        _input_names(self, input_features)
        return np.asarray(["hour_sin", "hour_cos"])
