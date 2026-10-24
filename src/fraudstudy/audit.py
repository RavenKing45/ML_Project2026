"""Resampling audit: what a technique's sampler steps do to TRAINING data (class counts, copies,
synthetic rows). Feeds ``artifacts/results/resampling_audit.json`` and the leakage tests.

This module never sees validation or test rows: callers pass training rows (or a CV training
fold) only.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.base import clone

from fraudstudy.errors import ConfigError, DataValidationError


@dataclass(frozen=True)
class ResamplingAudit:
    """Class counts of one training set before and after a technique's sampler steps.

    Rows are compared in the preprocessed feature space (the space the samplers see).

    Attributes:
        legit_before, fraud_before: Counts given to the pipeline.
        legit_after, fraud_after: Counts after the last sampler step.
        fraud_unique_after: Distinct fraud rows after resampling.
        fraud_duplicates_after: ``fraud_after - fraud_unique_after`` (exact copies, e.g. ROS).
        n_synthetic: Fraud rows after resampling that equal no row given to the samplers
            (new SMOTE points); 0 for ROS, RUS and techniques without samplers.
    """

    legit_before: int
    fraud_before: int
    legit_after: int
    fraud_after: int
    fraud_unique_after: int
    fraud_duplicates_after: int
    n_synthetic: int

    @property
    def ratio_after(self) -> float:
        """``fraud_after / legit_after`` (the achieved imblearn ratio)."""
        return self.fraud_after / self.legit_after if self.legit_after else float("inf")


def row_hashes(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """One 64-bit hash per row of values (index ignored); equal rows get equal hashes.

    Args:
        X: Feature table.

    Returns:
        ``uint64`` array of length ``len(X)``.
    """
    return pd.util.hash_pandas_object(pd.DataFrame(np.asarray(X)), index=False).to_numpy()


def _counts(y: np.ndarray) -> tuple[int, int]:
    """``(legit, fraud)`` counts of a label vector."""
    return int((y == 0).sum()), int((y == 1).sum())


def resampling_audit(
    pipeline_without_model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series
) -> ResamplingAudit:
    """Run a clone of ``pipeline_without_model`` on TRAINING rows and count what it produced.

    Steps run in order on a clone (the argument is left unfitted): transformers use
    ``fit_transform``, samplers use ``fit_resample``. Calling the steps one by one (instead of
    ``Pipeline.fit``) is the only way to see the resampled rows; it is allowed here because the
    audit is given training data only and never feeds a model.

    Args:
        pipeline_without_model: E.g. ``fraudstudy.pipeline.without_model(build_pipeline(...))``.
        X_train: Training features (a CV training fold or the full training set).
        y_train: Training labels.

    Returns:
        A :class:`ResamplingAudit`.

    Raises:
        ConfigError: If the pipeline still ends with a ``model`` step.
        DataValidationError: If ``X_train`` and ``y_train`` differ in length.
    """
    if pipeline_without_model.steps[-1][0] == "model":
        raise ConfigError(
            "resampling_audit needs the pipeline without its model; use "
            "fraudstudy.pipeline.without_model(pipe)."
        )
    if len(X_train) != len(y_train):
        raise DataValidationError(
            f"X_train has {len(X_train)} rows but y_train has {len(y_train)}."
        )
    pipe = clone(pipeline_without_model)
    y_before = np.asarray(y_train)
    Xt: pd.DataFrame | np.ndarray = X_train
    yt: np.ndarray = y_before
    sampler_input: pd.DataFrame | np.ndarray | None = None
    for _, step in pipe.steps:
        if hasattr(step, "fit_resample"):
            if sampler_input is None:
                sampler_input = Xt
            # Audit only, on training rows only: see the docstring.
            Xt, yt_out = step.fit_resample(Xt, yt)
            yt = np.asarray(yt_out)
        else:
            Xt = step.fit_transform(Xt, yt)
    originals = set(row_hashes(sampler_input if sampler_input is not None else Xt).tolist())

    fraud_hashes = row_hashes(Xt)[yt == 1]
    legit_before, fraud_before = _counts(y_before)
    legit_after, fraud_after = _counts(yt)
    unique_after = len(set(fraud_hashes.tolist()))
    return ResamplingAudit(
        legit_before=legit_before,
        fraud_before=fraud_before,
        legit_after=legit_after,
        fraud_after=fraud_after,
        fraud_unique_after=unique_after,
        fraud_duplicates_after=fraud_after - unique_after,
        n_synthetic=int(sum(h not in originals for h in fraud_hashes.tolist())),
    )
