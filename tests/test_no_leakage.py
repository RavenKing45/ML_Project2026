"""Proof that resampling never touches validation or test rows (synthetic sample only).

Row identity is the DataFrame index: the preprocessing ``ColumnTransformer`` outputs pandas with
the original index, so the index a sampler receives says exactly which rows it saw.

a. A spy sampler inside the real pipeline records every index it is given during
   ``cross_validate``: each fold's records equal that fold's TRAIN rows and never include its
   VALIDATION rows.
b. After fitting, ``predict_proba(X_test)`` returns one row per test row and the spy records
   nothing (samplers are skipped at prediction time).
c. A deliberately WRONG protocol (SMOTE on all data, then split) is shown to create training
   points interpolated from TEST frauds; the same detector finds none for the correct protocol.
"""

from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold, cross_validate

from fraudstudy.config import get_dataset_config, get_experiment_config, get_models_config
from fraudstudy.data import Split, deduplicate, load_dataset, split_dataset
from fraudstudy.pipeline import build_pipeline

RATIO = 0.1
TECHNIQUE_PARAMS: dict[str, dict[str, float]] = {
    "ros": {"sampling_strategy": RATIO},
    "rus": {"sampling_strategy": RATIO},
    "smote": {"sampling_strategy": RATIO},
    "smote_rus": {"smote_sampling_strategy": 0.05, "rus_sampling_strategy": 0.5},
}


class SpySampler(BaseEstimator):
    """imblearn-compatible sampler that wraps a real one and logs the index of every X it gets.

    Logs live in a class-level dict keyed by ``key`` because ``cross_validate`` clones the
    pipeline (a list passed as a parameter would be deep-copied, and the records lost).
    """

    log: ClassVar[dict[str, list[pd.Index]]] = {}

    def __init__(self, sampler: Any = None, key: str = "spy") -> None:
        self.sampler = sampler
        self.key = key

    def _record(self, X: pd.DataFrame) -> None:
        assert isinstance(X, pd.DataFrame), "preprocess must output pandas (keeps the index)"
        SpySampler.log.setdefault(self.key, []).append(X.index.copy())

    def fit(self, X: pd.DataFrame, y: Any) -> "SpySampler":
        self._record(X)
        self.sampler_ = clone(self.sampler).fit(X, y)
        return self

    def fit_resample(self, X: pd.DataFrame, y: Any) -> tuple[Any, Any]:
        self._record(X)
        self.sampler_ = clone(self.sampler)
        return self.sampler_.fit_resample(X, y)


@pytest.fixture(scope="module")
def cfg() -> dict[str, Any]:
    return get_dataset_config()


@pytest.fixture(scope="module")
def exp_cfg() -> dict[str, Any]:
    return {**get_experiment_config(), "use_hour_feature": False}


@pytest.fixture(scope="module")
def deduped(cfg: dict[str, Any]) -> pd.DataFrame:
    return deduplicate(load_dataset("sample", cfg=cfg))[0]


@pytest.fixture(scope="module")
def split(deduped: pd.DataFrame, cfg: dict[str, Any], exp_cfg: dict[str, Any]) -> Split:
    return split_dataset(deduped, cfg, exp_cfg)


def _spied_pipeline(technique: str, key: str, cfg: dict[str, Any], exp_cfg: dict[str, Any]) -> Any:
    """``build_pipeline`` output with its first sampler step wrapped in a :class:`SpySampler`."""
    model_params = {"max_iter": get_models_config()["logreg"]["max_iter"][0]}
    pipe = build_pipeline(
        "logreg", technique, model_params, TECHNIQUE_PARAMS[technique], cfg, exp_cfg
    )
    name, sampler = pipe.steps[1]
    SpySampler.log.pop(key, None)
    return pipe.set_params(**{name: SpySampler(sampler, key)})


# --- a. samplers only ever see the fold's training rows -----------------------------------------


@pytest.mark.parametrize("technique", sorted(TECHNIQUE_PARAMS))
def test_sampler_sees_only_training_fold_rows(
    technique: str, split: Split, cfg: dict[str, Any], exp_cfg: dict[str, Any]
) -> None:
    key = f"cv-{technique}"
    pipe = _spied_pipeline(technique, key, cfg, exp_cfg)
    X, y = split.X_train, split.y_train
    cv = StratifiedKFold(n_splits=exp_cfg["cv_folds"], shuffle=True, random_state=exp_cfg["seed"])
    result = cross_validate(
        pipe, X, y, cv=cv, scoring="average_precision", return_indices=True, error_score="raise"
    )
    records = SpySampler.log[key]
    folds = list(zip(result["indices"]["train"], result["indices"]["test"], strict=True))
    assert len(records) == len(folds) == exp_cfg["cv_folds"]  # one sampler fit per fold
    for seen, (train_pos, val_pos) in zip(records, folds, strict=True):
        train_rows, val_rows = set(X.index[train_pos]), set(X.index[val_pos])
        assert set(seen) <= train_rows
        assert set(seen).isdisjoint(val_rows)
        assert len(seen) == len(train_rows)  # every training row, each once
    assert np.all(np.isfinite(result["test_score"]))


# --- b. prediction passes rows straight through -------------------------------------------------


@pytest.mark.parametrize("technique", sorted(TECHNIQUE_PARAMS))
def test_prediction_skips_samplers(
    technique: str, split: Split, cfg: dict[str, Any], exp_cfg: dict[str, Any]
) -> None:
    key = f"predict-{technique}"
    pipe = _spied_pipeline(technique, key, cfg, exp_cfg)
    pipe.fit(split.X_train, split.y_train)
    assert len(SpySampler.log[key]) == 1
    assert set(SpySampler.log[key][0]) == set(split.X_train.index)

    scores = pipe.predict_proba(split.X_test)
    labels = pipe.predict(split.X_test)
    assert scores.shape == (len(split.X_test), 2)
    assert len(labels) == len(split.X_test)
    assert len(SpySampler.log[key]) == 1  # nothing recorded while predicting


# --- c. the leak the pipeline prevents ----------------------------------------------------------


def _parents(synthetic: np.ndarray, frauds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each synthetic row, the fraud pair (i, j) whose segment it lies on.

    SMOTE makes ``s = x_i + g * (x_j - x_i)`` with ``g`` in [0, 1). For every pair we project
    ``s - x_i`` on ``x_j - x_i`` (t clipped to [0, 1]) and take the pair with the smallest
    residual distance; for a true SMOTE point that residual is ~0.

    Returns:
        ``(pairs of shape (n, 2), residuals of shape (n,))``.
    """
    diffs = frauds[None, :, :] - frauds[:, None, :]  # diffs[i, j] = x_j - x_i
    norm2 = (diffs**2).sum(axis=-1)
    norm2[norm2 == 0] = np.inf  # i == j: t = 0, residual = distance to x_i
    pairs, residuals = [], []
    for s in synthetic:
        v = s - frauds  # v[i] = s - x_i
        t = np.clip(np.einsum("ijd,id->ij", diffs, v) / norm2, 0.0, 1.0)
        r = np.linalg.norm(v[:, None, :] - t[..., None] * diffs, axis=-1)
        i, j = np.unravel_index(np.argmin(r), r.shape)
        pairs.append((i, j))
        residuals.append(r[i, j])
    return np.asarray(pairs), np.asarray(residuals)


def _synthetic_from_test_frauds(
    X_resampled: pd.DataFrame, n_original: int, frauds: pd.DataFrame, test_index: pd.Index
) -> int:
    """Number of synthetic rows (after the first ``n_original``) built from a test fraud."""
    synthetic = X_resampled.iloc[n_original:].to_numpy(dtype=float)
    assert len(synthetic) > 0
    pairs, residuals = _parents(synthetic, frauds.to_numpy(dtype=float))
    # The detector found the exact segment for every synthetic row.
    assert residuals.max() < 1e-6 * np.abs(frauds.to_numpy()).max()
    is_test = frauds.index.isin(test_index)
    return int(is_test[pairs].any(axis=1).sum())


def test_smote_before_split_leaks_test_frauds_into_training(
    deduped: pd.DataFrame, split: Split, cfg: dict[str, Any], exp_cfg: dict[str, Any]
) -> None:
    """The wrong protocol leaks; the correct one (SMOTE on training rows only) cannot.

    WRONG: SMOTE on the full dataset, then split. Each test fraud is a SMOTE neighbour of other
    frauds, so synthetic points on segments ending at a test fraud land in the TRAINING set:
    part of the test answer is baked into training and the test score measures memory.
    RIGHT: inside the pipeline SMOTE runs in ``fit`` on training rows only (tests a/b above),
    so every synthetic point is interpolated between two TRAINING frauds.
    """
    X_all = deduped[list(split.X_train.columns)]
    y_all = deduped[cfg["target"]]
    frauds_all = X_all[y_all == 1]
    test_index = split.X_test.index

    def smote() -> SMOTE:
        return SMOTE(
            sampling_strategy=RATIO,
            k_neighbors=exp_cfg["smote_k_neighbors"],
            random_state=exp_cfg["seed"],
        )

    # Deliberately WRONG: fit_resample on ALL rows before splitting. Every synthetic row would
    # go to training (it has no original row to be split with).
    X_wrong, _ = smote().fit_resample(X_all, y_all)
    leaked = _synthetic_from_test_frauds(X_wrong, len(X_all), frauds_all, test_index)
    assert leaked > 0
    n_wrong = len(X_wrong) - len(X_all)
    assert leaked / n_wrong > 0.1  # a sizeable share, not a numerical fluke

    # Correct protocol, made visible: fit_resample on the TRAINING split only (what the
    # pipeline's SMOTE step does inside fit). Same detector, searching ALL frauds.
    X_right, _ = smote().fit_resample(split.X_train, split.y_train)
    assert _synthetic_from_test_frauds(X_right, len(split.X_train), frauds_all, test_index) == 0
