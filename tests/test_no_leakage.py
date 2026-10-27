"""Proof that no information from validation or test rows reaches training (fake sample only).

A row's identity is its DataFrame index: the preprocess step outputs pandas with the original
index, so the index a sampler receives says exactly which rows it saw.
"""

import json

import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold, cross_validate

from fraudstudy import evaluation, experiments
from fraudstudy.data import SEED, load_data, remove_duplicates, split_data

SEEN_ROWS = []  # every index the SpySampler was given, in order


class SpySampler(BaseEstimator):
    """Wraps a real sampler and writes down the index of every X it is asked to resample.

    The list lives outside the object because cross_validate copies the pipeline for each fold.
    """

    def __init__(self, sampler: object = None) -> None:
        self.sampler = sampler

    def fit_resample(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        SEEN_ROWS.append(X.index.copy())
        return clone(self.sampler).fit_resample(X, y)


@pytest.fixture(scope="module")
def split() -> tuple:
    df, _, _ = remove_duplicates(load_data("sample"))
    return split_data(df)


def spied_pipeline() -> object:
    """The real logreg + SMOTE pipeline, with its SMOTE step wrapped in a SpySampler."""
    pipeline = experiments.build_pipeline("logreg", "smote")
    smote = SMOTE(sampling_strategy=0.1, k_neighbors=experiments.SMOTE_K, random_state=SEED)
    SEEN_ROWS.clear()
    return pipeline.set_params(smote=SpySampler(smote), model__max_iter=2000)


def test_sampler_sees_only_the_training_part_of_each_fold(split: tuple) -> None:
    X_train, _, y_train, _ = split
    cv = StratifiedKFold(n_splits=experiments.CV_FOLDS, shuffle=True, random_state=SEED)
    result = cross_validate(spied_pipeline(), X_train, y_train, cv=cv,
                            scoring="average_precision", return_indices=True)
    folds = zip(result["indices"]["train"], result["indices"]["test"], strict=True)
    assert len(SEEN_ROWS) == experiments.CV_FOLDS  # one resampling per fold
    for seen, (train_positions, validation_positions) in zip(SEEN_ROWS, folds, strict=True):
        assert set(seen) == set(X_train.index[train_positions])
        assert set(seen).isdisjoint(X_train.index[validation_positions])


def test_predicting_never_calls_the_sampler(split: tuple) -> None:
    X_train, X_test, y_train, _ = split
    pipeline = spied_pipeline().fit(X_train, y_train)
    assert len(SEEN_ROWS) == 1  # once, during fit
    scores = pipeline.predict_proba(X_test)[:, 1]
    assert len(SEEN_ROWS) == 1  # nothing more while predicting
    assert scores.shape == (len(X_test),)


def test_scaler_ignores_extreme_test_values(split: tuple) -> None:
    X_train, X_test, y_train, _ = split
    pipeline = experiments.build_pipeline("logreg", "baseline").fit(X_train, y_train)
    scaler = pipeline.named_steps["preprocess"].named_transformers_["v_columns"]["scale"]
    center, scale = scaler.center_.copy(), scaler.scale_.copy()
    pipeline.predict_proba(X_test * 1e6)  # absurd values in every test column
    assert np.array_equal(scaler.center_, center)
    assert np.array_equal(scaler.scale_, scale)


def test_train_and_test_share_no_row_and_keep_the_fraud_rate(split: tuple) -> None:
    X_train, X_test, y_train, y_test = split
    assert set(X_train.index).isdisjoint(X_test.index)
    # remove_duplicates ran first, so no test row has an identical copy in train either
    assert X_train.merge(X_test, how="inner").empty
    assert abs(y_train.mean() - y_test.mean()) < 0.003  # 0.3 percentage points


def test_second_test_run_needs_a_reason(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(experiments, "ARTIFACTS_DIR", tmp_path)
    metrics = tmp_path / "sample" / "metrics"
    (metrics / "cells").mkdir(parents=True)
    (metrics / "cells" / "logreg__baseline.json").write_text(json.dumps({"model": "logreg"}))
    (metrics / evaluation.GUARD_FILE).write_text(json.dumps({"openings": []}))
    with pytest.raises(RuntimeError, match="rerun-reason"):
        evaluation.evaluate_test_set(source="sample")
