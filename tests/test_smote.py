"""Checks of what the resamplers produce: our from-scratch SMOTE (model_build/) and the imblearn
samplers inside the study's pipeline."""

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors
from smote_from_scratch import SMOTE as ScratchSMOTE

from fraudstudy.data import load_data, remove_duplicates, split_data
from fraudstudy.experiments import build_pipeline, counts_after_resampling

K = 5


@pytest.fixture(scope="module")
def train() -> tuple:
    df, _, _ = remove_duplicates(load_data("sample"))
    X_train, _, y_train, _ = split_data(df)
    return X_train, y_train


def test_scratch_smote_balances_keeps_originals_and_interpolates() -> None:
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (60, 3)), rng.normal(3, 1, (12, 3))])
    y = np.array([0] * 60 + [1] * 12)
    X_new, y_new = ScratchSMOTE(k_neighbors=K, random_state=0).fit_resample(X, y)

    assert (y_new == 0).sum() == (y_new == 1).sum() == 60
    assert np.array_equal(X_new[: len(X)], X) and np.array_equal(y_new[: len(y)], y)

    minority = X[y == 1]
    # K + 1 because each point's nearest neighbour is itself
    _, neighbours = NearestNeighbors(n_neighbors=K + 1).fit(minority).kneighbors(minority)
    for point in X_new[len(X):]:
        on_a_segment = False
        for i in range(len(minority)):
            for j in neighbours[i][1:]:
                start, step = minority[i], minority[j] - minority[i]
                t = np.dot(point - start, step) / np.dot(step, step)
                if 0 <= t <= 1 and np.allclose(start + t * step, point):
                    on_a_segment = True
        assert on_a_segment, "a synthetic point is not between a fraud and one of its neighbours"


@pytest.mark.parametrize("technique", ["ros", "smote", "rus"])
@pytest.mark.parametrize("ratio", [0.05, 0.1, 0.5])
def test_sampler_reaches_its_ratio(train: tuple, technique: str, ratio: float) -> None:
    pipeline = build_pipeline("logreg", technique)
    pipeline.set_params(**{f"{technique}__sampling_strategy": ratio})
    counts = counts_after_resampling(pipeline, *train)
    assert counts["fraud"] / counts["legit"] == pytest.approx(ratio, rel=0.01)


def test_smote_then_rus_ends_at_the_rus_ratio(train: tuple) -> None:
    pipeline = build_pipeline("logreg", "smote_rus")
    pipeline.set_params(smote__sampling_strategy=0.05, rus__sampling_strategy=0.5)
    counts = counts_after_resampling(pipeline, *train)
    assert counts["fraud"] / counts["legit"] == pytest.approx(0.5, rel=0.01)


def test_class_weighting_leaves_the_data_unchanged(train: tuple) -> None:
    X_train, y_train = train
    pipeline = build_pipeline("logreg", "class_weight")
    assert [name for name, _ in pipeline.steps] == ["preprocess", "model"]
    counts = counts_after_resampling(pipeline, X_train, y_train)
    assert counts == {"legit": int((y_train == 0).sum()), "fraud": int((y_train == 1).sum())}
