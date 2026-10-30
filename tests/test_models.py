"""Checks that every model in MODELS trains and scores, and that class weighting for LightGBM
uses the same ratios as the samplers (fake sample only).

The models are made small (20 trees) so the whole file runs in a few seconds.
"""

import pytest

from fraudstudy import experiments
from fraudstudy.data import load_data, remove_duplicates, split_data

SMALL = {  # a fast version of each model; the real settings come from the search
    "logreg": {"model__max_iter": 2000},
    "random_forest": {"model__n_estimators": 20},
    "lightgbm": {"model__n_estimators": 20},
}


@pytest.fixture(scope="module")
def split() -> tuple:
    df, _, _ = remove_duplicates(load_data("sample"))
    return split_data(df)


def test_every_model_fits_and_scores(split: tuple) -> None:
    X_train, X_test, y_train, _ = split
    for name in experiments.MODELS:
        pipeline = experiments.build_pipeline(name, "baseline").set_params(**SMALL[name])
        scores = pipeline.fit(X_train, y_train).predict_proba(X_test)[:, 1]
        assert scores.shape == (len(X_test),), name
        assert ((scores >= 0) & (scores <= 1)).all(), name


def test_lightgbm_weights_frauds_by_the_ratio(split: tuple) -> None:
    _, _, y_train, _ = split
    base = (y_train == 1).sum() / (y_train == 0).sum()
    expected = [r / base for r in experiments.RATIO_GRID if r > base * 1.05]
    space = experiments.search_space("lightgbm", "class_weight", y_train)
    assert space["model__scale_pos_weight"] == expected
