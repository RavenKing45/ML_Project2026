"""Tests for fraudstudy.sampling, fraudstudy.audit, the techniques in build_pipeline and
scripts/run_resampling_audit.py (synthetic sample only).

Some tests call ``fit_resample`` directly to inspect resampled rows. They do so on the
TRAINING split only; library code only ever runs samplers inside a pipeline.
"""

import importlib.util
import logging
from typing import Any

import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from fraudstudy.audit import ResamplingAudit, resampling_audit, row_hashes
from fraudstudy.config import get_dataset_config, get_experiment_config, get_models_config
from fraudstudy.contracts import ResamplingAudit as ResamplingAuditFile
from fraudstudy.data import Split, deduplicate, load_dataset, split_dataset
from fraudstudy.errors import ConfigError, DataValidationError
from fraudstudy.features import build_preprocessor
from fraudstudy.io import load_json
from fraudstudy.paths import PROJECT_ROOT
from fraudstudy.pipeline import build_pipeline, without_model
from fraudstudy.sampling import (
    TECHNIQUES,
    base_ratio,
    build_sampler_steps,
    class_weight_model_params,
    fraud_weight,
    technique_search_space,
    valid_ratios,
)

RATIO = 0.1  # a grid ratio valid for the sample (base ratio ~0.015)
REL_TOL = 0.01

_spec = importlib.util.spec_from_file_location(
    "run_resampling_audit", PROJECT_ROOT / "scripts" / "run_resampling_audit.py"
)
assert _spec is not None and _spec.loader is not None
_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_script)


@pytest.fixture(scope="module")
def cfg() -> dict[str, Any]:
    return get_dataset_config()


@pytest.fixture(scope="module")
def exp_cfg() -> dict[str, Any]:
    return {**get_experiment_config(), "use_hour_feature": False}


@pytest.fixture(scope="module")
def split(cfg: dict[str, Any], exp_cfg: dict[str, Any]) -> Split:
    deduped = deduplicate(load_dataset("sample", cfg=cfg))[0]
    return split_dataset(deduped, cfg, exp_cfg)


@pytest.fixture(scope="module")
def base(split: Split) -> float:
    return base_ratio(split.y_train)


@pytest.fixture(scope="module")
def train_pre(
    split: Split, cfg: dict[str, Any], exp_cfg: dict[str, Any]
) -> tuple[pd.DataFrame, pd.Series]:
    """Preprocessed TRAINING rows (the space the samplers work in)."""
    pre = build_preprocessor(cfg, exp_cfg)
    return pre.fit_transform(split.X_train), split.y_train


def _logreg_params() -> dict[str, Any]:
    return {"max_iter": get_models_config()["logreg"]["max_iter"][0]}


def _ratio(y: pd.Series | np.ndarray) -> float:
    labels = np.asarray(y)
    return float((labels == 1).sum() / (labels == 0).sum())


def _audit(
    technique: str, params: dict[str, Any], split: Split, cfg: dict[str, Any], exp_cfg: dict
) -> ResamplingAudit:
    pipe = build_pipeline("logreg", technique, None, params, cfg, exp_cfg)
    return resampling_audit(without_model(pipe), split.X_train, split.y_train)


# --- build_sampler_steps ------------------------------------------------------------------------


def test_sampler_steps_names_types_and_seed() -> None:
    steps = {t: build_sampler_steps(t, {}, 7, 3) for t in TECHNIQUES}
    assert steps["baseline"] == [] and steps["class_weight"] == []
    assert [(n, type(s)) for n, s in steps["ros"]] == [("ros", RandomOverSampler)]
    assert [(n, type(s)) for n, s in steps["rus"]] == [("rus", RandomUnderSampler)]
    assert [(n, type(s)) for n, s in steps["smote"]] == [("smote", SMOTE)]
    assert [(n, type(s)) for n, s in steps["smote_rus"]] == [
        ("smote", SMOTE),
        ("rus", RandomUnderSampler),
    ]
    for technique in ("ros", "rus", "smote", "smote_rus"):
        for _, sampler in steps[technique]:
            assert sampler.random_state == 7
    assert steps["smote"][0][1].k_neighbors == 3
    assert steps["smote_rus"][0][1].k_neighbors == 3


def test_sampler_steps_pass_ratios() -> None:
    ((_, ros),) = build_sampler_steps("ros", {"sampling_strategy": 0.1}, 0, 5)
    assert ros.sampling_strategy == 0.1
    smote, rus = build_sampler_steps(
        "smote_rus", {"smote_sampling_strategy": 0.05, "rus_sampling_strategy": 0.5}, 0, 5
    )
    assert smote[1].sampling_strategy == 0.05 and rus[1].sampling_strategy == 0.5
    assert build_sampler_steps("class_weight", {"fraud_weight": 3.0}, 0, 5) == []


@pytest.mark.parametrize(
    ("technique", "params", "match"),
    [
        ("baseline", {"ratio": 0.1}, "no parameters"),
        ("ros", {"ratio": 0.1}, "unknown parameters ratio"),
        ("smote", {"sampling_strategy": 0.0}, r"ratio in \(0, 1\]"),
        ("rus", {"sampling_strategy": 1.5}, r"ratio in \(0, 1\]"),
        ("rus", {"sampling_strategy": "0.1"}, r"ratio in \(0, 1\]"),
        (
            "smote_rus",
            {"smote_sampling_strategy": 0.5, "rus_sampling_strategy": 0.1},
            "a < rus ratio b",
        ),
        ("adasyn", {}, "technique 'adasyn'"),
    ],
)
def test_sampler_steps_reject_bad_input(technique: str, params: dict, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        build_sampler_steps(technique, params, 0, 5)


# --- ratios and weights -------------------------------------------------------------------------


def test_base_ratio(split: Split) -> None:
    y = split.y_train
    assert base_ratio(y) == pytest.approx((y == 1).sum() / (y == 0).sum())
    with pytest.raises(DataValidationError, match="both classes"):
        base_ratio(np.zeros(5))


def test_valid_ratios_drops_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="fraudstudy.sampling"):
        kept = valid_ratios([0.005, 0.01, 0.0105, 0.0106, 0.1], 0.01)
    assert kept == [0.0106, 0.1]  # 0.0105 == 0.01 * 1.05 is not strictly greater
    assert "Dropped ratios [0.005, 0.01, 0.0105]" in caplog.text
    with pytest.raises(ConfigError, match="ratio_grid"):
        valid_ratios([0.001], 0.01)


def test_fraud_weight_and_model_params() -> None:
    assert fraud_weight(0.1, 0.002) == pytest.approx(50.0)
    assert class_weight_model_params("logreg", 50.0) == {"class_weight": {0: 1.0, 1: 50.0}}
    assert class_weight_model_params("random_forest", "balanced") == {"class_weight": "balanced"}
    assert class_weight_model_params("lightgbm", 50.0) == {"scale_pos_weight": 50.0}
    with pytest.raises(ConfigError, match="no 'balanced'"):
        class_weight_model_params("lightgbm", "balanced")
    with pytest.raises(ConfigError, match="positive number"):
        class_weight_model_params("logreg", -1.0)
    with pytest.raises(ConfigError, match="svm"):
        class_weight_model_params("svm", 1.0)


# --- technique_search_space ---------------------------------------------------------------------


def test_search_space_single_samplers_share_the_grid(exp_cfg: dict[str, Any], base: float) -> None:
    expected = valid_ratios(list(exp_cfg["ratio_grid"]), base)
    assert expected == [r for r in exp_cfg["ratio_grid"] if r > base * 1.05]
    assert technique_search_space("baseline", exp_cfg, base) == {}
    for technique in ("ros", "rus", "smote"):
        space = technique_search_space(technique, exp_cfg, base)
        assert space == {f"{technique}__sampling_strategy": expected}


def test_search_space_smote_rus_pairs(exp_cfg: dict[str, Any], base: float) -> None:
    space = technique_search_space("smote_rus", exp_cfg, base)
    assert isinstance(space, list) and space
    pairs = [(d["smote__sampling_strategy"], d["rus__sampling_strategy"]) for d in space]
    hybrid = exp_cfg["hybrid"]
    expected = [
        ([a], [b])
        for a in hybrid["smote_ratios"]
        for b in hybrid["rus_ratios"]
        if a < b and a > base * 1.05 and b > base * 1.05
    ]
    assert pairs == expected
    # On the real data (base ~0.0017) every configured pair survives.
    assert len(technique_search_space("smote_rus", exp_cfg, 0.0017)) == 9


def test_search_space_class_weight(exp_cfg: dict[str, Any], base: float) -> None:
    ratios = valid_ratios(list(exp_cfg["ratio_grid"]), base)
    space = technique_search_space("class_weight", exp_cfg, base)
    assert isinstance(space, dict)
    values = space["model__class_weight"]
    assert values[-1] == "balanced"
    assert [v[1] for v in values[:-1]] == pytest.approx([r / base for r in ratios])
    assert all(v[0] == 1.0 for v in values[:-1])
    lgbm = technique_search_space("class_weight", exp_cfg, base, model_name="lightgbm")
    assert lgbm == {"model__scale_pos_weight": pytest.approx([r / base for r in ratios])}
    with pytest.raises(ConfigError, match="Unknown technique"):
        technique_search_space("adasyn", exp_cfg, base)


def test_search_spaces_are_accepted_by_randomized_search(
    split: Split, cfg: dict[str, Any], exp_cfg: dict[str, Any], base: float
) -> None:
    for technique in ("smote_rus", "class_weight"):
        pipe = build_pipeline("logreg", technique, _logreg_params(), cfg=cfg, exp_cfg=exp_cfg)
        search = RandomizedSearchCV(
            pipe,
            technique_search_space(technique, exp_cfg, base),
            n_iter=2,
            scoring="average_precision",
            cv=StratifiedKFold(2, shuffle=True, random_state=0),
            random_state=0,
            error_score="raise",
        )
        search.fit(split.X_train, split.y_train)
        assert 0.0 < search.best_score_ <= 1.0


# --- build_pipeline -----------------------------------------------------------------------------


EXPECTED_STEPS = {
    "baseline": ["preprocess", "model"],
    "ros": ["preprocess", "ros", "model"],
    "rus": ["preprocess", "rus", "model"],
    "smote": ["preprocess", "smote", "model"],
    "smote_rus": ["preprocess", "smote", "rus", "model"],
    "class_weight": ["preprocess", "model"],
}


@pytest.mark.parametrize("technique", TECHNIQUES)
def test_build_pipeline_every_technique_with_logreg(
    technique: str, split: Split, cfg: dict[str, Any], exp_cfg: dict[str, Any]
) -> None:
    pipe = build_pipeline("logreg", technique, _logreg_params(), cfg=cfg, exp_cfg=exp_cfg)
    assert isinstance(pipe, ImbPipeline)
    assert [name for name, _ in pipe.steps] == EXPECTED_STEPS[technique]
    if "smote" in pipe.named_steps:
        assert pipe.named_steps["smote"].k_neighbors == exp_cfg["smote_k_neighbors"]
    space = technique_search_space(technique, exp_cfg, base_ratio(split.y_train))
    first = space[0] if isinstance(space, list) else space
    pipe.set_params(**{k: v[0] for k, v in first.items()})
    pipe.fit(split.X_train, split.y_train)
    scores = pipe.predict_proba(split.X_test)
    assert scores.shape == (len(split.X_test), 2)
    assert np.all(np.isfinite(scores))


def test_build_pipeline_class_weight(cfg: dict[str, Any], exp_cfg: dict[str, Any]) -> None:
    pipe = build_pipeline("logreg", "class_weight", cfg=cfg, exp_cfg=exp_cfg)
    assert pipe.named_steps["model"].class_weight == "balanced"
    pipe = build_pipeline(
        "logreg", "class_weight", technique_params={"fraud_weight": 20.0}, cfg=cfg, exp_cfg=exp_cfg
    )
    assert pipe.named_steps["model"].class_weight == {0: 1.0, 1: 20.0}
    assert (
        build_pipeline("logreg", "smote", cfg=cfg, exp_cfg=exp_cfg)
        .named_steps["model"]
        .class_weight
        is None
    )
    with pytest.raises(ConfigError, match="class weighting is technique"):
        build_pipeline(
            "logreg", "smote", model_params={"class_weight": "balanced"}, cfg=cfg, exp_cfg=exp_cfg
        )


def test_without_model(cfg: dict[str, Any], exp_cfg: dict[str, Any]) -> None:
    pipe = build_pipeline("logreg", "smote_rus", cfg=cfg, exp_cfg=exp_cfg)
    assert [n for n, _ in without_model(pipe).steps] == ["preprocess", "smote", "rus"]
    with pytest.raises(ConfigError, match="last step is 'model'"):
        without_model(without_model(pipe))


# --- what each technique does to the training data ----------------------------------------------


def test_ros_hits_ratio_with_exact_duplicates(
    train_pre: tuple[pd.DataFrame, pd.Series], exp_cfg: dict[str, Any]
) -> None:
    X, y = train_pre
    # fit_resample on the TRAINING split only, to inspect the rows.
    X_res, y_res = RandomOverSampler(
        sampling_strategy=RATIO, random_state=exp_cfg["seed"]
    ).fit_resample(X, y)
    assert _ratio(y_res) == pytest.approx(RATIO, rel=REL_TOL)
    fraud_hashes = row_hashes(X_res[np.asarray(y_res) == 1])
    assert len(set(fraud_hashes)) < len(fraud_hashes)  # exact copies exist
    assert set(fraud_hashes) <= set(row_hashes(X[np.asarray(y) == 1]))  # and nothing new


def test_smote_rows_are_new_and_inside_the_fraud_range(
    train_pre: tuple[pd.DataFrame, pd.Series], exp_cfg: dict[str, Any]
) -> None:
    X, y = train_pre
    smote = SMOTE(
        sampling_strategy=RATIO,
        k_neighbors=exp_cfg["smote_k_neighbors"],
        random_state=exp_cfg["seed"],
    )
    # fit_resample on the TRAINING split only, to inspect the rows.
    X_res, y_res = smote.fit_resample(X, y)
    assert _ratio(y_res) == pytest.approx(RATIO, rel=REL_TOL)
    pd.testing.assert_frame_equal(
        X_res.iloc[: len(X)].reset_index(drop=True), X.reset_index(drop=True)
    )
    new = X_res.iloc[len(X) :]
    assert len(new) > 0 and (np.asarray(y_res)[len(X) :] == 1).all()
    assert not set(row_hashes(new)) & set(row_hashes(X))
    frauds = X[np.asarray(y) == 1]
    tol = 1e-9
    assert (new.to_numpy() >= frauds.min().to_numpy() - tol).all()
    assert (new.to_numpy() <= frauds.max().to_numpy() + tol).all()


def test_rus_keeps_only_original_rows(
    train_pre: tuple[pd.DataFrame, pd.Series], exp_cfg: dict[str, Any]
) -> None:
    X, y = train_pre
    rus = RandomUnderSampler(sampling_strategy=RATIO, random_state=exp_cfg["seed"])
    # fit_resample on the TRAINING split only, to inspect the rows.
    X_res, y_res = rus.fit_resample(X, y)
    assert _ratio(y_res) == pytest.approx(RATIO, rel=REL_TOL)
    assert (np.asarray(y_res) == 0).sum() < (np.asarray(y) == 0).sum()
    assert (np.asarray(y_res) == 1).sum() == (np.asarray(y) == 1).sum()
    assert X_res.index.is_unique and X_res.index.isin(X.index).all()
    pd.testing.assert_frame_equal(X_res, X.loc[X_res.index])
    pd.testing.assert_series_equal(
        pd.Series(np.asarray(y_res), index=X_res.index), y.loc[X_res.index], check_names=False
    )


def test_audit_ros_rus_smote(split: Split, cfg: dict[str, Any], exp_cfg: dict[str, Any]) -> None:
    before_legit = int((split.y_train == 0).sum())
    before_fraud = int((split.y_train == 1).sum())
    ros = _audit("ros", {"sampling_strategy": RATIO}, split, cfg, exp_cfg)
    assert (ros.legit_before, ros.fraud_before) == (before_legit, before_fraud)
    assert ros.ratio_after == pytest.approx(RATIO, rel=REL_TOL)
    assert ros.fraud_unique_after == before_fraud and ros.n_synthetic == 0
    assert ros.fraud_duplicates_after == ros.fraud_after - before_fraud > 0

    rus = _audit("rus", {"sampling_strategy": RATIO}, split, cfg, exp_cfg)
    assert rus.legit_after < before_legit and rus.fraud_after == before_fraud
    assert rus.n_synthetic == 0 and rus.fraud_duplicates_after == 0

    smote = _audit("smote", {"sampling_strategy": RATIO}, split, cfg, exp_cfg)
    assert smote.ratio_after == pytest.approx(RATIO, rel=REL_TOL)
    assert smote.n_synthetic == smote.fraud_after - before_fraud > 0
    assert smote.fraud_duplicates_after == 0


@pytest.mark.parametrize("b", [0.1, 0.5, 1.0])
def test_smote_rus_final_ratio_is_b(
    b: float, split: Split, cfg: dict[str, Any], exp_cfg: dict[str, Any]
) -> None:
    a = 0.05
    result = _audit(
        "smote_rus",
        {"smote_sampling_strategy": a, "rus_sampling_strategy": b},
        split,
        cfg,
        exp_cfg,
    )
    assert result.ratio_after == pytest.approx(b, rel=REL_TOL)
    # SMOTE first brought fraud to a * (all legit); RUS then only removed legit rows.
    assert result.fraud_after == pytest.approx(a * result.legit_before, abs=1)
    assert result.n_synthetic == result.fraud_after - result.fraud_before


@pytest.mark.parametrize("technique", ["baseline", "class_weight"])
def test_no_sampler_techniques_leave_data_unchanged(
    technique: str,
    split: Split,
    cfg: dict[str, Any],
    exp_cfg: dict[str, Any],
    train_pre: tuple[pd.DataFrame, pd.Series],
) -> None:
    params = {"fraud_weight": 10.0} if technique == "class_weight" else {}
    pipe = without_model(build_pipeline("logreg", technique, None, params, cfg, exp_cfg))
    result = resampling_audit(pipe, split.X_train, split.y_train)
    assert (result.legit_after, result.fraud_after) == (result.legit_before, result.fraud_before)
    assert result.n_synthetic == 0 and result.fraud_duplicates_after == 0
    assert not hasattr(pipe.named_steps["preprocess"], "transformers_")  # argument not fitted
    out = pipe.fit_transform(split.X_train, split.y_train)
    pd.testing.assert_frame_equal(out, train_pre[0])


def test_audit_rejects_bad_input(split: Split, cfg: dict[str, Any], exp_cfg: dict) -> None:
    pipe = build_pipeline("logreg", "ros", cfg=cfg, exp_cfg=exp_cfg)
    with pytest.raises(ConfigError, match="without its model"):
        resampling_audit(pipe, split.X_train, split.y_train)
    with pytest.raises(DataValidationError, match="rows"):
        resampling_audit(without_model(pipe), split.X_train, split.y_train.iloc[:-1])


# --- scripts/run_resampling_audit.py ------------------------------------------------------------


def test_audit_settings_use_the_shared_grid(exp_cfg: dict[str, Any], base: float) -> None:
    settings = _script.audit_settings(exp_cfg, base)
    ratios = valid_ratios(list(exp_cfg["ratio_grid"]), base)
    by_technique: dict[str, list[float | None]] = {}
    for s in settings:
        by_technique.setdefault(s.technique, []).append(s.ratio)
    assert by_technique["baseline"] == [None]
    for technique in ("ros", "rus", "smote", "class_weight"):
        assert by_technique[technique] == ratios
    assert set(by_technique["smote_rus"]) <= set(ratios)


def test_run_audit_sample(split: Split, cfg: dict[str, Any], exp_cfg: dict[str, Any]) -> None:
    audit = _script.run_audit(split.X_train, split.y_train, cfg, exp_cfg, 3)
    assert audit.all_checks_passed and audit.test_rows_resampled == 0
    assert all(e.validation_untouched for e in audit.entries)
    assert audit.dropped_ratios == [
        r for r in exp_cfg["ratio_grid"] if r <= audit.base_ratio * 1.05
    ]
    for e in audit.entries:
        assert e.validation.legit + e.validation.fraud + e.train_before.legit + (
            e.train_before.fraud
        ) == len(split.y_train)
        if e.technique in ("baseline", "class_weight"):
            assert e.train_after == e.train_before
        elif e.ratio is not None:
            assert e.train_after.fraud / e.train_after.legit == pytest.approx(e.ratio, rel=REL_TOL)
    ResamplingAuditFile.model_validate(audit.model_dump(mode="json"))


def test_main_quick_sample_has_too_few_frauds(
    tmp_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """10% of the sample leaves fewer frauds per fold than SMOTE needs: a clear error."""
    assert _script.main(["--source", "sample", "--quick", "--results-dir", str(tmp_path)]) == 1
    assert "smote_k_neighbors" in capsys.readouterr().err
    assert not (tmp_path / "resampling_audit.json").exists()


def test_run_audit_needs_enough_frauds(
    split: Split, cfg: dict[str, Any], exp_cfg: dict[str, Any]
) -> None:
    y = split.y_train.copy()
    y[:] = 0
    y.iloc[:2] = 1
    with pytest.raises(DataValidationError, match="fewer than 5 CV folds"):
        _script.run_audit(split.X_train, y, cfg, exp_cfg, 5)


def test_main_writes_valid_audit(tmp_path: Any, capsys: pytest.CaptureFixture[str]) -> None:
    code = _script.main(["--source", "sample", "--results-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0 and "all passed" in out
    audit = ResamplingAuditFile.model_validate(load_json(tmp_path / "resampling_audit.json"))
    assert audit.is_example is False and audit.all_checks_passed
    assert {e.technique for e in audit.entries} == set(TECHNIQUES)
