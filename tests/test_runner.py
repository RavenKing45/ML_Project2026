"""Tests for the model builders, the equal-budget experiment runner and result merging.

Synthetic sample only. One quick run (2 models x 3 techniques) is shared by the module; it must
finish in under 90 s, re-running must skip it, and ``force`` must reproduce it exactly.
"""

import importlib.util
import inspect
import shutil
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from fraudstudy.errors import ConfigError, DataValidationError
from fraudstudy.evaluation import best_threshold
from fraudstudy.experiment import (
    Configs,
    build_search_space,
    get_protocol,
    is_cell_done,
    load_cell_record,
    load_training_data,
    merge_results,
    run_cell,
    run_experiments,
    space_size,
)
from fraudstudy.experiment.results import (
    CELLS_DIR,
    FOLD_COLUMNS,
    cell_paths,
    read_fold_scores,
    scientific_view,
    upsert_fold_scores,
)
from fraudstudy.experiment.runner import (
    fold_id,
    model_search_space,
    ratio_info,
    scalar_params,
    stratified_subsample,
)
from fraudstudy.io import load_json, save_json
from fraudstudy.models import MODEL_NAMES, get_model
from fraudstudy.paths import resolve
from fraudstudy.pipeline import build_pipeline

MODELS = ["logreg", "lightgbm"]
TECHNIQUES = ["baseline", "smote", "class_weight"]
FORCE_MODELS = ["logreg", "lightgbm"]
FORCE_TECHNIQUES = ["smote"]
TIME_LIMIT_S = 90.0


def _load_script(name: str) -> ModuleType:
    """Import ``scripts/<name>.py`` as a module (scripts are not a package)."""
    spec = importlib.util.spec_from_file_location(name, resolve(f"scripts/{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cfgs(test_cfgs: Configs) -> Configs:
    return test_cfgs  # model_n_jobs capped for the sample (tests/conftest.py)


@pytest.fixture(scope="module")
def training(cfgs: Configs) -> tuple[pd.DataFrame, pd.Series, str]:
    return load_training_data("sample", True, cfgs)


@pytest.fixture(scope="module")
def quick_run(tmp_path_factory: pytest.TempPathFactory, cfgs: Configs) -> dict[str, Any]:
    """One quick run of 2 models x 3 techniques on the sample, timed."""
    root = tmp_path_factory.mktemp("quick_run")
    results, models = root / "results", root / "models"
    lines: list[str] = []
    start = time.perf_counter()
    out = run_experiments(
        "sample", True, results, models, MODELS, TECHNIQUES, cfgs=cfgs, log=lines.append
    )
    return {
        "results": results,
        "models": models,
        "out": out,
        "lines": lines,
        "seconds": time.perf_counter() - start,
    }


# --- models ------------------------------------------------------------------------------------


def test_get_model_types_and_seed() -> None:
    lr = get_model("logreg", 7, C=0.1)
    rf = get_model("random_forest", 7, max_depth=8)
    lgbm = get_model("lightgbm", 7, num_leaves=15)
    assert isinstance(lr, LogisticRegression) and lr.solver == "lbfgs" and lr.C == 0.1
    assert isinstance(rf, RandomForestClassifier) and rf.n_jobs == -1 and rf.max_depth == 8
    assert isinstance(lgbm, LGBMClassifier) and lgbm.verbose == -1 and lgbm.num_leaves == 15
    assert all(m.random_state == 7 for m in (lr, rf, lgbm))
    assert get_model("random_forest", 7, n_jobs=2).n_jobs == 2
    assert get_model("logreg", 7, n_jobs=2).n_jobs is None  # lbfgs (binary) is single-threaded


def test_get_model_rejects_unknown_and_random_state() -> None:
    with pytest.raises(ConfigError, match="Unknown model 'svm'"):
        get_model("svm", 1)
    with pytest.raises(ConfigError, match="random_state"):
        get_model("logreg", 1, random_state=3)


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_pipeline_builds_every_model(model: str, cfgs: Configs) -> None:
    pipe = build_pipeline(model, "smote_rus", cfg=cfgs.dataset, exp_cfg=cfgs.experiment)
    assert [n for n, _ in pipe.steps] == ["preprocess", "smote", "rus", "model"]
    assert pipe.named_steps["model"].random_state == cfgs.experiment["seed"]


# --- search space and protocol ------------------------------------------------------------------


def test_model_space_is_prefixed_and_rejects_class_weight(cfgs: Configs) -> None:
    space = model_search_space("logreg", cfgs.models)
    assert space == {f"model__{k}": v for k, v in cfgs.models["logreg"].items()}
    with pytest.raises(ConfigError, match="class_weight"):
        model_search_space("logreg", {"logreg": {"C": [1], "class_weight": ["balanced"]}})
    with pytest.raises(ConfigError, match="non-empty list"):
        model_search_space("logreg", {"logreg": {"C": 1}})


def test_search_space_merges_model_space_into_each_hybrid_dict(cfgs: Configs) -> None:
    base = 0.0017
    space = build_search_space("logreg", "smote_rus", cfgs.models, cfgs.experiment, base)
    assert isinstance(space, list) and len(space) == 9  # 3 a x 3 b, all a < b on real data
    for d in space:
        assert set(d) == {
            "model__C",
            "model__max_iter",
            "smote__sampling_strategy",
            "rus__sampling_strategy",
        }
    n_c = len(cfgs.models["logreg"]["C"])
    assert space_size(space) == 9 * n_c
    baseline = build_search_space("logreg", "baseline", cfgs.models, cfgs.experiment, base)
    assert space_size(baseline) == n_c
    ros = build_search_space("random_forest", "ros", cfgs.models, cfgs.experiment, base)
    rf_size = int(np.prod([len(v) for v in cfgs.models["random_forest"].values()]))
    assert space_size(ros) == rf_size * len(cfgs.experiment["ratio_grid"])


def test_protocol_quick_and_full(cfgs: Configs) -> None:
    exp = cfgs.experiment
    quick, full = get_protocol(exp, True), get_protocol(exp, False)
    assert (quick.cv_folds, quick.cv_repeats, quick.n_iter) == (
        exp["quick"]["cv_folds"],
        exp["quick"]["cv_repeats"],
        exp["quick"]["n_iter"],
    )
    assert (full.cv_folds, full.cv_repeats, full.n_iter, full.train_fraction) == (
        exp["cv_folds"],
        exp["cv_repeats"],
        exp["tuning"]["n_iter"],
        1.0,
    )


def test_scalar_params_and_ratio_info() -> None:
    params = scalar_params({"model__class_weight": {0: 1.0, 1: 50.0}, "model__C": np.float64(1)})
    assert params == {"model__class_weight": 50.0, "model__C": 1.0}
    base = 0.01
    assert ratio_info("baseline", {}, base) == (None, None, None)
    assert ratio_info("rus", {"rus__sampling_strategy": 0.1}, base) == (0.1, None, None)
    pair = {"smote__sampling_strategy": 0.05, "rus__sampling_strategy": 0.5}
    assert ratio_info("smote_rus", pair, base) == (0.5, 0.05, None)
    assert ratio_info("class_weight", {"model__class_weight": 10.0}, base) == (
        pytest.approx(0.1),
        None,
        10.0,
    )
    assert ratio_info("class_weight", {"model__class_weight": "balanced"}, base) == (
        1.0,
        None,
        "balanced",
    )
    assert ratio_info("class_weight", {"model__scale_pos_weight": 50.0}, base)[0] == (
        pytest.approx(0.5)
    )


def test_fold_id_ignores_order() -> None:
    assert fold_id(pd.Index([3, 1, 2])) == fold_id(pd.Index([1, 2, 3]))
    assert fold_id(pd.Index([1, 2, 3])) != fold_id(pd.Index([1, 2, 4]))


def test_stratified_subsample_keeps_fraud_rate(training: tuple[Any, ...]) -> None:
    X, y, _ = training
    X_sub, y_sub = stratified_subsample(X, y, 0.5, 42)
    assert len(y_sub) == pytest.approx(len(y) / 2, abs=1)
    assert y_sub.mean() == pytest.approx(y.mean(), abs=0.002)
    assert stratified_subsample(X, y, 1.0, 42)[0] is X
    with pytest.raises(ConfigError):
        stratified_subsample(X, y, 0.0, 42)


def test_run_cell_accepts_training_data_only() -> None:
    params = list(inspect.signature(run_cell).parameters)
    assert params == ["model", "technique", "X_train", "y_train", "cfgs", "quick"]


def test_run_cell_rejects_too_few_frauds(training: tuple[Any, ...], cfgs: Configs) -> None:
    X, y, _ = training
    keep = (y == 0) | (y.cumsum() <= 8)  # 8 frauds: training folds hold ~5 <= k_neighbors
    with pytest.raises(DataValidationError, match="frauds"):
        run_cell("logreg", "smote", X[keep], y[keep], cfgs, quick=True)


def test_training_data_excludes_the_test_split(training: tuple[Any, ...], cfgs: Configs) -> None:
    X, y, sha = training
    expected_train = round(5980 * (1 - cfgs.experiment["test_size"]))  # 6000 rows - 20 dupes
    assert len(X) == len(y) == expected_train
    assert len(sha) == 64


# --- quick run ----------------------------------------------------------------------------------


def test_quick_run_is_fast_and_writes_valid_files(
    quick_run: dict[str, Any], training: tuple[Any, ...]
) -> None:
    X, y, _ = training
    assert quick_run["seconds"] < TIME_LIMIT_S
    assert [s for _, s in quick_run["out"]] == ["ran"] * 6
    results, models = quick_run["results"], quick_run["models"]
    for m in MODELS:
        for t in TECHNIQUES:
            paths = cell_paths(m, t, results, models)
            rec = load_cell_record(paths["record"])
            assert (rec.model, rec.technique, rec.source, rec.quick) == (m, t, "sample", True)
            oof = np.load(paths["oof"])
            assert oof.shape == (len(y),) and np.all((oof >= 0) & (oof <= 1))
            assert rec.thresholds == {
                "f1": best_threshold(y, oof, "f1"),
                "f2": best_threshold(y, oof, "f2"),
            }
            pipe = joblib.load(paths["pipeline"])
            assert pipe.predict_proba(X.iloc[:5]).shape == (5, 2)
            assert len(rec.fold_scores) == rec.cv_folds * rec.cv_repeats
            assert 0 < rec.cv_pr_auc_mean <= 1 and rec.cv_pr_auc_std >= 0
    folds = read_fold_scores(results / "fold_scores.csv")
    assert list(folds.columns) == list(FOLD_COLUMNS)
    assert not folds.duplicated(subset=["model", "technique", "repeat", "fold"]).any()
    meta = load_json(results / "run_meta.json")
    for key in (
        "configs",
        "git_commit",
        "library_versions",
        "data_sha256",
        "started_at",
        "finished_at",
    ):
        assert key in meta
    assert all(line.endswith(" s") for line in quick_run["lines"][1:])  # elapsed time per cell


def test_equal_budget_across_cells(quick_run: dict[str, Any], cfgs: Configs) -> None:
    budget = get_protocol(cfgs.experiment, True).n_iter
    for rec, _ in quick_run["out"]:
        assert rec.n_iter_budget == budget
        assert rec.n_candidates_evaluated == min(budget, rec.search_space_size)
        assert len(rec.candidates) == rec.n_candidates_evaluated


def test_fold_assignments_identical_across_cells(quick_run: dict[str, Any]) -> None:
    folds = read_fold_scores(quick_run["results"] / "fold_scores.csv")
    sets = {
        cell: frozenset(zip(g["repeat"], g["fold"], g["fold_id"], strict=True))
        for cell, g in folds.groupby(["model", "technique"])
    }
    assert len(sets) == 6
    assert len(set(sets.values())) == 1
    assert len(next(iter(sets.values()))) == len(set(folds["fold_id"]))  # folds differ


def test_resampling_audit_in_record(quick_run: dict[str, Any]) -> None:
    recs = {(r.model, r.technique): r for r, _ in quick_run["out"]}
    smote = recs[("logreg", "smote")]
    assert smote.n_train_after_resampling.fraud > smote.n_train_frauds
    assert smote.n_synthetic > 0
    assert smote.chosen_ratio is not None
    for t in ("baseline", "class_weight"):
        r = recs[("lightgbm", t)]
        assert r.n_train_after_resampling.fraud == r.n_train_frauds
    assert recs[("logreg", "baseline")].chosen_ratio is None
    assert recs[("lightgbm", "class_weight")].fraud_weight is not None


def test_rerun_skips_and_force_reproduces(quick_run: dict[str, Any], cfgs: Configs) -> None:
    results, models = quick_run["results"], quick_run["models"]
    before = {(r.model, r.technique): r for r, _ in quick_run["out"]}
    oof_before = {k: np.load(cell_paths(*k, results)["oof"]) for k in before}
    n_rows = len(read_fold_scores(results / "fold_scores.csv"))

    again = run_experiments(
        "sample", True, results, models, MODELS, TECHNIQUES, cfgs=cfgs, log=lambda _: None
    )
    assert [s for _, s in again] == ["skipped"] * 6

    forced = run_experiments(
        "sample",
        True,
        results,
        models,
        FORCE_MODELS,
        FORCE_TECHNIQUES,
        force=True,
        cfgs=cfgs,
        log=lambda _: None,
    )
    assert [s for _, s in forced] == ["ran"] * len(forced)
    for rec, _ in forced:
        key = (rec.model, rec.technique)
        assert scientific_view(rec) == scientific_view(before[key])
        np.testing.assert_array_equal(np.load(cell_paths(*key, results)["oof"]), oof_before[key])
    assert len(read_fold_scores(results / "fold_scores.csv")) == n_rows  # replaced, not appended


def test_cell_from_other_mode_is_not_done(quick_run: dict[str, Any]) -> None:
    results = quick_run["results"]
    assert is_cell_done("logreg", "baseline", results, quick=True, source="sample", seed=42)
    assert not is_cell_done("logreg", "baseline", results, quick=False, source="sample", seed=42)
    assert not is_cell_done("logreg", "baseline", results, quick=True, source="raw", seed=42)
    assert not is_cell_done("logreg", "rus", results, quick=True, source="sample", seed=42)


def test_upsert_replaces_rows_of_the_same_cell(tmp_path: Path) -> None:
    def rows(model: str, value: float) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "model": model,
                    "technique": "baseline",
                    "repeat": 0,
                    "fold": f,
                    "fold_id": "x",
                    "pr_auc": value,
                    "roc_auc": value,
                }
                for f in range(2)
            ]
        )

    path = tmp_path / "fold_scores.csv"
    upsert_fold_scores(rows("logreg", 0.1), path)
    upsert_fold_scores(rows("lightgbm", 0.2), path)
    upsert_fold_scores(rows("logreg", 0.3), path)
    df = read_fold_scores(path)
    assert len(df) == 4
    assert set(df.loc[df["model"] == "logreg", "pr_auc"]) == {0.3}


# --- merging ------------------------------------------------------------------------------------


def _split_copy(src: Path, dest: Path, models: list[str]) -> Path:
    """Copy the cells of ``models`` from results folder ``src`` to ``dest`` (a "machine")."""
    (dest / CELLS_DIR).mkdir(parents=True)
    for path in (src / CELLS_DIR).iterdir():
        if path.name.split("__")[0] in models:
            shutil.copy2(path, dest / CELLS_DIR / path.name)
    return dest


def test_merge_combines_machines_and_reports_missing(
    quick_run: dict[str, Any], tmp_path: Path, cfgs: Configs
) -> None:
    src = quick_run["results"]
    a = _split_copy(src, tmp_path / "a", ["logreg"])
    b = _split_copy(src, tmp_path / "b", ["lightgbm"])
    c = _split_copy(src, tmp_path / "c", ["lightgbm"])  # same cells again
    out = tmp_path / "merged"
    exp = cfgs.experiment
    report = merge_results([a, b, c], out, list(exp["models"]), list(exp["techniques"]))
    assert len(report.cells) == 6
    assert len(report.missing) == 18 - 6 and ("random_forest", "smote") in report.missing
    assert sorted(report.duplicates) == sorted(("lightgbm", t) for t in TECHNIQUES)
    assert report.conflicts == []
    summary = load_json(out / "cv_summary.json")
    assert summary["n_cells"] == 6 and len(summary["missing"]) == 12
    merged = read_fold_scores(out / "fold_scores.csv")
    original = read_fold_scores(src / "fold_scores.csv")
    key = ["model", "technique", "repeat", "fold"]
    pd.testing.assert_frame_equal(
        merged.sort_values(key).reset_index(drop=True),
        original.sort_values(key).reset_index(drop=True),
    )
    assert len(list((out / CELLS_DIR).glob("*_oof.npy"))) == 6


def test_merge_rejects_incomparable_cells(quick_run: dict[str, Any], tmp_path: Path) -> None:
    a = _split_copy(quick_run["results"], tmp_path / "a", ["logreg"])
    b = _split_copy(quick_run["results"], tmp_path / "b", ["lightgbm"])
    path = b / CELLS_DIR / "lightgbm__baseline.json"
    data = load_json(path)
    data["train_sha256"] = "0" * 64
    save_json(data, path)
    with pytest.raises(DataValidationError, match="train_sha256"):
        merge_results([a, b], tmp_path / "out", MODELS, TECHNIQUES)
    with pytest.raises(DataValidationError, match="No cell records"):
        merge_results([tmp_path / "empty"], tmp_path / "out", MODELS, TECHNIQUES)


# --- scripts ------------------------------------------------------------------------------------


def test_scripts_resume_and_merge(
    quick_run: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results, models = quick_run["results"], quick_run["models"]
    run = _load_script("run_experiments")
    code = run.main(
        [
            "--source",
            "sample",
            "--quick",
            "--models",
            "logreg",
            "--techniques",
            "baseline",
            "--results-dir",
            str(results),
            "--models-dir",
            str(models),
        ]
    )
    assert code == 0
    assert "skipped" in capsys.readouterr().out

    merge = _load_script("merge_results")
    code = merge.main(["--inputs", str(results), "--output", str(tmp_path / "merged")])
    assert code == 2  # 12 of the 18 cells are missing
    out = capsys.readouterr().out
    assert "merged     6 of 18" in out and "random_forest__baseline" in out
