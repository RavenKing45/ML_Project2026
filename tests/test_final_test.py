"""Tests for fraudstudy.final_test: the single test-set evaluation, its guard, and the temporal
sensitivity check.

Synthetic sample only. A small quick run (3 cells) is shared by the module; the guard is
exercised on the sample by passing an explicit guard file (sample runs are exempt by default).
The bootstrap is cut to 50 resamples here to keep the suite fast; tests/test_metrics.py tests
the bootstrap itself.
"""

import copy
import importlib.util
import shutil
from pathlib import Path
from types import ModuleType
from typing import Any

import joblib
import numpy as np
import pytest

from fraudstudy import experiment, final_test, metrics
from fraudstudy.config import resolve
from fraudstudy.contracts import ComparisonTable, CurvesFile
from fraudstudy.experiment import RUN_META_FILE, cell_paths, run_experiments
from fraudstudy.final_test import (
    COMPARISON_FILE,
    CURVES_FILE,
    DETAILS_FILE,
    GUARD_FILE,
    SAMPLE_EVAL_DIR,
    TEMPORAL_FILE,
    evaluate_test,
    load_test_split,
    open_test_set,
    temporal_check,
)
from fraudstudy.io import load_json, save_json
from fraudstudy.metrics import compute_metrics

CELLS = [("logreg", "baseline"), ("logreg", "smote"), ("lightgbm", "baseline")]


def _load_script(name: str) -> ModuleType:
    """Import ``scripts/<name>.py`` as a module (scripts are not a package)."""
    spec = importlib.util.spec_from_file_location(name, resolve(f"scripts/{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cfgs(test_cfgs: dict) -> dict:
    """test_cfgs (model_n_jobs capped, tests/conftest.py) with a 50-resample bootstrap."""
    experiment = copy.deepcopy(test_cfgs["experiment"])
    experiment["bootstrap"]["n_resamples"] = 50
    return {**test_cfgs, "experiment": experiment}


@pytest.fixture(scope="module")
def cells_run(tmp_path_factory: pytest.TempPathFactory, cfgs: dict) -> dict[str, Path]:
    """Quick-run 3 cells on the sample into a temporary folder (no test set involved)."""
    root = tmp_path_factory.mktemp("cells")
    results, models = root / "results", root / "models"
    for model, technique in CELLS:
        run_experiments(
            "sample", True, results, models, [model], [technique], cfgs=cfgs, log=lambda _: None
        )
    return {"results": results, "models": models}


@pytest.fixture
def workdir(cells_run: dict[str, Path], tmp_path: Path) -> dict[str, Path]:
    """A private copy of the run (each test may write guard / output files) and decisions.md."""
    results, models = tmp_path / "results", tmp_path / "models"
    shutil.copytree(cells_run["results"], results)
    shutil.copytree(cells_run["models"], models)
    decisions = tmp_path / "decisions.md"
    shutil.copy2(resolve("docs/decisions.md"), decisions)
    return {"results": results, "models": models, "decisions": decisions}


def _evaluate(w: dict[str, Path], cfgs: dict, **kwargs: Any) -> Any:
    return evaluate_test(
        "sample",
        w["results"],
        w["models"],
        decisions_path=w["decisions"],
        cfgs=cfgs,
        log=lambda _: None,
        **kwargs,
    )


# --- evaluation ---------------------------------------------------------------------------------


def test_sample_outputs_validate_against_contracts(workdir: dict[str, Path], cfgs: dict) -> None:
    report = _evaluate(workdir, cfgs)
    out = workdir["results"] / SAMPLE_EVAL_DIR
    table = ComparisonTable.model_validate(load_json(out / COMPARISON_FILE))
    curves = CurvesFile.model_validate(load_json(out / CURVES_FILE))
    assert table.dataset_source == "synthetic" and not table.is_example
    assert [(r.model, r.technique) for r in table.records] == CELLS
    assert {(c.model, c.technique) for c in curves.cells} == set(CELLS)
    assert report["evaluated"] == CELLS
    details = load_json(out / DETAILS_FILE)
    assert all("test_at_f2" in d for d in details["cells"])
    # sample runs are exempt from the guard and never touch the real results folder
    assert not (workdir["results"] / GUARD_FILE).exists()
    assert not (workdir["results"] / COMPARISON_FILE).exists()


def test_thresholds_come_from_cell_files_not_the_test_set(
    workdir: dict[str, Path], cfgs: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*_: Any, **__: Any) -> float:
        raise AssertionError("thresholds must not be recomputed during test evaluation")

    monkeypatch.setattr(metrics, "best_threshold", forbidden)
    monkeypatch.setattr(experiment, "best_threshold", forbidden)
    report = _evaluate(workdir, cfgs)
    split = load_test_split("sample", True, cfgs)
    cells = load_json(report["paths"]["details"])["cells"]
    details = {(d["model"], d["technique"]): d for d in cells}
    for rec in report["comparison"]["records"]:
        key = (rec["model"], rec["technique"])
        paths = cell_paths(*key, workdir["results"], workdir["models"])
        cell = load_json(paths["record"])
        assert rec["threshold_f1"] == cell["thresholds"]["f1"]
        assert rec["threshold_f2"] == cell["thresholds"]["f2"]
        scores = joblib.load(paths["pipeline"]).predict_proba(split["X_test"])[:, 1]
        y_test = split["y_test"]
        assert rec["test_at_tuned"] == compute_metrics(y_test, scores, cell["thresholds"]["f1"])
        assert rec["test_at_0_5"] == compute_metrics(y_test, scores, 0.5)
        f2 = compute_metrics(y_test, scores, cell["thresholds"]["f2"])
        assert details[key]["test_at_f2"] == f2
        assert rec["pr_auc_ci"][0] <= rec["test_at_tuned"]["pr_auc"] <= rec["pr_auc_ci"][1]
        assert rec["cv_pr_auc_mean"] == cell["cv_pr_auc_mean"]


def test_guard_blocks_second_run_without_reason(workdir: dict[str, Path], cfgs: dict) -> None:
    guard = workdir["results"] / GUARD_FILE
    before = workdir["decisions"].read_text(encoding="utf-8")
    _evaluate(workdir, cfgs, guard_path=guard)
    first = load_json(guard)
    assert len(first["openings"]) == 1 and first["openings"][0]["cells"][0] == "logreg__baseline"
    assert workdir["decisions"].read_text(encoding="utf-8") == before  # first opening: no row

    with pytest.raises(ValueError, match="--rerun-reason"):
        _evaluate(workdir, cfgs, guard_path=guard)
    with pytest.raises(ValueError):
        _evaluate(workdir, cfgs, guard_path=guard, rerun_reason="   ")
    assert len(load_json(guard)["openings"]) == 1  # refusals record nothing

    _evaluate(workdir, cfgs, guard_path=guard, rerun_reason="fixed a | bug\nin export")
    openings = load_json(guard)["openings"]
    assert len(openings) == 2 and openings[1]["reason"] == "fixed a / bug in export"
    added = workdir["decisions"].read_text(encoding="utf-8")[len(before.rstrip("\n")) :]
    assert added.count("\n| ") == 1 and "fixed a / bug in export" in added
    assert "Test set re-opened (evaluate_test" in added


def test_training_hash_mismatch_is_refused(workdir: dict[str, Path], cfgs: dict) -> None:
    meta_path = workdir["results"] / RUN_META_FILE
    meta = load_json(meta_path)
    meta["train_sha256"] = "0" * 64
    save_json(meta, meta_path)
    with pytest.raises(ValueError, match="run_meta"):
        _evaluate(workdir, cfgs, guard_path=workdir["results"] / GUARD_FILE)
    assert not (workdir["results"] / GUARD_FILE).exists()  # refused before opening


def test_cells_subset_keeps_earlier_records(workdir: dict[str, Path], cfgs: dict) -> None:
    _evaluate(workdir, cfgs)
    report = _evaluate(workdir, cfgs, cells=["logreg__smote"])
    assert report["evaluated"] == [("logreg", "smote")]
    assert {(r["model"], r["technique"]) for r in report["comparison"]["records"]} == set(CELLS)
    with pytest.raises(ValueError, match="random_forest__rus"):
        _evaluate(workdir, cfgs, cells=["random_forest__rus"])


def test_open_test_set_first_opening_needs_no_reason(tmp_path: Path) -> None:
    decisions = tmp_path / "decisions.md"
    decisions.write_text("| date | decision | reason | alternatives |\n", encoding="utf-8")
    guard = open_test_set(tmp_path / GUARD_FILE, CELLS, None, decisions)
    assert guard["openings"][0]["reason"] is None
    assert decisions.read_text(encoding="utf-8").count("\n") == 1


# --- temporal check -----------------------------------------------------------------------------


def test_temporal_check_is_labelled_and_uses_cell_thresholds(
    workdir: dict[str, Path], cfgs: dict
) -> None:
    result = temporal_check(
        "sample", workdir["results"], workdir["models"], cfgs=cfgs, log=lambda _: None
    )
    written = load_json(workdir["results"] / SAMPLE_EVAL_DIR / TEMPORAL_FILE)
    assert written == result
    assert result["is_sensitivity_analysis"] is True
    assert "SENSITIVITY ANALYSIS" in result["label"]
    assert [c["technique"] for c in result["cells"]] == ["baseline"]  # lightgbm cells only
    cell = load_json(cell_paths("lightgbm", "baseline", workdir["results"])["record"])
    assert result["cells"][0]["threshold_f1"] == cell["thresholds"]["f1"]
    split = result["split"]
    assert split["train"]["rows"] + split["test"]["rows"] == 5980
    assert np.isfinite(result["cells"][0]["temporal_at_tuned"]["pr_auc"])


def test_temporal_check_needs_the_test_evaluation_first(
    workdir: dict[str, Path], cfgs: dict
) -> None:
    guard = workdir["results"] / GUARD_FILE
    kwargs: dict[str, Any] = {
        "guard_path": guard,
        "decisions_path": workdir["decisions"],
        "cfgs": cfgs,
        "log": lambda _: None,
    }
    with pytest.raises(ValueError, match="evaluate_test.py first"):
        temporal_check("sample", workdir["results"], workdir["models"], **kwargs)
    _evaluate(workdir, cfgs, guard_path=guard)
    temporal_check("sample", workdir["results"], workdir["models"], **kwargs)
    assert [o["action"] for o in load_json(guard)["openings"]] == [
        "evaluate_test",
        "temporal_check",
    ]
    with pytest.raises(ValueError, match="already opened by temporal_check"):
        temporal_check("sample", workdir["results"], workdir["models"], **kwargs)


# --- scripts ------------------------------------------------------------------------------------


def test_scripts_run_on_sample(
    workdir: dict[str, Path], cfgs: dict, capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(final_test, "load_configs", lambda: cfgs)  # the small bootstrap
    common = [
        "--source",
        "sample",
        "--results-dir",
        str(workdir["results"]),
        "--models-dir",
        str(workdir["models"]),
        "--decisions",
        str(workdir["decisions"]),
    ]
    assert _load_script("evaluate_test").main(common) == 0
    out = capsys.readouterr().out
    assert "logreg        smote" in out and "comparison" in out
    assert _load_script("run_temporal_check").main(common) == 0
    assert "SENSITIVITY ANALYSIS" in capsys.readouterr().out
    bad = _load_script("evaluate_test").main([*common, "--cells", "nope"])
    assert bad == 1
