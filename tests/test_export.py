"""Tests for the deploy export: manifest, deployed pipelines, calibrators, reference, presets and
demo batch (synthetic sample only).

The module quick-runs three cells into a temporary folder, evaluates them on the sample's test
split and exports them once. LightGBM's search space is cut to one point and the bootstrap to a
few resamples to keep the module fast; the export does not depend on either.
"""

import copy
import dataclasses
import importlib.util
import shutil
from pathlib import Path
from types import ModuleType
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from fraudstudy import config, export
from fraudstudy.config import resolve
from fraudstudy.contracts import Manifest, PresetsFile
from fraudstudy.evaluation import fit_isotonic
from fraudstudy.experiment import Configs, load_cell_record, run_experiments
from fraudstudy.experiment.results import cell_paths
from fraudstudy.experiment.test_eval import evaluate_test, open_test_set
from fraudstudy.explain import FeatureReference, compute_contributions, explain
from fraudstudy.export import (
    DEMO_BATCH_FILE,
    EXPORT_ACTION,
    FEATURE_REFERENCE_FILE,
    MANIFEST_FILE,
    PRESETS_FILE,
    ExportPaths,
    PresetRules,
    deploy_settings,
    export_artifacts,
    load_export_data,
    pick_presets,
    size_warnings,
)
from fraudstudy.io import load_json, sha256_file

DEFAULT = "lightgbm__smote_rus"
DEPLOY_CELLS = ["lightgbm__baseline", "lightgbm__smote_rus", "logreg__<technique>"]
EXPORTED = ["lightgbm__baseline", "lightgbm__smote_rus", "logreg__smote_rus"]
TINY_LIGHTGBM = {
    "n_estimators": [40],
    "learning_rate": [0.1],
    "num_leaves": [15],
    "min_child_samples": [20],
    "subsample": [1.0],
    "subsample_freq": [1],
    "colsample_bytree": [1.0],
}
N_CHECK_ROWS = 100


def _load_script(name: str) -> ModuleType:
    """Import ``scripts/<name>.py`` as a module (scripts are not a package)."""
    spec = importlib.util.spec_from_file_location(name, resolve(f"scripts/{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _noop(_: str) -> None:
    return None


@pytest.fixture(scope="module")
def cfgs(test_cfgs: Configs) -> Configs:
    """test_cfgs with three deploy cells, a one-point LightGBM space and a tiny bootstrap."""
    experiment = copy.deepcopy(test_cfgs.experiment)
    experiment["deploy"]["cells"] = list(DEPLOY_CELLS)
    experiment["bootstrap"]["n_resamples"] = 20
    models = {**copy.deepcopy(test_cfgs.models), "lightgbm": TINY_LIGHTGBM}
    return dataclasses.replace(test_cfgs, experiment=experiment, models=models)


def _out(root: Path) -> ExportPaths:
    return ExportPaths(root / "models_out", root / "demo_out", root / "demo_out" / DEMO_BATCH_FILE)


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory, cfgs: Configs) -> dict[str, Any]:
    """Quick-run, evaluate (sample) and export the three cells once."""
    root = tmp_path_factory.mktemp("export")
    results, models = root / "results", root / "models"
    for model, techniques in (("lightgbm", ["baseline", "smote_rus"]), ("logreg", ["smote_rus"])):
        run_experiments("sample", True, results, models, [model], techniques, cfgs=cfgs, log=_noop)
    decisions = root / "decisions.md"
    shutil.copy2(resolve("docs/decisions.md"), decisions)
    evaluate_test("sample", results, models, decisions_path=decisions, cfgs=cfgs, log=_noop)
    out = _out(root)
    report = export_artifacts(
        "sample", results, models, default_cell=DEFAULT, out=out, cfgs=cfgs, log=_noop
    )
    data = load_export_data("sample", True, cfgs, verify_hash=False)
    return {
        "root": root,
        "results": results,
        "models": models,
        "decisions": decisions,
        "out": out,
        "report": report,
        "data": data,
    }


@pytest.fixture
def workdir(run: dict[str, Any], tmp_path: Path) -> dict[str, Path]:
    """A private copy of the cells (a test may change files or write a guard)."""
    results, models = tmp_path / "results", tmp_path / "models"
    shutil.copytree(run["results"], results)
    shutil.copytree(run["models"], models)
    decisions = tmp_path / "decisions.md"
    shutil.copy2(run["decisions"], decisions)
    return {"results": results, "models": models, "decisions": decisions, "root": tmp_path}


# --- default cell and deploy settings ------------------------------------------------------------


def test_refuses_without_a_default_cell(run: dict[str, Any], cfgs: Configs) -> None:
    assert cfgs.experiment["deploy"]["default_cell"] is None
    with pytest.raises(ValueError, match="TEAM chooses it"):
        export_artifacts("sample", run["results"], run["models"], out=run["out"], cfgs=cfgs)


def test_deploy_settings_resolve_the_logreg_placeholder(cfgs: Configs) -> None:
    exp = cfgs.experiment
    assert deploy_settings(exp, DEFAULT).cells == EXPORTED
    assert deploy_settings(exp, "logreg__smote_rus").cells == EXPORTED
    assert deploy_settings(exp, "lightgbm__baseline").cells[-1] == "logreg__baseline"
    configured = copy.deepcopy(exp)
    configured["deploy"]["default_cell"] = DEFAULT
    assert deploy_settings(configured).default_cell == DEFAULT
    with pytest.raises(ValueError, match="disagrees"):
        deploy_settings(configured, "lightgbm__baseline")


@pytest.mark.parametrize(
    ("change", "default", "message"),
    [
        ({}, "random_forest__smote", "cannot be deployed"),
        ({"cells": ["lightgbm__baseline", "random_forest__baseline"]}, DEFAULT, "cannot be"),
        ({}, "lightgbm__ros", "must be one of deploy.cells"),
        ({}, "lightgbm__adasyn", "technique from experiment.yaml"),
        ({"default_threshold": "f3"}, DEFAULT, "f1 or f2"),
        ({"demo_batch": {"rows": 5, "frauds": 10}}, DEFAULT, "frauds <= rows"),
    ],
)
def test_deploy_settings_reject_bad_input(
    cfgs: Configs, change: dict[str, Any], default: str, message: str
) -> None:
    exp = copy.deepcopy(cfgs.experiment)
    exp["deploy"].update(change)
    with pytest.raises(ValueError, match=message):
        deploy_settings(exp, default)


def test_deploy_settings_check_the_input_range_percentiles(cfgs: Configs) -> None:
    exp = copy.deepcopy(cfgs.experiment)
    assert deploy_settings(exp, DEFAULT).input_range_percentiles == (0.1, 99.9)
    exp["deploy"]["input_range_percentiles"] = [99.9, 0.1]
    with pytest.raises(ValueError, match="Range percentiles"):
        deploy_settings(exp, DEFAULT)


def test_deploy_block_is_required(cfgs: Configs) -> None:
    exp = {k: v for k, v in cfgs.experiment.items() if k != "deploy"}
    with pytest.raises(ValueError, match="deploy"):
        deploy_settings(exp, DEFAULT)


# --- manifest and deployed files ------------------------------------------------------------------


def test_manifest_validates_and_sha256_match_files(run: dict[str, Any]) -> None:
    out: ExportPaths = run["out"]
    manifest = Manifest.model_validate(load_json(out.models / MANIFEST_FILE))
    assert not manifest.is_example and manifest.default_model_id == DEFAULT
    assert [m.id for m in manifest.models] == EXPORTED
    assert manifest.dataset.source == "synthetic" and manifest.quick_mode
    for m in manifest.models:
        record = load_cell_record(cell_paths(m.model, m.technique, run["results"])["record"])
        assert m.sha256 == sha256_file(out.models / m.file)
        assert m.calibrator_file is not None and m.has_calibrator
        assert m.calibrator_sha256 == sha256_file(out.models / m.calibrator_file)
        assert (m.threshold_f1, m.threshold_f2) == (
            record.thresholds["f1"],
            record.thresholds["f2"],
        )
        assert m.default_threshold == "f1" and m.data_sha256 == record.data_sha256
        assert m.features == [f"V{i}" for i in range(1, 29)] + ["Amount"]
        assert m.model_label and m.technique_label
        assert {"scikit-learn", "lightgbm", "imbalanced-learn"} <= set(m.library_versions)


def test_deployed_pipelines_score_identically_to_the_originals(run: dict[str, Any]) -> None:
    rows = run["data"].X_test.iloc[:N_CHECK_ROWS]
    manifest = Manifest.model_validate(load_json(run["out"].models / MANIFEST_FILE))
    for m in manifest.models:
        original = joblib.load(
            cell_paths(m.model, m.technique, run["results"], run["models"])["pipeline"]
        )
        deployed = joblib.load(run["out"].models / m.file)
        assert np.array_equal(original.predict_proba(rows), deployed.predict_proba(rows))


def test_calibrators_are_fitted_on_out_of_fold_scores(run: dict[str, Any]) -> None:
    grid = np.linspace(0.0, 1.0, 201)
    y_train = run["data"].y_train
    manifest = Manifest.model_validate(load_json(run["out"].models / MANIFEST_FILE))
    for m in manifest.models:
        oof = np.load(cell_paths(m.model, m.technique, run["results"])["oof"])
        calibrator = joblib.load(run["out"].models / str(m.calibrator_file))
        expected = fit_isotonic(oof, y_train).predict(grid)
        np.testing.assert_array_equal(calibrator.predict(grid), expected)
        assert np.all(np.diff(expected) >= 0) and 0.0 <= expected.min() <= expected.max() <= 1.0


def test_feature_reference_explains_every_deployed_model(run: dict[str, Any]) -> None:
    out: ExportPaths = run["out"]
    reference = FeatureReference.model_validate(load_json(out.models / FEATURE_REFERENCE_FILE))
    assert reference.train_sha256 == run["data"].train_sha256
    assert reference.n_legit == int((run["data"].y_train == 0).sum())
    rows = run["data"].X_test.iloc[:N_CHECK_ROWS]
    manifest = Manifest.model_validate(load_json(out.models / MANIFEST_FILE))
    for m in manifest.models:
        pipe = joblib.load(out.models / m.file)
        c = compute_contributions(pipe, rows, reference)
        np.testing.assert_allclose(c.base + c.contributions.sum(axis=1), c.score, atol=1e-6)
        assert len(explain(pipe, rows.iloc[[0]], reference).top_features) == 8


def test_feature_reference_has_training_ranges(run: dict[str, Any], cfgs: Configs) -> None:
    reference = FeatureReference.model_validate(
        load_json(run["out"].models / FEATURE_REFERENCE_FILE)
    )
    percentiles = tuple(cfgs.experiment["deploy"]["input_range_percentiles"])
    assert reference.range_percentiles == percentiles
    X_train = run["data"].X_train
    for f in reference.features:
        expected = np.percentile(X_train[f.raw_column].to_numpy(dtype=float), percentiles)
        assert f.train_range == pytest.approx(tuple(expected))


def test_report_lists_every_file_and_the_total_size(run: dict[str, Any]) -> None:
    report = run["report"]
    names = {p.name for p in report.sizes}
    assert {MANIFEST_FILE, FEATURE_REFERENCE_FILE, PRESETS_FILE, DEMO_BATCH_FILE} <= names
    assert {f"{c}.joblib" for c in EXPORTED} <= names
    assert report.total_mb == pytest.approx(sum(p.stat().st_size for p in report.sizes) / 2**20)
    assert report.warnings == []


# --- presets and demo batch -----------------------------------------------------------------------


def test_presets_are_real_test_rows_that_follow_the_rules(
    run: dict[str, Any], cfgs: Configs
) -> None:
    out: ExportPaths = run["out"]
    presets = PresetsFile.model_validate(load_json(out.demo / PRESETS_FILE)).presets
    manifest = Manifest.model_validate(load_json(out.models / MANIFEST_FILE))
    entry = next(m for m in manifest.models if m.id == DEFAULT)
    pipe, thr = joblib.load(out.models / entry.file), entry.threshold_f1
    rules = deploy_settings(cfgs.experiment, DEFAULT).presets
    test_rows = run["data"].test_rows
    columns = ["Time", *cfgs.dataset["pca_features"], "Amount"]
    ids = [p.id for p in presets] + run["report"].skipped_presets
    assert sorted(ids) == sorted(export.PRESET_TEXT)
    matched = []
    for p in presets:
        values = p.features.model_dump()
        hit = (test_rows[columns] == pd.Series(values)[columns]).all(axis=1)
        assert hit.sum() == 1, p.id  # an unchanged test-set row
        row = test_rows[hit]
        matched.append(row.index[0])
        assert int(row["Class"].iloc[0]) == p.expected_class
        score = pipe.predict_proba(row[columns[1:]])[0, 1]
        amount = values["Amount"]
        rule = {
            "legit_everyday": p.expected_class == 0 and 5 <= amount <= 50 and score < thr,
            "legit_large": p.expected_class == 0 and score < thr,
            "fraud_typical": p.expected_class == 1 and score >= thr,
            "fraud_small_amount": p.expected_class == 1 and amount <= rules.small_amount_max,
            "borderline": abs(score - thr) <= rules.borderline_window,
            "missed_fraud": p.expected_class == 1 and score < thr,
        }[p.id]
        assert rule, p.id
        assert DEFAULT in p.description and "Synthetic test-set payment" in p.description
    assert len(set(matched)) == len(matched)


def _handmade(n: int = 40) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Test-like rows where every preset has candidates, with their scores."""
    cfg = {
        "amount_column": "Amount",
        "target": "Class",
        "time_column": "Time",
        "pca_features": [f"V{i}" for i in range(1, 29)],
    }
    rng = np.random.default_rng(0)
    rows = pd.DataFrame(rng.normal(size=(n, 28)), columns=cfg["pca_features"])
    rows.insert(0, "Time", np.arange(n, dtype=float))
    rows["Amount"] = np.r_[[20.0] * 10, [5000.0] * 10, [1.0] * 20]
    rows["Class"] = np.r_[[0] * 20, [1] * 20]
    scores = np.r_[np.linspace(0.0, 0.2, 20), np.linspace(0.1, 0.95, 20)]
    return rows, scores, cfg


def test_pick_presets_is_deterministic_and_covers_all_six() -> None:
    rows, scores, cfg = _handmade()
    rules = PresetRules((5.0, 50.0), 5.0, 0.5, 0.05)
    first, skipped = pick_presets(rows, scores, 0.5, rules, DEFAULT, cfg)
    again, _ = pick_presets(rows, scores, 0.5, rules, DEFAULT, cfg)
    assert skipped == [] and first == again
    assert [p.id for p in first.presets] == list(export.PRESET_TEXT)
    by_id = {p.id: p for p in first.presets}
    assert by_id["legit_large"].features.Amount == 5000.0
    # rows are used once: the closest score (0.503) went to fraud_small_amount, so 0.458 is next
    assert "risk score 0.458" in by_id["borderline"].description
    assert by_id["missed_fraud"].expected_class == 1


def test_pick_presets_skips_presets_without_candidates() -> None:
    rows, scores, cfg = _handmade()
    legit_only = rows["Class"] == 0
    rules = PresetRules((5.0, 50.0), 5.0, 0.5, 0.05)
    presets, skipped = pick_presets(rows[legit_only], scores[legit_only], 0.5, rules, DEFAULT, cfg)
    assert [p.id for p in presets.presets] == ["legit_everyday", "legit_large"]
    assert skipped == ["fraud_typical", "fraud_small_amount", "borderline", "missed_fraud"]


def test_demo_batch_has_class_column_and_some_fraud(run: dict[str, Any], cfgs: Configs) -> None:
    batch = pd.read_csv(run["out"].batch)
    test_rows = run["data"].test_rows
    settings = deploy_settings(cfgs.experiment, DEFAULT)
    assert list(batch.columns) == list(test_rows.columns)
    assert len(batch) == settings.demo_batch_rows
    assert batch["Class"].sum() == min(settings.demo_batch_frauds, int(test_rows["Class"].sum()))
    assert len(batch.merge(test_rows, how="inner")) == len(batch)  # every row is a test row
    assert run["out"].batch.read_bytes().count(b"\r") == 0


# --- guard and integrity checks -------------------------------------------------------------------


def test_guard_requires_the_test_evaluation_and_a_reason_to_repeat(
    workdir: dict[str, Path], cfgs: Configs
) -> None:
    guard = workdir["root"] / "TEST_SET_OPENED.json"
    kwargs: dict[str, Any] = {
        "default_cell": DEFAULT,
        "out": _out(workdir["root"]),
        "guard_path": guard,
        "decisions_path": workdir["decisions"],
        "cfgs": cfgs,
        "log": _noop,
    }
    with pytest.raises(ValueError, match="evaluate_test.py first"):
        export_artifacts("sample", workdir["results"], workdir["models"], **kwargs)
    open_test_set(guard, [("lightgbm", "baseline")], None, workdir["decisions"])
    export_artifacts("sample", workdir["results"], workdir["models"], **kwargs)
    openings = load_json(guard)["openings"]
    assert [o["action"] for o in openings] == ["evaluate_test", EXPORT_ACTION]
    assert openings[-1]["cells"] == EXPORTED
    with pytest.raises(ValueError, match="--rerun-reason"):
        export_artifacts("sample", workdir["results"], workdir["models"], **kwargs)


def test_refuses_a_pipeline_changed_after_the_test_evaluation(
    workdir: dict[str, Path], cfgs: Configs
) -> None:
    path = cell_paths("lightgbm", "baseline", workdir["results"], workdir["models"])["pipeline"]
    joblib.dump(joblib.load(path), path, compress=1)
    with pytest.raises(ValueError, match="changed after the test evaluation"):
        export_artifacts(
            "sample",
            workdir["results"],
            workdir["models"],
            DEFAULT,
            _out(workdir["root"]),
            cfgs=cfgs,
            log=_noop,
        )


def test_refuses_cells_that_did_not_run(workdir: dict[str, Path], cfgs: Configs) -> None:
    exp = copy.deepcopy(cfgs.experiment)
    exp["deploy"]["cells"] = [*DEPLOY_CELLS, "lightgbm__ros"]
    with pytest.raises(ValueError, match="lightgbm__ros"):
        export_artifacts(
            "sample",
            workdir["results"],
            workdir["models"],
            DEFAULT,
            _out(workdir["root"]),
            cfgs=dataclasses.replace(cfgs, experiment=exp),
            log=_noop,
        )


def test_refuses_other_library_versions(
    workdir: dict[str, Path], cfgs: Configs, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = export.library_versions()
    monkeypatch.setattr(export, "library_versions", lambda: {**real, "lightgbm": "0.0.1"})
    with pytest.raises(ValueError, match="lightgbm"):
        export_artifacts(
            "sample",
            workdir["results"],
            workdir["models"],
            DEFAULT,
            _out(workdir["root"]),
            cfgs=cfgs,
            log=_noop,
        )


def test_size_warnings_flag_large_files(tmp_path: Path) -> None:
    sizes = {tmp_path / "big.joblib": 60 * 2**20, tmp_path / "small.joblib": 1024}
    warnings = size_warnings(sizes, 50.0)
    assert len(warnings) == 1 and "big.joblib is 60.0 MB" in warnings[0]


# --- command line ---------------------------------------------------------------------------------


def test_cli_exports_and_refuses_without_default(
    workdir: dict[str, Path],
    cfgs: Configs,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_dir = workdir["root"] / "configs"
    config_dir.mkdir()
    for name, data in (("dataset", cfgs.dataset), ("experiment", cfgs.experiment)):
        (config_dir / f"{name}.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    (config_dir / "models.yaml").write_text(yaml.safe_dump(cfgs.models), encoding="utf-8")
    monkeypatch.setattr(config, "CONFIG_DIR", config_dir)
    out = workdir["root"] / "cli"
    args = [
        "--source", "sample",
        "--results-dir", str(workdir["results"]),
        "--models-dir", str(workdir["models"]),
        "--out-models", str(out / "models"),
        "--out-demo", str(out / "demo"),
        "--out-batch", str(out / "demo" / DEMO_BATCH_FILE),
    ]  # fmt: skip
    script = _load_script("export_artifacts")
    assert script.main(args) == 1
    assert "TEAM chooses it" in capsys.readouterr().err
    assert not (out / "models" / MANIFEST_FILE).exists()
    assert script.main([*args, "--default-cell", DEFAULT]) == 0
    printed = capsys.readouterr().out
    assert "total" in printed and DEFAULT in printed
    Manifest.model_validate(load_json(out / "models" / MANIFEST_FILE))
