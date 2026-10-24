"""Small real models for the prediction tests (Prompt 13), written as the exporter writes them.

:func:`build_real_artifacts` fits the six LightGBM techniques and logistic regression (with the
default cell's technique) on the synthetic sample's training split, without tuning, and writes what
``scripts/export_artifacts.py`` writes: ``deploy/*.joblib`` with calibrators,
``feature_reference.json``, ``manifest.json``, ``presets.json`` and ``demo_batch.csv``. The JSON
results are the example files. Thresholds and calibrators come from in-sample training scores,
which is fine for API tests (the exporter uses out-of-fold scores).

A plain module, not a conftest, so test modules can import its names (pytest's default "prepend"
import mode puts this folder on ``sys.path``); the fixtures live in ``conftest.py``.
"""

import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from backend.app.settings import Settings
from fraudstudy.contracts import (
    MODEL_LABELS,
    TECHNIQUE_LABELS,
    DatasetInfo,
    Manifest,
    ManifestEntry,
    PresetsFile,
)
from fraudstudy.data import deduplicate, load_dataset, split_dataset
from fraudstudy.evaluation import best_threshold, fit_isotonic
from fraudstudy.experiment import Configs
from fraudstudy.experiment.runner import library_versions, training_sha256
from fraudstudy.explain import FeatureReference, build_feature_reference
from fraudstudy.export import (
    DEMO_BATCH_FILE,
    FEATURE_REFERENCE_FILE,
    MANIFEST_FILE,
    PRESETS_FILE,
    demo_batch,
    deploy_settings,
    pick_presets,
)
from fraudstudy.io import save_json, sha256_file
from fraudstudy.paths import PROJECT_ROOT
from fraudstudy.pipeline import build_pipeline

EXAMPLES = PROJECT_ROOT / "docs" / "contracts" / "examples"
RESULT_FILES = ("comparison", "curves", "stats", "eda", "resampling_audit")
DEFAULT_ID = "lightgbm__smote_rus"
LOGREG_ID = "logreg__smote_rus"
UNCALIBRATED_ID = "lightgbm__baseline"
DATA_SHA = "cd" * 32
TECHNIQUE_PARAMS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "ros": {"sampling_strategy": 0.1},
    "rus": {"sampling_strategy": 0.1},
    "smote": {"sampling_strategy": 0.1},
    "smote_rus": {"smote_sampling_strategy": 0.05, "rus_sampling_strategy": 0.1},
    "class_weight": {"fraud_weight": 5.0},
}
TINY_LIGHTGBM = {"n_estimators": 25, "num_leaves": 15}
LOGREG_PARAMS = {"C": 0.1, "max_iter": 2000}


@dataclass(frozen=True)
class RealArtifacts:
    """The fitted models and the files written for them.

    Attributes:
        artifacts: ``ARTIFACTS_DIR`` (``results/`` example files, ``demo/presets.json``).
        models: ``MODELS_DIR`` (``manifest.json``, ``feature_reference.json``, ``deploy/``).
        demo_batch: ``demo_batch.csv`` (test rows with ``Class``).
        manifest, presets, reference: What was written.
        pipelines, calibrators: Fitted objects by manifest id (no calibrator for
            ``UNCALIBRATED_ID``).
        test_rows: The sample's test rows with every column.
    """

    artifacts: Path
    models: Path
    demo_batch: Path
    manifest: Manifest
    presets: PresetsFile
    reference: FeatureReference
    pipelines: dict[str, Any]
    calibrators: dict[str, Any]
    test_rows: pd.DataFrame


def _fit_cells(cfgs: Configs, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    """The seven deployed cells, fitted on ``X, y`` with small fixed hyper-parameters."""
    cfg, exp = cfgs.dataset, cfgs.experiment
    lightgbm = {**TINY_LIGHTGBM, "n_jobs": exp["parallel"]["model_n_jobs"]}
    cells = {f"lightgbm__{t}": ("lightgbm", t, lightgbm) for t in TECHNIQUE_PARAMS}
    cells[LOGREG_ID] = ("logreg", LOGREG_ID.partition("__")[2], LOGREG_PARAMS)
    fitted: dict[str, Any] = {}
    for cell_id, (model, technique, params) in cells.items():
        pipeline = build_pipeline(model, technique, params, TECHNIQUE_PARAMS[technique], cfg, exp)
        fitted[cell_id] = pipeline.fit(X, y)
    return fitted


def _write_cell(
    cell_id: str, pipeline: Any, X: pd.DataFrame, y: pd.Series, deploy: Path
) -> tuple[ManifestEntry, Any]:
    """Dump one cell (and its calibrator, except ``UNCALIBRATED_ID``) and build its entry."""
    model, _, technique = cell_id.partition("__")
    model_path = deploy / f"{cell_id}.joblib"
    joblib.dump(pipeline, model_path)
    scores = pipeline.predict_proba(X)[:, 1]
    calibrator = calibrator_file = calibrator_sha = None
    if cell_id != UNCALIBRATED_ID:
        calibrator = fit_isotonic(scores, y)
        calibrator_path = deploy / f"{cell_id}.calibrator.joblib"
        joblib.dump(calibrator, calibrator_path)
        calibrator_file = f"{deploy.name}/{calibrator_path.name}"
        calibrator_sha = sha256_file(calibrator_path)
    entry = ManifestEntry(
        id=cell_id,
        model=model,
        technique=technique,
        model_label=MODEL_LABELS[model],
        technique_label=TECHNIQUE_LABELS[technique],
        file=f"{deploy.name}/{model_path.name}",
        sha256=sha256_file(model_path),
        threshold_f1=best_threshold(y, scores, "f1"),
        threshold_f2=best_threshold(y, scores, "f2"),
        has_calibrator=calibrator is not None,
        calibrator_file=calibrator_file,
        calibrator_sha256=calibrator_sha,
        features=[str(c) for c in pipeline.named_steps["preprocess"].feature_names_in_],
        data_sha256=DATA_SHA,
        library_versions=library_versions(),
    )
    return entry, calibrator


def build_real_artifacts(root: Path, cfgs: Configs) -> RealArtifacts:
    """Fit the deployed cells on the sample and write a real-mode artifact tree under ``root``.

    Args:
        root: An empty folder.
        cfgs: The project configs (with the test thread cap).

    Returns:
        The :class:`RealArtifacts`.
    """
    cfg, exp = cfgs.dataset, cfgs.experiment
    artifacts, models = root / "artifacts", root / "models"
    deploy = models / "deploy"
    for folder in (artifacts / "results", artifacts / "demo", deploy):
        folder.mkdir(parents=True)
    for name in RESULT_FILES:
        shutil.copy(EXAMPLES / f"{name}.json", artifacts / "results" / f"{name}.json")

    df, _ = deduplicate(load_dataset("sample", cfg=cfg), target=cfg["target"])
    split = split_dataset(df, cfg, exp)
    X, y = split.X_train, split.y_train
    pipelines = _fit_cells(cfgs, X, y)
    entries: list[ManifestEntry] = []
    calibrators: dict[str, Any] = {}
    for cell_id, pipeline in pipelines.items():
        entry, calibrator = _write_cell(cell_id, pipeline, X, y, deploy)
        entries.append(entry)
        if calibrator is not None:
            calibrators[cell_id] = calibrator

    reference = build_feature_reference(
        pipelines[DEFAULT_ID].named_steps["preprocess"],
        X,
        y,
        source="sample",
        data_sha256=DATA_SHA,
        train_sha256=training_sha256(X, y),
        range_percentiles=exp["deploy"]["input_range_percentiles"],
    )
    save_json(reference.model_dump(mode="json"), models / FEATURE_REFERENCE_FILE)

    settings = deploy_settings(exp, DEFAULT_ID)
    test_rows = df.loc[split.X_test.index]
    default_entry = next(e for e in entries if e.id == DEFAULT_ID)
    presets, _ = pick_presets(
        test_rows,
        pipelines[DEFAULT_ID].predict_proba(split.X_test)[:, 1],
        default_entry.threshold_f1,
        settings.presets,
        DEFAULT_ID,
        cfg,
        origin="Synthetic",
    )
    save_json(presets.model_dump(mode="json"), artifacts / "demo" / PRESETS_FILE)
    batch_path = root / DEMO_BATCH_FILE
    batch = demo_batch(
        test_rows,
        cfg["target"],
        settings.demo_batch_rows,
        settings.demo_batch_frauds,
        int(exp["seed"]),
    )
    batch.to_csv(batch_path, index=False, lineterminator="\n")

    manifest = Manifest(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        git_commit=None,
        seed=int(exp["seed"]),
        quick_mode=True,
        dataset=DatasetInfo(name=cfg["name"], source="synthetic", sha256=DATA_SHA, n_rows=len(df)),
        default_model_id=DEFAULT_ID,
        models=entries,
    )
    save_json(manifest.model_dump(mode="json"), models / MANIFEST_FILE)
    return RealArtifacts(
        artifacts=artifacts,
        models=models,
        demo_batch=batch_path,
        manifest=manifest,
        presets=presets,
        reference=reference,
        pipelines=pipelines,
        calibrators=calibrators,
        test_rows=test_rows,
    )


def real_settings(real: RealArtifacts, **overrides: Any) -> Settings:
    """Settings that read ``real`` (no ``.env``, development mode, rate limits off);
    ``overrides`` win."""
    values: dict[str, Any] = {
        "use_example_artifacts": False,
        "examples_dir": EXAMPLES,
        "artifacts_dir": real.artifacts,
        "models_dir": real.models,
        "allowed_origins": "",
        "environment": "development",
        "default_model_id": None,
        "log_level": "INFO",
        "rate_limit_enabled": False,
    }
    return Settings(_env_file=None, **(values | overrides))
