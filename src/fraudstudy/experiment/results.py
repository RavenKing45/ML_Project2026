"""Saving, resuming and merging experiment cells (master guide Part 6, "What the experiment
produces").

Layout (directories are arguments, so tests can write to a temporary folder):

* ``<results>/cells/<model>__<technique>.json``: :class:`CellRecord` (written LAST, so its
  presence marks a finished cell).
* ``<results>/cells/<model>__<technique>_oof.npy``: out-of-fold scores, in training-row order.
* ``<results>/fold_scores.csv``: ``model, technique, repeat, fold, fold_id, pr_auc, roc_auc``;
  a cell's rows are replaced when it is re-run, never duplicated.
* ``<models>/cells/<model>__<technique>.joblib``: the refitted best pipeline (git-ignored).
* ``<results>/run_meta.json``: config snapshot, git hash, library versions, data hash, times.
* ``<results>/cv_summary.json``: written by :func:`merge_results`.

None of these files contain test-set information.
"""

import logging
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from pydantic import BaseModel, ConfigDict, ValidationError

from fraudstudy.contracts import ClassCounts
from fraudstudy.errors import DataValidationError
from fraudstudy.io import load_json, save_json

logger = logging.getLogger(__name__)

CELLS_DIR = "cells"
FOLD_SCORES_FILE = "fold_scores.csv"
RUN_META_FILE = "run_meta.json"
CV_SUMMARY_FILE = "cv_summary.json"
FOLD_COLUMNS: tuple[str, ...] = (
    "model",
    "technique",
    "repeat",
    "fold",
    "fold_id",
    "pr_auc",
    "roc_auc",
)
FOLD_KEY: list[str] = ["model", "technique", "repeat", "fold"]
# Fields that legitimately differ between two runs of the same cell (clock, machine, commit).
VOLATILE_FIELDS: frozenset[str] = frozenset(
    {
        "search_seconds",
        "fit_seconds",
        "predict_seconds",
        "total_seconds",
        "started_at",
        "finished_at",
        "git_commit",
        "library_versions",
    }
)
Scalar = str | int | float | bool | None


class _Strict(BaseModel):
    """Base model: unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")


class FoldScore(_Strict):
    """One validation fold of the repeated CV.

    ``fold_id`` is a hash of the fold's validation row labels: equal ids mean equal folds, which
    the paired statistics rely on.
    """

    repeat: int
    fold: int
    fold_id: str
    pr_auc: float
    roc_auc: float


class Candidate(_Strict):
    """One configuration tried by the randomized search (from ``cv_results_``)."""

    params: dict[str, Scalar]
    mean_test_score: float
    std_test_score: float
    rank_test_score: int
    split_test_scores: list[float]
    mean_fit_time: float


class CellRecord(_Strict):
    """Everything known about one (model, technique) cell after tuning, on training data only.

    Scores are average precision (``pr_auc``) and ROC-AUC. Standard deviations use ``ddof=0``.
    ``best_params`` holds pipeline parameter names with scalar values: a ``class_weight`` dict
    ``{0: 1, 1: w}`` is stored as ``w`` (``"balanced"`` stays as is). ``chosen_ratio`` follows
    ``ComparisonRecord``: the sampler ratio; for ``smote_rus`` the RUS ratio b (a in
    ``smote_ratio``); for ``class_weight`` ``w x base_ratio`` (``"balanced"`` = 1.0, equal
    weighted class mass); ``None`` for the baseline.
    """

    model: str
    technique: str
    source: str | None = None
    quick: bool
    seed: int
    cv_folds: int
    cv_repeats: int
    n_iter_budget: int
    search_space_size: int
    n_candidates_evaluated: int
    n_train: int
    n_train_frauds: int
    train_sha256: str
    base_ratio: float
    dropped_ratios: list[float]
    best_params: dict[str, Scalar]
    chosen_ratio: float | None
    smote_ratio: float | None
    fraud_weight: float | str | None
    search_best_pr_auc: float
    candidates: list[Candidate]
    oof_pr_auc: float
    oof_roc_auc: float
    thresholds: dict[str, float]
    cv_pr_auc_mean: float
    cv_pr_auc_std: float
    cv_roc_auc_mean: float
    cv_roc_auc_std: float
    fold_scores: list[FoldScore]
    n_train_after_resampling: ClassCounts
    n_synthetic: int
    fraud_duplicates_after: int
    search_seconds: float
    fit_seconds: float
    predict_seconds: float
    total_seconds: float
    started_at: str
    finished_at: str
    data_sha256: str | None = None
    git_commit: str | None = None
    library_versions: dict[str, str]


@dataclass
class CellResult:
    """Output of :func:`fraudstudy.experiment.runner.run_cell`.

    Attributes:
        record: The JSON-serialisable summary.
        oof_scores: Out-of-fold fraud scores of the best configuration, in training-row order.
        pipeline: The best pipeline refitted on the whole training set.
    """

    record: CellRecord
    oof_scores: np.ndarray
    pipeline: Pipeline = field(repr=False)


@dataclass(frozen=True)
class MergeReport:
    """What :func:`merge_results` combined.

    Attributes:
        cells: ``(model, technique)`` pairs merged, in config order.
        missing: Expected pairs (``models`` x ``techniques``) with no finished cell.
        duplicates: Pairs found in several inputs with identical results.
        conflicts: Pairs found with different results (the latest ``finished_at`` is kept).
        paths: Files written.
    """

    cells: list[tuple[str, str]]
    missing: list[tuple[str, str]]
    duplicates: list[tuple[str, str]]
    conflicts: list[tuple[str, str]]
    paths: dict[str, Path]


def cell_id(model: str, technique: str) -> str:
    """File stem of a cell: ``"<model>__<technique>"`` (the manifest's id format)."""
    return f"{model}__{technique}"


def cell_paths(
    model: str, technique: str, results_dir: Path, models_dir: Path | None = None
) -> dict[str, Path]:
    """Paths of a cell's files.

    Args:
        model: Model id.
        technique: Technique id.
        results_dir: E.g. ``artifacts/results``.
        models_dir: E.g. ``artifacts/models``; the ``"pipeline"`` path is omitted if ``None``.

    Returns:
        ``{"record", "oof"[, "pipeline"]}`` -> path.
    """
    stem = cell_id(model, technique)
    cells = Path(results_dir) / CELLS_DIR
    paths = {"record": cells / f"{stem}.json", "oof": cells / f"{stem}_oof.npy"}
    if models_dir is not None:
        paths["pipeline"] = Path(models_dir) / CELLS_DIR / f"{stem}.joblib"
    return paths


def fold_frame(record: CellRecord) -> pd.DataFrame:
    """The record's fold scores as rows of ``fold_scores.csv`` (:data:`FOLD_COLUMNS`)."""
    rows = [
        {"model": record.model, "technique": record.technique, **f.model_dump()}
        for f in record.fold_scores
    ]
    return pd.DataFrame(rows, columns=list(FOLD_COLUMNS))


def read_fold_scores(path: Path) -> pd.DataFrame:
    """Read ``fold_scores.csv``; an empty frame with :data:`FOLD_COLUMNS` if it does not exist.

    Raises:
        DataValidationError: If the file lacks required columns.
    """
    if not Path(path).is_file():
        return pd.DataFrame(columns=list(FOLD_COLUMNS))
    df = pd.read_csv(path, dtype={"fold_id": str})
    missing = [c for c in FOLD_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"{path} is missing columns {', '.join(missing)}; delete it and re-run with --force."
        )
    return df[list(FOLD_COLUMNS)]


def upsert_fold_scores(new_rows: pd.DataFrame, path: Path) -> Path:
    """Write ``new_rows`` into ``fold_scores.csv``, replacing earlier rows of the same cells.

    Args:
        new_rows: Rows with :data:`FOLD_COLUMNS`.
        path: The CSV file (created with its parent folder if missing).

    Returns:
        The path written.
    """
    old = read_fold_scores(path)
    cells = set(zip(new_rows["model"], new_rows["technique"], strict=True))
    keep = ~pd.Series(
        [(m, t) in cells for m, t in zip(old["model"], old["technique"], strict=True)],
        index=old.index,
        dtype=bool,
    )
    frames = [df for df in (old[keep], new_rows) if not df.empty]
    out = pd.concat(frames, ignore_index=True) if frames else new_rows
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return Path(path)


def save_cell(result: CellResult, results_dir: Path, models_dir: Path) -> dict[str, Path]:
    """Save a finished cell: OOF scores, pipeline, fold rows, then the JSON record (last).

    Writing the record last means a crash mid-save leaves the cell "not done", so a resumed run
    redoes it instead of trusting partial files.

    Args:
        result: From ``run_cell``.
        results_dir: E.g. ``artifacts/results``.
        models_dir: E.g. ``artifacts/models``.

    Returns:
        ``{"record", "oof", "pipeline", "fold_scores"}`` -> path written.
    """
    rec = result.record
    paths = cell_paths(rec.model, rec.technique, results_dir, models_dir)
    paths["oof"].parent.mkdir(parents=True, exist_ok=True)
    np.save(paths["oof"], np.asarray(result.oof_scores, dtype=float))
    paths["pipeline"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.pipeline, paths["pipeline"])
    paths["fold_scores"] = upsert_fold_scores(fold_frame(rec), Path(results_dir) / FOLD_SCORES_FILE)
    save_json(rec.model_dump(mode="json"), paths["record"])
    return paths


def load_cell_record(path: Path) -> CellRecord:
    """Read and validate a cell JSON file.

    Raises:
        DataValidationError: If the file is not a valid :class:`CellRecord`.
    """
    try:
        return CellRecord.model_validate(load_json(path))
    except (ValidationError, ValueError) as exc:
        raise DataValidationError(
            f"{path} is not a valid cell record ({exc}); re-run the cell with --force."
        ) from exc


def is_cell_done(
    model: str,
    technique: str,
    results_dir: Path,
    *,
    quick: bool,
    source: str,
    seed: int,
) -> bool:
    """Whether a cell has finished files that match the requested run settings.

    A cell saved by a different mode (quick vs full), source or seed is NOT done, so a full run
    never silently reuses quick results.

    Args:
        model: Model id.
        technique: Technique id.
        results_dir: E.g. ``artifacts/results``.
        quick: Requested quick mode.
        source: Requested data source (``raw`` / ``sample``).
        seed: Requested seed.

    Returns:
        ``True`` if the record and OOF file exist, the record is valid and its settings match.
    """
    paths = cell_paths(model, technique, results_dir)
    if not (paths["record"].is_file() and paths["oof"].is_file()):
        return False
    try:
        rec = load_cell_record(paths["record"])
    except DataValidationError as exc:
        logger.warning("%s", exc)
        return False
    if (rec.quick, rec.source, rec.seed) != (quick, source, seed):
        logger.info(
            "%s exists but was run with quick=%s, source=%s, seed=%s; re-running.",
            cell_id(model, technique),
            rec.quick,
            rec.source,
            rec.seed,
        )
        return False
    return True


def write_run_meta(meta: dict[str, Any], results_dir: Path) -> Path:
    """Write ``run_meta.json`` (overwrites the previous run's file).

    Args:
        meta: Config snapshot, git hash, library versions, data hash, start/end time, cells.
        results_dir: E.g. ``artifacts/results``.

    Returns:
        The path written.
    """
    return save_json(meta, Path(results_dir) / RUN_META_FILE)


def scientific_view(record: CellRecord) -> dict[str, Any]:
    """The record without clock / machine fields (:data:`VOLATILE_FIELDS`, fit times).

    Two runs of the same cell with the same seed and data must give equal views.
    """
    data = record.model_dump(mode="json", exclude=set(VOLATILE_FIELDS))
    for cand in data["candidates"]:
        cand.pop("mean_fit_time")
    return data


def _summary_row(rec: CellRecord) -> dict[str, Any]:
    """One ``cv_summary.json`` entry."""
    return {
        "model": rec.model,
        "technique": rec.technique,
        "best_params": rec.best_params,
        "chosen_ratio": rec.chosen_ratio,
        "smote_ratio": rec.smote_ratio,
        "fraud_weight": rec.fraud_weight,
        "cv_pr_auc_mean": rec.cv_pr_auc_mean,
        "cv_pr_auc_std": rec.cv_pr_auc_std,
        "cv_roc_auc_mean": rec.cv_roc_auc_mean,
        "cv_roc_auc_std": rec.cv_roc_auc_std,
        "oof_pr_auc": rec.oof_pr_auc,
        "thresholds": rec.thresholds,
        "n_candidates_evaluated": rec.n_candidates_evaluated,
        "fit_seconds": rec.fit_seconds,
        "n_train_after_resampling": rec.n_train_after_resampling.model_dump(),
    }


def _check_comparable(records: list[CellRecord]) -> None:
    """Raise unless every record shares the protocol, training data and fold assignment."""
    keys = ("seed", "quick", "source", "cv_folds", "cv_repeats", "train_sha256")
    for key in keys:
        values = {getattr(r, key) for r in records}
        if len(values) > 1:
            raise DataValidationError(
                f"Cells disagree on {key} ({sorted(map(str, values))}); they come from "
                "different protocols or data and cannot be compared. Re-run the odd ones."
            )
    fold_sets = {frozenset(f.fold_id for f in r.fold_scores) for r in records}
    if len(fold_sets) > 1:
        raise DataValidationError(
            "Cells have different repeated-CV folds, so paired tests are invalid. Re-run them "
            "with the same seed and data."
        )


def merge_results(
    input_dirs: list[Path],
    output_dir: Path,
    models: list[str],
    techniques: list[str],
) -> MergeReport:
    """Combine cells from several result folders into ``cv_summary.json`` and ``fold_scores.csv``.

    Fold rows are rebuilt from the cell records (the authoritative copy) and de-duplicated on
    ``(model, technique, repeat, fold)``. Cell JSON and OOF files are copied into
    ``output_dir/cells`` when they come from another folder. Pipelines (``.joblib``) are not
    copied; they stay in each machine's ``artifacts/models/cells``.

    Args:
        input_dirs: Result folders (each with a ``cells/`` sub-folder).
        output_dir: Where to write; may be one of ``input_dirs``.
        models: Expected model ids (``models`` in experiment.yaml).
        techniques: Expected technique ids (``techniques`` in experiment.yaml).

    Returns:
        A :class:`MergeReport`.

    Raises:
        DataValidationError: If no cell is found, a file is invalid, or cells are not comparable
            (different seed, mode, data or folds).
    """
    found: dict[tuple[str, str], tuple[CellRecord, Path]] = {}
    duplicates: list[tuple[str, str]] = []
    conflicts: list[tuple[str, str]] = []
    for folder in input_dirs:
        for path in sorted((Path(folder) / CELLS_DIR).glob("*__*.json")):
            rec = load_cell_record(path)
            key = (rec.model, rec.technique)
            if key in found:
                old = found[key][0]
                if scientific_view(old) == scientific_view(rec):
                    duplicates.append(key)
                else:
                    conflicts.append(key)
                    logger.warning("%s differs between inputs; keeping the latest.", cell_id(*key))
                if rec.finished_at <= old.finished_at:
                    continue
            found[key] = (rec, path)
    if not found:
        raise DataValidationError(
            f"No cell records found under {[str(d) for d in input_dirs]} (expected "
            f"<dir>/{CELLS_DIR}/<model>__<technique>.json). Run scripts/run_experiments.py first."
        )
    order = {(m, t): i for i, (m, t) in enumerate((m, t) for m in models for t in techniques)}
    keys = sorted(found, key=lambda k: (order.get(k, len(order)), k))
    records = [found[k][0] for k in keys]
    _check_comparable(records)

    out_cells = Path(output_dir) / CELLS_DIR
    out_cells.mkdir(parents=True, exist_ok=True)
    for key in keys:
        rec, src = found[key]
        dest = cell_paths(rec.model, rec.technique, Path(output_dir))
        src_oof = src.with_name(dest["oof"].name)
        if src.resolve() != dest["record"].resolve():
            shutil.copy2(src, dest["record"])
            if src_oof.is_file():
                shutil.copy2(src_oof, dest["oof"])

    folds = pd.concat([fold_frame(r) for r in records], ignore_index=True)
    folds = folds.drop_duplicates(subset=FOLD_KEY, keep="last")
    fold_path = Path(output_dir) / FOLD_SCORES_FILE
    folds.to_csv(fold_path, index=False)

    missing = [k for k in order if k not in found]
    first = records[0]
    summary = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "primary_metric": "average_precision",
        "seed": first.seed,
        "quick": first.quick,
        "source": first.source,
        "cv_folds": first.cv_folds,
        "cv_repeats": first.cv_repeats,
        "n_cells": len(records),
        "missing": [{"model": m, "technique": t} for m, t in missing],
        "cells": [_summary_row(r) for r in records],
    }
    summary_path = save_json(summary, Path(output_dir) / CV_SUMMARY_FILE)
    return MergeReport(
        cells=keys,
        missing=missing,
        duplicates=duplicates,
        conflicts=conflicts,
        paths={"cv_summary": summary_path, "fold_scores": fold_path},
    )
