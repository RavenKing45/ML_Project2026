"""The single evaluation of finished cells on the locked test set, and the temporal sensitivity
check.

Nothing here tunes anything: thresholds come from the cell files (chosen on out-of-fold training
scores in Prompt 08) and the pipelines are the ones refitted on the training set. The test set
is scored once per cell, and every opening of the real test set is recorded in
``TEST_SET_OPENED.json`` (the "guard"). Re-opening it needs an explicit reason, which is also
appended to ``docs/decisions.md``. Runs on the synthetic sample are exempt and write elsewhere.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score

from fraudstudy.config import resolve
from fraudstudy.contracts import (
    MODEL_LABELS,
    TECHNIQUE_LABELS,
    ComparisonRecord,
    ComparisonTable,
    Curves,
    CurvesFile,
)
from fraudstudy.data import deduplicate, load_dataset, split_dataset
from fraudstudy.evaluation import bootstrap_ci, compute_metrics, make_curves
from fraudstudy.experiment.results import (
    CELLS_DIR,
    RUN_META_FILE,
    CellRecord,
    cell_id,
    cell_paths,
    load_cell_record,
)
from fraudstudy.experiment.runner import (
    Configs,
    get_protocol,
    stratified_subsample,
    training_sha256,
)
from fraudstudy.io import git_hash, load_json, save_json, sha256_file

logger = logging.getLogger(__name__)

GUARD_FILE = "TEST_SET_OPENED.json"
COMPARISON_FILE = "comparison.json"
CURVES_FILE = "curves.json"
DETAILS_FILE = "test_eval_details.json"
TEMPORAL_FILE = "temporal_check.json"
SAMPLE_EVAL_DIR = "sample_test_eval"
DECISIONS_FILE = "docs/decisions.md"
DATASET_SOURCE: dict[str, str] = {"raw": "real", "sample": "synthetic"}
DEFAULT_THRESHOLD = 0.5
TUNED_OBJECTIVE = "f1"  # ComparisonRecord.test_at_tuned uses threshold_f1
TEMPORAL_LABEL = (
    "SENSITIVITY ANALYSIS (not the main result): the CV-chosen configuration of each cell, "
    "refitted on the earlier part of a temporal split and scored on the later part. Thresholds "
    "and hyper-parameters are unchanged from the stratified-CV cells; nothing is tuned here."
)


@dataclass(frozen=True)
class TestSplit:
    """The locked test part, plus the hash of the training part it was split from.

    Attributes:
        X_test, y_test: Test features and labels (the same rows every run, by seed).
        train_sha256: :func:`training_sha256` of the training rows the cells were tuned on
            (after the quick-mode subsample on the raw source).
    """

    X_test: pd.DataFrame
    y_test: pd.Series
    train_sha256: str


@dataclass(frozen=True)
class EvalReport:
    """What :func:`evaluate_test` did.

    Attributes:
        comparison: The written ``ComparisonTable`` (evaluated cells plus kept earlier ones).
        curves: The written ``CurvesFile``.
        evaluated: Cells scored in this run.
        paths: Files written.
    """

    comparison: ComparisonTable
    curves: CurvesFile
    evaluated: list[tuple[str, str]]
    paths: dict[str, Path]


def output_dir_for(source: str, results_dir: Path) -> Path:
    """Where evaluation files go: ``results_dir`` for raw, ``results_dir/sample_test_eval``
    for the sample (so synthetic numbers never overwrite real ones)."""
    return Path(results_dir) if source == "raw" else Path(results_dir) / SAMPLE_EVAL_DIR


def load_test_split(source: str, quick: bool, cfgs: Configs, verify_hash: bool = True) -> TestSplit:
    """Rebuild the training run's split (same load, dedup, stratified split, subsample).

    Args:
        source: ``"raw"`` or ``"sample"``.
        quick: Whether the cells were run in quick mode (affects the training subsample only).
        cfgs: The three configs.
        verify_hash: Check the raw file's SHA-256.

    Returns:
        A :class:`TestSplit`.
    """
    df = load_dataset(source, verify_hash=verify_hash, cfg=cfgs.dataset)
    deduped, _ = deduplicate(df, target=cfgs.dataset["target"])
    split = split_dataset(deduped, cfgs.dataset, cfgs.experiment, strategy="stratified")
    X_train, y_train = split["X_train"], split["y_train"]
    if quick and source == "raw":
        fraction = get_protocol(cfgs.experiment, quick=True).train_fraction
        X_train, y_train = stratified_subsample(
            X_train, y_train, fraction, int(cfgs.experiment["seed"])
        )
    return TestSplit(split["X_test"], split["y_test"], training_sha256(X_train, y_train))


def check_training_hash(
    train_sha256: str, records: list[CellRecord], run_meta: dict[str, Any] | None
) -> None:
    """Raise unless the rebuilt training set is the one every cell (and run_meta) was tuned on.

    Raises:
        ValueError: On any mismatch (a different split would put training rows in the test
            set).
    """
    if run_meta is not None and run_meta.get("train_sha256") != train_sha256:
        raise ValueError(
            f"Training-set hash {train_sha256[:12]}... does not match run_meta.json "
            f"({str(run_meta.get('train_sha256'))[:12]}...). The data, dedup, split or seed "
            "changed since the cells were run; re-run scripts/run_experiments.py --force."
        )
    bad = [cell_id(r.model, r.technique) for r in records if r.train_sha256 != train_sha256]
    if bad:
        raise ValueError(
            f"Cells {', '.join(bad)} were tuned on a different training set than the current "
            "split; re-run them with scripts/run_experiments.py --force."
        )


def _clean_reason(reason: str) -> str:
    """One-line reason, safe inside a Markdown table cell."""
    return " ".join(reason.replace("|", "/").split())


def _opening(action: str, cells: list[tuple[str, str]], reason: str | None) -> dict[str, Any]:
    """One guard-file entry."""
    return {
        "opened_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "action": action,
        "git_commit": git_hash(),
        "cells": [cell_id(m, t) for m, t in cells],
        "reason": reason,
    }


def _log_decision(decisions_path: Path, entry: dict[str, Any]) -> None:
    """Append a decisions.md row recording a deliberate re-opening of the test set."""
    cells_text = ", ".join(entry["cells"]) or "none"
    row = (
        f"| {entry['opened_at'][:10]} | Test set re-opened ({entry['action']}, commit "
        f"{str(entry['git_commit'])[:7]}) for {cells_text}: {entry['reason']} | Logged "
        "automatically by the test-set guard; no tuned value was changed | Not re-opening it |\n"
    )
    text = Path(decisions_path).read_text(encoding="utf-8")
    Path(decisions_path).write_text(text.rstrip("\n") + "\n" + row, encoding="utf-8")


def open_test_set(
    guard_path: Path,
    cells: list[tuple[str, str]],
    rerun_reason: str | None,
    decisions_path: Path,
    action: str = "evaluate_test",
) -> dict[str, Any]:
    """Record an opening of the real test set; refuse a repeat without a reason.

    ``evaluate_test`` may run once: the first call creates ``guard_path``. A second
    ``evaluate_test`` (or a second ``temporal_check``) needs a non-empty ``rerun_reason``; the
    entry is then appended to the guard file and a row (date, reason, cells) to
    ``decisions_path``. A first ``temporal_check`` requires the guard to exist already (it must
    come after the single evaluation) and needs no reason.

    Args:
        guard_path: ``artifacts/results/TEST_SET_OPENED.json``.
        cells: Cells being evaluated.
        rerun_reason: Why the test set is opened again (required for a repeat).
        decisions_path: ``docs/decisions.md``.
        action: ``"evaluate_test"`` or ``"temporal_check"``.

    Returns:
        The guard file contents after the update.

    Raises:
        ValueError: If a repeat has no reason, or ``temporal_check`` runs before
            ``evaluate_test``.
    """
    check_guard(guard_path, rerun_reason, action)
    guard_path = Path(guard_path)
    reason = _clean_reason(rerun_reason or "") or None
    if not guard_path.is_file():
        guard = {
            "warning": "The locked test set has been scored. Do not tune anything afterwards.",
            "openings": [_opening(action, cells, reason)],
        }
        save_json(guard, guard_path)
        return guard
    guard = load_json(guard_path)
    earlier = [o for o in guard["openings"] if o.get("action") == action]
    entry = _opening(action, cells, reason)
    guard["openings"].append(entry)
    if earlier:
        _log_decision(decisions_path, entry)
    save_json(guard, guard_path)
    return guard


def check_guard(guard_path: Path | None, rerun_reason: str | None, action: str) -> None:
    """Raise the guard's refusal early (before loading data) without recording anything.

    Raises:
        ValueError: As :func:`open_test_set` would.
    """
    if guard_path is None:
        return
    path = Path(guard_path)
    if not path.is_file():
        if action != "evaluate_test":
            raise ValueError(
                "The temporal check re-uses rows of the locked test set; run "
                "scripts/evaluate_test.py first, so the single test evaluation comes before it."
            )
        return
    earlier = [o for o in load_json(path)["openings"] if o.get("action") == action]
    if earlier and not _clean_reason(rerun_reason or ""):
        raise ValueError(
            f"The test set was already opened by {action} ({earlier[0]['opened_at']}, see "
            f"{path}). Re-opening it biases the final estimate; if you must, pass "
            '--rerun-reason "<why>" (logged there and in docs/decisions.md). Never change a '
            "tuned value because of test results."
        )


def finished_cells(results_dir: Path, source: str) -> list[CellRecord]:
    """All valid cell records in ``results_dir/cells`` produced from ``source``."""
    records = [load_cell_record(p) for p in sorted((Path(results_dir) / CELLS_DIR).glob("*.json"))]
    return [r for r in records if r.source == source]


def _select_records(
    results_dir: Path, source: str, cells: list[str] | None, cfgs: Configs
) -> list[CellRecord]:
    """The records to evaluate, in config order; every requested cell must exist."""
    if cells:
        records = []
        for name in cells:
            model, sep, technique = name.partition("__")
            path = cell_paths(model, technique, results_dir)["record"]
            if not sep or not path.is_file():
                raise ValueError(
                    f"No finished cell {name!r} (expected {path}); run scripts/run_experiments.py "
                    "for it first, or check the <model>__<technique> spelling."
                )
            records.append(load_cell_record(path))
    else:
        records = finished_cells(results_dir, source)
    wrong = [cell_id(r.model, r.technique) for r in records if r.source != source]
    if wrong:
        raise ValueError(
            f"Cells {', '.join(wrong)} were run on a different source than {source!r}."
        )
    if not records:
        raise ValueError(
            f"No finished {source} cells in {Path(results_dir) / CELLS_DIR}; run "
            "scripts/run_experiments.py first."
        )
    if len({r.quick for r in records}) > 1:
        raise ValueError("Cells mix quick and full runs; evaluate one mode at a time.")
    order = {
        (m, t): i
        for i, (m, t) in enumerate(
            (m, t) for m in cfgs.experiment["models"] for t in cfgs.experiment["techniques"]
        )
    }
    return sorted(records, key=lambda r: order.get((r.model, r.technique), len(order)))


def _cis(
    y: pd.Series, scores: np.ndarray, exp_cfg: dict[str, Any]
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Bootstrap CIs ``(pr_auc_ci, roc_auc_ci)`` with the config's resamples, confidence, seed."""
    boot = exp_cfg["bootstrap"]
    args = (int(boot["n_resamples"]), float(boot["confidence"]), int(exp_cfg["seed"]))
    return (
        bootstrap_ci(y, scores, average_precision_score, *args),
        bootstrap_ci(y, scores, roc_auc_score, *args),
    )


def evaluate_cell(
    record: CellRecord,
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    exp_cfg: dict[str, Any],
) -> tuple[ComparisonRecord, Curves, dict[str, Any]]:
    """Score the test set once with a fitted pipeline and build the contract objects.

    Thresholds are read from ``record.thresholds`` (tuned on out-of-fold training scores);
    they are never recomputed here.

    Args:
        record: The cell's training-time record.
        pipeline: Its fitted best pipeline.
        X_test: Test features.
        y_test: Test labels.
        exp_cfg: Experiment config (``bootstrap``, ``seed``).

    Returns:
        ``(ComparisonRecord, Curves, details)``; ``details`` holds the F2-threshold metrics and
        test counts (not part of the 1.2 contract).

    Raises:
        ValueError: If the record lacks an F1 or F2 threshold.
    """
    missing = [obj for obj in ("f1", "f2") if obj not in record.thresholds]
    if missing:
        raise ValueError(
            f"{cell_id(record.model, record.technique)} has no {', '.join(missing)} threshold; "
            "re-run it with scripts/run_experiments.py --force."
        )
    scores = np.asarray(pipeline.predict_proba(X_test)[:, 1], dtype=float)
    t_f1, t_f2 = record.thresholds["f1"], record.thresholds["f2"]
    at_tuned, at_f2 = compute_metrics(y_test, scores, t_f1), compute_metrics(y_test, scores, t_f2)
    pr_ci, roc_ci = _cis(y_test, scores, exp_cfg)
    comparison = ComparisonRecord(
        model=record.model,
        technique=record.technique,
        model_label=MODEL_LABELS[record.model],
        technique_label=TECHNIQUE_LABELS[record.technique],
        best_params=record.best_params,
        chosen_ratio=record.chosen_ratio,
        cv_pr_auc_mean=record.cv_pr_auc_mean,
        cv_pr_auc_std=record.cv_pr_auc_std,
        cv_roc_auc_mean=record.cv_roc_auc_mean,
        threshold_f1=t_f1,
        threshold_f2=t_f2,
        test_at_0_5=compute_metrics(y_test, scores, DEFAULT_THRESHOLD),
        test_at_tuned=at_tuned,
        pr_auc_ci=pr_ci,
        roc_auc_ci=roc_ci,
        fit_seconds=record.fit_seconds,
        n_train_after_resampling=record.n_train_after_resampling,
    )
    curves = make_curves(y_test, scores, model=record.model, technique=record.technique)
    details = {
        "model": record.model,
        "technique": record.technique,
        "n_test": len(y_test),
        "n_test_frauds": int((np.asarray(y_test) == 1).sum()),
        "threshold_f1": t_f1,
        "threshold_f2": t_f2,
        "threshold_source": "cell record (out-of-fold training scores)",
        "test_at_f2": at_f2.model_dump(),
    }
    return comparison, curves, details


def _upsert(old: list[Any], new: list[Any]) -> list[Any]:
    """Replace items of ``old`` that share a ``(model, technique)`` with ``new``; keep the rest."""
    keys = {(n.model, n.technique) for n in new}
    return [o for o in old if (o.model, o.technique) not in keys] + new


def evaluate_test(
    source: str,
    results_dir: Path,
    models_dir: Path,
    cells: list[str] | None = None,
    rerun_reason: str | None = None,
    guard_path: Path | None = None,
    decisions_path: Path | None = None,
    output_dir: Path | None = None,
    cfgs: Configs | None = None,
    verify_hash: bool = True,
    log: Callable[[str], None] = print,
) -> EvalReport:
    """Evaluate finished cells on the locked test set and write comparison.json / curves.json.

    Steps: select cells -> check the guard (raw: refuse a re-opening without a reason) ->
    rebuild the split and check the training hash against run_meta.json and every cell ->
    record the opening -> score each cell once -> write validated contract files (earlier
    records of other cells in the same files are kept).

    Args:
        source: ``"raw"`` or ``"sample"``.
        results_dir: Folder with ``cells/`` and ``run_meta.json`` (``artifacts/results``).
        models_dir: Folder with ``cells/*.joblib`` (``artifacts/models``).
        cells: ``"<model>__<technique>"`` ids (default: every finished cell of ``source``).
        rerun_reason: Required when the guard file already exists.
        guard_path: Guard file; default ``results_dir/TEST_SET_OPENED.json`` for raw and none
            for the sample (exempt). Pass one to exercise the guard on the sample in tests.
        decisions_path: Where re-opening reasons are appended (default ``docs/decisions.md``).
        output_dir: Default :func:`output_dir_for`.
        cfgs: Configs (loaded if omitted).
        verify_hash: Check the raw file's SHA-256.
        log: Progress printer.

    Returns:
        An :class:`EvalReport`.

    Raises:
        ValueError: If the guard refuses, or the training set differs from the cells'.
        ValueError / ValueError: If cells, files or data are missing or invalid.
    """
    cfgs = cfgs if cfgs is not None else Configs.load()
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir is not None else output_dir_for(source, results_dir)
    if guard_path is None and source == "raw":
        guard_path = results_dir / GUARD_FILE
    decisions_path = Path(decisions_path) if decisions_path else resolve(DECISIONS_FILE)

    records = _select_records(results_dir, source, cells, cfgs)
    keys = [(r.model, r.technique) for r in records]
    check_guard(guard_path, rerun_reason, "evaluate_test")
    pipelines = {}
    for r in records:
        path = cell_paths(r.model, r.technique, results_dir, models_dir)["pipeline"]
        if not path.is_file():
            raise ValueError(
                f"Fitted pipeline {path} is missing; copy it from the machine that ran the cell "
                "or re-run the cell."
            )
        pipelines[(r.model, r.technique)] = path

    meta_path = results_dir / RUN_META_FILE
    run_meta = load_json(meta_path) if meta_path.is_file() else None
    split = load_test_split(source, records[0].quick, cfgs, verify_hash)
    check_training_hash(split.train_sha256, records, run_meta)

    if guard_path is not None:
        open_test_set(guard_path, keys, rerun_reason, decisions_path)
    log(
        f"test set: {len(split.y_test):,} rows, {int((split.y_test == 1).sum()):,} frauds; "
        f"scoring {len(records)} cells once"
    )
    comp_records, curve_cells, details = [], [], []
    for r in records:
        key = (r.model, r.technique)
        pipeline = joblib.load(pipelines[key])
        comp, curves, extra = evaluate_cell(
            r, pipeline, split.X_test, split.y_test, cfgs.experiment
        )
        extra["pipeline_sha256"] = sha256_file(pipelines[key])
        comp_records.append(comp)
        curve_cells.append(curves)
        details.append(extra)
        log(
            f"{cell_id(*key):<26} test PR-AUC {comp.test_at_tuned.pr_auc:.4f} "
            f"[{comp.pr_auc_ci[0]:.3f}, {comp.pr_auc_ci[1]:.3f}]"
        )

    dataset_source = DATASET_SOURCE[source]
    comp_path, curves_path = output_dir / COMPARISON_FILE, output_dir / CURVES_FILE
    old_comp: list[ComparisonRecord] = []
    old_curves: list[Curves] = []
    if comp_path.is_file():
        previous = ComparisonTable.model_validate(load_json(comp_path))
        if previous.dataset_source == dataset_source:
            old_comp = previous.records
    if curves_path.is_file() and old_comp:
        old_curves = CurvesFile.model_validate(load_json(curves_path)).cells
    table = ComparisonTable(dataset_source=dataset_source, records=_upsert(old_comp, comp_records))
    curves_file = CurvesFile(cells=_upsert(old_curves, curve_cells))
    save_json(table.model_dump(mode="json"), comp_path)
    save_json(curves_file.model_dump(mode="json"), curves_path)
    details_path = save_json(
        {
            "note": "Extra test-set facts not in the 1.2 contract (metrics at threshold_f2).",
            "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "source": source,
            "quick": records[0].quick,
            "train_sha256": split.train_sha256,
            "cells": details,
        },
        output_dir / DETAILS_FILE,
    )
    paths = {"comparison": comp_path, "curves": curves_path, "details": details_path}
    if guard_path is not None:
        paths["guard"] = Path(guard_path)
    return EvalReport(table, curves_file, keys, paths)


def temporal_check(
    source: str,
    results_dir: Path,
    models_dir: Path,
    model: str = "lightgbm",
    techniques: list[str] | None = None,
    rerun_reason: str | None = None,
    guard_path: Path | None = None,
    decisions_path: Path | None = None,
    output_dir: Path | None = None,
    cfgs: Configs | None = None,
    verify_hash: bool = True,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Temporal-split sensitivity analysis of the CV-chosen configurations of one model.

    For each technique: clone the cell's fitted pipeline (same hyper-parameters, ratio, weight),
    refit it on the earlier ``1 - test_size`` of the data by ``Time``, score the later part, and
    report metrics at 0.5 and at the cell's stored F1 threshold plus bootstrap CIs.

    The temporal split mixes rows of the locked stratified test set into both parts, so on the
    raw source this may only run AFTER :func:`evaluate_test` (the guard file must exist), and
    the run is logged in it. Nothing it produces may feed back into any tuned value.

    Args:
        source: ``"raw"`` or ``"sample"``.
        results_dir: Folder with ``cells/``.
        models_dir: Folder with ``cells/*.joblib``.
        model: Model whose cells to check.
        techniques: Techniques (default: every finished cell of ``model``).
        rerun_reason: Required on raw when the temporal check already ran (it re-reads test rows).
        guard_path: Default ``results_dir/TEST_SET_OPENED.json`` on raw; none on the sample.
        decisions_path: Where re-run reasons are appended.
        output_dir: Default :func:`output_dir_for`.
        cfgs: Configs (loaded if omitted).
        verify_hash: Check the raw file's SHA-256.
        log: Progress printer.

    Returns:
        The ``temporal_check.json`` contents (also written).

    Raises:
        ValueError: On raw, if the test set was not evaluated yet, or the check already ran
            and no reason was given.
    """
    cfgs = cfgs if cfgs is not None else Configs.load()
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir is not None else output_dir_for(source, results_dir)
    if guard_path is None and source == "raw":
        guard_path = results_dir / GUARD_FILE
    decisions_path = Path(decisions_path) if decisions_path else resolve(DECISIONS_FILE)
    techniques = list(techniques or cfgs.experiment["techniques"])
    records = [
        r
        for r in finished_cells(results_dir, source)
        if r.model == model and r.technique in techniques
    ]
    if not records:
        raise ValueError(
            f"No finished {source} cells for model {model!r} in {results_dir / CELLS_DIR}."
        )
    order = {t: i for i, t in enumerate(cfgs.experiment["techniques"])}
    records.sort(key=lambda r: order.get(r.technique, len(order)))
    keys = [(r.model, r.technique) for r in records]
    if guard_path is not None:
        open_test_set(guard_path, keys, rerun_reason, decisions_path, action="temporal_check")

    df = load_dataset(source, verify_hash=verify_hash, cfg=cfgs.dataset)
    deduped, _ = deduplicate(df, target=cfgs.dataset["target"])
    split = split_dataset(deduped, cfgs.dataset, cfgs.experiment, strategy="temporal")
    time_col = cfgs.dataset["time_column"]
    cutoff = float(deduped.loc[split["X_test"].index, time_col].min())
    y_train, y_test = split["y_train"], split["y_test"]
    log(
        f"temporal split: train {len(y_train):,} rows ({int(y_train.sum())} frauds), "
        f"test {len(y_test):,} rows ({int(y_test.sum())} frauds), Time cutoff "
        f"{cutoff:.0f} s"
    )

    cells_out: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for r in records:
        name = cell_id(r.model, r.technique)
        path = cell_paths(r.model, r.technique, results_dir, models_dir)["pipeline"]
        if not path.is_file():
            skipped.append({"cell": name, "reason": f"pipeline file {path.name} missing"})
            continue
        pipe = clone(joblib.load(path))
        try:
            pipe.fit(split["X_train"], split["y_train"])
        except ValueError as exc:  # e.g. a ratio at or below the temporal train ratio
            skipped.append({"cell": name, "reason": f"refit failed: {exc}"})
            log(f"{name:<26} skipped ({exc})")
            continue
        scores = np.asarray(pipe.predict_proba(split["X_test"])[:, 1], dtype=float)
        pr_ci, roc_ci = _cis(split["y_test"], scores, cfgs.experiment)
        at_half = compute_metrics(split["y_test"], scores, DEFAULT_THRESHOLD)
        at_tuned = compute_metrics(split["y_test"], scores, r.thresholds["f1"])
        cells_out.append(
            {
                "model": r.model,
                "technique": r.technique,
                "best_params": r.best_params,
                "threshold_f1": r.thresholds["f1"],
                "cv_pr_auc_mean": r.cv_pr_auc_mean,
                "temporal_at_0_5": at_half.model_dump(),
                "temporal_at_tuned": at_tuned.model_dump(),
                "pr_auc_ci": list(pr_ci),
                "roc_auc_ci": list(roc_ci),
            }
        )
        log(f"{name:<26} temporal PR-AUC {at_tuned.pr_auc:.4f} [{pr_ci[0]:.3f}, {pr_ci[1]:.3f}]")

    result = {
        "analysis": "temporal_split_sensitivity_check",
        "is_sensitivity_analysis": True,
        "label": TEMPORAL_LABEL,
        "dataset_source": DATASET_SOURCE[source],
        "model": model,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_hash(),
        "split": {
            "strategy": "temporal",
            "time_cutoff_seconds": cutoff,
            "train": {"rows": len(split["y_train"]), "frauds": int(split["y_train"].sum())},
            "test": {"rows": len(split["y_test"]), "frauds": int(split["y_test"].sum())},
        },
        "cells": cells_out,
        "skipped": skipped,
    }
    save_json(result, output_dir / TEMPORAL_FILE)
    return result
