"""The ONE opening of the locked test set, and the temporal sensitivity check that follows it.

scripts/evaluate_test.py calls evaluate_test(), scripts/run_temporal_check.py calls
temporal_check(), export.py reuses the guard and the hash check. Reads the cell files and
pipelines written by experiment.py; writes comparison.json, curves.json, test_eval_details.json,
temporal_check.json and, for the real data, the guard file TEST_SET_OPENED.json. Nothing here
tunes anything: thresholds come from the cell files. (Not named test_*.py so nobody mistakes it
for a pytest file.)
"""

from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score

from fraudstudy.config import load_configs, resolve
from fraudstudy.data import deduplicate, load_dataset, split_dataset
from fraudstudy.experiment import (
    MODEL_LABELS,
    RUN_META_FILE,
    TECHNIQUE_LABELS,
    cell_id,
    cell_paths,
    protocol,
    stratified_subsample,
    training_sha256,
)
from fraudstudy.io import git_hash, load_json, save_json, sha256_file
from fraudstudy.metrics import bootstrap_ci, compute_metrics, make_curves

CONTRACT_VERSION = "1.6"  # docs/contracts.md; written into comparison.json and curves.json
GUARD_FILE = "TEST_SET_OPENED.json"
COMPARISON_FILE = "comparison.json"
CURVES_FILE = "curves.json"
DETAILS_FILE = "test_eval_details.json"
TEMPORAL_FILE = "temporal_check.json"
SAMPLE_EVAL_DIR = "sample_test_eval"
DECISIONS_FILE = "docs/decisions.md"
DATASET_SOURCE = {"raw": "real", "sample": "synthetic"}
TEMPORAL_LABEL = (
    "SENSITIVITY ANALYSIS (not the main result): the CV-chosen configuration of each cell, "
    "refitted on the earlier part of a temporal split and scored on the later part. Thresholds "
    "and hyper-parameters are unchanged from the stratified-CV cells; nothing is tuned here."
)


def output_dir_for(source: str, results_dir: Path) -> Path:
    """results_dir for the real data, results_dir/sample_test_eval for the sample (so synthetic
    numbers never overwrite real ones)."""
    return Path(results_dir) if source == "raw" else Path(results_dir) / SAMPLE_EVAL_DIR


def load_test_split(source: str, quick: bool, cfgs: dict, verify_hash: bool = True) -> dict:
    """Rebuild the experiment's split; return X_test, y_test and the hash of the training part."""
    df = load_dataset(source, verify_hash=verify_hash, cfg=cfgs["dataset"])
    deduped, _ = deduplicate(df, target=cfgs["dataset"]["target"])
    split = split_dataset(deduped, cfgs["dataset"], cfgs["experiment"], strategy="stratified")
    X_train, y_train = split["X_train"], split["y_train"]
    if quick and source == "raw":
        fraction = protocol(cfgs["experiment"], quick=True)["train_fraction"]
        X_train, y_train = stratified_subsample(X_train, y_train, fraction,
                                                int(cfgs["experiment"]["seed"]))
    return {"X_test": split["X_test"], "y_test": split["y_test"],
            "train_sha256": training_sha256(X_train, y_train)}


def check_training_hash(train_sha256: str, records: list, run_meta: dict | None) -> None:
    """Stop unless the rebuilt training rows are the ones every cell was tuned on.

    A different split would put training rows into the test set.
    """
    if run_meta is not None and run_meta.get("train_sha256") != train_sha256:
        raise ValueError(
            f"Training-set hash {train_sha256[:12]}... does not match run_meta.json "
            f"({str(run_meta.get('train_sha256'))[:12]}...). The data, dedup, split or seed "
            "changed since the cells were run; re-run scripts/run_experiments.py --force."
        )
    bad = [cell_id(r["model"], r["technique"]) for r in records
           if r["train_sha256"] != train_sha256]
    if bad:
        raise ValueError(f"Cells {', '.join(bad)} were tuned on a different training set than the "
                         "current split; re-run them with scripts/run_experiments.py --force.")


def clean_reason(reason: str | None) -> str:
    """The re-run reason on one line, safe inside a Markdown table cell."""
    return " ".join((reason or "").replace("|", "/").split())


def check_guard(guard_path: Path | None, rerun_reason: str | None, action: str) -> None:
    """Refuse (before any data is loaded) a second opening without a reason.

    The temporal check and the export re-read test rows, so they must come after the single
    evaluation. guard_path None means the sample, which is exempt.
    """
    if guard_path is None:
        return
    if not Path(guard_path).is_file():
        if action != "evaluate_test":
            raise ValueError(
                "The temporal check re-uses rows of the locked test set; run "
                "scripts/evaluate_test.py first, so the single test evaluation comes before it."
            )
        return
    earlier = [o for o in load_json(guard_path)["openings"] if o["action"] == action]
    if earlier and not clean_reason(rerun_reason):
        raise ValueError(
            f"The test set was already opened by {action} ({earlier[0]['opened_at']}, see "
            f"{guard_path}). Re-opening it biases the final estimate; if you must, pass "
            '--rerun-reason "<why>" (logged there and in docs/decisions.md). Never change a '
            "tuned value because of test results."
        )


def open_test_set(guard_path: Path, cells: list, rerun_reason: str | None, decisions_path: Path,
                  action: str = "evaluate_test") -> dict:
    """Record an opening of the real test set in the guard file; refuse a repeat without reason.

    A repeat (same action again) is allowed only with a reason, which is also appended as a
    row (date, reason, cells) to docs/decisions.md.
    """
    check_guard(guard_path, rerun_reason, action)
    entry = {"opened_at": datetime.now(UTC).isoformat(timespec="seconds"), "action": action,
             "git_commit": git_hash(), "cells": [cell_id(m, t) for m, t in cells],
             "reason": clean_reason(rerun_reason) or None}
    if not Path(guard_path).is_file():
        guard = {"warning": "The locked test set has been scored. Do not tune anything afterwards.",
                 "openings": [entry]}
        save_json(guard, guard_path)
        return guard
    guard = load_json(guard_path)
    repeat = any(o["action"] == action for o in guard["openings"])
    guard["openings"].append(entry)
    if repeat:
        row = (f"| {entry['opened_at'][:10]} | Test set re-opened ({action}, commit "
               f"{str(entry['git_commit'])[:7]}) for {', '.join(entry['cells']) or 'none'}: "
               f"{entry['reason']} | Logged automatically by the test-set guard; no tuned value "
               "was changed | Not re-opening it |\n")
        text = Path(decisions_path).read_text(encoding="utf-8")
        Path(decisions_path).write_text(text.rstrip("\n") + "\n" + row, encoding="utf-8")
    save_json(guard, guard_path)
    return guard


def finished_cells(results_dir: Path, source: str) -> list:
    """Every cell record in results_dir/cells that was run on `source`."""
    records = [load_json(p) for p in sorted((Path(results_dir) / "cells").glob("*.json"))]
    return [r for r in records if r["source"] == source]


def select_cells(results_dir: Path, source: str, cells: list | None, cfgs: dict) -> list:
    """The cell records to evaluate, in config order; every requested cell must exist."""
    if cells:
        records = []
        for name in cells:
            model, _, technique = name.partition("__")
            path = cell_paths(model, technique, results_dir)["record"]
            if not path.is_file():
                raise ValueError(f"No finished cell {name!r} (expected {path}); run "
                                 "scripts/run_experiments.py for it first, or check the "
                                 "<model>__<technique> spelling.")
            records.append(load_json(path))
    else:
        records = finished_cells(results_dir, source)
    wrong = [cell_id(r["model"], r["technique"]) for r in records if r["source"] != source]
    if wrong:
        raise ValueError(f"Cells {', '.join(wrong)} were run on another source than {source!r}.")
    if not records:
        raise ValueError(f"No finished {source} cells in {Path(results_dir) / 'cells'}; run "
                         "scripts/run_experiments.py first.")
    if len({r["quick"] for r in records}) > 1:
        raise ValueError("Cells mix quick and full runs; evaluate one mode at a time.")
    exp_cfg = cfgs["experiment"]
    order = [(m, t) for m in exp_cfg["models"] for t in exp_cfg["techniques"]]
    return sorted(records, key=lambda r: order.index((r["model"], r["technique"]))
                  if (r["model"], r["technique"]) in order else len(order))


def confidence_intervals(y: pd.Series, scores: np.ndarray, exp_cfg: dict) -> tuple[list, list]:
    """Bootstrap CIs of PR-AUC and ROC-AUC, with the config's resamples, coverage and seed."""
    settings = (int(exp_cfg["bootstrap"]["n_resamples"]), float(exp_cfg["bootstrap"]["confidence"]),
                int(exp_cfg["seed"]))
    return (list(bootstrap_ci(y, scores, average_precision_score, *settings)),
            list(bootstrap_ci(y, scores, roc_auc_score, *settings)))


def evaluate_cell(record: dict, pipeline: object, X_test: pd.DataFrame, y_test: pd.Series,
                  exp_cfg: dict) -> tuple[dict, dict, dict]:
    """Score the test set once with a fitted pipeline: (comparison record, curves, details).

    The F1 / F2 thresholds are READ from the cell record (chosen on out-of-fold training
    scores); they are never recomputed on test data.
    """
    scores = np.asarray(pipeline.predict_proba(X_test)[:, 1], dtype=float)
    threshold_f1, threshold_f2 = record["thresholds"]["f1"], record["thresholds"]["f2"]
    pr_auc_ci, roc_auc_ci = confidence_intervals(y_test, scores, exp_cfg)
    comparison = {
        "model": record["model"], "technique": record["technique"],
        "model_label": MODEL_LABELS[record["model"]],
        "technique_label": TECHNIQUE_LABELS[record["technique"]],
        "best_params": record["best_params"], "chosen_ratio": record["chosen_ratio"],
        "cv_pr_auc_mean": record["cv_pr_auc_mean"], "cv_pr_auc_std": record["cv_pr_auc_std"],
        "cv_roc_auc_mean": record["cv_roc_auc_mean"],
        "threshold_f1": threshold_f1, "threshold_f2": threshold_f2,
        "test_at_0_5": compute_metrics(y_test, scores, 0.5),
        "test_at_tuned": compute_metrics(y_test, scores, threshold_f1),
        "pr_auc_ci": pr_auc_ci, "roc_auc_ci": roc_auc_ci,
        "fit_seconds": record["fit_seconds"],
        "n_train_after_resampling": record["n_train_after_resampling"],
    }
    curves = make_curves(y_test, scores, record["model"], record["technique"])
    details = {
        "model": record["model"], "technique": record["technique"], "n_test": len(y_test),
        "n_test_frauds": int((np.asarray(y_test) == 1).sum()),
        "threshold_f1": threshold_f1, "threshold_f2": threshold_f2,
        "threshold_source": "cell record (out-of-fold training scores)",
        "test_at_f2": compute_metrics(y_test, scores, threshold_f2),
    }
    return comparison, curves, details


def replace_cells(old: list, new: list) -> list:
    """Items of `old` whose (model, technique) is not in `new`, followed by `new`."""
    keys = {(n["model"], n["technique"]) for n in new}
    return [o for o in old if (o["model"], o["technique"]) not in keys] + new


def evaluate_test(source: str, results_dir: Path, models_dir: Path, cells: list | None = None,
                  rerun_reason: str | None = None, guard_path: Path | None = None,
                  decisions_path: Path | None = None, output_dir: Path | None = None,
                  cfgs: dict | None = None, verify_hash: bool = True,
                  log: object = print) -> dict:
    """Score finished cells on the locked test set, once, and write comparison.json / curves.json.

    Order: pick the cells -> check the guard -> rebuild the split and check the training hash
    against run_meta.json and every cell -> record the opening -> score each cell -> write the
    files (earlier records of other cells in them are kept). The guard is on for the real data;
    the sample is exempt unless a guard_path is passed (the tests do that).
    Returns comparison, curves, evaluated (cells scored now) and paths.
    """
    cfgs = cfgs or load_configs()
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else output_dir_for(source, results_dir)
    if guard_path is None and source == "raw":
        guard_path = results_dir / GUARD_FILE
    decisions_path = Path(decisions_path) if decisions_path else resolve(DECISIONS_FILE)

    records = select_cells(results_dir, source, cells, cfgs)
    keys = [(r["model"], r["technique"]) for r in records]
    check_guard(guard_path, rerun_reason, "evaluate_test")
    pipelines = {key: cell_paths(*key, results_dir, models_dir)["pipeline"] for key in keys}
    for path in pipelines.values():
        if not path.is_file():
            raise ValueError(f"Fitted pipeline {path} is missing; copy it from the machine that "
                             "ran the cell or re-run the cell.")
    meta_path = results_dir / RUN_META_FILE
    run_meta = load_json(meta_path) if meta_path.is_file() else None
    split = load_test_split(source, records[0]["quick"], cfgs, verify_hash)
    check_training_hash(split["train_sha256"], records, run_meta)

    if guard_path is not None:
        open_test_set(guard_path, keys, rerun_reason, decisions_path)
    X_test, y_test = split["X_test"], split["y_test"]
    log(f"test set: {len(y_test):,} rows, {int((y_test == 1).sum()):,} frauds; scoring "
        f"{len(records)} cells once")
    comparisons, curves, details = [], [], []
    for record, key in zip(records, keys, strict=True):
        comparison, cell_curves, extra = evaluate_cell(record, joblib.load(pipelines[key]), X_test,
                                                       y_test, cfgs["experiment"])
        extra["pipeline_sha256"] = sha256_file(pipelines[key])
        comparisons.append(comparison)
        curves.append(cell_curves)
        details.append(extra)
        low, high = comparison["pr_auc_ci"]
        log(f"{cell_id(*key):<26} test PR-AUC {comparison['test_at_tuned']['pr_auc']:.4f} "
            f"[{low:.3f}, {high:.3f}]")

    dataset_source = DATASET_SOURCE[source]
    comparison_path, curves_path = output_dir / COMPARISON_FILE, output_dir / CURVES_FILE
    old_comparisons, old_curves = [], []
    if comparison_path.is_file():
        previous = load_json(comparison_path)
        if previous["dataset_source"] == dataset_source:
            old_comparisons = previous["records"]
    if curves_path.is_file() and old_comparisons:
        old_curves = load_json(curves_path)["cells"]
    table = {"contract_version": CONTRACT_VERSION, "is_example": False,
             "primary_metric": "average_precision", "dataset_source": dataset_source,
             "records": replace_cells(old_comparisons, comparisons)}
    curves_file = {"contract_version": CONTRACT_VERSION, "is_example": False,
                   "cells": replace_cells(old_curves, curves)}
    save_json(table, comparison_path)
    save_json(curves_file, curves_path)
    details_path = save_json({
        "note": "Extra test-set facts not in the 1.2 contract (metrics at threshold_f2).",
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"), "source": source,
        "quick": records[0]["quick"], "train_sha256": split["train_sha256"], "cells": details,
    }, output_dir / DETAILS_FILE)
    paths = {"comparison": comparison_path, "curves": curves_path, "details": details_path}
    if guard_path is not None:
        paths["guard"] = Path(guard_path)
    return {"comparison": table, "curves": curves_file, "evaluated": keys, "paths": paths}


def temporal_check(source: str, results_dir: Path, models_dir: Path, model: str = "lightgbm",
                   techniques: list | None = None, rerun_reason: str | None = None,
                   guard_path: Path | None = None, decisions_path: Path | None = None,
                   output_dir: Path | None = None, cfgs: dict | None = None,
                   verify_hash: bool = True, log: object = print) -> dict:
    """SENSITIVITY ANALYSIS: refit each CV-chosen pipeline of one model on the earlier part of a
    temporal split (by Time) and score the later part, at 0.5 and at the cell's F1 threshold.

    The temporal parts contain locked test rows, so on the real data this runs only AFTER
    evaluate_test and is logged in the guard file. Nothing it produces may change a tuned value.
    Returns the temporal_check.json contents (also written).
    """
    cfgs = cfgs or load_configs()
    exp_cfg = cfgs["experiment"]
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else output_dir_for(source, results_dir)
    if guard_path is None and source == "raw":
        guard_path = results_dir / GUARD_FILE
    decisions_path = Path(decisions_path) if decisions_path else resolve(DECISIONS_FILE)
    techniques = list(techniques or exp_cfg["techniques"])
    records = [r for r in finished_cells(results_dir, source)
               if r["model"] == model and r["technique"] in techniques]
    if not records:
        raise ValueError(f"No finished {source} cells for model {model!r} in "
                         f"{results_dir / 'cells'}.")
    records.sort(key=lambda r: exp_cfg["techniques"].index(r["technique"]))
    if guard_path is not None:
        open_test_set(guard_path, [(r["model"], r["technique"]) for r in records], rerun_reason,
                      decisions_path, action="temporal_check")

    df = load_dataset(source, verify_hash=verify_hash, cfg=cfgs["dataset"])
    deduped, _ = deduplicate(df, target=cfgs["dataset"]["target"])
    split = split_dataset(deduped, cfgs["dataset"], exp_cfg, strategy="temporal")
    X_train, y_train, X_test, y_test = (split["X_train"], split["y_train"], split["X_test"],
                                        split["y_test"])
    cutoff = float(deduped.loc[X_test.index, cfgs["dataset"]["time_column"]].min())
    log(f"temporal split: train {len(y_train):,} rows ({int(y_train.sum())} frauds), test "
        f"{len(y_test):,} rows ({int(y_test.sum())} frauds), Time cutoff {cutoff:.0f} s")

    cells, skipped = [], []
    for r in records:
        name = cell_id(r["model"], r["technique"])
        path = cell_paths(r["model"], r["technique"], results_dir, models_dir)["pipeline"]
        if not path.is_file():
            skipped.append({"cell": name, "reason": f"pipeline file {path.name} missing"})
            continue
        pipe = clone(joblib.load(path))  # same settings, not yet fitted
        try:
            pipe.fit(X_train, y_train)
        except ValueError as error:  # e.g. a ratio at or below the temporal training ratio
            skipped.append({"cell": name, "reason": f"refit failed: {error}"})
            log(f"{name:<26} skipped ({error})")
            continue
        scores = np.asarray(pipe.predict_proba(X_test)[:, 1], dtype=float)
        pr_auc_ci, roc_auc_ci = confidence_intervals(y_test, scores, exp_cfg)
        at_tuned = compute_metrics(y_test, scores, r["thresholds"]["f1"])
        cells.append({"model": r["model"], "technique": r["technique"],
                      "best_params": r["best_params"], "threshold_f1": r["thresholds"]["f1"],
                      "cv_pr_auc_mean": r["cv_pr_auc_mean"],
                      "temporal_at_0_5": compute_metrics(y_test, scores, 0.5),
                      "temporal_at_tuned": at_tuned, "pr_auc_ci": pr_auc_ci,
                      "roc_auc_ci": roc_auc_ci})
        log(f"{name:<26} temporal PR-AUC {at_tuned['pr_auc']:.4f} "
            f"[{pr_auc_ci[0]:.3f}, {pr_auc_ci[1]:.3f}]")

    result = {
        "analysis": "temporal_split_sensitivity_check", "is_sensitivity_analysis": True,
        "label": TEMPORAL_LABEL, "dataset_source": DATASET_SOURCE[source], "model": model,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"), "git_commit": git_hash(),
        "split": {"strategy": "temporal", "time_cutoff_seconds": cutoff,
                  "train": {"rows": len(y_train), "frauds": int(y_train.sum())},
                  "test": {"rows": len(y_test), "frauds": int(y_test.sum())}},
        "cells": cells, "skipped": skipped,
    }
    save_json(result, output_dir / TEMPORAL_FILE)
    return result
