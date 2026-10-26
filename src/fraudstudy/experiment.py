"""The experiment: tune and score every (model, technique) cell on the TRAINING split, save each
cell, and merge cells run on several laptops.

scripts/run_experiments.py calls run_experiments(), scripts/merge_results.py calls
merge_results(); final_test.py and export.py read the saved cells. Writes, per cell,
results/cells/<cell>.json (a plain dict, written last so its presence means "finished"),
results/cells/<cell>_oof.npy, rows of results/fold_scores.csv and models/cells/<cell>.joblib,
plus results/run_meta.json. The test split is never loaded here.
"""

import hashlib
import logging
import math
import platform
import shutil
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)

from fraudstudy.config import load_configs, resolve, set_seed
from fraudstudy.data import deduplicate, load_dataset, split_dataset
from fraudstudy.io import git_hash, load_json, save_json, sha256_file
from fraudstudy.metrics import best_threshold
from fraudstudy.models import PARALLEL_MODELS
from fraudstudy.pipeline import build_pipeline
from fraudstudy.sampling import (
    CLASS_WEIGHT_PARAM,
    base_ratio,
    resampling_audit,
    technique_search_space,
    valid_ratios,
)

logger = logging.getLogger(__name__)

MODEL_LABELS = {"logreg": "Logistic regression", "random_forest": "Random forest",
                "lightgbm": "LightGBM"}
TECHNIQUE_LABELS = {
    "baseline": "A. Baseline (no resampling)",
    "ros": "B. Random oversampling",
    "rus": "C. Random undersampling",
    "smote": "D. SMOTE",
    "smote_rus": "E. SMOTE + random undersampling",
    "class_weight": "F. Class weighting",
}
RUN_META_FILE = "run_meta.json"
FOLD_COLUMNS = ["model", "technique", "repeat", "fold", "fold_id", "pr_auc", "roc_auc"]
# Cell fields that differ between two runs of the same cell (clock, machine, commit).
CLOCK_FIELDS = ["search_seconds", "fit_seconds", "predict_seconds", "total_seconds",
                "started_at", "finished_at", "git_commit", "library_versions"]
LIBRARIES = ["numpy", "pandas", "scikit-learn", "imbalanced-learn", "lightgbm", "scipy",
             "joblib", "pydantic"]


def cell_id(model: str, technique: str) -> str:
    """File name stem of a cell, e.g. "lightgbm__smote" (the manifest uses the same ids)."""
    return f"{model}__{technique}"


def cell_paths(model: str, technique: str, results_dir: Path,
               models_dir: Path | None = None) -> dict:
    """Paths of a cell's record (.json), out-of-fold scores (.npy) and, if asked, pipeline."""
    name = cell_id(model, technique)
    paths = {"record": Path(results_dir) / "cells" / f"{name}.json",
             "oof": Path(results_dir) / "cells" / f"{name}_oof.npy"}
    if models_dir is not None:
        paths["pipeline"] = Path(models_dir) / "cells" / f"{name}.joblib"
    return paths


def protocol(exp_cfg: dict, quick: bool) -> dict:
    """CV folds, repeats, tuning budget and training share: the `quick` block or the full run."""
    if quick:
        q = exp_cfg["quick"]
        return {"cv_folds": int(q["cv_folds"]), "cv_repeats": int(q["cv_repeats"]),
                "n_iter": int(q["n_iter"]), "train_fraction": float(q["train_fraction"])}
    return {"cv_folds": int(exp_cfg["cv_folds"]), "cv_repeats": int(exp_cfg["cv_repeats"]),
            "n_iter": int(exp_cfg["tuning"]["n_iter"]), "train_fraction": 1.0}


def build_search_space(model: str, technique: str, models_cfg: dict, exp_cfg: dict,
                       base: float) -> dict | list:
    """The model's grid from models.yaml (as model__<param>) plus the technique's ratio grid.

    smote_rus has a list of grids (one per ratio pair), so the model grid joins each of them.
    """
    if CLASS_WEIGHT_PARAM[model] in models_cfg[model]:
        raise ValueError(f"configs/models.yaml must not tune {CLASS_WEIGHT_PARAM[model]!r} for "
                         f"{model}; class weighting is technique 'class_weight'.")
    model_space = {f"model__{key}": list(values) for key, values in models_cfg[model].items()}
    technique_space = technique_search_space(technique, exp_cfg, base, model_name=model)
    if isinstance(technique_space, list):
        return [{**model_space, **grid} for grid in technique_space]
    return {**model_space, **technique_space}


def space_size(space: dict | list) -> int:
    """Number of different configurations in a search space."""
    grids = space if isinstance(space, list) else [space]
    return sum(math.prod(len(values) for values in grid.values()) for grid in grids)


def fold_id(index: pd.Index) -> str:
    """Short hash of a validation fold's row labels: equal folds get equal ids."""
    text = ",".join(map(str, sorted(index.tolist())))
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def training_sha256(X: pd.DataFrame, y: pd.Series) -> str:
    """SHA-256 of the training rows (index, values, labels): proves cells saw the same data."""
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(X, index=True).to_numpy().tobytes())
    digest.update(np.asarray(y).astype(np.int64).tobytes())
    return digest.hexdigest()


def library_versions() -> dict:
    """Installed versions of Python and the modelling libraries."""
    versions = {"python": platform.python_version()}
    for name in LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def plain_params(params: dict) -> dict:
    """Search parameters as plain scalars: a class weight {0: 1, 1: w} is stored as w."""
    out = {}
    for key, value in params.items():
        if isinstance(value, dict):
            value = value[1]
        out[key] = value.item() if isinstance(value, np.generic) else value
    return out


def ratio_info(technique: str, best_params: dict, base: float) -> tuple:
    """(chosen_ratio, smote_ratio, fraud_weight) of the best configuration.

    chosen_ratio is the sampler ratio (for smote_rus the RUS ratio b); for class_weight it is
    w * base ("balanced" counts as 1.0: equal weighted class mass); None for the baseline.
    """
    if technique in ("ros", "rus", "smote"):
        return float(best_params[f"{technique}__sampling_strategy"]), None, None
    if technique == "smote_rus":
        return (float(best_params["rus__sampling_strategy"]),
                float(best_params["smote__sampling_strategy"]), None)
    if technique == "class_weight":
        weight = next(v for k, v in best_params.items()
                      if k in ("model__class_weight", "model__scale_pos_weight"))
        if weight == "balanced":
            return 1.0, None, "balanced"
        return float(weight) * base, None, float(weight)
    return None, None, None


def stratified_subsample(X: pd.DataFrame, y: pd.Series, fraction: float,
                         seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """Keep a stratified `fraction` of the rows (quick mode on the real data); all if >= 1."""
    if fraction >= 1.0:
        return X, y
    X_sub, _, y_sub, _ = train_test_split(X, y, train_size=fraction, stratify=y, random_state=seed)
    return X_sub, y_sub


def load_training_data(source: str, quick: bool, cfgs: dict,
                       verify_hash: bool = True) -> tuple[pd.DataFrame, pd.Series, str]:
    """Load, deduplicate and split, and return ONLY the training part plus the data file's hash.

    In quick mode on the real data a stratified share of the training rows is kept (the
    synthetic sample is already small, so it is used whole).
    """
    df = load_dataset(source, verify_hash=verify_hash, cfg=cfgs["dataset"])
    deduped, _ = deduplicate(df, target=cfgs["dataset"]["target"])
    split = split_dataset(deduped, cfgs["dataset"], cfgs["experiment"], strategy="stratified")
    X, y = split["X_train"], split["y_train"]
    if quick and source == "raw":
        fraction = protocol(cfgs["experiment"], quick=True)["train_fraction"]
        X, y = stratified_subsample(X, y, fraction, int(cfgs["experiment"]["seed"]))
    path_key = "raw_path" if source == "raw" else "sample_path"
    return X, y, sha256_file(resolve(cfgs["dataset"][path_key]))


def run_cell(model: str, technique: str, X_train: pd.DataFrame, y_train: pd.Series, cfgs: dict,
             quick: bool) -> tuple[dict, np.ndarray, object]:
    """Tune and score one cell on TRAINING rows only (there is no test-set argument on purpose).

    1. RandomizedSearchCV over model grid + technique ratios, n_iter = min(budget, space size),
       scored by average precision on StratifiedKFold(cv_folds).
    2. Out-of-fold scores of the best configuration on the same folds; F1 / F2 thresholds.
    3. RepeatedStratifiedKFold fold scores. The folds depend only on the seed and the labels,
       so every cell gets the same folds and the statistics can pair them.
    4. Resampling audit of the best configuration, and fit / predict times.
    Returns (record, out-of-fold scores, best pipeline refitted on all training rows).
    """
    started, start = datetime.now(UTC), time.perf_counter()
    exp_cfg = cfgs["experiment"]
    seed, k = int(exp_cfg["seed"]), int(exp_cfg["smote_k_neighbors"])
    proto = protocol(exp_cfg, quick)
    model_jobs = exp_cfg["parallel"]["model_n_jobs"]
    search_jobs = exp_cfg["parallel"]["search_n_jobs"]
    base = base_ratio(y_train)
    cv = StratifiedKFold(n_splits=proto["cv_folds"], shuffle=True, random_state=seed)
    repeated = list(RepeatedStratifiedKFold(n_splits=proto["cv_folds"],
                                            n_repeats=proto["cv_repeats"],
                                            random_state=seed).split(X_train, y_train))
    labels = np.asarray(y_train)
    fewest = min(int((labels[train] == 1).sum())
                 for train, _ in list(cv.split(X_train, y_train)) + repeated)
    if fewest == 0 or (technique.startswith("smote") and fewest <= k):
        raise ValueError(f"A CV training fold has only {fewest} frauds; {technique} needs more "
                         f"than smote_k_neighbors={k}. Use more data (drop --quick or use "
                         "--source raw).")

    # Same budget for every cell; only a space with fewer configurations gets fewer tries.
    space = build_search_space(model, technique, cfgs["models"], exp_cfg, base)
    size = space_size(space)
    n_iter = min(proto["n_iter"], size)
    if n_iter < proto["n_iter"]:
        logger.info("%s: search space has %d configurations, fewer than the budget %d; trying "
                    "all.", cell_id(model, technique), size, proto["n_iter"])
    model_params = {"n_jobs": model_jobs} if model in PARALLEL_MODELS else {}
    # Technique F starts at weight 1; the search then sets the real weight.
    technique_params = {"fraud_weight": 1.0} if technique == "class_weight" else {}
    pipe = build_pipeline(model, technique, model_params, technique_params, cfgs["dataset"],
                          exp_cfg)
    search = RandomizedSearchCV(pipe, space, n_iter=n_iter, scoring="average_precision", cv=cv,
                                refit=True, random_state=seed, n_jobs=search_jobs,
                                error_score="raise")
    search_start = time.perf_counter()
    search.fit(X_train, y_train)
    search_seconds = time.perf_counter() - search_start
    best = search.best_estimator_
    predict_start = time.perf_counter()
    best.predict_proba(X_train)
    predict_seconds = time.perf_counter() - predict_start

    oof = cross_val_predict(clone(best), X_train, y_train, cv=cv, method="predict_proba",
                            n_jobs=search_jobs)[:, 1]
    thresholds = {str(name): best_threshold(y_train, oof, name)
                  for name in exp_cfg["threshold"]["objectives"]}
    scores = cross_validate(clone(best), X_train, y_train, cv=repeated,
                            scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc"},
                            n_jobs=search_jobs, error_score="raise")
    folds = [{"repeat": i // proto["cv_folds"], "fold": i % proto["cv_folds"],
              "fold_id": fold_id(X_train.index[validation]),
              "pr_auc": float(scores["test_pr_auc"][i]),
              "roc_auc": float(scores["test_roc_auc"][i])}
             for i, (_, validation) in enumerate(repeated)]
    pr_auc = np.array([f["pr_auc"] for f in folds])
    roc_auc = np.array([f["roc_auc"] for f in folds])

    audit = resampling_audit(best, X_train, y_train)
    best_params = plain_params(search.best_params_)
    chosen_ratio, smote_ratio, fraud_weight = ratio_info(technique, best_params, base)
    grid = [float(r) for r in exp_cfg["ratio_grid"]]
    kept = valid_ratios(grid, base)
    results = search.cv_results_
    candidates = [{"params": plain_params(params),
                   "mean_test_score": float(results["mean_test_score"][i]),
                   "std_test_score": float(results["std_test_score"][i]),
                   "rank_test_score": int(results["rank_test_score"][i]),
                   "split_test_scores": [float(results[f"split{s}_test_score"][i])
                                         for s in range(proto["cv_folds"])],
                   "mean_fit_time": float(results["mean_fit_time"][i])}
                  for i, params in enumerate(results["params"])]
    record = {
        "model": model, "technique": technique, "source": None, "quick": quick, "seed": seed,
        "cv_folds": proto["cv_folds"], "cv_repeats": proto["cv_repeats"],
        "n_iter_budget": proto["n_iter"], "search_space_size": size,
        "n_candidates_evaluated": len(results["params"]),
        "n_train": len(y_train), "n_train_frauds": int((labels == 1).sum()),
        "train_sha256": training_sha256(X_train, y_train), "base_ratio": base,
        "dropped_ratios": [r for r in grid if r not in kept],
        "best_params": best_params, "chosen_ratio": chosen_ratio, "smote_ratio": smote_ratio,
        "fraud_weight": fraud_weight, "search_best_pr_auc": float(search.best_score_),
        "candidates": candidates,
        "oof_pr_auc": float(average_precision_score(y_train, oof)),
        "oof_roc_auc": float(roc_auc_score(y_train, oof)), "thresholds": thresholds,
        "cv_pr_auc_mean": float(pr_auc.mean()), "cv_pr_auc_std": float(pr_auc.std()),
        "cv_roc_auc_mean": float(roc_auc.mean()), "cv_roc_auc_std": float(roc_auc.std()),
        "fold_scores": folds,
        "n_train_after_resampling": {"legit": audit["legit_after"], "fraud": audit["fraud_after"]},
        "n_synthetic": audit["n_synthetic"],
        "fraud_duplicates_after": audit["fraud_duplicates_after"],
        "search_seconds": search_seconds, "fit_seconds": float(search.refit_time_),
        "predict_seconds": predict_seconds, "total_seconds": time.perf_counter() - start,
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data_sha256": None, "git_commit": None, "library_versions": library_versions(),
    }
    return record, np.asarray(oof, dtype=float), best


def read_fold_scores(path: Path) -> pd.DataFrame:
    """fold_scores.csv, or an empty table with the right columns if it does not exist yet."""
    if not Path(path).is_file():
        return pd.DataFrame(columns=FOLD_COLUMNS)
    return pd.read_csv(path, dtype={"fold_id": str})[FOLD_COLUMNS]


def fold_rows(record: dict) -> pd.DataFrame:
    """A cell's repeated-CV fold scores as rows of fold_scores.csv."""
    rows = [{"model": record["model"], "technique": record["technique"], **fold}
            for fold in record["fold_scores"]]
    return pd.DataFrame(rows, columns=FOLD_COLUMNS)


def save_fold_scores(new_rows: pd.DataFrame, path: Path) -> None:
    """Write new_rows into fold_scores.csv, replacing older rows of the same cells."""
    old = read_fold_scores(path)
    cells = set(zip(new_rows["model"], new_rows["technique"], strict=True))
    keep = [(m, t) not in cells for m, t in zip(old["model"], old["technique"], strict=True)]
    frames = [df for df in (old[keep], new_rows) if not df.empty]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    (pd.concat(frames, ignore_index=True) if frames else new_rows).to_csv(path, index=False)


def save_cell(record: dict, oof: np.ndarray, pipeline: object, results_dir: Path,
              models_dir: Path) -> dict:
    """Save a finished cell: OOF scores, pipeline, fold rows, and the record LAST.

    If the run crashes mid-save there is no record, so a resumed run redoes the cell instead of
    trusting half-written files.
    """
    paths = cell_paths(record["model"], record["technique"], results_dir, models_dir)
    paths["oof"].parent.mkdir(parents=True, exist_ok=True)
    np.save(paths["oof"], np.asarray(oof, dtype=float))
    paths["pipeline"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, paths["pipeline"])
    save_fold_scores(fold_rows(record), Path(results_dir) / "fold_scores.csv")
    save_json(record, paths["record"])
    return paths


def is_cell_done(model: str, technique: str, results_dir: Path, quick: bool, source: str,
                 seed: int) -> bool:
    """True if the cell's files exist and were made with the same mode, source and seed.

    So a full run never silently reuses quick results.
    """
    paths = cell_paths(model, technique, results_dir)
    if not (paths["record"].is_file() and paths["oof"].is_file()):
        return False
    record = load_json(paths["record"])
    return (record["quick"], record["source"], record["seed"]) == (quick, source, seed)


def run_experiments(source: str, quick: bool, results_dir: Path, models_dir: Path,
                    models: list | None = None, techniques: list | None = None,
                    force: bool = False, verify_hash: bool = True, cfgs: dict | None = None,
                    log: object = print) -> list:
    """Run every requested cell, skipping finished ones unless force, and save each at once.

    Returns (record, "ran" or "skipped") per cell, in model-then-technique order, and writes
    run_meta.json (configs, git hash, library versions, data hashes, start and end time).
    """
    cfgs = cfgs or load_configs()
    exp_cfg = cfgs["experiment"]
    models = list(models or exp_cfg["models"])
    techniques = list(techniques or exp_cfg["techniques"])
    seed = int(exp_cfg["seed"])
    set_seed(seed)
    started, start = datetime.now(UTC), time.perf_counter()

    X, y, data_sha = load_training_data(source, quick, cfgs, verify_hash)
    commit = git_hash()
    cells = [(m, t) for m in models for t in techniques]
    log(f"source {source}{' (quick)' if quick else ''}: {len(y):,} training rows, "
        f"{int((y == 1).sum()):,} frauds; {len(cells)} cells; test split not used")
    out = []
    for i, (model, technique) in enumerate(cells, start=1):
        prefix = f"[{i}/{len(cells)}] {cell_id(model, technique):<26}"
        if not force and is_cell_done(model, technique, results_dir, quick, source, seed):
            out.append((load_json(cell_paths(model, technique, results_dir)["record"]), "skipped"))
            log(f"{prefix} skipped (finished; use --force to re-run)")
            continue
        cell_start = time.perf_counter()
        record, oof, pipeline = run_cell(model, technique, X, y, cfgs, quick)
        record.update(source=source, data_sha256=data_sha, git_commit=commit)
        save_cell(record, oof, pipeline, results_dir, models_dir)
        out.append((record, "ran"))
        log(f"{prefix} PR-AUC {record['cv_pr_auc_mean']:.4f} +- {record['cv_pr_auc_std']:.4f}  "
            f"tries {record['n_candidates_evaluated']:>2}  "
            f"{time.perf_counter() - cell_start:6.1f} s")

    save_json({
        "source": source, "quick": quick, "protocol": protocol(exp_cfg, quick),
        "parallel": dict(exp_cfg["parallel"]),
        "configs": {"dataset": cfgs["dataset"], "experiment": exp_cfg, "models": cfgs["models"]},
        "git_commit": commit, "library_versions": library_versions(), "data_sha256": data_sha,
        "train_sha256": training_sha256(X, y), "n_train": len(y),
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "elapsed_seconds": time.perf_counter() - start,
        "cells_run": [cell_id(r["model"], r["technique"]) for r, s in out if s == "ran"],
        "cells_skipped": [cell_id(r["model"], r["technique"]) for r, s in out if s == "skipped"],
    }, Path(results_dir) / RUN_META_FILE)
    return out


def scientific_view(record: dict) -> dict:
    """The record without clock and machine fields: two runs of a cell must give equal views."""
    view = {key: value for key, value in record.items() if key not in CLOCK_FIELDS}
    view["candidates"] = [{k: v for k, v in c.items() if k != "mean_fit_time"}
                          for c in record["candidates"]]
    return view


def merge_results(input_dirs: list, output_dir: Path, models: list, techniques: list) -> dict:
    """Combine cells from several laptops' results folders into one folder.

    Copies the cell .json / _oof.npy files into output_dir/cells (pipelines stay on each
    laptop), rebuilds fold_scores.csv from the cell records, and writes cv_summary.json.
    Cells must share seed, mode, source, CV settings, training data and folds, or the paired
    statistics would be invalid. The same cell found twice: identical is fine, different keeps
    the one that finished last. Returns cells, missing, duplicates, conflicts and paths.
    """
    found, duplicates, conflicts = {}, [], []
    for folder in input_dirs:
        for path in sorted((Path(folder) / "cells").glob("*__*.json")):
            record = load_json(path)
            key = (record["model"], record["technique"])
            if key in found:
                old = found[key][0]
                same = scientific_view(old) == scientific_view(record)
                (duplicates if same else conflicts).append(key)
                if not same:
                    logger.warning("%s differs between inputs; keeping the latest.", cell_id(*key))
                if record["finished_at"] <= old["finished_at"]:
                    continue
            found[key] = (record, path)
    if not found:
        raise ValueError(f"No cell records found under {[str(d) for d in input_dirs]} (expected "
                         "<dir>/cells/<model>__<technique>.json). Run scripts/run_experiments.py "
                         "first.")
    order = [(m, t) for m in models for t in techniques]
    keys = sorted(found, key=lambda k: (order.index(k) if k in order else len(order), k))
    records = [found[k][0] for k in keys]
    for field in ("seed", "quick", "source", "cv_folds", "cv_repeats", "train_sha256"):
        values = {str(r[field]) for r in records}
        if len(values) > 1:
            raise ValueError(f"Cells disagree on {field} ({sorted(values)}); they come from "
                             "different protocols or data and cannot be compared. Re-run the odd "
                             "ones.")
    if len({frozenset(f["fold_id"] for f in r["fold_scores"]) for r in records}) > 1:
        raise ValueError("Cells have different repeated-CV folds, so paired tests are invalid. "
                         "Re-run them with the same seed and data.")

    output_dir = Path(output_dir)
    (output_dir / "cells").mkdir(parents=True, exist_ok=True)
    for key in keys:
        record, source_path = found[key]
        target = cell_paths(*key, output_dir)
        if source_path.resolve() != target["record"].resolve():
            shutil.copy2(source_path, target["record"])
            source_oof = source_path.with_name(target["oof"].name)
            if source_oof.is_file():
                shutil.copy2(source_oof, target["oof"])
    folds = pd.concat([fold_rows(r) for r in records], ignore_index=True)
    folds = folds.drop_duplicates(subset=["model", "technique", "repeat", "fold"], keep="last")
    folds.to_csv(output_dir / "fold_scores.csv", index=False)

    missing = [k for k in order if k not in found]
    summary_keys = ["model", "technique", "best_params", "chosen_ratio", "smote_ratio",
                    "fraud_weight", "cv_pr_auc_mean", "cv_pr_auc_std", "cv_roc_auc_mean",
                    "cv_roc_auc_std", "oof_pr_auc", "thresholds", "n_candidates_evaluated",
                    "fit_seconds", "n_train_after_resampling"]
    first = records[0]
    summary = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "primary_metric": "average_precision", "seed": first["seed"], "quick": first["quick"],
        "source": first["source"], "cv_folds": first["cv_folds"],
        "cv_repeats": first["cv_repeats"], "n_cells": len(records),
        "missing": [{"model": m, "technique": t} for m, t in missing],
        "cells": [{key: r[key] for key in summary_keys} for r in records],
    }
    summary_path = save_json(summary, output_dir / "cv_summary.json")
    return {"cells": keys, "missing": missing, "duplicates": duplicates, "conflicts": conflicts,
            "paths": {"cv_summary": summary_path, "fold_scores": output_dir / "fold_scores.csv"}}


def source_fold_scores(results_dir: Path, source: str) -> pd.DataFrame:
    """fold_scores.csv rows of the finished cells run on `source` (never mix sample and real)."""
    records = [load_json(p) for p in sorted((Path(results_dir) / "cells").glob("*.json"))]
    cells = {(r["model"], r["technique"]) for r in records if r["source"] == source}
    scores = read_fold_scores(Path(results_dir) / "fold_scores.csv")
    keep = [(m, t) in cells for m, t in zip(scores["model"], scores["technique"], strict=True)]
    scores = scores[keep].reset_index(drop=True)
    if scores.empty:
        raise ValueError(f"No fold scores of finished {source} cells in "
                         f"{Path(results_dir) / 'fold_scores.csv'}; run "
                         f"scripts/run_experiments.py --source {source} first.")
    return scores
