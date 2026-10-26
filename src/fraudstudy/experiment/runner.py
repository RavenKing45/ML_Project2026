"""Equal-budget tuning and evaluation of one (model, technique) cell on TRAINING data only.

Protocol per cell (master guide Part 6, "Tuning, fairly" to "Measuring uncertainty"):

1. Search space = the model's space from ``configs/models.yaml`` (prefixed ``model__``) plus the
   technique's ratio space (:func:`fraudstudy.sampling.technique_search_space`).
2. ``RandomizedSearchCV`` with ``n_iter = min(budget, space size)``, scored by average
   precision on ``StratifiedKFold(cv_folds, shuffle=True, random_state=seed)``, ``refit=True``.
3. Out-of-fold scores of the best configuration (same folds), and F1 / F2 thresholds on them.
4. ``RepeatedStratifiedKFold(cv_folds, cv_repeats, random_state=seed)`` fold scores. The folds
   depend only on the seed and the training labels, so every cell gets the same folds.
5. Resampling audit of the best configuration on the training set; fit and predict times.

No early stopping is used; the only held-out data are the CV folds inside each step. The test
set is never passed to, or loaded by, :func:`run_cell`.
"""

import hashlib
import logging
import math
import platform
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

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

from fraudstudy.config import (
    get_dataset_config,
    get_experiment_config,
    get_models_config,
    resolve,
    set_seed,
)
from fraudstudy.contracts import ClassCounts
from fraudstudy.data import deduplicate, load_dataset, split_dataset
from fraudstudy.evaluation import best_threshold
from fraudstudy.experiment.results import (
    Candidate,
    CellRecord,
    CellResult,
    FoldScore,
    cell_id,
    cell_paths,
    is_cell_done,
    load_cell_record,
    save_cell,
    write_run_meta,
)
from fraudstudy.io import git_hash, sha256_file
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

SCORING = "average_precision"
FOLD_SCORING: dict[str, str] = {"pr_auc": "average_precision", "roc_auc": "roc_auc"}
FOLD_ID_CHARS = 16
LIBRARIES: tuple[str, ...] = (
    "numpy",
    "pandas",
    "scikit-learn",
    "imbalanced-learn",
    "lightgbm",
    "scipy",
    "joblib",
    "pydantic",
)
# Starting fraud weight for technique F; always overridden by the searched weight.
NEUTRAL_FRAUD_WEIGHT = 1.0


@dataclass(frozen=True)
class Configs:
    """The three YAML configs (``dataset``, ``experiment``, ``models``)."""

    dataset: dict[str, Any]
    experiment: dict[str, Any]
    models: dict[str, Any]

    @classmethod
    def load(cls) -> "Configs":
        """Load all three from ``configs/``."""
        return cls(get_dataset_config(), get_experiment_config(), get_models_config())


@dataclass(frozen=True)
class Protocol:
    """CV and budget settings of a run (full or quick).

    Attributes:
        cv_folds: Folds of the search / OOF CV and of each repeat.
        cv_repeats: Repeats of the repeated CV.
        n_iter: Tuning budget (configurations tried per cell, before capping by space size).
        train_fraction: Share of training rows kept (raw source only; 1.0 for full runs).
    """

    cv_folds: int
    cv_repeats: int
    n_iter: int
    train_fraction: float


def get_protocol(exp_cfg: dict[str, Any], quick: bool) -> Protocol:
    """Protocol from ``experiment.yaml``: the ``quick`` block, or the full settings.

    Args:
        exp_cfg: Contents of ``configs/experiment.yaml``.
        quick: Use the ``quick`` block.

    Returns:
        The :class:`Protocol`.
    """
    if quick:
        q = exp_cfg["quick"]
        return Protocol(
            int(q["cv_folds"]), int(q["cv_repeats"]), int(q["n_iter"]), float(q["train_fraction"])
        )
    return Protocol(
        int(exp_cfg["cv_folds"]), int(exp_cfg["cv_repeats"]), int(exp_cfg["tuning"]["n_iter"]), 1.0
    )


def parallel_settings(exp_cfg: dict[str, Any]) -> tuple[int, int]:
    """``(model_n_jobs, search_n_jobs)`` from ``parallel`` in ``experiment.yaml``.

    Models use several cores while the search and CV fit one candidate at a time, so memory
    holds one resampled training fold at once.

    Raises:
        ValueError: If the ``parallel`` block is missing or incomplete.
    """
    block = exp_cfg.get("parallel")
    if not isinstance(block, dict) or not {"model_n_jobs", "search_n_jobs"} <= set(block):
        raise ValueError(
            "experiment.yaml needs `parallel: {model_n_jobs: -1, search_n_jobs: 1}` "
            "(cores per model fit, and parallel search fits)."
        )
    return int(block["model_n_jobs"]), int(block["search_n_jobs"])


def model_search_space(model: str, models_cfg: dict[str, Any]) -> dict:
    """The model's hyper-parameter grid from ``models.yaml``, keys prefixed with ``model__``.

    Args:
        model: Model id.
        models_cfg: Contents of ``configs/models.yaml``.

    Returns:
        ``{"model__<param>": [values]}``.

    Raises:
        ValueError: If the model has no space, a value is not a non-empty list, or the space
            sets the class weight (that is technique F's job).
    """
    space = models_cfg.get(model)
    if not isinstance(space, dict) or not space:
        raise ValueError(f"configs/models.yaml has no search space for model {model!r}.")
    weight_param = CLASS_WEIGHT_PARAM.get(model)
    if weight_param in space:
        raise ValueError(
            f"configs/models.yaml must not tune {weight_param!r} for {model}; class weighting "
            "is technique 'class_weight'."
        )
    for key, values in space.items():
        if not isinstance(values, list) or not values:
            raise ValueError(
                f"configs/models.yaml {model}.{key} must be a non-empty list; got {values!r}."
            )
    return {f"model__{key}": list(values) for key, values in space.items()}


def build_search_space(
    model: str,
    technique: str,
    models_cfg: dict[str, Any],
    exp_cfg: dict[str, Any],
    base: float,
) -> dict | list:
    """Model space + technique space; for a list-of-dicts technique space, merged into each dict.

    Args:
        model: Model id.
        technique: Technique id.
        models_cfg: Contents of ``configs/models.yaml``.
        exp_cfg: Contents of ``configs/experiment.yaml``.
        base: ``frauds / legit`` of the training set.

    Returns:
        A ``RandomizedSearchCV`` parameter space (dict, or list of dicts for ``smote_rus``).
    """
    model_space = model_search_space(model, models_cfg)
    tech_space = technique_search_space(technique, exp_cfg, base, model_name=model)
    if isinstance(tech_space, list):
        return [{**model_space, **d} for d in tech_space]
    return {**model_space, **tech_space}


def space_size(space: dict | list) -> int:
    """Number of distinct configurations: ``sum over dicts of prod(len(values))``."""
    dicts = space if isinstance(space, list) else [space]
    return sum(math.prod(len(v) for v in d.values()) for d in dicts)


def fold_id(index: pd.Index) -> str:
    """Short SHA-256 of a validation fold's sorted row labels (equal folds -> equal ids)."""
    text = ",".join(map(str, sorted(index.tolist())))
    return hashlib.sha256(text.encode()).hexdigest()[:FOLD_ID_CHARS]


def training_sha256(X: pd.DataFrame, y: pd.Series) -> str:
    """SHA-256 of the training rows (index, values and labels), to prove cells saw equal data."""
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(X, index=True).to_numpy().tobytes())
    digest.update(np.asarray(y).astype(np.int64).tobytes())
    return digest.hexdigest()


def library_versions() -> dict[str, str]:
    """Installed versions of Python and the modelling libraries."""
    versions = {"python": platform.python_version()}
    for name in LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def _builtin(value: Any) -> Any:
    """NumPy scalar -> Python scalar; anything else unchanged."""
    return value.item() if isinstance(value, np.generic) else value


def scalar_params(params: dict[str, Any]) -> dict[str, Any]:
    """Pipeline parameters with scalar values: a ``{0: 1, 1: w}`` class weight becomes ``w``."""
    out: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, dict):
            value = value.get(1, value.get(1.0))
        out[key] = _builtin(value)
    return out


def ratio_info(
    technique: str, best_params: dict[str, Any], base: float
) -> tuple[float | None, float | None, float | str | None]:
    """``(chosen_ratio, smote_ratio, fraud_weight)`` of the best configuration.

    ``chosen_ratio``: the sampler ratio; for ``smote_rus`` the RUS ratio b; for ``class_weight``
    ``w x base`` (``"balanced"`` gives 1.0: equal weighted class mass); ``None`` for baseline.

    Args:
        technique: Technique id.
        best_params: Scalar best parameters (:func:`scalar_params`).
        base: ``frauds / legit`` of the training set.

    Returns:
        The triple described above.
    """
    if technique in ("ros", "rus", "smote"):
        return float(best_params[f"{technique}__sampling_strategy"]), None, None
    if technique == "smote_rus":
        b = float(best_params["rus__sampling_strategy"])
        return b, float(best_params["smote__sampling_strategy"]), None
    if technique == "class_weight":
        weight_keys = {f"model__{param}" for param in CLASS_WEIGHT_PARAM.values()}
        weight = next(v for k, v in best_params.items() if k in weight_keys)
        if weight == "balanced":
            return 1.0, None, "balanced"
        return float(weight) * base, None, float(weight)
    return None, None, None


def _check_enough_frauds(
    y: pd.Series, splits: list[tuple[np.ndarray, np.ndarray]], technique: str, k: int
) -> None:
    """Raise unless every training fold can be resampled (SMOTE needs more than k frauds)."""
    y_arr = np.asarray(y)
    fewest = min(int((y_arr[train] == 1).sum()) for train, _ in splits)
    if fewest == 0 or (technique.startswith("smote") and fewest <= k):
        raise ValueError(
            f"A CV training fold has only {fewest} frauds; {technique} needs more than "
            f"smote_k_neighbors={k}. Use more data (drop --quick or use --source raw)."
        )


def _candidates(cv_results: dict[str, Any], n_splits: int) -> list[Candidate]:
    """``cv_results_`` as a list of :class:`Candidate`, in search order."""
    out: list[Candidate] = []
    for i, params in enumerate(cv_results["params"]):
        out.append(
            Candidate(
                params=scalar_params(params),
                mean_test_score=float(cv_results["mean_test_score"][i]),
                std_test_score=float(cv_results["std_test_score"][i]),
                rank_test_score=int(cv_results["rank_test_score"][i]),
                split_test_scores=[
                    float(cv_results[f"split{s}_test_score"][i]) for s in range(n_splits)
                ],
                mean_fit_time=float(cv_results["mean_fit_time"][i]),
            )
        )
    return out


def run_cell(
    model: str,
    technique: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfgs: Configs,
    quick: bool,
) -> CellResult:
    """Tune and evaluate one cell on the training data (see the module docstring for the steps).

    Takes TRAINING rows only; there is deliberately no test-set argument. ``train_fraction`` is
    not applied here (the caller subsamples raw data in quick mode).

    Args:
        model: Model id (``models`` in experiment.yaml).
        technique: Technique id (``techniques`` in experiment.yaml).
        X_train: Training features.
        y_train: Training labels (1 = fraud).
        cfgs: The three configs.
        quick: Use the ``quick`` protocol (fewer folds, repeats and tries).

    Returns:
        A :class:`CellResult` (record, out-of-fold scores, refitted best pipeline).

    Raises:
        ValueError: If the model, technique or config is invalid.
        ValueError: If the data are unusable (too few frauds per fold, size mismatch).
    """
    started = datetime.now(UTC)
    t_start = time.perf_counter()
    if len(X_train) != len(y_train):
        raise ValueError(f"X_train has {len(X_train)} rows but y_train {len(y_train)}.")
    exp_cfg = cfgs.experiment
    seed = int(exp_cfg["seed"])
    proto = get_protocol(exp_cfg, quick)
    model_n_jobs, search_n_jobs = parallel_settings(exp_cfg)
    base = base_ratio(y_train)

    cv = StratifiedKFold(n_splits=proto.cv_folds, shuffle=True, random_state=seed)
    rcv = RepeatedStratifiedKFold(
        n_splits=proto.cv_folds, n_repeats=proto.cv_repeats, random_state=seed
    )
    repeated_splits = list(rcv.split(X_train, y_train))
    k = int(exp_cfg["smote_k_neighbors"])
    _check_enough_frauds(y_train, list(cv.split(X_train, y_train)) + repeated_splits, technique, k)

    space = build_search_space(model, technique, cfgs.models, exp_cfg, base)
    size = space_size(space)
    n_iter = min(proto.n_iter, size)
    if n_iter < proto.n_iter:
        logger.info(
            "%s: search space has %d configurations, fewer than the budget %d; trying all.",
            cell_id(model, technique),
            size,
            proto.n_iter,
        )
    model_params = {"n_jobs": model_n_jobs} if model in PARALLEL_MODELS else {}
    technique_params: dict[str, Any] = (
        {"fraud_weight": NEUTRAL_FRAUD_WEIGHT} if technique == "class_weight" else {}
    )
    pipe = build_pipeline(model, technique, model_params, technique_params, cfgs.dataset, exp_cfg)

    search = RandomizedSearchCV(
        pipe,
        space,
        n_iter=n_iter,
        scoring=SCORING,
        cv=cv,
        refit=True,
        random_state=seed,
        n_jobs=search_n_jobs,
        error_score="raise",
    )
    t0 = time.perf_counter()
    search.fit(X_train, y_train)
    search_seconds = time.perf_counter() - t0
    best = search.best_estimator_
    t0 = time.perf_counter()
    best.predict_proba(X_train)
    predict_seconds = time.perf_counter() - t0

    oof = cross_val_predict(
        clone(best), X_train, y_train, cv=cv, method="predict_proba", n_jobs=search_n_jobs
    )[:, 1]
    thresholds = {
        str(obj): best_threshold(y_train, oof, obj) for obj in exp_cfg["threshold"]["objectives"]
    }

    rep = cross_validate(
        clone(best),
        X_train,
        y_train,
        cv=repeated_splits,
        scoring=FOLD_SCORING,
        n_jobs=search_n_jobs,
        error_score="raise",
    )
    folds = [
        FoldScore(
            repeat=i // proto.cv_folds,
            fold=i % proto.cv_folds,
            fold_id=fold_id(X_train.index[val]),
            pr_auc=float(rep["test_pr_auc"][i]),
            roc_auc=float(rep["test_roc_auc"][i]),
        )
        for i, (_, val) in enumerate(repeated_splits)
    ]
    pr = np.array([f.pr_auc for f in folds])
    roc = np.array([f.roc_auc for f in folds])

    audit = resampling_audit(best, X_train, y_train)
    best_params = scalar_params(search.best_params_)
    chosen, smote_ratio, weight = ratio_info(technique, best_params, base)
    grid = [float(r) for r in exp_cfg["ratio_grid"]]
    kept = set(valid_ratios(grid, base))

    record = CellRecord(
        model=model,
        technique=technique,
        quick=quick,
        seed=seed,
        cv_folds=proto.cv_folds,
        cv_repeats=proto.cv_repeats,
        n_iter_budget=proto.n_iter,
        search_space_size=size,
        n_candidates_evaluated=len(search.cv_results_["params"]),
        n_train=len(y_train),
        n_train_frauds=int((np.asarray(y_train) == 1).sum()),
        train_sha256=training_sha256(X_train, y_train),
        base_ratio=base,
        dropped_ratios=[r for r in grid if r not in kept],
        best_params=best_params,
        chosen_ratio=chosen,
        smote_ratio=smote_ratio,
        fraud_weight=weight,
        search_best_pr_auc=float(search.best_score_),
        candidates=_candidates(search.cv_results_, proto.cv_folds),
        oof_pr_auc=float(average_precision_score(y_train, oof)),
        oof_roc_auc=float(roc_auc_score(y_train, oof)),
        thresholds=thresholds,
        cv_pr_auc_mean=float(pr.mean()),
        cv_pr_auc_std=float(pr.std()),
        cv_roc_auc_mean=float(roc.mean()),
        cv_roc_auc_std=float(roc.std()),
        fold_scores=folds,
        n_train_after_resampling=ClassCounts(
            legit=audit["legit_after"], fraud=audit["fraud_after"]
        ),
        n_synthetic=audit["n_synthetic"],
        fraud_duplicates_after=audit["fraud_duplicates_after"],
        search_seconds=search_seconds,
        fit_seconds=float(search.refit_time_),
        predict_seconds=predict_seconds,
        total_seconds=time.perf_counter() - t_start,
        started_at=started.isoformat(timespec="seconds"),
        finished_at=datetime.now(UTC).isoformat(timespec="seconds"),
        library_versions=library_versions(),
    )
    return CellResult(record=record, oof_scores=np.asarray(oof, dtype=float), pipeline=best)


def stratified_subsample(
    X: pd.DataFrame, y: pd.Series, fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Keep a stratified ``fraction`` of the rows (quick mode on the raw source).

    Args:
        X: Training features.
        y: Training labels.
        fraction: Share to keep, in (0, 1]; ``>= 1`` returns the inputs unchanged.
        seed: Random seed.

    Returns:
        ``(X_sub, y_sub)``.

    Raises:
        ValueError: If ``fraction`` is not positive.
    """
    if fraction <= 0:
        raise ValueError(f"quick.train_fraction must be in (0, 1]; got {fraction}.")
    if fraction >= 1.0:
        return X, y
    X_sub, _, y_sub, _ = train_test_split(X, y, train_size=fraction, stratify=y, random_state=seed)
    return X_sub, y_sub


def load_training_data(
    source: str, quick: bool, cfgs: Configs, verify_hash: bool = True
) -> tuple[pd.DataFrame, pd.Series, str]:
    """Load -> deduplicate -> stratified split, and return the TRAINING part only.

    In quick mode on the raw source, a stratified ``quick.train_fraction`` of the training rows
    is kept (the sample is already small, so it is used whole).

    Args:
        source: ``"raw"`` or ``"sample"``.
        quick: Quick mode.
        cfgs: The three configs.
        verify_hash: Check the raw file's SHA-256 (see ``load_dataset``).

    Returns:
        ``(X_train, y_train, data_sha256)`` where ``data_sha256`` is the source file's hash.
    """
    df = load_dataset(source, verify_hash=verify_hash, cfg=cfgs.dataset)
    deduped, _ = deduplicate(df, target=cfgs.dataset["target"])
    split = split_dataset(deduped, cfgs.dataset, cfgs.experiment, strategy="stratified")
    X, y = split["X_train"], split["y_train"]
    del split  # the test part is not used anywhere in the experiment runner
    if quick and source == "raw":
        fraction = get_protocol(cfgs.experiment, quick=True).train_fraction
        X, y = stratified_subsample(X, y, fraction, int(cfgs.experiment["seed"]))
    path_key = "raw_path" if source == "raw" else "sample_path"
    return X, y, sha256_file(resolve(cfgs.dataset[path_key]))


def _check_ids(kind: str, requested: list[str], allowed: list[str]) -> None:
    """Raise ``ValueError`` if ``requested`` has ids outside ``allowed``."""
    unknown = [r for r in requested if r not in allowed]
    if unknown:
        raise ValueError(
            f"Unknown {kind} {', '.join(unknown)}; choose from {', '.join(allowed)} "
            "(configs/experiment.yaml)."
        )


def run_experiments(
    source: str,
    quick: bool,
    results_dir: Path,
    models_dir: Path,
    models: list[str] | None = None,
    techniques: list[str] | None = None,
    force: bool = False,
    verify_hash: bool = True,
    cfgs: Configs | None = None,
    log: Callable[[str], None] = print,
) -> list[tuple[CellRecord, str]]:
    """Run (or skip, if finished) every requested cell, saving each as soon as it finishes.

    Args:
        source: ``"raw"`` or ``"sample"``.
        quick: Quick protocol (and ``train_fraction`` on the raw source).
        results_dir: E.g. ``artifacts/results``.
        models_dir: E.g. ``artifacts/models``.
        models: Model ids (default: all in experiment.yaml).
        techniques: Technique ids (default: all in experiment.yaml).
        force: Re-run finished cells.
        verify_hash: Check the raw file's SHA-256.
        cfgs: Configs (loaded if omitted).
        log: Progress printer (one line per cell, with elapsed seconds).

    Returns:
        ``(record, "ran" | "skipped")`` per requested cell, in model-then-technique order.

    Raises:
        ValueError: If a requested model or technique is not in experiment.yaml.
    """
    cfgs = cfgs if cfgs is not None else Configs.load()
    exp_cfg = cfgs.experiment
    models = list(models or exp_cfg["models"])
    techniques = list(techniques or exp_cfg["techniques"])
    _check_ids("model(s)", models, list(exp_cfg["models"]))
    _check_ids("technique(s)", techniques, list(exp_cfg["techniques"]))
    seed = int(exp_cfg["seed"])
    set_seed(seed)
    started = datetime.now(UTC)
    t_run = time.perf_counter()

    X, y, data_sha = load_training_data(source, quick, cfgs, verify_hash)
    commit = git_hash()
    cells = [(m, t) for m in models for t in techniques]
    log(
        f"source {source}{' (quick)' if quick else ''}: {len(y):,} training rows, "
        f"{int((y == 1).sum()):,} frauds; {len(cells)} cells; test split not used"
    )
    out: list[tuple[CellRecord, str]] = []
    for i, (m, t) in enumerate(cells, start=1):
        name = cell_id(m, t)
        prefix = f"[{i}/{len(cells)}] {name:<26}"
        if not force and is_cell_done(m, t, results_dir, quick=quick, source=source, seed=seed):
            out.append((load_cell_record(cell_paths(m, t, results_dir)["record"]), "skipped"))
            log(f"{prefix} skipped (finished; use --force to re-run)")
            continue
        t0 = time.perf_counter()
        result = run_cell(m, t, X, y, cfgs, quick)
        result.record = result.record.model_copy(
            update={"source": source, "data_sha256": data_sha, "git_commit": commit}
        )
        save_cell(result, results_dir, models_dir)
        rec = result.record
        out.append((rec, "ran"))
        log(
            f"{prefix} PR-AUC {rec.cv_pr_auc_mean:.4f} +- {rec.cv_pr_auc_std:.4f}  "
            f"tries {rec.n_candidates_evaluated:>2}  {time.perf_counter() - t0:6.1f} s"
        )

    write_run_meta(
        {
            "source": source,
            "quick": quick,
            "protocol": asdict(get_protocol(exp_cfg, quick)),
            "parallel": dict(exp_cfg["parallel"]),
            "configs": {"dataset": cfgs.dataset, "experiment": exp_cfg, "models": cfgs.models},
            "git_commit": commit,
            "library_versions": library_versions(),
            "data_sha256": data_sha,
            "train_sha256": training_sha256(X, y),
            "n_train": len(y),
            "started_at": started.isoformat(timespec="seconds"),
            "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "elapsed_seconds": time.perf_counter() - t_run,
            "cells_run": [cell_id(r.model, r.technique) for r, s in out if s == "ran"],
            "cells_skipped": [cell_id(r.model, r.technique) for r, s in out if s == "skipped"],
        },
        results_dir,
    )
    return out
