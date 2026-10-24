"""Package the deployable models, explanation reference and demo presets for the backend (master
guide Parts 7, 9 and 11).

Inputs, all produced earlier: finished cells (``<results>/cells``: record and out-of-fold scores),
their refitted pipelines (``<models>/cells``) and, on the raw source, the single test evaluation
(``scripts/evaluate_test.py``). Outputs:

* ``deploy/<id>.joblib``: the cell's fitted pipeline, unchanged (``joblib`` ``compress=3``), so
  it is the exact object the test evaluation scored. Its sampler steps are inert at prediction
  time (imblearn runs them only inside ``fit``).
* ``deploy/<id>.calibrator.joblib`` (optional): isotonic calibrator fitted on the cell's
  OUT-OF-FOLD training scores, never on test scores.
* ``feature_reference.json`` (:class:`fraudstudy.explain.FeatureReference`): legitimate-row
  quantiles, scaled training means and each raw column's ``deploy.input_range_percentiles``
  range, from TRAINING rows only.
* ``presets.json`` (``PresetsFile``) and ``demo_batch.csv``: real TEST-set rows, picked with the
  default cell's scores after the test evaluation. Picking them tunes nothing, but they are
  test rows, so on the raw source each export is recorded in the test-set guard (like the
  temporal check) and a repeat needs a reason.
* ``manifest.json`` (``Manifest``), written LAST, so its presence marks a complete export.

Only logreg and LightGBM are deployable (512 MB server); random forest stays in the JSON results.
The default cell is never chosen here: the team passes it or sets ``deploy.default_cell``.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from fraudstudy.contracts import (
    MODEL_LABELS,
    TECHNIQUE_LABELS,
    DatasetInfo,
    Manifest,
    ManifestEntry,
    Preset,
    PresetsFile,
    Transaction,
)
from fraudstudy.data import deduplicate, load_dataset, split_dataset
from fraudstudy.errors import ConfigError, DataValidationError, LeakageError
from fraudstudy.evaluation import fit_isotonic
from fraudstudy.experiment.results import RUN_META_FILE, CellRecord, cell_paths, load_cell_record
from fraudstudy.experiment.runner import (
    Configs,
    get_protocol,
    library_versions,
    stratified_subsample,
    training_sha256,
)
from fraudstudy.experiment.test_eval import (
    DATASET_SOURCE,
    DECISIONS_FILE,
    DETAILS_FILE,
    GUARD_FILE,
    check_guard,
    check_training_hash,
    open_test_set,
    output_dir_for,
)
from fraudstudy.explain import (
    FeatureReference,
    build_feature_reference,
    check_range_percentiles,
    feature_map,
)
from fraudstudy.io import git_commit_hash, load_json, save_json, sha256_file
from fraudstudy.paths import resolve

logger = logging.getLogger(__name__)

MODELS_OUT = "artifacts/models"
DEMO_OUT = "artifacts/demo"
DEMO_BATCH_OUT = "data/sample/demo_batch.csv"
SAMPLE_SUBDIR = "sample"  # sample exports go to <models>/sample and <demo>/sample
DEPLOY_DIR = "deploy"
MANIFEST_FILE = "manifest.json"
FEATURE_REFERENCE_FILE = "feature_reference.json"
PRESETS_FILE = "presets.json"
DEMO_BATCH_FILE = "demo_batch.csv"
CALIBRATOR_SUFFIX = ".calibrator.joblib"
JOBLIB_COMPRESS = 3
TECHNIQUE_PLACEHOLDER = "<technique>"
DEPLOYABLE_MODELS: tuple[str, ...] = ("logreg", "lightgbm")
THRESHOLD_OBJECTIVES: tuple[str, ...] = ("f1", "f2")
# Libraries whose pickles must be written and read by the same version.
PICKLE_LIBRARIES: tuple[str, ...] = ("numpy", "scikit-learn", "imbalanced-learn", "lightgbm")
EXPORT_ACTION = "export_presets"
SCORE_CHECK_ROWS = 100
BYTES_PER_MB = 1024 * 1024
DEPLOY_KEYS: tuple[str, ...] = (
    "cells",
    "default_cell",
    "default_threshold",
    "calibrate",
    "max_file_mb",
    "presets",
    "demo_batch",
    "input_range_percentiles",
)
PRESET_TEXT: dict[str, tuple[str, str]] = {
    "legit_everyday": ("Everyday purchase", "An everyday legitimate purchase."),
    "legit_large": ("Large legitimate payment", "A large payment that is legitimate."),
    "fraud_typical": ("Typical fraud", "A fraud the default model catches."),
    "fraud_small_amount": ("Small-amount fraud", "A fraud with a very small amount."),
    "borderline": ("Borderline case", "A payment scored close to the threshold."),
    "missed_fraud": (
        "Missed fraud",
        "A real fraud the default model misses (a false negative, shown on purpose).",
    ),
}


@dataclass(frozen=True)
class PresetRules:
    """Preset selection rules (``deploy.presets`` in experiment.yaml).

    Attributes:
        everyday_amount: ``[low, high]`` Amount range of the everyday legitimate preset.
        small_amount_max: Largest Amount of the small-amount fraud preset.
        large_amount_quantile: "Large" = Amount at or above this quantile of legit test rows.
        borderline_window: Borderline = ``|score - threshold| <= window``.
    """

    everyday_amount: tuple[float, float]
    small_amount_max: float
    large_amount_quantile: float
    borderline_window: float


@dataclass(frozen=True)
class DeploySettings:
    """The ``deploy`` block with the default cell resolved.

    Attributes:
        cells: Deployed cell ids in config order (placeholder resolved, duplicates dropped).
        default_cell: The team's default cell (one of ``cells``).
        default_threshold: ``"f1"`` or ``"f2"``: the threshold applied when a request has none.
        calibrate: Fit an isotonic calibrator per cell.
        max_file_mb: Warn when a written file is larger.
        presets: :class:`PresetRules`.
        demo_batch_rows: Rows in ``demo_batch.csv``.
        demo_batch_frauds: Frauds among them (fewer if the test set has fewer).
        input_range_percentiles: ``(low, high)`` percentiles of each raw column over the
            training rows; the backend warns about inputs outside that range.
    """

    cells: list[str]
    default_cell: str
    default_threshold: str
    calibrate: bool
    max_file_mb: float
    presets: PresetRules
    demo_batch_rows: int
    demo_batch_frauds: int
    input_range_percentiles: tuple[float, float]


@dataclass(frozen=True)
class ExportPaths:
    """Where the export writes.

    Attributes:
        models: ``manifest.json``, ``feature_reference.json`` and ``deploy/``.
        demo: ``presets.json``.
        batch: ``demo_batch.csv``.
    """

    models: Path
    demo: Path
    batch: Path


@dataclass(frozen=True)
class ExportData:
    """The training rows the cells were tuned on and the locked test rows.

    Attributes:
        X_train, y_train: Training features and labels (quick subsample applied on raw).
        X_test, y_test: Test features and labels.
        test_rows: The test rows with every dataset column (``Time`` and ``Class`` included).
        n_rows: Rows in the dataset file as loaded.
        train_sha256: :func:`training_sha256` of the training rows.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    test_rows: pd.DataFrame
    n_rows: int
    train_sha256: str


@dataclass
class ExportReport:
    """What :func:`export_artifacts` wrote.

    Attributes:
        manifest: The written manifest.
        presets: The written presets.
        skipped_presets: Preset ids with no matching test row.
        sizes: Written file -> size in bytes.
        warnings: Size and stale-file warnings.
    """

    manifest: Manifest
    presets: PresetsFile
    skipped_presets: list[str]
    sizes: dict[Path, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def total_mb(self) -> float:
        """Total size of the written files in MB."""
        return sum(self.sizes.values()) / BYTES_PER_MB


def export_paths(source: str) -> ExportPaths:
    """Default output folders: the real locations for raw, ``.../sample`` for the sample (so
    synthetic artifacts never overwrite real ones)."""
    if source == "raw":
        return ExportPaths(resolve(MODELS_OUT), resolve(DEMO_OUT), resolve(DEMO_BATCH_OUT))
    demo = resolve(DEMO_OUT) / SAMPLE_SUBDIR
    return ExportPaths(resolve(MODELS_OUT) / SAMPLE_SUBDIR, demo, demo / DEMO_BATCH_FILE)


def parse_cell(name: str, techniques: list[str]) -> tuple[str, str]:
    """Split a deployable ``"<model>__<technique>"`` id.

    Raises:
        ConfigError: If the id is malformed, the model is not deployable, or the technique is
            not in experiment.yaml.
    """
    model, sep, technique = str(name).partition("__")
    if not sep or technique not in techniques:
        raise ConfigError(
            f"{name!r} is not a <model>__<technique> id with a technique from experiment.yaml "
            f"({', '.join(techniques)})."
        )
    if model not in DEPLOYABLE_MODELS:
        raise ConfigError(
            f"{name!r} cannot be deployed: only {', '.join(DEPLOYABLE_MODELS)} fit on the 512 MB "
            "server (master guide Part 11). Random forest results stay in comparison.json."
        )
    return model, technique


def deploy_settings(exp_cfg: dict[str, Any], default_cell: str | None = None) -> DeploySettings:
    """Read and check the ``deploy`` block; resolve the default cell and the placeholder.

    The default cell comes from ``default_cell`` (the CLI's ``--default-cell``) or
    ``deploy.default_cell``; one of them is required, and they must agree if both are set.
    ``"<technique>"`` in a cell id is replaced by the default cell's technique.

    Args:
        exp_cfg: Contents of ``configs/experiment.yaml``.
        default_cell: The team's choice, or ``None`` to use the config value.

    Returns:
        The :class:`DeploySettings`.

    Raises:
        ConfigError: If the block is missing or invalid (including ``input_range_percentiles``
            not being ``[low, high]`` with ``0 <= low < high <= 100``), or no default cell is
            given.
    """
    block = exp_cfg.get("deploy")
    if not isinstance(block, dict) or not set(DEPLOY_KEYS) <= set(block):
        raise ConfigError(
            f"experiment.yaml needs a `deploy` block with {', '.join(DEPLOY_KEYS)} "
            "(see scripts/export_artifacts.py)."
        )
    configured = block["default_cell"]
    if default_cell and configured and default_cell != configured:
        raise ConfigError(
            f"--default-cell {default_cell} disagrees with deploy.default_cell {configured} in "
            "experiment.yaml; give one or make them equal."
        )
    chosen = default_cell or configured
    if not chosen:
        raise ConfigError(
            "No default cell given. The TEAM chooses it from the cross-validation results "
            "(artifacts/results/cv_summary.json); it is never picked automatically. Pass "
            "--default-cell <model>__<technique> or set deploy.default_cell in "
            "configs/experiment.yaml."
        )
    techniques = list(exp_cfg["techniques"])
    _, default_technique = parse_cell(chosen, techniques)
    cells: list[str] = []
    for raw in block["cells"]:
        name = str(raw).replace(TECHNIQUE_PLACEHOLDER, default_technique)
        parse_cell(name, techniques)
        if name not in cells:
            cells.append(name)
    if chosen not in cells:
        raise ConfigError(f"Default cell {chosen} must be one of deploy.cells ({cells}).")
    threshold = str(block["default_threshold"])
    if threshold not in THRESHOLD_OBJECTIVES:
        raise ConfigError(f"deploy.default_threshold must be f1 or f2; got {threshold!r}.")
    p, batch = block["presets"], block["demo_batch"]
    low, high = (float(v) for v in p["everyday_amount"])
    rows, frauds = int(batch["rows"]), int(batch["frauds"])
    if not 0 <= frauds <= rows:
        raise ConfigError(f"deploy.demo_batch needs 0 <= frauds <= rows; got {frauds}, {rows}.")
    return DeploySettings(
        cells=cells,
        default_cell=chosen,
        default_threshold=threshold,
        calibrate=bool(block["calibrate"]),
        max_file_mb=float(block["max_file_mb"]),
        presets=PresetRules(
            everyday_amount=(low, high),
            small_amount_max=float(p["small_amount_max"]),
            large_amount_quantile=float(p["large_amount_quantile"]),
            borderline_window=float(p["borderline_window"]),
        ),
        demo_batch_rows=rows,
        demo_batch_frauds=frauds,
        input_range_percentiles=check_range_percentiles(block["input_range_percentiles"]),
    )


def load_cells(
    cells: list[str], results_dir: Path, models_dir: Path, source: str
) -> dict[str, CellRecord]:
    """Load and cross-check the records of the deployed cells.

    Raises:
        ConfigError: If a cell has no finished record, pipeline or out-of-fold file.
        DataValidationError: If a cell ran on another source, or the cells disagree on seed,
            mode, training data or data file.
    """
    records: dict[str, CellRecord] = {}
    missing: list[str] = []
    for name in cells:
        model, _, technique = name.partition("__")
        paths = cell_paths(model, technique, results_dir, models_dir)
        if not all(p.is_file() for p in paths.values()):
            missing.append(name)
            continue
        records[name] = load_cell_record(paths["record"])
    if missing:
        raise ConfigError(
            f"Cells {', '.join(missing)} have no finished record, out-of-fold scores and pipeline "
            f"under {results_dir} / {models_dir}; run scripts/run_experiments.py for them (or "
            "copy them from the machine that did)."
        )
    wrong = [n for n, r in records.items() if r.source != source]
    if wrong:
        raise DataValidationError(f"Cells {', '.join(wrong)} were not run on source {source!r}.")
    for key in ("seed", "quick", "train_sha256", "data_sha256"):
        values = {str(getattr(r, key)) for r in records.values()}
        if len(values) > 1:
            raise DataValidationError(
                f"Deployed cells disagree on {key} ({sorted(values)}); re-run the odd ones so "
                "every deployed model comes from the same protocol and data."
            )
    if next(iter(records.values())).data_sha256 is None:
        raise DataValidationError("Cell records lack data_sha256; re-run them with the runner.")
    return records


def check_library_versions(records: dict[str, CellRecord]) -> dict[str, str]:
    """Raise unless this environment has the training versions of :data:`PICKLE_LIBRARIES`.

    Returns:
        The current :func:`library_versions` (written to the manifest).

    Raises:
        ConfigError: On any mismatch.
    """
    current = library_versions()
    bad = [
        f"{name}: {lib} {r.library_versions.get(lib)} at training, {current.get(lib)} here"
        for name, r in records.items()
        for lib in PICKLE_LIBRARIES
        if r.library_versions.get(lib) != current.get(lib)
    ]
    if bad:
        raise ConfigError(
            "Export with the library versions the cells were trained with (pip install -r "
            f"requirements.txt), so the pickles load identically: {'; '.join(bad)}."
        )
    return current


def check_test_evaluated(
    records: dict[str, CellRecord], details_path: Path, models_dir: Path, required: bool
) -> None:
    """Check that every deployed pipeline is byte-identical to the one the test set scored.

    Uses ``pipeline_sha256`` in ``test_eval_details.json``. Skipped when the file is absent and
    ``required`` is false (sample runs).

    Raises:
        DataValidationError: If the details are missing (when required), a cell was not
            evaluated, or a pipeline file changed since the evaluation.
    """
    if not details_path.is_file():
        if required:
            raise DataValidationError(
                f"{details_path} not found; run scripts/evaluate_test.py before exporting."
            )
        return
    evaluated = {
        f"{d['model']}__{d['technique']}": d.get("pipeline_sha256")
        for d in load_json(details_path)["cells"]
    }
    for name, r in records.items():
        if name not in evaluated:
            raise DataValidationError(
                f"{name} was not evaluated on the test set ({details_path}); deploy only "
                "evaluated cells, so the served model is the one whose metrics are reported."
            )
        pipeline = cell_paths(r.model, r.technique, Path(), models_dir)["pipeline"]
        if sha256_file(pipeline) != evaluated[name]:
            raise DataValidationError(
                f"{pipeline} changed after the test evaluation; restore the evaluated file."
            )


def load_export_data(source: str, quick: bool, cfgs: Configs, verify_hash: bool) -> ExportData:
    """Rebuild the training run's split: load -> dedup -> stratified split -> quick subsample.

    The same steps as ``load_training_data`` / ``load_test_split``; the caller checks the
    training hash against the cell records.
    """
    df = load_dataset(source, verify_hash=verify_hash, cfg=cfgs.dataset)
    deduped, _ = deduplicate(df, target=cfgs.dataset["target"])
    split = split_dataset(deduped, cfgs.dataset, cfgs.experiment, strategy="stratified")
    X_train, y_train = split.X_train, split.y_train
    if quick and source == "raw":
        fraction = get_protocol(cfgs.experiment, quick=True).train_fraction
        X_train, y_train = stratified_subsample(
            X_train, y_train, fraction, int(cfgs.experiment["seed"])
        )
    return ExportData(
        X_train=X_train,
        y_train=y_train,
        X_test=split.X_test,
        y_test=split.y_test,
        test_rows=deduped.loc[split.X_test.index],
        n_rows=len(df),
        train_sha256=training_sha256(X_train, y_train),
    )


def save_deploy_pipeline(pipeline: Any, path: Path, X_check: pd.DataFrame) -> str:
    """Dump ``pipeline`` with ``compress=3``, reload it and check identical scores.

    Args:
        pipeline: The cell's fitted pipeline.
        path: Destination ``.joblib``.
        X_check: Rows to compare scores on (training rows).

    Returns:
        The file's SHA-256.

    Raises:
        DataValidationError: If the reloaded pipeline scores ``X_check`` differently.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path, compress=JOBLIB_COMPRESS)
    before = pipeline.predict_proba(X_check)[:, 1]
    after = joblib.load(path).predict_proba(X_check)[:, 1]
    if not np.array_equal(before, after):
        raise DataValidationError(f"{path} scores differently after reloading; do not deploy it.")
    return sha256_file(path)


def check_same_preprocessing(
    pipelines: dict[str, Any], reference: FeatureReference, X_train: pd.DataFrame
) -> None:
    """Raise unless every pipeline's preprocessor gives the reference's features and scaled
    training means (each was fitted on the same training rows, so they must agree)."""
    expected = [(f.feature, f.raw_column) for f in reference.features]
    means = np.array([f.train_mean_scaled for f in reference.features])
    for name, pipe in pipelines.items():
        pre = pipe.named_steps["preprocess"]
        got = np.asarray(pre.transform(X_train), dtype=float).mean(axis=0)
        if feature_map(pre) != expected or not np.allclose(got, means, rtol=1e-9, atol=1e-12):
            raise DataValidationError(
                f"{name}'s preprocessor differs from the default cell's; all deployed cells "
                "must be fitted on the same training rows."
            )


def _pick(index: np.ndarray, scores: np.ndarray, threshold: float, how: str) -> int:
    """Position of the chosen candidate: lower-median score, or closest to the threshold."""
    if how == "closest":
        return int(index[np.argmin(np.abs(scores[index] - threshold))])
    ordered = index[np.argsort(scores[index], kind="stable")]
    return int(ordered[(len(ordered) - 1) // 2])


def pick_presets(
    rows: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    rules: PresetRules,
    cell: str,
    cfg: dict[str, Any],
    objective: str = "f1",
    origin: str = "Real",
) -> tuple[PresetsFile, list[str]]:
    """Choose up to six real test-set payments for the demo, deterministically.

    "Below" means ``score < threshold`` (predicted legitimate). Rules, in order (a row is used
    once):

    * ``legit_everyday``: Class 0, Amount in ``everyday_amount``, below; lower-median score.
    * ``legit_large``: Class 0, Amount >= the ``large_amount_quantile`` of legit test amounts,
      below; lower-median score.
    * ``fraud_typical``: Class 1, ``score >= threshold``; lower-median score.
    * ``fraud_small_amount``: Class 1, Amount <= ``small_amount_max``; lower-median score.
    * ``borderline``: ``|score - threshold| <= borderline_window``; closest to the threshold.
    * ``missed_fraud``: Class 1, below; lower-median score.

    Args:
        rows: Test rows with every dataset column (raw units, ``Time`` and ``Class`` included).
        scores: The default cell's risk scores of ``rows``.
        threshold: Its default threshold.
        rules: :class:`PresetRules`.
        cell: The default cell id (named in the descriptions).
        cfg: Contents of ``configs/dataset.yaml``.
        objective: ``"f1"`` or ``"f2"`` (named in the descriptions).
        origin: ``"Real"`` or ``"Synthetic"`` (the data, named in the descriptions).

    Returns:
        ``(PresetsFile, skipped preset ids)``; a preset is skipped when no row matches.
    """
    amount = rows[cfg["amount_column"]].to_numpy(dtype=float)
    label = rows[cfg["target"]].to_numpy(dtype=int)
    s = np.asarray(scores, dtype=float)
    legit, fraud, below = label == 0, label == 1, s < threshold
    low, high = rules.everyday_amount
    large = np.quantile(amount[legit], rules.large_amount_quantile) if legit.any() else np.inf
    candidates = [
        ("legit_everyday", legit & (amount >= low) & (amount <= high) & below, "median"),
        ("legit_large", legit & (amount >= large) & below, "median"),
        ("fraud_typical", fraud & ~below, "median"),
        ("fraud_small_amount", fraud & (amount <= rules.small_amount_max), "median"),
        ("borderline", np.abs(s - threshold) <= rules.borderline_window, "closest"),
        ("missed_fraud", fraud & below, "median"),
    ]
    columns = [cfg["time_column"], *cfg["pca_features"], cfg["amount_column"]]
    used = np.zeros(len(rows), dtype=bool)
    presets: list[Preset] = []
    skipped: list[str] = []
    for preset_id, mask, how in candidates:
        index = np.flatnonzero(mask & ~used)
        if index.size == 0:
            skipped.append(preset_id)
            continue
        pos = _pick(index, s, threshold, how)
        used[pos] = True
        row = rows.iloc[pos]
        title, text = PRESET_TEXT[preset_id]
        verdict = "flagged as fraud" if s[pos] >= threshold else "passed as legitimate"
        presets.append(
            Preset(
                id=preset_id,
                label=title,
                description=(
                    f"{text} {origin} test-set payment, Amount {amount[pos]:,.2f}; {cell} "
                    f"risk score {s[pos]:.3f} vs its {objective.upper()} threshold "
                    f"{threshold:.3f} ({verdict})."
                ),
                expected_class=int(label[pos]),
                features=Transaction(**{c: float(row[c]) for c in columns}),
            )
        )
    return PresetsFile(presets=presets), skipped


def demo_batch(
    rows: pd.DataFrame, target: str, n_rows: int, n_frauds: int, seed: int
) -> pd.DataFrame:
    """Sample test rows for the batch demo: ``n_frauds`` frauds (or all, if fewer) plus legit rows
    up to ``n_rows``, sampled per class with ``seed`` and shuffled. Columns and values are
    unchanged (``Class`` kept).
    """
    fraud, legit = rows[rows[target] == 1], rows[rows[target] == 0]
    k_fraud = min(n_frauds, len(fraud))
    k_legit = min(n_rows - k_fraud, len(legit))
    parts = [fraud.sample(k_fraud, random_state=seed), legit.sample(k_legit, random_state=seed)]
    return pd.concat(parts).sample(frac=1.0, random_state=seed)


def size_warnings(sizes: dict[Path, int], max_mb: float) -> list[str]:
    """A warning for each file larger than ``max_mb`` MB."""
    return [
        f"{path.name} is {size / BYTES_PER_MB:.1f} MB (> {max_mb:g} MB); it may not fit the "
        "512 MB server. Consider a smaller configuration."
        for path, size in sizes.items()
        if size > max_mb * BYTES_PER_MB
    ]


def export_artifacts(
    source: str,
    results_dir: Path,
    models_dir: Path,
    default_cell: str | None = None,
    out: ExportPaths | None = None,
    rerun_reason: str | None = None,
    guard_path: Path | None = None,
    decisions_path: Path | None = None,
    cfgs: Configs | None = None,
    verify_hash: bool = True,
    log: Callable[[str], None] = print,
) -> ExportReport:
    """Export the deployable models, calibrators, feature reference, presets and demo batch.

    Steps: settings and default cell -> deployed cells, library versions, test-evaluation
    checks -> guard (raw: the test set must have been evaluated; a repeat needs a reason) ->
    rebuild the split, check the training hash -> write deploy files, reference, presets, demo
    batch, then the manifest.

    Args:
        source: ``"raw"`` or ``"sample"``.
        results_dir: Folder with ``cells/`` and ``run_meta.json`` (``artifacts/results``).
        models_dir: Folder with ``cells/*.joblib`` (``artifacts/models``).
        default_cell: The team's default cell (else ``deploy.default_cell``); never guessed.
        out: Output folders (default :func:`export_paths`).
        rerun_reason: Required on raw when an export already read the test set.
        guard_path: Default ``results_dir/TEST_SET_OPENED.json`` on raw; none on the sample.
        decisions_path: Where re-run reasons are appended (default ``docs/decisions.md``).
        cfgs: Configs (loaded if omitted).
        verify_hash: Check the raw file's SHA-256.
        log: Progress printer (never prints feature values).

    Returns:
        An :class:`ExportReport`.

    Raises:
        ConfigError: If no default cell is given or the deploy settings, cells or library
            versions are invalid.
        DataValidationError: If files are missing, inconsistent or changed since evaluation.
        LeakageError: If the guard refuses or the training set differs from the cells'.
    """
    cfgs = cfgs if cfgs is not None else Configs.load()
    settings = deploy_settings(cfgs.experiment, default_cell)
    results_dir, models_dir = Path(results_dir), Path(models_dir)
    out = out if out is not None else export_paths(source)
    if guard_path is None and source == "raw":
        guard_path = results_dir / GUARD_FILE
    decisions_path = Path(decisions_path) if decisions_path else resolve(DECISIONS_FILE)

    records = load_cells(settings.cells, results_dir, models_dir, source)
    versions = check_library_versions(records)
    details = output_dir_for(source, results_dir) / DETAILS_FILE
    check_test_evaluated(records, details, models_dir, required=source == "raw")
    if guard_path is not None:
        if not Path(guard_path).is_file():
            raise LeakageError(
                "Presets and the demo batch are test-set rows; run scripts/evaluate_test.py "
                "first, so the single test evaluation comes before them."
            )
        check_guard(guard_path, rerun_reason, EXPORT_ACTION)

    first = next(iter(records.values()))
    data = load_export_data(source, first.quick, cfgs, verify_hash)
    run_meta_path = results_dir / RUN_META_FILE
    run_meta = load_json(run_meta_path) if run_meta_path.is_file() else None
    check_training_hash(data.train_sha256, list(records.values()), run_meta)
    if guard_path is not None:
        keys = [(r.model, r.technique) for r in records.values()]
        open_test_set(guard_path, keys, rerun_reason, decisions_path, EXPORT_ACTION)

    pipelines = {
        name: joblib.load(cell_paths(r.model, r.technique, results_dir, models_dir)["pipeline"])
        for name, r in records.items()
    }
    default = pipelines[settings.default_cell]
    reference = build_feature_reference(
        default.named_steps["preprocess"],
        data.X_train,
        data.y_train,
        source=source,
        data_sha256=first.data_sha256,
        train_sha256=data.train_sha256,
        range_percentiles=settings.input_range_percentiles,
    )
    check_same_preprocessing(pipelines, reference, data.X_train)

    deploy_dir = out.models / DEPLOY_DIR
    sizes: dict[Path, int] = {}
    entries: list[ManifestEntry] = []
    X_check = data.X_train.iloc[:SCORE_CHECK_ROWS]
    for name, r in records.items():
        model_path = deploy_dir / f"{name}.joblib"
        model_sha = save_deploy_pipeline(pipelines[name], model_path, X_check)
        sizes[model_path] = model_path.stat().st_size
        cal_file = cal_sha = None
        if settings.calibrate:
            oof = np.load(cell_paths(r.model, r.technique, results_dir)["oof"])
            if len(oof) != len(data.y_train):
                raise DataValidationError(
                    f"{name}: {len(oof)} out-of-fold scores for {len(data.y_train)} training rows."
                )
            cal_path = deploy_dir / f"{name}{CALIBRATOR_SUFFIX}"
            joblib.dump(fit_isotonic(oof, data.y_train), cal_path, compress=JOBLIB_COMPRESS)
            sizes[cal_path] = cal_path.stat().st_size
            cal_file, cal_sha = cal_path.relative_to(out.models).as_posix(), sha256_file(cal_path)
        inputs = pipelines[name].named_steps["preprocess"].feature_names_in_
        entries.append(
            ManifestEntry(
                id=name,
                model=r.model,
                technique=r.technique,
                model_label=MODEL_LABELS[r.model],
                technique_label=TECHNIQUE_LABELS[r.technique],
                file=model_path.relative_to(out.models).as_posix(),
                sha256=model_sha,
                threshold_f1=r.thresholds["f1"],
                threshold_f2=r.thresholds["f2"],
                default_threshold=settings.default_threshold,
                has_calibrator=cal_file is not None,
                calibrator_file=cal_file,
                calibrator_sha256=cal_sha,
                features=[str(c) for c in inputs],
                data_sha256=str(r.data_sha256),
                library_versions=versions,
            )
        )
        log(f"deployed   {name:<26} {sizes[model_path] / BYTES_PER_MB:6.2f} MB")

    ref_path = save_json(reference.model_dump(mode="json"), out.models / FEATURE_REFERENCE_FILE)
    threshold = records[settings.default_cell].thresholds[settings.default_threshold]
    scores = default.predict_proba(data.X_test)[:, 1]
    presets, skipped = pick_presets(
        data.test_rows,
        scores,
        threshold,
        settings.presets,
        settings.default_cell,
        cfgs.dataset,
        settings.default_threshold,
        "Real" if source == "raw" else "Synthetic",
    )
    presets_path = save_json(presets.model_dump(mode="json"), out.demo / PRESETS_FILE)
    for preset_id in skipped:
        logger.warning("No test row matches preset %s; it is left out.", preset_id)
    batch = demo_batch(
        data.test_rows,
        cfgs.dataset["target"],
        settings.demo_batch_rows,
        settings.demo_batch_frauds,
        int(cfgs.experiment["seed"]),
    )
    out.batch.parent.mkdir(parents=True, exist_ok=True)
    batch.to_csv(out.batch, index=False, lineterminator="\n")

    manifest = Manifest(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        git_commit=git_commit_hash(),
        seed=first.seed,
        quick_mode=first.quick,
        dataset=DatasetInfo(
            name=str(cfgs.dataset["name"]),
            source=DATASET_SOURCE[source],
            sha256=str(first.data_sha256),
            n_rows=data.n_rows,
        ),
        default_model_id=settings.default_cell,
        models=entries,
    )
    manifest_path = save_json(manifest.model_dump(mode="json"), out.models / MANIFEST_FILE)
    for path in (ref_path, presets_path, out.batch, manifest_path):
        sizes[path] = path.stat().st_size

    warnings = size_warnings(sizes, settings.max_file_mb)
    stale = sorted(p for p in deploy_dir.glob("*.joblib") if p not in sizes)
    warnings += [
        f"{p.name} is not in the new manifest; delete it before building the image." for p in stale
    ]
    for text in warnings:
        logger.warning("%s", text)
    return ExportReport(manifest, presets, skipped, sizes, warnings)
