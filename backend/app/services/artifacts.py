"""Load, validate and hold every artifact the API serves, once, at start-up.

Two sources (``USE_EXAMPLE_ARTIFACTS``):

* real (default): ``ARTIFACTS_DIR/results/{comparison,curves,stats,eda,resampling_audit,
  temporal_check}.json``, ``ARTIFACTS_DIR/demo/presets.json`` and ``MODELS_DIR/manifest.json``
  plus the model and calibrator files it lists;
* examples: the fake fixtures in ``EXAMPLES_DIR`` (``docs/contracts/examples``). The manifest is
  validated, but no model file is loaded: the examples have none.

Every JSON file is validated against its contract (``temporal_check.json`` has none; only its
``analysis`` tag is checked). Each model file must exist inside ``MODELS_DIR``, match its
manifest SHA-256 and come from the same versions of the pickle libraries as this environment;
otherwise it is not loaded. Loading never raises: every problem is recorded as a message saying
what to fix, ``/health`` reports "degraded", and an endpoint whose artifact is unavailable
answers 503 with the error shape.

Required artifacts: comparison, curves, stats (``/results``), eda (``/eda``), presets
(``/presets``) and manifest (``/models``). Optional: resampling_audit (in ``/eda``) and
temporal_check (in ``/results``); a missing optional file is not a problem, an invalid one is.
With the real source, ``MODELS_DIR/feature_reference.json`` (explanations and range warnings of
``/predict``; an internal file of :mod:`fraudstudy.explain`, not a contract) follows the same
optional rule; one without training ranges is a problem, because it predates them.
"""

import json
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

import joblib
from fastapi import Depends, Request
from pydantic import ValidationError

from backend.app.errors import ARTIFACT_UNAVAILABLE, ApiError, ArtifactError
from backend.app.logging import LOGGER_NAME
from backend.app.settings import Settings
from fraudstudy.contracts import (
    ComparisonTable,
    CurvesFile,
    EdaSummary,
    Manifest,
    ManifestEntry,
    PresetsFile,
    ResamplingAudit,
    StatsResult,
    TopLevel,
)
from fraudstudy.experiment.runner import library_versions
from fraudstudy.explain import FeatureReference
from fraudstudy.export import FEATURE_REFERENCE_FILE, PICKLE_LIBRARIES
from fraudstudy.io import git_commit_hash, sha256_file
from fraudstudy.paths import PROJECT_ROOT

logger = logging.getLogger(f"{LOGGER_NAME}.artifacts")

RESULTS_SUBDIR = "results"
DEMO_SUBDIR = "demo"
JSON_SUFFIX = ".json"
RESULTS_FILES: tuple[str, ...] = (
    "comparison",
    "curves",
    "stats",
    "eda",
    "resampling_audit",
    "temporal_check",
)
REQUIRED: tuple[str, ...] = ("comparison", "curves", "stats", "eda", "presets", "manifest")
OPTIONAL: tuple[str, ...] = ("resampling_audit", "temporal_check")
ARTIFACTS: tuple[str, ...] = REQUIRED + OPTIONAL
CONTRACTS: dict[str, type[TopLevel]] = {
    "comparison": ComparisonTable,
    "curves": CurvesFile,
    "stats": StatsResult,
    "eda": EdaSummary,
    "resampling_audit": ResamplingAudit,
    "presets": PresetsFile,
    "manifest": Manifest,
}
PRODUCERS: dict[str, str] = {
    "comparison": "scripts/evaluate_test.py",
    "curves": "scripts/evaluate_test.py",
    "stats": "scripts/run_stats.py",
    "eda": "scripts/run_eda.py",
    "resampling_audit": "scripts/run_resampling_audit.py",
    "temporal_check": "scripts/run_temporal_check.py",
    "presets": "scripts/export_artifacts.py",
    "manifest": "scripts/export_artifacts.py",
}
TEMPORAL_ANALYSIS = "temporal_split_sensitivity_check"
MAX_REPORTED_ERRORS = 3
# Threads per prediction. One row gains nothing from threads, and every request thread that calls
# LightGBM with n_jobs > 1 keeps its own OpenMP team of worker threads alive; more OpenMP threads
# than CPUs make every OpenMP computation in the process (training included) far slower.
PREDICT_N_JOBS = 1
Source = Literal["real", "examples"]


@dataclass(frozen=True)
class LoadedModel:
    """A deployed model ready to score.

    Attributes:
        entry: Its manifest entry (thresholds, features, labels).
        pipeline: The fitted pipeline (has ``predict_proba``).
        calibrator: The isotonic calibrator, or ``None`` (none exported, or it failed to load).
    """

    entry: ManifestEntry
    pipeline: Any
    calibrator: Any | None


@dataclass
class ArtifactStore:
    """Everything loaded at start-up, plus the problems met while loading.

    Attributes:
        source: ``"real"`` or ``"examples"``.
        comparison, curves, stats, eda, resampling_audit, presets, manifest: Validated
            contracts, or ``None`` when unavailable.
        temporal_check: ``temporal_check.json`` as written (no contract), or ``None``.
        feature_reference: ``MODELS_DIR/feature_reference.json``, or ``None`` (examples, or not
            exported: predictions then carry no explanation).
        models: Loaded models by manifest id (empty with example artifacts).
        default_model_id: Model used when a request names none, or ``None``.
        problems: Messages saying what is wrong and how to fix it.
        unavailable: Artifact name -> why it is unavailable (for the 503 body).
        git_hash: The backend's git commit, or ``None``.
        started_at: ``time.monotonic()`` at start-up.
    """

    source: Source
    comparison: ComparisonTable | None = None
    curves: CurvesFile | None = None
    stats: StatsResult | None = None
    eda: EdaSummary | None = None
    resampling_audit: ResamplingAudit | None = None
    presets: PresetsFile | None = None
    manifest: Manifest | None = None
    temporal_check: dict[str, Any] | None = None
    feature_reference: FeatureReference | None = None
    models: dict[str, LoadedModel] = field(default_factory=dict)
    default_model_id: str | None = None
    problems: list[str] = field(default_factory=list)
    unavailable: dict[str, str] = field(default_factory=dict)
    git_hash: str | None = None
    started_at: float = field(default_factory=time.monotonic)

    @property
    def is_example(self) -> bool:
        """True when serving the example fixtures."""
        return self.source == "examples"

    @property
    def status(self) -> Literal["ok", "degraded"]:
        """``"degraded"`` when any problem was recorded, else ``"ok"``."""
        return "degraded" if self.problems else "ok"

    def uptime_s(self) -> float:
        """Seconds since start-up."""
        return time.monotonic() - self.started_at

    def require(self, *names: str) -> None:
        """Raise a 503 :class:`ApiError` naming every unavailable artifact among ``names``.

        Args:
            names: Artifact names, e.g. ``"comparison"``.

        Raises:
            ApiError: 503 ``artifact_unavailable`` with ``details = {"artifacts": [...],
                "problems": [...]}`` if any of them was not loaded.
        """
        missing = [name for name in names if getattr(self, name) is None]
        if not missing:
            return
        files = ", ".join(f"{name}{JSON_SUFFIX}" for name in missing)
        reasons = [
            self.unavailable.get(name, f"{name}{JSON_SUFFIX} was not loaded.") for name in missing
        ]
        raise ApiError(
            503,
            ARTIFACT_UNAVAILABLE,
            f"Not available: {files}. See /api/v1/health for what is wrong and how to fix it.",
            {"artifacts": missing, "problems": reasons},
        )


def display_path(path: Path) -> str:
    """``path`` relative to the project root when inside it (keeps local folders out of
    ``/health``), else as given, with forward slashes."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def artifact_paths(settings: Settings) -> dict[str, Path]:
    """Where each artifact is read from.

    Args:
        settings: Uses ``use_example_artifacts``, ``examples_dir``, ``artifacts_dir`` and
            ``models_dir``.

    Returns:
        Artifact name -> JSON file path, for every name in :data:`ARTIFACTS`.
    """
    if settings.use_example_artifacts:
        return {name: settings.examples_dir / f"{name}{JSON_SUFFIX}" for name in ARTIFACTS}
    results = settings.artifacts_dir / RESULTS_SUBDIR
    paths = {name: results / f"{name}{JSON_SUFFIX}" for name in RESULTS_FILES}
    paths["presets"] = settings.artifacts_dir / DEMO_SUBDIR / f"presets{JSON_SUFFIX}"
    paths["manifest"] = settings.models_dir / f"manifest{JSON_SUFFIX}"
    return paths


def _first_errors(exc: ValidationError) -> str:
    """The first few validation errors as ``loc: msg`` (never the offending values)."""
    errors = exc.errors(include_url=False, include_input=False, include_context=False)
    shown = [
        f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
        for err in errors[:MAX_REPORTED_ERRORS]
    ]
    more = len(errors) - len(shown)
    return "; ".join(shown) + (f"; and {more} more" if more > 0 else "")


def _reject_constant(constant: str) -> float:
    """``json.loads`` hook: refuse ``NaN`` / ``Infinity`` (responses must be strict JSON)."""
    raise ValueError(f"contains {constant}")


def read_artifact(name: str, path: Path) -> Any:
    """Read one artifact and validate it.

    Args:
        name: Artifact name (a key of :data:`PRODUCERS`).
        path: Its JSON file.

    Returns:
        The validated contract model, or for ``temporal_check`` the parsed JSON object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ArtifactError: If it is not valid JSON, contains NaN / Infinity, or does not match its
            contract.
    """
    where, producer = display_path(path), PRODUCERS[name]
    if not path.is_file():
        raise FileNotFoundError(where)
    try:
        data = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (OSError, UnicodeDecodeError) as exc:
        raise ArtifactError(f"{where} could not be read ({type(exc).__name__}).") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactError(
            f"{where} is not valid JSON ({exc.msg} at line {exc.lineno}); re-run {producer}."
        ) from exc
    except ValueError as exc:  # from _reject_constant
        raise ArtifactError(
            f"{where} {exc} (not finite, so not valid in a response); re-run {producer}."
        ) from exc
    if name == "temporal_check":
        if not isinstance(data, dict) or data.get("analysis") != TEMPORAL_ANALYSIS:
            raise ArtifactError(
                f"{where} is not a temporal-check result (expected analysis="
                f"{TEMPORAL_ANALYSIS!r}); re-run {producer}."
            )
        return data
    contract = CONTRACTS[name]
    try:
        return contract.model_validate(data)
    except ValidationError as exc:
        raise ArtifactError(
            f"{where} does not match the {contract.__name__} contract ({_first_errors(exc)}); "
            f"re-run {producer} or fix the file."
        ) from exc


def load_json_artifacts(
    paths: dict[str, Path], source: Source
) -> tuple[dict[str, Any], dict[str, str]]:
    """Read every artifact in ``paths``; never raises.

    Args:
        paths: From :func:`artifact_paths`.
        source: ``"examples"`` changes the hint for a missing file.

    Returns:
        ``(loaded, unavailable)``: artifact name -> value, and artifact name -> problem message.
        A missing OPTIONAL file is neither loaded nor a problem.
    """
    loaded: dict[str, Any] = {}
    unavailable: dict[str, str] = {}
    for name, path in paths.items():
        try:
            loaded[name] = read_artifact(name, path)
        except FileNotFoundError:
            if name in OPTIONAL:
                continue
            hint = (
                "restore it from git"
                if source == "examples"
                else f"run {PRODUCERS[name]} (or set USE_EXAMPLE_ARTIFACTS=true to serve the "
                "example files)"
            )
            unavailable[name] = f"{name}{JSON_SUFFIX} not found at {display_path(path)}; {hint}."
        except ArtifactError as exc:
            unavailable[name] = str(exc)
    return loaded, unavailable


def load_feature_reference(models_dir: Path) -> tuple[FeatureReference | None, list[str]]:
    """Read and validate ``MODELS_DIR/feature_reference.json``; never raises.

    Args:
        models_dir: ``MODELS_DIR``.

    Returns:
        ``(reference or None, problems)``. A missing file is not a problem (each prediction then
        says it has no explanation); an unreadable or invalid file is, and so is a valid one
        without training ranges (exported before contract 1.6), which is still returned.
    """
    path = models_dir / FEATURE_REFERENCE_FILE
    where = display_path(path)
    fix = "re-run scripts/export_artifacts.py"
    if not path.is_file():
        return None, []
    try:
        data = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
        reference = FeatureReference.model_validate(data)
    except (OSError, UnicodeDecodeError, ValueError) as exc:  # JSONDecodeError is a ValueError
        reason = _first_errors(exc) if isinstance(exc, ValidationError) else type(exc).__name__
        return None, [
            f"{where} could not be used ({reason}); {fix}. Predictions are served without "
            "explanations or range warnings."
        ]
    if reference.range_percentiles is None:
        return reference, [
            f"{where} has no training ranges, so /predict cannot warn about values outside the "
            f"range seen in training; {fix}."
        ]
    return reference, []


def version_mismatches(recorded: dict[str, str], current: dict[str, str]) -> list[str]:
    """Pickle libraries (:data:`fraudstudy.export.PICKLE_LIBRARIES`) whose version differs.

    Args:
        recorded: ``ManifestEntry.library_versions`` (versions at export).
        current: This environment's ``library_versions()``.

    Returns:
        One ``"<lib> <export version> at export, <current> here"`` per mismatch.
    """
    return [
        f"{lib} {recorded.get(lib, 'not recorded')} at export, {current.get(lib)} here"
        for lib in PICKLE_LIBRARIES
        if recorded.get(lib) != current.get(lib)
    ]


def load_checked_file(models_dir: Path, relative: str, sha256: str, what: str) -> Any:
    """``joblib.load`` a file listed in the manifest after checking where it is and its hash.

    Args:
        models_dir: ``MODELS_DIR``; the file must resolve inside it.
        relative: Path relative to ``models_dir`` (``ManifestEntry.file``).
        sha256: Expected SHA-256 from the manifest.
        what: What it is, for messages (e.g. ``"model lightgbm__smote"``).

    Returns:
        The unpickled object.

    Raises:
        ArtifactError: If the path leaves ``models_dir``, the file is missing, its hash differs,
            or it cannot be unpickled.
    """
    base = models_dir.resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base):
        raise ArtifactError(f"{what}: {relative!r} points outside MODELS_DIR; fix manifest.json.")
    if not path.is_file():
        raise ArtifactError(
            f"{what}: {display_path(path)} not found; run scripts/export_artifacts.py or copy "
            "the deploy/ folder next to manifest.json."
        )
    if sha256_file(path) != sha256:
        raise ArtifactError(
            f"{what}: {display_path(path)} does not match its manifest SHA-256 (changed or "
            "corrupted); re-export it with scripts/export_artifacts.py."
        )
    try:
        return joblib.load(path)
    except Exception as exc:  # unpickling can fail in many library-specific ways
        raise ArtifactError(
            f"{what}: {display_path(path)} could not be loaded ({type(exc).__name__}); check "
            "the library versions (pip install -r requirements.txt) or re-export it."
        ) from exc


def single_threaded(pipeline: Any) -> Any:
    """Set ``n_jobs = PREDICT_N_JOBS`` on every step that has it (LightGBM, logistic regression,
    the ``ColumnTransformer``, a random forest).

    ``n_jobs`` only changes how many threads compute a prediction, never its value.

    Args:
        pipeline: A loaded pipeline, or a bare estimator.

    Returns:
        ``pipeline``, changed in place.
    """
    steps = [step for _, step in getattr(pipeline, "steps", [])] or [pipeline]
    for step in steps:
        get_params = getattr(step, "get_params", None)
        if callable(get_params) and "n_jobs" in get_params(deep=False):
            step.set_params(n_jobs=PREDICT_N_JOBS)
    return pipeline


def load_model(
    entry: ManifestEntry, models_dir: Path, current: dict[str, str]
) -> tuple[LoadedModel | None, list[str]]:
    """Load one manifest model and its calibrator; never raises.

    A model is not loaded when its library versions differ, its file is missing, changed or
    unloadable, or it has no ``predict_proba``. A calibrator that fails is a problem, but the
    model is still served (without calibrated probabilities). Loaded models predict
    single-threaded (:func:`single_threaded`).

    Args:
        entry: The manifest entry.
        models_dir: ``MODELS_DIR``.
        current: This environment's ``library_versions()``.

    Returns:
        ``(model or None, problems)``.
    """
    mismatches = version_mismatches(entry.library_versions, current)
    if mismatches:
        return None, [
            f"Model {entry.id} not loaded: library versions differ from the export "
            f"({'; '.join(mismatches)}). Install the pinned versions: pip install -r "
            "requirements.txt."
        ]
    try:
        pipeline = load_checked_file(models_dir, entry.file, entry.sha256, f"Model {entry.id}")
    except ArtifactError as exc:
        return None, [f"{exc} It is not loaded."]
    if not callable(getattr(pipeline, "predict_proba", None)):
        return None, [
            f"Model {entry.id}: {entry.file} has no predict_proba, so it is not a fitted "
            "classification pipeline; re-export it. It is not loaded."
        ]
    single_threaded(pipeline)
    problems: list[str] = []
    calibrator = None
    if entry.calibrator_file is not None and entry.calibrator_sha256 is not None:
        what = f"Calibrator of {entry.id}"
        try:
            calibrator = load_checked_file(
                models_dir, entry.calibrator_file, entry.calibrator_sha256, what
            )
        except ArtifactError as exc:
            problems.append(f"{exc} The model is served without calibrated probabilities.")
    elif entry.has_calibrator:
        problems.append(
            f"Model {entry.id}: has_calibrator is true but no calibrator_file is listed; "
            "re-export with scripts/export_artifacts.py. Served without calibrated probabilities."
        )
    return LoadedModel(entry, pipeline, calibrator), problems


def load_models(manifest: Manifest, models_dir: Path) -> tuple[dict[str, LoadedModel], list[str]]:
    """Load every model of the manifest (:func:`load_model`).

    Args:
        manifest: The validated manifest.
        models_dir: ``MODELS_DIR``.

    Returns:
        ``(loaded models by id, problems)``.
    """
    current = library_versions()
    models: dict[str, LoadedModel] = {}
    problems: list[str] = []
    if not manifest.models:
        problems.append("manifest.json lists no models; re-run scripts/export_artifacts.py.")
    for entry in manifest.models:
        model, model_problems = load_model(entry, models_dir, current)
        problems += model_problems
        if model is not None:
            models[entry.id] = model
    return models, problems


def resolve_default_model(
    manifest: Manifest, override: str | None, loaded: Iterable[str] | None
) -> tuple[str | None, list[str]]:
    """The default model: ``DEFAULT_MODEL_ID`` if it is in the manifest, else the manifest's.

    Args:
        manifest: The validated manifest.
        override: ``settings.default_model_id``.
        loaded: Ids of loaded models, or ``None`` to skip the "is it loaded" check (examples).

    Returns:
        ``(default id or None, problems)``.
    """
    problems: list[str] = []
    ids = [entry.id for entry in manifest.models]
    chosen = manifest.default_model_id
    if override is not None and override in ids:
        chosen = override
    elif override is not None:
        problems.append(
            f"DEFAULT_MODEL_ID={override!r} is not in manifest.json (models: "
            f"{', '.join(ids) or 'none'}); using the manifest default {chosen!r} instead."
        )
    if chosen is None:
        problems.append(
            "No default model: manifest.json has default_model_id null and DEFAULT_MODEL_ID is "
            "not set. Re-export with --default-cell or set DEFAULT_MODEL_ID."
        )
    elif loaded is not None and chosen not in set(loaded):
        problems.append(f"The default model {chosen} is not loaded (see the problem above).")
    return chosen, problems


def load_artifacts(settings: Settings) -> ArtifactStore:
    """Load and check everything the API serves; never raises.

    Steps: settings problems (:meth:`Settings.config_problems`) -> JSON artifacts
    (:func:`load_json_artifacts`) -> models and feature reference (real source only;
    :func:`load_models`, :func:`load_feature_reference`) -> default model
    (:func:`resolve_default_model`) -> git hash.

    Args:
        settings: The backend settings.

    Returns:
        The :class:`ArtifactStore`.
    """
    source: Source = "examples" if settings.use_example_artifacts else "real"
    store = ArtifactStore(source=source, problems=settings.config_problems())
    try:
        loaded, unavailable = load_json_artifacts(artifact_paths(settings), source)
        for name, value in loaded.items():
            setattr(store, name, value)
        store.unavailable = unavailable
        store.problems += list(unavailable.values())
        if store.manifest is not None:
            loaded_ids: list[str] | None = None
            if source == "real":
                store.models, model_problems = load_models(store.manifest, settings.models_dir)
                store.problems += model_problems
                loaded_ids = list(store.models)
                store.feature_reference, reference_problems = load_feature_reference(
                    settings.models_dir
                )
                store.problems += reference_problems
            store.default_model_id, default_problems = resolve_default_model(
                store.manifest, settings.default_model_id, loaded_ids
            )
            store.problems += default_problems
    except Exception as exc:  # last resort: start degraded rather than crash
        logger.exception("artifact loading failed")
        store.problems.append(
            f"Unexpected {type(exc).__name__} while loading artifacts; see the server log."
        )
    store.git_hash = git_commit_hash()
    return store


def get_store(request: Request) -> ArtifactStore:
    """FastAPI dependency: the store loaded at start-up.

    Raises:
        ApiError: 503 ``service_unavailable`` if start-up has not run.
    """
    store = getattr(request.app.state, "store", None)
    if not isinstance(store, ArtifactStore):
        raise ApiError(
            503,
            "service_unavailable",
            "The API is still starting up; try again in a few seconds.",
        )
    return store


StoreDep = Annotated[ArtifactStore, Depends(get_store)]
