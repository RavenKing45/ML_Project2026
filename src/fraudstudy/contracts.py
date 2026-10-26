"""Data contracts between research code, backend and frontend (Pydantic v2).

This module is the single source of truth for every saved artifact and API payload;
docs/contracts.md describes each one and docs/contracts/examples/ holds a fake example of each.
Later code imports these models to write (``model_dump(mode="json")``) and read
(``model_validate``) artifacts. API shapes follow Part 9 of the master guide.

Conventions:

* Field names are snake_case. Every model forbids unknown fields, so a typo fails validation,
  and rejects NaN/infinity.
* Every top-level file/payload model has ``contract_version`` and ``is_example`` (``True`` only
  in the fixture files, so the dashboard can show an "example data" banner).
* Probabilities, rates and areas are floats in [0, 1]; counts are non-negative ints.
"""

from typing import Annotated, Any, Literal, Required

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)
from typing_extensions import TypedDict  # pydantic needs this one on Python < 3.12

CONTRACT_VERSION = "1.6"
MAX_CURVE_POINTS = 101
AMOUNT_MAX = 1_000_000.0

ModelId = Literal["logreg", "random_forest", "lightgbm"]
TechniqueId = Literal["baseline", "ros", "rus", "smote", "smote_rus", "class_weight"]
ThresholdSource = Literal[
    "tuned_on_validation_F1", "tuned_on_validation_F2", "user_supplied", "default_0_5"
]
PredictedLabel = Literal["FRAUD", "LEGITIMATE"]
InputSource = Literal["preset", "edited", "manual"]
DatasetSource = Literal["real", "synthetic"]

MODEL_LABELS: dict[str, str] = {
    "logreg": "Logistic regression",
    "random_forest": "Random forest",
    "lightgbm": "LightGBM",
}
TECHNIQUE_LABELS: dict[str, str] = {
    "baseline": "A. Baseline (no resampling)",
    "ros": "B. Random oversampling",
    "rus": "C. Random undersampling",
    "smote": "D. SMOTE",
    "smote_rus": "E. SMOTE + random undersampling",
    "class_weight": "F. Class weighting",
}

Unit = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegInt = Annotated[int, Field(ge=0)]
NonNegFloat = Annotated[float, Field(ge=0.0)]
Percentile = Annotated[float, Field(ge=0.0, le=100.0)]
Interval = tuple[Unit, Unit]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
JsonScalar = str | int | float | bool | None


class StrictModel(BaseModel):
    """Base for every contract model: unknown fields and non-finite floats are rejected."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TopLevel(StrictModel):
    """Base for models that are a whole JSON file or API payload."""

    contract_version: str = CONTRACT_VERSION
    is_example: bool = False


def _check_interval(name: str, interval: tuple[float, float] | None) -> None:
    """Raise ``ValueError`` if ``interval`` is not ``[low, high]`` with ``low <= high``."""
    if interval is not None and interval[0] > interval[1]:
        raise ValueError(f"{name} must be [low, high] with low <= high; got {list(interval)}.")


def _check_label(label: str, risk_score: float, threshold: float) -> None:
    """Raise ``ValueError`` unless ``label`` is ``"FRAUD"`` exactly when
    ``risk_score >= threshold``."""
    expected = "FRAUD" if risk_score >= threshold else "LEGITIMATE"
    if label != expected:
        raise ValueError(f"label must be {expected} for risk_score/threshold given.")


def _check_unique_cells(cells: list[Any], what: str) -> None:
    """Raise ``ValueError`` if two items share the same ``(model, technique)``."""
    keys = [(c.model, c.technique) for c in cells]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    if dupes:
        raise ValueError(f"{what} has duplicate (model, technique) cells: {dupes}.")


# --- shared pieces ------------------------------------------------------------------------------


class ClassCounts(StrictModel):
    """Number of rows per class."""

    legit: NonNegInt
    fraud: NonNegInt


class Metrics(StrictModel):
    """Classification metrics at one threshold, plus threshold-free areas.

    ``f2`` is F-beta with beta=2 (recall weighted 4x precision): (5PR)/(4P+R).
    ``fpr = fp/(fp+tn)``, ``fnr = fn/(fn+tp)``. ``pr_auc`` is average precision.
    """

    precision: Unit
    recall: Unit
    f1: Unit
    f2: Unit
    fpr: Unit
    fnr: Unit
    mcc: Annotated[float, Field(ge=-1.0, le=1.0)]
    accuracy: Unit
    tp: NonNegInt
    fp: NonNegInt
    tn: NonNegInt
    fn: NonNegInt
    pr_auc: Unit
    roc_auc: Unit


class CellRef(StrictModel):
    """Identifies one (model, technique) cell of the experiment matrix."""

    model: ModelId
    technique: TechniqueId


# --- comparison.json ----------------------------------------------------------------------------


class ComparisonRecord(CellRef):
    """Final result for one (model, technique) cell.

    ``chosen_ratio`` is the tuned minority:majority ``sampling_strategy`` of the last resampling
    step (for ``smote_rus`` the RUS ratio; the SMOTE ratio is in ``best_params``), or the ratio the
    class weights were derived from for ``class_weight``; ``None`` for ``baseline``.
    ``test_at_tuned`` uses ``threshold_f1``. CIs are bootstrap intervals on the test set.
    """

    model_label: str
    technique_label: str
    best_params: dict[str, JsonScalar]
    chosen_ratio: Annotated[float, Field(gt=0.0)] | None
    cv_pr_auc_mean: Unit
    cv_pr_auc_std: NonNegFloat
    cv_roc_auc_mean: Unit
    threshold_f1: Unit
    threshold_f2: Unit
    test_at_0_5: Metrics
    test_at_tuned: Metrics
    pr_auc_ci: Interval
    roc_auc_ci: Interval
    fit_seconds: NonNegFloat
    n_train_after_resampling: ClassCounts

    @model_validator(mode="after")
    def _intervals_ordered(self) -> "ComparisonRecord":
        _check_interval("pr_auc_ci", self.pr_auc_ci)
        _check_interval("roc_auc_ci", self.roc_auc_ci)
        return self


class ComparisonTable(TopLevel):
    """``comparison.json``: one record per cell (18 for the full matrix)."""

    primary_metric: Literal["average_precision"] = "average_precision"
    dataset_source: DatasetSource
    records: list[ComparisonRecord]

    @model_validator(mode="after")
    def _unique(self) -> "ComparisonTable":
        _check_unique_cells(self.records, "comparison records")
        return self


# --- curves.json --------------------------------------------------------------------------------


def _check_lengths(name: str, **arrays: list[float]) -> None:
    """Raise ``ValueError`` unless all arrays have the same length in [1, MAX_CURVE_POINTS]."""
    lengths = {k: len(v) for k, v in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{name}: arrays must have equal length; got {lengths}.")
    n = next(iter(lengths.values()))
    if not 1 <= n <= MAX_CURVE_POINTS:
        raise ValueError(f"{name}: need 1..{MAX_CURVE_POINTS} points; got {n}. Downsample.")


class RocCurve(StrictModel):
    """ROC curve points."""

    fpr: list[Unit]
    tpr: list[Unit]

    @model_validator(mode="after")
    def _lengths(self) -> "RocCurve":
        _check_lengths("roc", fpr=self.fpr, tpr=self.tpr)
        return self


class PrCurve(StrictModel):
    """Precision-recall curve points."""

    recall: list[Unit]
    precision: list[Unit]

    @model_validator(mode="after")
    def _lengths(self) -> "PrCurve":
        _check_lengths("pr", recall=self.recall, precision=self.precision)
        return self


class ThresholdTable(StrictModel):
    """Precision, recall and FPR at a grid of thresholds (for the threshold slider)."""

    threshold: list[Unit]
    precision: list[Unit]
    recall: list[Unit]
    fpr: list[Unit]

    @model_validator(mode="after")
    def _lengths(self) -> "ThresholdTable":
        _check_lengths(
            "threshold_table",
            threshold=self.threshold,
            precision=self.precision,
            recall=self.recall,
            fpr=self.fpr,
        )
        return self


class Curves(CellRef):
    """Test-set curves for one cell, each downsampled to at most ``MAX_CURVE_POINTS`` points."""

    roc: RocCurve
    pr: PrCurve
    threshold_table: ThresholdTable


class CurvesFile(TopLevel):
    """``curves.json``: curves for every cell."""

    cells: list[Curves]

    @model_validator(mode="after")
    def _unique(self) -> "CurvesFile":
        _check_unique_cells(self.cells, "curves")
        return self


# --- stats.json ---------------------------------------------------------------------------------


class PairwiseTest(StrictModel):
    """One statistical comparison of two techniques under the same model.

    ``difference`` is metric(technique_a) - metric(technique_b); ``difference_ci`` its bootstrap CI
    (``None`` for tests that do not produce one, e.g. Wilcoxon or McNemar). For a paired test on
    fold scores, ``difference`` is the mean of the per-fold differences.

    Since 1.3 (optional): ``n_pairs`` paired blocks, ``median_difference`` the median per-block
    difference, and ``wins`` / ``ties`` / ``losses`` the blocks where technique_a scored higher /
    equal / lower than technique_b.
    """

    model: ModelId
    technique_a: TechniqueId
    technique_b: TechniqueId
    test: Literal["wilcoxon", "paired_bootstrap", "mcnemar"]
    metric: str
    difference: float
    difference_ci: tuple[float, float] | None
    statistic: float
    p_value: Unit
    p_value_adjusted: Unit
    significant: bool
    n_pairs: NonNegInt | None = None
    median_difference: float | None = None
    wins: NonNegInt | None = None
    ties: NonNegInt | None = None
    losses: NonNegInt | None = None

    @model_validator(mode="after")
    def _ci(self) -> "PairwiseTest":
        _check_interval("difference_ci", self.difference_ci)
        counts = (self.wins, self.ties, self.losses)
        if self.n_pairs is not None and None not in counts and sum(counts) != self.n_pairs:
            raise ValueError(f"wins + ties + losses must equal n_pairs ({self.n_pairs}).")
        return self


class FriedmanTest(StrictModel):
    """Friedman test across all techniques, plus what a critical-difference diagram needs.

    ``scope`` is a model id (blocks = CV folds of that model) or ``"all"`` (blocks = every
    model x fold). Rank 1 is best. ``critical_difference`` is the Nemenyi CD at ``alpha``.
    """

    scope: ModelId | Literal["all"]
    metric: str
    n_blocks: Annotated[int, Field(ge=2)]
    statistic: float
    p_value: Unit
    average_ranks: dict[TechniqueId, Annotated[float, Field(ge=1.0)]]
    critical_difference: NonNegFloat | None


class StatsResult(TopLevel):
    """``stats.json``: Friedman tests and pairwise tests with multiple-comparison correction.

    Since 1.3 (optional): ``dataset_source`` of the fold scores and ``notes``, caveats the
    dashboard and report must show next to the tests (e.g. that fold scores are not independent).
    """

    alpha: Annotated[float, Field(gt=0.0, lt=1.0)]
    correction: Literal["holm", "bonferroni", "none"]
    n_bootstrap: NonNegInt
    confidence: Annotated[float, Field(gt=0.0, lt=1.0)]
    seed: int
    friedman: list[FriedmanTest]
    pairwise: list[PairwiseTest]
    dataset_source: DatasetSource | None = None
    notes: list[str] = Field(default_factory=list)


# --- eda.json -----------------------------------------------------------------------------------


class Distribution(StrictModel):
    """Five-number summary plus mean, std and zero count of one column for one class."""

    count: NonNegInt
    mean: float
    std: NonNegFloat
    min: float
    q1: float
    median: float
    q3: float
    max: float
    n_zero: NonNegInt


class ClassSplit(StrictModel):
    """A ``Distribution`` per class."""

    legit: Distribution
    fraud: Distribution


class Histogram(StrictModel):
    """Per-class counts over shared bins; ``len(bin_edges) == len(counts) + 1``."""

    bin_edges: list[float]
    legit_counts: list[NonNegInt]
    fraud_counts: list[NonNegInt]

    @model_validator(mode="after")
    def _shape(self) -> "Histogram":
        if not (len(self.legit_counts) == len(self.fraud_counts) == len(self.bin_edges) - 1):
            raise ValueError(
                "Histogram needs len(bin_edges) == len(legit_counts) + 1 == len(fraud_counts) + 1."
            )
        return self


class FeatureSeparation(StrictModel):
    """How strongly one feature separates the classes (sorted by ``ks_statistic`` desc)."""

    feature: str
    legit_mean: float
    fraud_mean: float
    legit_std: NonNegFloat
    fraud_std: NonNegFloat
    ks_statistic: Unit


class TrainingCounts(StrictModel):
    """Training-set class counts before and after one technique's resampler (at its tuned ratio,
    on the full training set). Equal before/after for ``baseline`` and ``class_weight``."""

    technique: TechniqueId
    ratio: Annotated[float, Field(gt=0.0)] | None
    before: ClassCounts
    after: ClassCounts


class AmountStats(StrictModel):
    """Summary of ``Amount`` (both classes pooled). ``skewness`` is the bias-corrected
    Fisher-Pearson coefficient G1 (0 for a symmetric distribution)."""

    min: float
    median: float
    mean: float
    max: float
    skewness: float


class AmountSummary(StrictModel):
    """``Amount`` statistics on the raw scale and after ``log1p`` (``log(1 + Amount)``)."""

    raw: AmountStats
    log1p: AmountStats


class EffectSize(StrictModel):
    """Standardised difference of one feature between classes (Cohen's d).

    ``d = (fraud_mean - legit_mean) / s_pooled`` with
    ``s_pooled = sqrt(((n_f - 1) s_f^2 + (n_l - 1) s_l^2) / (n_f + n_l - 2))``; positive means
    fraud rows have larger values. ``|d|`` of 0.2 / 0.5 / 0.8 is conventionally small / medium /
    large.
    """

    feature: str
    cohens_d: float
    legit_mean: float
    fraud_mean: float


class FeatureHistogram(Histogram):
    """Per-class histogram of one feature over shared bins, with per-class fractions.

    ``legit_fraction[i] = legit_counts[i] / sum(legit_counts)`` (likewise for fraud), so the two
    classes can be compared despite the imbalance. The bins span the union of each class's
    ``bin_range_quantiles`` range; values beyond it are counted in the outermost bins.
    """

    feature: str
    legit_fraction: list[Unit]
    fraud_fraction: list[Unit]
    bin_range_quantiles: Interval

    @model_validator(mode="after")
    def _fractions(self) -> "FeatureHistogram":
        n = len(self.legit_counts)
        if not (len(self.legit_fraction) == len(self.fraud_fraction) == n):
            raise ValueError("FeatureHistogram fractions need one value per bin.")
        _check_interval("bin_range_quantiles", self.bin_range_quantiles)
        return self


HOUR_ASSUMPTION = "assumes Time=0 is midnight; not documented by the data provider"
HOURS_PER_DAY = 24


class HourOfDay(StrictModel):
    """Transactions per hour of day, ``hour = floor(Time / 3600) mod 24``.

    Index ``i`` of each list is hour ``i`` (0-23). ``hour_assumption`` states that the clock time
    is a guess (the provider never said when ``Time=0`` is).
    """

    hour_assumption: str
    fraud_counts: list[NonNegInt]
    legit_counts: list[NonNegInt]

    @model_validator(mode="after")
    def _shape(self) -> "HourOfDay":
        if not (len(self.fraud_counts) == len(self.legit_counts) == HOURS_PER_DAY):
            raise ValueError(f"HourOfDay needs {HOURS_PER_DAY} counts per class.")
        return self


class EdaSummary(TopLevel):
    """``eda.json``: dataset summary plus before/after resampling counts per technique.

    ``n_rows``, ``n_columns``, ``class_counts``, ``fraud_rate``, ``imbalance_ratio`` and
    ``missing_values`` describe the raw file; every distributional field describes the
    deduplicated data. Fields added in 1.1 are optional so 1.0 files still validate.
    ``training_counts`` is filled by the experiment code (it needs tuned ratios) and may be empty.
    """

    dataset_source: DatasetSource
    dataset_sha256: str
    n_rows: NonNegInt
    n_columns: NonNegInt
    class_counts: ClassCounts
    fraud_rate: Unit
    n_duplicate_rows: NonNegInt
    n_duplicate_frauds: NonNegInt
    missing_values: dict[str, NonNegInt]
    amount: ClassSplit
    amount_histogram: Histogram
    time_histogram: Histogram
    feature_separation: list[FeatureSeparation]
    training_counts: list[TrainingCounts] = Field(default_factory=list)
    # --- added in 1.1 (all optional) ---
    imbalance_ratio: Annotated[str, Field(pattern=r"^1 : [0-9]+$")] | None = None
    class_counts_dedup: ClassCounts | None = None
    amount_summary: AmountSummary | None = None
    time_span_hours: NonNegFloat | None = None
    effect_sizes: list[EffectSize] = Field(default_factory=list)
    feature_histograms: list[FeatureHistogram] = Field(default_factory=list)
    hour_of_day: HourOfDay | None = None

    @model_validator(mode="after")
    def _sorted(self) -> "EdaSummary":
        ks = [f.ks_statistic for f in self.feature_separation]
        if ks != sorted(ks, reverse=True):
            raise ValueError("feature_separation must be sorted by ks_statistic, descending.")
        d = [abs(e.cohens_d) for e in self.effect_sizes]
        if d != sorted(d, reverse=True):
            raise ValueError("effect_sizes must be sorted by |cohens_d|, descending.")
        return self


# --- resampling_audit.json ----------------------------------------------------------------------


class ResamplingAuditEntry(CellRef):
    """Row counts before/after resampling in one CV training fold, and the untouched val fold.

    ``validation_untouched`` is True when the validation fold's rows and labels are identical to
    the pre-resampling split (the leakage check).

    Since 1.2 (optional): for ``smote_rus``, ``ratio`` is the final RUS ratio b and
    ``smote_ratio`` the SMOTE ratio a; ``fraud_weight`` is technique F's weight ``ratio /
    base_ratio``; ``fraud_unique_after`` / ``fraud_duplicates_after`` count distinct and
    exactly-copied fraud rows after resampling.
    """

    fold: NonNegInt
    ratio: Annotated[float, Field(gt=0.0)] | None
    train_before: ClassCounts
    train_after: ClassCounts
    validation: ClassCounts
    n_synthetic: NonNegInt
    validation_untouched: bool
    smote_ratio: Annotated[float, Field(gt=0.0)] | None = None
    fraud_weight: Annotated[float, Field(gt=0.0)] | None = None
    fraud_unique_after: NonNegInt | None = None
    fraud_duplicates_after: NonNegInt | None = None


class ResamplingAudit(TopLevel):
    """``resampling_audit.json``: proof that samplers only touched training folds."""

    seed: int
    cv_folds: Annotated[int, Field(ge=2)]
    test_rows_resampled: NonNegInt
    all_checks_passed: bool
    entries: list[ResamplingAuditEntry]
    base_ratio: Annotated[float, Field(gt=0.0)] | None = None
    dropped_ratios: list[float] = []


# --- artifacts/models/manifest.json -------------------------------------------------------------


class ManifestEntry(CellRef):
    """One saved model pipeline.

    ``id`` is ``"<model>__<technique>"``; ``file`` is relative to ``artifacts/models/``.
    ``features`` is the exact column order the pipeline expects; the backend builds its input
    table in this order. ``library_versions`` must match the backend's at load time.

    Optional since 1.4: ``model_label`` / ``technique_label`` (display names);
    ``default_threshold`` (which tuned threshold applies when a request gives none);
    ``calibrator_file`` (relative to ``artifacts/models/``) and its ``calibrator_sha256``, set
    together and only when ``has_calibrator``.
    """

    id: str
    file: str
    sha256: Sha256
    threshold_f1: Unit
    threshold_f2: Unit
    has_calibrator: bool
    features: list[str]
    data_sha256: str
    library_versions: dict[str, str]
    model_label: str | None = None
    technique_label: str | None = None
    default_threshold: Literal["f1", "f2"] = "f1"
    calibrator_file: str | None = None
    calibrator_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def _id_matches_cell(self) -> "ManifestEntry":
        expected = f"{self.model}__{self.technique}"
        if self.id != expected:
            raise ValueError(f"Manifest id must be '{expected}' for this cell; got '{self.id}'.")
        if (self.calibrator_file is None) != (self.calibrator_sha256 is None):
            raise ValueError(f"{self.id}: set calibrator_file and calibrator_sha256 together.")
        if self.calibrator_file is not None and not self.has_calibrator:
            raise ValueError(f"{self.id}: calibrator_file is set but has_calibrator is false.")
        return self


class DatasetInfo(StrictModel):
    """Which dataset a run used."""

    name: str
    source: DatasetSource
    sha256: str
    n_rows: NonNegInt


class Manifest(TopLevel):
    """``artifacts/models/manifest.json``: the deployed models and their provenance."""

    created_at: str  # ISO 8601, UTC
    git_commit: str | None
    seed: int
    quick_mode: bool
    dataset: DatasetInfo
    default_model_id: str | None
    models: list[ManifestEntry]

    @model_validator(mode="after")
    def _ids(self) -> "Manifest":
        ids = [m.id for m in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Manifest model ids must be unique; got {ids}.")
        if self.default_model_id is not None and self.default_model_id not in ids:
            raise ValueError(
                f"default_model_id '{self.default_model_id}' is not in the manifest ({ids})."
            )
        return self


# --- features, presets, predict -----------------------------------------------------------------


class Transaction(StrictModel):
    """One payment's raw inputs. ``Time`` is accepted but ignored by the model.

    Column names keep the dataset's capitalisation so CSV rows map 1:1. All values must be
    finite; ``Amount`` is in [0, ``AMOUNT_MAX``].
    """

    Time: NonNegFloat | None = None
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: Annotated[float, Field(ge=0.0, le=AMOUNT_MAX)]


class Preset(StrictModel):
    """A real test-set payment for the one-click demo on the "Try a transaction" page."""

    id: str
    label: str
    description: str
    expected_class: Literal[0, 1]
    features: Transaction


class PresetsFile(TopLevel):
    """``artifacts/demo/presets.json``: 5-6 demo presets."""

    presets: list[Preset]


class ModelRef(StrictModel):
    """Which deployed model produced a prediction; ``id`` is the manifest id."""

    id: str
    model: ModelId
    technique: TechniqueId


class FeatureContribution(StrictModel):
    """One feature's contribution to a score.

    ``legit_percentile`` is where ``value`` falls among legit training rows (0-100).
    """

    feature: str
    value: float
    contribution: float
    legit_percentile: Percentile


class Explanation(StrictModel):
    """Per-prediction explanation; ``units`` says what ``contribution`` is measured in."""

    method: str
    units: str
    top_features: list[FeatureContribution]


class PredictRequest(StrictModel):
    """``POST /api/v1/predict`` body.

    ``model_id=None`` selects the default model; ``threshold=None`` uses its tuned threshold.

    Since 1.6 (optional): ``input_source`` says where the features came from: ``"preset"`` (an
    unchanged demo preset), ``"edited"`` (a preset with edited values) or ``"manual"`` (typed or
    pasted). Edited and manual inputs get a warning that they may not resemble real payments.
    """

    features: Transaction
    model_id: str | None = None
    threshold: Unit | None = None
    input_source: InputSource | None = None


class TechniqueScore(StrictModel):
    """The same payment scored by one deployed model of the technique comparison (since 1.6).

    Each model applies its OWN tuned default threshold (a user-supplied threshold is not reused:
    resampling shifts score scales, so one cut-off would not compare techniques fairly).
    ``label`` is ``"FRAUD"`` exactly when ``risk_score >= threshold``.
    """

    id: str
    model: ModelId
    technique: TechniqueId
    risk_score: Unit
    threshold: Unit
    label: PredictedLabel

    @model_validator(mode="after")
    def _label_consistent(self) -> "TechniqueScore":
        _check_label(self.label, self.risk_score, self.threshold)
        return self


class PredictResponse(TopLevel):
    """``POST /api/v1/predict`` response.

    ``label`` is ``"FRAUD"`` exactly when ``risk_score >= threshold``. ``risk_score`` is a ranking
    score, not a probability; ``calibrated_probability`` is ``None`` without a calibrator.

    Since 1.6 (optional): ``technique_scores``, the same payment scored by every deployed LightGBM
    technique, each at its own tuned threshold (:class:`TechniqueScore`).
    """

    label: PredictedLabel
    risk_score: Unit
    calibrated_probability: Unit | None
    threshold: Unit
    threshold_source: ThresholdSource
    model: ModelRef
    explanation: Explanation | None
    warnings: list[str]
    disclaimer: str
    latency_ms: NonNegFloat
    technique_scores: list[TechniqueScore] = Field(default_factory=list)

    @model_validator(mode="after")
    def _label_consistent(self) -> "PredictResponse":
        _check_label(self.label, self.risk_score, self.threshold)
        return self


class BatchRowResult(StrictModel):
    """Score for one row of an uploaded CSV (``row`` is 0-based, header excluded).

    Since 1.6 (optional): ``true_class``, the row's ``Class`` when the CSV had that column.
    """

    row: NonNegInt
    amount: NonNegFloat
    risk_score: Unit
    calibrated_probability: Unit | None
    label: PredictedLabel
    true_class: Literal[0, 1] | None = None


class BatchRowError(StrictModel):
    """A CSV row that could not be scored."""

    row: NonNegInt
    message: str


class BatchResponse(TopLevel):
    """``POST /api/v1/predict/batch`` response: summary plus row-level results.

    ``metrics`` is set only when the uploaded CSV had a ``Class`` column.
    """

    model: ModelRef
    threshold: Unit
    threshold_source: ThresholdSource
    n_rows: NonNegInt
    n_scored: NonNegInt
    n_flagged: NonNegInt
    flagged_fraction: Unit
    flagged_amount_total: NonNegFloat
    metrics: Metrics | None
    results: list[BatchRowResult]
    errors: list[BatchRowError]
    warnings: list[str]
    disclaimer: str
    latency_ms: NonNegFloat


class ErrorDetailBody(TypedDict, total=False):
    """Serialised form of :class:`ErrorDetail` (documents its JSON schema; ``details`` optional)."""

    code: Required[str]
    message: Required[str]
    details: Any


class ErrorDetail(StrictModel):
    """Machine-readable ``code`` (e.g. ``"unknown_model"``) and a human ``message``.

    Since 1.5 (optional): ``details``, any JSON value with more context (e.g. the valid model
    ids, or the field errors of a 422), ``null`` when there is nothing to add. The backend
    always sends it. It is serialised only when it was given, so a 1.4 body without it
    round-trips unchanged.
    """

    code: str
    message: str
    details: Any = None

    @model_serializer(mode="wrap")
    def _omit_unset_details(self, handler: SerializerFunctionWrapHandler) -> ErrorDetailBody:
        data = handler(self)
        if "details" not in self.model_fields_set:
            data.pop("details", None)
        return data


class ErrorResponse(StrictModel):
    """Body of every non-2xx API response:
    ``{"error": {"code": ..., "message": ..., "details": ...}}``."""

    error: ErrorDetail


# --- GET endpoints of the backend (since 1.5) ---------------------------------------------------

HealthStatus = Literal["ok", "degraded"]
DefaultThreshold = Literal["f1", "f2"]


class HealthResponse(TopLevel):
    """``GET /api/v1/health``: is the backend up, and did every artifact load?

    ``status`` is ``"degraded"`` exactly when ``problems`` is non-empty (validated); each problem
    says what is wrong and how to fix it. ``models_loaded`` counts model files loaded from the
    manifest (always 0 with the example artifacts, which have no model files).
    ``default_model`` is the manifest id used when a request names none (``DEFAULT_MODEL_ID``
    overrides the manifest's). ``data_sha256`` is the training data's hash from the manifest;
    ``git_hash`` the backend's commit (``None`` outside a git checkout); ``uptime_s`` seconds
    since start-up. ``is_example`` is true when the backend serves the example artifacts.
    """

    status: HealthStatus
    problems: list[str]
    models_loaded: NonNegInt
    default_model: str | None
    data_sha256: str | None
    git_hash: str | None
    version: str
    uptime_s: NonNegFloat

    @model_validator(mode="after")
    def _status_matches_problems(self) -> "HealthResponse":
        expected = "degraded" if self.problems else "ok"
        if self.status != expected:
            raise ValueError(f"status must be {expected!r} when problems has {len(self.problems)}.")
        return self


class ModelInfo(CellRef):
    """One manifest model as ``GET /api/v1/models`` lists it.

    ``loaded`` is true when the backend loaded its file at start-up (always false with the
    example artifacts); only loaded models can score requests. ``default_threshold`` says which
    tuned threshold applies when a request gives none. ``features`` is the pipeline's input
    column order.
    """

    id: str
    model_label: str
    technique_label: str
    threshold_f1: Unit
    threshold_f2: Unit
    default_threshold: DefaultThreshold
    has_calibrator: bool
    features: list[str]
    loaded: bool

    @model_validator(mode="after")
    def _id_matches_cell(self) -> "ModelInfo":
        expected = f"{self.model}__{self.technique}"
        if self.id != expected:
            raise ValueError(f"Model id must be '{expected}' for this cell; got '{self.id}'.")
        return self


class ModelsResponse(TopLevel):
    """``GET /api/v1/models``: the deployed models and the default one (one of ``models``)."""

    default_model_id: str | None
    models: list[ModelInfo]

    @model_validator(mode="after")
    def _ids(self) -> "ModelsResponse":
        ids = [m.id for m in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Model ids must be unique; got {ids}.")
        if self.default_model_id is not None and self.default_model_id not in ids:
            raise ValueError(f"default_model_id '{self.default_model_id}' is not in {ids}.")
        return self


class ResultsResponse(TopLevel):
    """``GET /api/v1/results``: ``comparison.json``, ``curves.json`` and ``stats.json`` as
    written, plus ``temporal_check.json`` when present.

    ``temporal_check`` is a labelled SENSITIVITY ANALYSIS, not the main result; its inner shape
    is internal to ``fraudstudy.final_test.temporal_check`` (not a contract) and it is
    ``None`` when the check has not been run. ``is_example`` is true if any part is an example.
    """

    comparison: ComparisonTable
    curves: CurvesFile
    stats: StatsResult
    temporal_check: dict[str, Any] | None = None


class EdaResponse(TopLevel):
    """``GET /api/v1/eda``: ``eda.json`` plus ``resampling_audit.json`` (``None`` if absent)."""

    eda: EdaSummary
    resampling_audit: ResamplingAudit | None = None


# --- SMOTE demo ---------------------------------------------------------------------------------


class SmoteDemoRequest(StrictModel):
    """``POST /api/v1/smote-demo`` body: a small 2-D toy dataset to resample.

    Since 1.6 (optional): ``oversample_percent`` (100-500 in steps of 100), the SMOTE paper's
    "N%": ``n_fraud * oversample_percent / 100`` synthetic points. When given, it replaces
    ``sampling_strategy`` (imblearn's target ``fraud / legit`` ratio), which is then ignored.
    """

    scenario: Literal["single_cluster", "two_clusters", "overlap"] = "two_clusters"
    n_legit: Annotated[int, Field(ge=2, le=2000)] = 300
    n_fraud: Annotated[int, Field(ge=2, le=200)] = 20
    k_neighbors: Annotated[int, Field(ge=1, le=15)] = 5
    sampling_strategy: Annotated[float, Field(gt=0.0, le=1.0)] = 0.5
    seed: int = 42
    oversample_percent: Annotated[int, Field(ge=100, le=500, multiple_of=100)] | None = None

    @model_validator(mode="after")
    def _enough_neighbours(self) -> "SmoteDemoRequest":
        if self.k_neighbors >= self.n_fraud:
            raise ValueError(
                f"k_neighbors ({self.k_neighbors}) must be smaller than n_fraud ({self.n_fraud})."
            )
        return self


class Point2D(StrictModel):
    """An original point of the toy dataset."""

    x: float
    y: float
    label: Literal[0, 1]


class SyntheticPoint(StrictModel):
    """A SMOTE point ``x_new = x + gap * (x_hat - x)`` (the guide's lambda is ``gap``), where
    ``x = original[parent]`` and ``x_hat = original[neighbor]``, both fraud points."""

    x: float
    y: float
    parent: NonNegInt
    neighbor: NonNegInt
    gap: Unit
    in_legit_region: bool


class FraudNeighbors(StrictModel):
    """The ``k_neighbors`` nearest other fraud points of one fraud point (nearest first), as
    indices into ``original`` (the nearest hit, the point itself, is dropped, as in imblearn)."""

    point: NonNegInt
    neighbors: list[NonNegInt]


class DecisionGrid(StrictModel):
    """Predicted labels of a small classifier on a regular grid over the toy data, trained on the
    original points (``before``) and on original plus synthetic points (``after``).

    ``before[i][j]`` and ``after[i][j]`` are the labels at ``(x[j], y[i])``; ``model`` names the
    classifier, e.g. ``"DecisionTreeClassifier(max_depth=4)"``.
    """

    model: str
    x: list[float]
    y: list[float]
    before: list[list[Literal[0, 1]]]
    after: list[list[Literal[0, 1]]]

    @model_validator(mode="after")
    def _shape(self) -> "DecisionGrid":
        for name, grid in (("before", self.before), ("after", self.after)):
            if len(grid) != len(self.y) or any(len(row) != len(self.x) for row in grid):
                raise ValueError(f"DecisionGrid.{name} must have len(y) rows of len(x) labels.")
        return self


class SmoteDemoResponse(TopLevel):
    """``POST /api/v1/smote-demo`` response; echoes the request.

    Since 1.6 (optional): ``fraud_neighbors`` (one entry per fraud point of ``original``) and
    ``decision_grid`` (a decision boundary before and after oversampling).
    """

    request: SmoteDemoRequest
    original: list[Point2D]
    synthetic: list[SyntheticPoint]
    n_synthetic_in_legit_region: NonNegInt
    warnings: list[str]
    fraud_neighbors: list[FraudNeighbors] = Field(default_factory=list)
    decision_grid: DecisionGrid | None = None


# --- registry -----------------------------------------------------------------------------------

EXAMPLE_MODELS: dict[str, type[TopLevel]] = {
    "comparison": ComparisonTable,
    "curves": CurvesFile,
    "stats": StatsResult,
    "eda": EdaSummary,
    "resampling_audit": ResamplingAudit,
    "manifest": Manifest,
    "presets": PresetsFile,
    "predict_response": PredictResponse,
    "batch_response": BatchResponse,
    "smote_demo": SmoteDemoResponse,
}
"""Example file stem in docs/contracts/examples/ -> model it must validate against."""
