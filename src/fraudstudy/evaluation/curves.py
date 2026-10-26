"""ROC / PR curves and the threshold-slider table, downsampled for ``curves.json``.

The full curves can have thousands of points; the contract allows at most
``MAX_CURVE_POINTS`` per array, so points are picked evenly by index, always keeping both
endpoints. Picked points are real curve points (no interpolation).
"""

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from fraudstudy.contracts import (
    MAX_CURVE_POINTS,
    Curves,
    ModelId,
    PrCurve,
    RocCurve,
    TechniqueId,
    ThresholdTable,
)
from fraudstudy.evaluation.metrics import check_inputs

MIN_CURVE_POINTS = 2  # both endpoints


def downsample_indices(n: int, max_points: int) -> np.ndarray:
    """Return at most ``max_points`` evenly spaced, sorted, unique indices into ``range(n)``.

    Indices ``0`` and ``n - 1`` are always included.

    Args:
        n: Length of the array being downsampled (>= 1).
        max_points: Maximum number of indices (>= 2).

    Returns:
        Integer index array.
    """
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.round(np.linspace(0, n - 1, max_points)).astype(int))


def threshold_table(
    y_true: object, scores: object, max_points: int = MAX_CURVE_POINTS
) -> ThresholdTable:
    """Precision, recall and FPR at thresholds placed at score quantiles.

    Thresholds are the distinct values of ``quantile(scores, q)`` for ``max_points`` evenly
    spaced ``q`` in [0, 1], ascending. At threshold ``t`` rows with ``score >= t`` are flagged:
    ``precision = TP/(TP+FP)``, ``recall = TP/(TP+FN)``, ``fpr = FP/(FP+TN)`` (0 when undefined).

    Args:
        y_true: Binary labels (1 = fraud).
        scores: Risk scores in [0, 1].
        max_points: Maximum number of thresholds.

    Returns:
        A validated ``ThresholdTable``.

    Raises:
        ValueError: If the inputs are invalid or ``max_points`` is out of range.
    """
    _check_max_points(max_points)
    y, s = _check_unit_scores(y_true, scores)
    thresholds = np.unique(np.quantile(s, np.linspace(0.0, 1.0, max_points)))
    pos, neg = np.sort(s[y == 1]), np.sort(s[y == 0])
    tp = pos.size - np.searchsorted(pos, thresholds, side="left")
    fp = neg.size - np.searchsorted(neg, thresholds, side="left")
    flagged = tp + fp
    precision = np.divide(tp, flagged, out=np.zeros(thresholds.size), where=flagged > 0)
    return ThresholdTable(
        threshold=thresholds.tolist(),
        precision=precision.tolist(),
        recall=(tp / pos.size).tolist(),
        fpr=(fp / neg.size).tolist(),
    )


def make_curves(
    y_true: object,
    scores: object,
    *,
    model: ModelId,
    technique: TechniqueId,
    max_points: int = MAX_CURVE_POINTS,
) -> Curves:
    """Build the downsampled ROC curve, PR curve and threshold table for one cell.

    ROC points come from ``roc_curve`` (fpr and tpr ascending, from (0, 0) to (1, 1)). PR points
    come from ``precision_recall_curve``, reversed so recall ascends: the first point is
    sklearn's (recall 0, precision 1) anchor and the last is the lowest threshold.

    Args:
        y_true: Binary labels (1 = fraud).
        scores: Risk scores in [0, 1].
        model: Model id of the cell.
        technique: Technique id of the cell.
        max_points: Maximum points per curve (2..``MAX_CURVE_POINTS``).

    Returns:
        A validated ``Curves`` object.

    Raises:
        ValueError: If the inputs are invalid or ``max_points`` is out of range.
    """
    _check_max_points(max_points)
    y, s = _check_unit_scores(y_true, scores)
    fpr, tpr, _ = roc_curve(y, s)
    keep = downsample_indices(fpr.size, max_points)
    precision, recall, _ = precision_recall_curve(y, s)
    precision, recall = precision[::-1], recall[::-1]
    keep_pr = downsample_indices(recall.size, max_points)
    return Curves(
        model=model,
        technique=technique,
        roc=RocCurve(fpr=fpr[keep].tolist(), tpr=tpr[keep].tolist()),
        pr=PrCurve(recall=recall[keep_pr].tolist(), precision=precision[keep_pr].tolist()),
        threshold_table=threshold_table(y, s, max_points),
    )


def _check_max_points(max_points: int) -> None:
    """Raise ``ValueError`` unless ``max_points`` is in [2, MAX_CURVE_POINTS]."""
    if not MIN_CURVE_POINTS <= max_points <= MAX_CURVE_POINTS:
        raise ValueError(
            f"max_points must be between {MIN_CURVE_POINTS} and {MAX_CURVE_POINTS} "
            f"(the curves.json contract limit); got {max_points}."
        )


def _check_unit_scores(y_true: object, scores: object) -> tuple[np.ndarray, np.ndarray]:
    """Validate inputs and require scores in [0, 1] (thresholds are stored as probabilities)."""
    y, s = check_inputs(y_true, scores)
    if s.min() < 0.0 or s.max() > 1.0:
        raise ValueError(
            f"Curves need scores in [0, 1]; got range [{s.min()}, {s.max()}]. "
            "Pass predict_proba(...)[:, 1]."
        )
    return y, s
