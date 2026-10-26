"""Threshold selection on OUT-OF-FOLD training scores (master guide Part 6, "Choosing a threshold").

The threshold is chosen once, on out-of-fold scores, then frozen and applied unchanged to the
test set. Never pass test-set scores here.
"""

from typing import Literal

import numpy as np
from sklearn.metrics import precision_recall_curve

from fraudstudy.evaluation.metrics import F2_BETA, check_inputs

Objective = Literal["f1", "f2"]
OBJECTIVE_BETA: dict[str, float] = {"f1": 1.0, "f2": F2_BETA}


def fbeta_from_pr(precision: np.ndarray, recall: np.ndarray, beta: float) -> np.ndarray:
    """Element-wise F-beta from precision and recall arrays.

    ``F_beta = (1 + beta^2) P R / (beta^2 P + R)``, and 0 where ``P = R = 0``.

    Args:
        precision: Precision values.
        recall: Recall values (same shape).
        beta: Weight of recall relative to precision (1 = F1, 2 = F2).

    Returns:
        The F-beta values.
    """
    b2 = beta**2
    num = (1.0 + b2) * precision * recall
    den = b2 * precision + recall
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)


def best_threshold(y_true: object, oof_scores: object, objective: Objective) -> float:
    """Return the score cut-off that maximises F1 or F2 on out-of-fold scores.

    Every distinct score is a candidate (``precision_recall_curve`` without dropping points);
    candidate ``t`` predicts fraud for ``score >= t``. Ties in the objective go to the HIGHER
    threshold (fewer alarms for the same score).

    Args:
        y_true: Binary training labels (1 = fraud).
        oof_scores: Out-of-fold scores for the same rows.
        objective: ``"f1"`` or ``"f2"`` (``threshold.objectives`` in experiment.yaml).

    Returns:
        The chosen threshold (one of the observed scores).

    Raises:
        ValueError: If ``objective`` is not ``"f1"`` or ``"f2"``.
        ValueError: If the inputs are invalid.
    """
    if objective not in OBJECTIVE_BETA:
        raise ValueError(
            f"Unknown threshold objective '{objective}'; use one of {sorted(OBJECTIVE_BETA)}."
        )
    y, s = check_inputs(y_true, oof_scores)
    precision, recall, thresholds = precision_recall_curve(y, s, drop_intermediate=False)
    # The last (precision=1, recall=0) point has no threshold; thresholds are ascending.
    f = fbeta_from_pr(precision[:-1], recall[:-1], OBJECTIVE_BETA[objective])
    best = np.flatnonzero(f == f.max())
    return float(thresholds[best].max())
