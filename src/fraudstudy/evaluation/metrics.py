"""Classification metrics at one threshold plus threshold-free areas (master guide Part 2).

A row is predicted FRAUD exactly when ``score >= threshold`` (the same rule as
``PredictResponse``). Every ratio uses ``zero_division=0``: an undefined ratio such as
precision with nothing flagged is reported as 0, never NaN.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fraudstudy.contracts import Metrics

F2_BETA = 2.0
ZERO_DIVISION = 0.0


def check_inputs(y_true: object, scores: object) -> tuple[np.ndarray, np.ndarray]:
    """Validate labels and scores and return them as 1-D numpy arrays.

    Args:
        y_true: Binary labels (1 = fraud, 0 = legitimate).
        scores: Risk scores, one per label; higher means more likely fraud.

    Returns:
        ``(y, s)``: integer labels and float scores.

    Raises:
        ValueError: If the inputs are empty, of different lengths, not 1-D, the labels
            are not 0/1, a class is missing, or a score is NaN/infinite.
    """
    y = np.asarray(y_true).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    if y.size == 0 or y.size != s.size:
        raise ValueError(
            f"Need one score per label and at least one row; got {y.size} labels and "
            f"{s.size} scores."
        )
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Labels must be 0 (legitimate) or 1 (fraud). Check y_true.")
    y = y.astype(int)
    n_fraud = int(y.sum())
    if n_fraud == 0 or n_fraud == y.size:
        raise ValueError(
            f"Need both classes to evaluate; got {n_fraud} frauds and {y.size - n_fraud} "
            "legitimate rows. Use a stratified split."
        )
    if not np.isfinite(s).all():
        raise ValueError("Scores contain NaN or infinite values. Check the model output.")
    return y, s


def compute_metrics(y_true: object, scores: object, threshold: float) -> Metrics:
    """Compute every ``Metrics`` field for one threshold.

    With TP/FP/TN/FN from ``predicted = score >= threshold``:
    precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2PR/(P+R), F2 = 5PR/(4P+R),
    FPR = FP/(FP+TN), FNR = FN/(FN+TP), accuracy = (TP+TN)/n,
    MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)).
    ``pr_auc`` is average precision ``sum_k (R_k - R_{k-1}) P_k`` and ``roc_auc`` the area under
    the ROC curve; both ignore the threshold.

    Args:
        y_true: Binary labels (1 = fraud).
        scores: Risk scores.
        threshold: Cut-off; scores at or above it are predicted fraud.

    Returns:
        A validated ``Metrics`` object.

    Raises:
        ValueError: If the inputs are invalid (see :func:`check_inputs`) or the
            threshold is not finite.
    """
    y, s = check_inputs(y_true, scores)
    if not np.isfinite(threshold):
        raise ValueError(f"Threshold must be a finite number; got {threshold}.")
    pred = (s >= threshold).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    return Metrics(
        precision=float(precision_score(y, pred, zero_division=ZERO_DIVISION)),
        recall=float(recall_score(y, pred, zero_division=ZERO_DIVISION)),
        f1=float(f1_score(y, pred, zero_division=ZERO_DIVISION)),
        f2=float(fbeta_score(y, pred, beta=F2_BETA, zero_division=ZERO_DIVISION)),
        fpr=_ratio(fp, fp + tn),
        fnr=_ratio(fn, fn + tp),
        mcc=float(matthews_corrcoef(y, pred)),
        accuracy=float(accuracy_score(y, pred)),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        pr_auc=float(average_precision_score(y, s)),
        roc_auc=float(roc_auc_score(y, s)),
    )


def _ratio(numerator: int, denominator: int) -> float:
    """Return ``numerator / denominator``, or ``ZERO_DIVISION`` when the denominator is 0."""
    return numerator / denominator if denominator else ZERO_DIVISION
