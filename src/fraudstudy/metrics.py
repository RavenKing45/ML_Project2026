"""Metrics, threshold choice, curves, bootstrap intervals and calibration, from labels and scores.

experiment.py calls best_threshold() on out-of-fold training scores; final_test.py calls
compute_metrics(), make_curves() and bootstrap_ci() on test scores; export.py and the backend
call fit_calibrator() / apply_calibrator(). Every metric comes from sklearn.metrics. Reads and
writes no files.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

MAX_CURVE_POINTS = 101  # the most points per array that the curves.json contract allows


def compute_metrics(y_true: object, scores: object, threshold: float) -> dict:
    """Every metric at one threshold: a row is predicted fraud when score >= threshold.

    An undefined ratio (e.g. precision when nothing is flagged) is reported as 0, never NaN.
    pr_auc (average precision) and roc_auc ignore the threshold.
    """
    y = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = (int(n) for n in confusion_matrix(y, predicted, labels=[0, 1]).ravel())
    return {
        "precision": float(precision_score(y, predicted, zero_division=0.0)),
        "recall": float(recall_score(y, predicted, zero_division=0.0)),
        "f1": float(f1_score(y, predicted, zero_division=0.0)),
        "f2": float(fbeta_score(y, predicted, beta=2.0, zero_division=0.0)),
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "fnr": fn / (fn + tp) if fn + tp else 0.0,
        "mcc": float(matthews_corrcoef(y, predicted)),
        "accuracy": float(accuracy_score(y, predicted)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
    }


def best_threshold(y_true: object, oof_scores: object, objective: str) -> float:
    """The score cut-off that maximises F1 or F2 on OUT-OF-FOLD training scores.

    Never pass test scores here: the threshold is chosen once on training data, then frozen.
    Every distinct score is a candidate; ties go to the higher threshold (fewer alarms).
    F_beta = (1 + beta^2) P R / (beta^2 P + R), and 0 where P = R = 0.
    """
    betas = {"f1": 1.0, "f2": 2.0}
    if objective not in betas:
        raise ValueError(f"Unknown threshold objective '{objective}'; use one of {sorted(betas)}.")
    precision, recall, thresholds = precision_recall_curve(y_true, oof_scores)
    # The last point (precision 1, recall 0) has no threshold; thresholds are ascending.
    precision, recall = precision[:-1], recall[:-1]
    b2 = betas[objective] ** 2
    top, bottom = (1.0 + b2) * precision * recall, b2 * precision + recall
    f_beta = np.divide(top, bottom, out=np.zeros_like(top), where=bottom > 0)
    return float(thresholds[f_beta == f_beta.max()].max())


def even_indices(n: int) -> np.ndarray:
    """At most MAX_CURVE_POINTS evenly spaced indices into range(n), always keeping both ends."""
    if n <= MAX_CURVE_POINTS:
        return np.arange(n)
    return np.unique(np.round(np.linspace(0, n - 1, MAX_CURVE_POINTS)).astype(int))


def make_curves(y_true: object, scores: object, model: str, technique: str) -> dict:
    """ROC curve, PR curve and threshold table of one cell, each at most 101 points (curves.json).

    The curves are thinned by picking real points evenly by index (no interpolation). The PR
    curve is reversed so recall goes up. The threshold table (for the dashboard's slider) gives
    precision, recall and FPR at thresholds placed at score quantiles; a row is flagged when
    score >= threshold.
    """
    y = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    fpr, tpr, _ = roc_curve(y, scores)
    keep = even_indices(fpr.size)
    precision, recall, _ = precision_recall_curve(y, scores)
    precision, recall = precision[::-1], recall[::-1]
    keep_pr = even_indices(recall.size)

    table = {"threshold": [], "precision": [], "recall": [], "fpr": []}
    for threshold in np.unique(np.quantile(scores, np.linspace(0.0, 1.0, MAX_CURVE_POINTS))):
        tn, fp, fn, tp = confusion_matrix(y, scores >= threshold, labels=[0, 1]).ravel()
        table["threshold"].append(float(threshold))
        table["precision"].append(tp / (tp + fp) if tp + fp else 0.0)
        table["recall"].append(tp / (tp + fn))
        table["fpr"].append(fp / (fp + tn))
    return {
        "model": model,
        "technique": technique,
        "roc": {"fpr": fpr[keep].tolist(), "tpr": tpr[keep].tolist()},
        "pr": {"recall": recall[keep_pr].tolist(), "precision": precision[keep_pr].tolist()},
        "threshold_table": {key: [float(v) for v in values] for key, values in table.items()},
    }


def bootstrap_ci(y_true: object, scores: object, metric: object, n_resamples: int,
                 confidence: float, seed: int) -> tuple[float, float]:
    """Percentile confidence interval of `metric(y, scores)` under a stratified bootstrap.

    Args:
        y_true: Labels (1 = fraud).
        scores: Scores of the same rows.
        metric: A sklearn function such as average_precision_score.
        n_resamples: Number of resamples (bootstrap.n_resamples in experiment.yaml).
        confidence: Coverage, e.g. 0.95.
        seed: The experiment seed.

    Returns:
        (low, high): the (1 - confidence)/2 and (1 + confidence)/2 quantiles of the resamples.
    """
    y = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    fraud_rows, legit_rows = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    values = []
    # Stratified: each resample redraws the frauds from the frauds and the legit rows from the
    # legit rows, so every resample has the original class counts. Rows are drawn 32 resamples
    # at a time; keep that, or the same seed would give different intervals than before.
    for start in range(0, n_resamples, 32):
        block = min(32, n_resamples - start)
        fraud_draws = rng.integers(0, fraud_rows.size, size=(block, fraud_rows.size))
        legit_draws = rng.integers(0, legit_rows.size, size=(block, legit_rows.size))
        for fraud_pick, legit_pick in zip(fraud_draws, legit_draws, strict=True):
            rows = np.concatenate([fraud_rows[fraud_pick], legit_rows[legit_pick]])
            values.append(metric(y[rows], scores[rows]))
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail])
    return float(low), float(high)


def fit_calibrator(oof_scores: object, y: object) -> IsotonicRegression:
    """Fit an isotonic calibrator (scores -> fraud probability) on OUT-OF-FOLD training scores.

    Resampled models give inflated scores. The calibrator is a non-decreasing step function, so
    it never changes which transaction ranks above which; PR-AUC is reported on raw scores.
    """
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    return calibrator.fit(np.asarray(oof_scores, dtype=float), np.asarray(y).astype(int))


def apply_calibrator(calibrator: IsotonicRegression, scores: object) -> np.ndarray:
    """Map risk scores to calibrated fraud probabilities in [0, 1]."""
    return np.clip(calibrator.predict(np.asarray(scores, dtype=float).ravel()), 0.0, 1.0)
