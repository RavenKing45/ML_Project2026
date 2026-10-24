"""Stratified bootstrap confidence intervals on a fixed set of scores (master guide Part 6).

Each resample draws ``n_fraud`` fraud rows from the fraud rows and ``n_legit`` legitimate rows
from the legitimate rows, with replacement, so every resample has the original class counts
(and always contains frauds). The interval is the percentile interval: the
``(1 - confidence)/2`` and ``(1 + confidence)/2`` quantiles of the resampled metric.

For average precision and ROC-AUC (the metrics the study reports) the resamples are computed in
vectorised blocks: the scores are sorted once, and a resample is just a count of how often each
row was drawn, so both areas are cumulative sums over the sorted rows. Any other ``metric_fn``
is called once per resample. Both paths consume the random generator identically, so the same
seed gives the same interval either way.
"""

from collections.abc import Callable, Iterator

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from fraudstudy.errors import DataValidationError
from fraudstudy.evaluation.metrics import check_inputs

MetricFn = Callable[[np.ndarray, np.ndarray], float]
BLOCK_SIZE = 32  # resamples per vectorised block (memory ~ BLOCK_SIZE x n_rows floats)


def bootstrap_ci(
    y_true: object,
    scores: object,
    metric_fn: MetricFn,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    """Percentile confidence interval of ``metric_fn`` under a stratified bootstrap.

    Args:
        y_true: Binary labels (1 = fraud).
        scores: Risk scores.
        metric_fn: ``metric_fn(y, scores) -> float``, e.g. ``average_precision_score``.
        n_resamples: Number of bootstrap resamples (``bootstrap.n_resamples`` in config).
        confidence: Interval coverage in (0, 1) (``bootstrap.confidence`` in config).
        seed: Random seed (the experiment ``seed``).

    Returns:
        ``(low, high)``.

    Raises:
        DataValidationError: If the inputs are invalid, ``n_resamples < 1`` or ``confidence``
            is not in (0, 1).
    """
    if not 0.0 < confidence < 1.0:
        raise DataValidationError(f"confidence must be in (0, 1); got {confidence}.")
    stats = bootstrap_distribution(y_true, scores, metric_fn, n_resamples, seed)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(stats, [tail, 1.0 - tail])
    return float(low), float(high)


def bootstrap_distribution(
    y_true: object, scores: object, metric_fn: MetricFn, n_resamples: int, seed: int
) -> np.ndarray:
    """Return ``metric_fn`` on each of ``n_resamples`` stratified bootstrap resamples.

    Args:
        y_true: Binary labels (1 = fraud).
        scores: Risk scores.
        metric_fn: ``metric_fn(y, scores) -> float``.
        n_resamples: Number of resamples (>= 1).
        seed: Random seed.

    Returns:
        Array of shape ``(n_resamples,)``.

    Raises:
        DataValidationError: If the inputs are invalid or ``n_resamples < 1``.
    """
    y, s = check_inputs(y_true, scores)
    if n_resamples < 1:
        raise DataValidationError(f"n_resamples must be at least 1; got {n_resamples}.")
    fraud_idx, legit_idx = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    draws = _stratified_draws(fraud_idx, legit_idx, n_resamples, seed)
    if metric_fn is average_precision_score or metric_fn is roc_auc_score:
        return _vectorised(y, s, draws, area=metric_fn)
    y_boot = np.concatenate([np.ones(fraud_idx.size, int), np.zeros(legit_idx.size, int)])
    out = [
        float(metric_fn(y_boot, s[np.concatenate([f, g])]))
        for f_block, g_block in draws
        for f, g in zip(f_block, g_block, strict=True)
    ]
    return np.asarray(out)


def _stratified_draws(
    fraud_idx: np.ndarray, legit_idx: np.ndarray, n_resamples: int, seed: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield blocks of row indices ``(fraud rows, legit rows)``, each of shape (block, n_class)."""
    rng = np.random.default_rng(seed)
    for start in range(0, n_resamples, BLOCK_SIZE):
        block = min(BLOCK_SIZE, n_resamples - start)
        f = fraud_idx[rng.integers(0, fraud_idx.size, size=(block, fraud_idx.size))]
        g = legit_idx[rng.integers(0, legit_idx.size, size=(block, legit_idx.size))]
        yield f, g


def _vectorised(
    y: np.ndarray,
    s: np.ndarray,
    draws: Iterator[tuple[np.ndarray, np.ndarray]],
    area: MetricFn,
) -> np.ndarray:
    """Average precision or ROC-AUC for every resample, from draw counts over sorted scores.

    With rows sorted by descending score and grouped by tied score, let ``TP_k`` / ``FP_k`` be
    the drawn fraud / legit counts at or above group ``k`` (``TP_0 = FP_0 = 0``). Then
    ``AP = sum_k (TP_k - TP_{k-1}) / n_fraud * TP_k / (TP_k + FP_k)`` and
    ``ROC-AUC = sum_k (FPR_k - FPR_{k-1}) (TPR_k + TPR_{k-1}) / 2`` (trapezoid rule), exactly
    as scikit-learn computes them on the resampled rows.
    """
    n = y.size
    order = np.argsort(-s, kind="stable")
    rank = np.empty(n, dtype=np.intp)
    rank[order] = np.arange(n)
    sorted_s = s[order]
    group_end = np.flatnonzero(np.append(sorted_s[1:] != sorted_s[:-1], True))
    n_fraud, n_legit = int(y.sum()), int(n - y.sum())
    results = []
    for f, g in draws:
        tp = _cumulative_counts(rank[f], n)[:, group_end]
        fp = _cumulative_counts(rank[g], n)[:, group_end]
        if area is average_precision_score:
            flagged = tp + fp
            precision = np.divide(tp, flagged, out=np.zeros_like(tp), where=flagged > 0)
            gain = np.diff(tp, axis=1, prepend=0.0)
            results.append((gain * precision).sum(axis=1) / n_fraud)
        else:
            tpr = np.pad(tp / n_fraud, ((0, 0), (1, 0)))
            fpr = np.pad(fp / n_legit, ((0, 0), (1, 0)))
            results.append((np.diff(fpr, axis=1) * (tpr[:, 1:] + tpr[:, :-1]) / 2.0).sum(axis=1))
    return np.concatenate(results)


def _cumulative_counts(positions: np.ndarray, n: int) -> np.ndarray:
    """Per row of ``positions`` (sorted-order indices), cumulative draw counts over ``range(n)``."""
    block = positions.shape[0]
    flat = (positions + n * np.arange(block)[:, None]).ravel()
    counts = np.bincount(flat, minlength=block * n).reshape(block, n).astype(float)
    return np.cumsum(counts, axis=1)
