"""Tests for fraudstudy.evaluation (metrics, thresholds, curves, bootstrap, calibration).

All data here is small and generated in the test; no dataset file is needed.
"""

import ast
import time
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from fraudstudy.config import PROJECT_ROOT
from fraudstudy.contracts import MAX_CURVE_POINTS, Curves, Metrics, ThresholdTable
from fraudstudy.evaluation import (
    apply_calibrator,
    best_threshold,
    bootstrap_ci,
    bootstrap_distribution,
    check_inputs,
    compute_metrics,
    downsample_indices,
    fbeta_from_pr,
    fit_isotonic,
    make_curves,
    threshold_table,
)

SEED = 42
N_LEGIT_REAL, N_FRAUD_REAL = 284_315, 492  # ULB dataset class counts
# 10-row example: at threshold 0.5 -> TP 2, FP 1, TN 5, FN 2.
Y10 = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
S10 = np.array([0.9, 0.5, 0.4, 0.2, 0.7, 0.3, 0.1, 0.1, 0.05, 0.0])


def scored_data(n: int, fraud_rate: float, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Labels and overlapping scores in [0, 1] (frauds score higher on average)."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    y[:2] = (1, 0)  # both classes always present
    s = 1.0 / (1.0 + np.exp(-(rng.normal(size=n) + 2.0 * y - 2.0)))
    return y, s


def ci_from_loop(y: np.ndarray, s: np.ndarray, fn, n: int, conf: float) -> tuple[float, float]:
    """Force the per-resample loop path by wrapping a metric in a new function."""
    return bootstrap_ci(y, s, lambda a, b: fn(a, b), n, conf, SEED)


# --- metrics --------------------------------------------------------------------------------


def test_compute_metrics_matches_hand_computed_10_row_example() -> None:
    m = compute_metrics(Y10, S10, 0.5)  # 0.5 itself counts as fraud (score >= threshold)
    assert (m.tp, m.fp, m.tn, m.fn) == (2, 1, 5, 2)
    p, r = 2 / 3, 1 / 2
    assert m.precision == pytest.approx(p)
    assert m.recall == pytest.approx(r)
    assert m.f1 == pytest.approx(2 * p * r / (p + r))  # 4/7
    assert m.f2 == pytest.approx(5 * p * r / (4 * p + r))  # 10/19
    assert m.fpr == pytest.approx(1 / 6)
    assert m.fnr == pytest.approx(1 / 2)
    assert m.accuracy == pytest.approx(7 / 10)
    assert m.mcc == pytest.approx((2 * 5 - 1 * 2) / np.sqrt(3 * 4 * 6 * 7))
    assert isinstance(m, Metrics)


def test_always_zero_scorer_shows_the_accuracy_paradox() -> None:
    y = np.r_[np.zeros(N_LEGIT_REAL, int), np.ones(N_FRAUD_REAL, int)]
    m = compute_metrics(y, np.zeros(y.size), 0.5)
    assert round(m.accuracy, 5) == 0.99827
    assert m.recall == 0.0 and m.precision == 0.0 and m.f1 == 0.0 and m.f2 == 0.0
    assert m.fnr == 1.0 and m.fpr == 0.0 and m.mcc == 0.0
    assert m.pr_auc == pytest.approx(N_FRAUD_REAL / y.size)
    assert m.roc_auc == pytest.approx(0.5)


def test_check_inputs_rejects_bad_data() -> None:
    with pytest.raises(ValueError, match="one score per label"):
        check_inputs([0, 1], [0.1])
    with pytest.raises(ValueError, match="0 \\(legitimate\\) or 1"):
        check_inputs([0, 2], [0.1, 0.2])
    with pytest.raises(ValueError, match="both classes"):
        check_inputs([0, 0], [0.1, 0.2])
    with pytest.raises(ValueError, match="NaN"):
        check_inputs([0, 1], [0.1, np.nan])
    with pytest.raises(ValueError, match="finite"):
        compute_metrics(Y10, S10, float("nan"))


# --- thresholds -----------------------------------------------------------------------------


def test_best_threshold_returns_f1_maximising_cut_on_toy_example() -> None:
    y = [0, 0, 1, 0, 1, 1]
    s = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8]
    # F1 per cut: 0.8->0.5, 0.6->0.8, 0.4->0.667, 0.3->0.857, 0.2->0.75, 0.1->0.667
    assert best_threshold(y, s, "f1") == pytest.approx(0.3)
    assert best_threshold(y, s, "f2") == pytest.approx(0.3)


def test_best_threshold_agrees_with_brute_force() -> None:
    y, s = scored_data(400, 0.1)
    s = np.round(s, 2)  # creates ties between rows
    for objective in ("f1", "f2"):
        scores = {t: getattr(compute_metrics(y, s, t), objective) for t in np.unique(s)}
        best = max(scores.values())
        expected = max(t for t, v in scores.items() if np.isclose(v, best, rtol=0, atol=1e-12))
        assert best_threshold(y, s, objective) == pytest.approx(expected)


def test_best_threshold_breaks_ties_by_higher_threshold() -> None:
    # Cuts 0.9 (P 1, R 0.5) and 0.1 (P 0.5, R 1) both give F1 = 2/3, the maximum.
    assert best_threshold([1, 0, 0, 1], [0.9, 0.6, 0.5, 0.1], "f1") == pytest.approx(0.9)


def test_best_threshold_rejects_unknown_objective() -> None:
    with pytest.raises(ValueError, match="Unknown threshold objective"):
        best_threshold(Y10, S10, "accuracy")  # type: ignore[arg-type]


def test_fbeta_from_pr_handles_zeros() -> None:
    f = fbeta_from_pr(np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 0.5]), 2.0)
    assert f == pytest.approx([0.0, 0.5, 5 * 0.5 / (4 + 0.5)])


# --- curves ---------------------------------------------------------------------------------


def test_downsample_indices_keeps_endpoints_and_limit() -> None:
    idx = downsample_indices(10_000, MAX_CURVE_POINTS)
    assert idx.size <= MAX_CURVE_POINTS and idx[0] == 0 and idx[-1] == 9_999
    assert np.all(np.diff(idx) > 0)
    assert downsample_indices(5, MAX_CURVE_POINTS).tolist() == [0, 1, 2, 3, 4]


def test_make_curves_downsamples_and_validates_against_contract() -> None:
    y, s = scored_data(20_000, 0.05)
    c = make_curves(y, s, model="lightgbm", technique="smote")
    for arr in (c.roc.fpr, c.pr.recall, c.threshold_table.threshold):
        assert 2 <= len(arr) <= MAX_CURVE_POINTS
    assert (c.roc.fpr[0], c.roc.tpr[0], c.roc.fpr[-1], c.roc.tpr[-1]) == (0.0, 0.0, 1.0, 1.0)
    assert (c.pr.recall[0], c.pr.precision[0], c.pr.recall[-1]) == (0.0, 1.0, 1.0)
    assert np.all(np.diff(c.roc.fpr) >= 0) and np.all(np.diff(c.pr.recall) >= 0)
    assert Curves.model_validate(c.model_dump(mode="json")) == c


def test_make_curves_rejects_bad_max_points_and_scores() -> None:
    with pytest.raises(ValueError, match="max_points"):
        make_curves(Y10, S10, model="logreg", technique="baseline", max_points=MAX_CURVE_POINTS + 1)
    with pytest.raises(ValueError, match="scores in \\[0, 1\\]"):
        make_curves(Y10, S10 * 2, model="logreg", technique="baseline")


def test_threshold_table_matches_compute_metrics() -> None:
    y, s = scored_data(3_000, 0.05)
    table = threshold_table(y, s)
    assert isinstance(table, ThresholdTable) and len(table.threshold) <= MAX_CURVE_POINTS
    assert table.threshold == sorted(table.threshold)
    assert table.threshold[0] == pytest.approx(s.min()) and table.threshold[-1] == s.max()
    for i in (0, len(table.threshold) // 2, len(table.threshold) - 1):
        m = compute_metrics(y, s, table.threshold[i])
        assert table.precision[i] == pytest.approx(m.precision)
        assert table.recall[i] == pytest.approx(m.recall)
        assert table.fpr[i] == pytest.approx(m.fpr)


# --- bootstrap ------------------------------------------------------------------------------


def test_bootstrap_ci_contains_point_estimate_and_is_deterministic() -> None:
    y, s = scored_data(5_000, 0.02)
    for fn in (average_precision_score, roc_auc_score):
        low, high = bootstrap_ci(y, s, fn, 300, 0.95, SEED)
        assert low <= fn(y, s) <= high and low < high
        assert bootstrap_ci(y, s, fn, 300, 0.95, SEED) == (low, high)
    assert bootstrap_ci(y, s, average_precision_score, 300, 0.95, SEED + 1) != (low, high)


def test_bootstrap_ci_narrows_with_more_data() -> None:
    small = scored_data(2_000, 0.02)
    large = scored_data(40_000, 0.02)
    lo_s, hi_s = bootstrap_ci(*small, average_precision_score, 200, 0.95, SEED)
    lo_l, hi_l = bootstrap_ci(*large, average_precision_score, 200, 0.95, SEED)
    assert hi_l - lo_l < hi_s - lo_s


def test_bootstrap_is_stratified() -> None:
    y, s = scored_data(1_000, 0.01)
    seen: list[tuple[int, int]] = []

    def record(yb: np.ndarray, sb: np.ndarray) -> float:
        seen.append((int(yb.sum()), yb.size))
        return float(sb.mean())

    bootstrap_distribution(y, s, record, 50, SEED)
    assert seen == [(int(y.sum()), y.size)] * 50


@pytest.mark.parametrize("fn", [average_precision_score, roc_auc_score])
def test_vectorised_bootstrap_equals_per_resample_loop(fn) -> None:
    y, s = scored_data(3_000, 0.03)
    s = np.round(s, 2)  # ties exercise the grouped computation
    fast = bootstrap_distribution(y, s, fn, 70, SEED)
    slow = bootstrap_distribution(y, s, lambda a, b: fn(a, b), 70, SEED)
    np.testing.assert_allclose(fast, slow, rtol=1e-10, atol=1e-12)
    assert ci_from_loop(y, s, fn, 70, 0.9) == pytest.approx(bootstrap_ci(y, s, fn, 70, 0.9, SEED))


def test_bootstrap_1000_resamples_on_85k_rows_is_fast() -> None:
    y, s = scored_data(85_000, 0.0017)
    start = time.perf_counter()
    bootstrap_ci(y, s, average_precision_score, 1000, 0.95, SEED)
    assert time.perf_counter() - start < 20.0


def test_bootstrap_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ci(Y10, S10, average_precision_score, 10, 1.0, SEED)
    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap_ci(Y10, S10, average_precision_score, 0, 0.95, SEED)


# --- calibration ----------------------------------------------------------------------------


def test_isotonic_calibration_maps_to_probabilities_and_keeps_order() -> None:
    y, s = scored_data(5_000, 0.05)
    cal = fit_isotonic(s, y)
    p = apply_calibrator(cal, s)
    assert p.min() >= 0.0 and p.max() <= 1.0
    assert np.all(np.diff(p[np.argsort(s)]) >= 0)  # never reverses the order
    assert p.mean() == pytest.approx(y.mean(), abs=1e-9)  # isotonic preserves the mean
    clipped = apply_calibrator(cal, [-1.0, 2.0])
    assert clipped.tolist() == [p.min(), p.max()]  # out-of-range scores are clipped


# --- independence ---------------------------------------------------------------------------


def test_evaluation_does_not_import_modelling_code() -> None:
    forbidden = ("fraudstudy.pipeline", "fraudstudy.sampling", "fraudstudy.preprocess", "imblearn")
    for path in Path(PROJECT_ROOT, "src", "fraudstudy", "evaluation").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            else:
                continue
            assert not any(n.startswith(forbidden) for n in names), f"{path.name} imports {names}"
