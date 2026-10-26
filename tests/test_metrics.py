"""Tests for fraudstudy.metrics: metrics, thresholds, curves, bootstrap and calibration."""

import ast
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from fraudstudy.config import PROJECT_ROOT
from fraudstudy.contracts import Curves, Metrics
from fraudstudy.metrics import (
    MAX_CURVE_POINTS,
    apply_calibrator,
    best_threshold,
    bootstrap_ci,
    compute_metrics,
    even_indices,
    fit_calibrator,
    make_curves,
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


# --- metrics --------------------------------------------------------------------------------


def test_compute_metrics_matches_hand_computed_10_row_example() -> None:
    m = compute_metrics(Y10, S10, 0.5)  # 0.5 itself counts as fraud (score >= threshold)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 1, 5, 2)
    p, r = 2 / 3, 1 / 2
    assert m["precision"] == pytest.approx(p)
    assert m["recall"] == pytest.approx(r)
    assert m["f1"] == pytest.approx(2 * p * r / (p + r))  # 4/7
    assert m["f2"] == pytest.approx(5 * p * r / (4 * p + r))  # 10/19
    assert m["fpr"] == pytest.approx(1 / 6)
    assert m["fnr"] == pytest.approx(1 / 2)
    assert m["accuracy"] == pytest.approx(7 / 10)
    assert m["mcc"] == pytest.approx((2 * 5 - 1 * 2) / np.sqrt(3 * 4 * 6 * 7))
    assert list(m) == ["precision", "recall", "f1", "f2", "fpr", "fnr", "mcc", "accuracy", "tp",
                       "fp", "tn", "fn", "pr_auc", "roc_auc"]
    Metrics.model_validate(m)  # the plain dict has the contract shape


def test_always_zero_scorer_shows_the_accuracy_paradox() -> None:
    y = np.r_[np.zeros(N_LEGIT_REAL, int), np.ones(N_FRAUD_REAL, int)]
    m = compute_metrics(y, np.zeros(y.size), 0.5)
    assert round(m["accuracy"], 5) == 0.99827
    assert m["recall"] == 0.0 and m["precision"] == 0.0 and m["f1"] == 0.0 and m["f2"] == 0.0
    assert m["fnr"] == 1.0 and m["fpr"] == 0.0 and m["mcc"] == 0.0
    assert m["pr_auc"] == pytest.approx(N_FRAUD_REAL / y.size)
    assert m["roc_auc"] == pytest.approx(0.5)


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
        scores = {t: compute_metrics(y, s, t)[objective] for t in np.unique(s)}
        best = max(scores.values())
        expected = max(t for t, v in scores.items() if np.isclose(v, best, rtol=0, atol=1e-12))
        assert best_threshold(y, s, objective) == pytest.approx(expected)


def test_best_threshold_breaks_ties_by_higher_threshold() -> None:
    # Cuts 0.9 (P 1, R 0.5) and 0.1 (P 0.5, R 1) both give F1 = 2/3, the maximum.
    assert best_threshold([1, 0, 0, 1], [0.9, 0.6, 0.5, 0.1], "f1") == pytest.approx(0.9)


def test_best_threshold_rejects_unknown_objective() -> None:
    with pytest.raises(ValueError, match="Unknown threshold objective"):
        best_threshold(Y10, S10, "accuracy")


# --- curves ---------------------------------------------------------------------------------


def test_even_indices_keep_endpoints_and_limit() -> None:
    idx = even_indices(10_000)
    assert idx.size <= MAX_CURVE_POINTS and idx[0] == 0 and idx[-1] == 9_999
    assert np.all(np.diff(idx) > 0)
    assert even_indices(5).tolist() == [0, 1, 2, 3, 4]


def test_make_curves_downsamples_and_has_the_contract_shape() -> None:
    y, s = scored_data(20_000, 0.05)
    c = make_curves(y, s, model="lightgbm", technique="smote")
    Curves.model_validate(c)  # the plain dict has the curves.json contract shape
    for arr in (c["roc"]["fpr"], c["pr"]["recall"], c["threshold_table"]["threshold"]):
        assert 2 <= len(arr) <= MAX_CURVE_POINTS
    roc, pr = c["roc"], c["pr"]
    assert (roc["fpr"][0], roc["tpr"][0], roc["fpr"][-1], roc["tpr"][-1]) == (0.0, 0.0, 1.0, 1.0)
    assert (pr["recall"][0], pr["precision"][0], pr["recall"][-1]) == (0.0, 1.0, 1.0)
    assert np.all(np.diff(roc["fpr"]) >= 0) and np.all(np.diff(pr["recall"]) >= 0)


def test_threshold_table_matches_compute_metrics() -> None:
    y, s = scored_data(3_000, 0.05)
    table = make_curves(y, s, model="logreg", technique="baseline")["threshold_table"]
    assert len(table["threshold"]) <= MAX_CURVE_POINTS
    assert table["threshold"] == sorted(table["threshold"])
    assert table["threshold"][0] == pytest.approx(s.min()) and table["threshold"][-1] == s.max()
    for i in (0, len(table["threshold"]) // 2, len(table["threshold"]) - 1):
        m = compute_metrics(y, s, table["threshold"][i])
        assert table["precision"][i] == pytest.approx(m["precision"])
        assert table["recall"][i] == pytest.approx(m["recall"])
        assert table["fpr"][i] == pytest.approx(m["fpr"])


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

    bootstrap_ci(y, s, record, 50, 0.95, SEED)
    assert seen == [(int(y.sum()), y.size)] * 50  # every resample keeps the class counts


# --- calibration ----------------------------------------------------------------------------


def test_isotonic_calibration_maps_to_probabilities_and_keeps_order() -> None:
    y, s = scored_data(5_000, 0.05)
    cal = fit_calibrator(s, y)
    p = apply_calibrator(cal, s)
    assert p.min() >= 0.0 and p.max() <= 1.0
    assert np.all(np.diff(p[np.argsort(s)]) >= 0)  # never reverses the order
    assert p.mean() == pytest.approx(y.mean(), abs=1e-9)  # isotonic preserves the mean
    clipped = apply_calibrator(cal, [-1.0, 2.0])
    assert clipped.tolist() == [p.min(), p.max()]  # out-of-range scores are clipped


# --- independence ---------------------------------------------------------------------------


def test_metrics_do_not_import_modelling_code() -> None:
    forbidden = ("fraudstudy.pipeline", "fraudstudy.sampling", "fraudstudy.preprocess", "imblearn")
    for path in [Path(PROJECT_ROOT, "src", "fraudstudy", "metrics.py")]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            else:
                continue
            assert not any(n.startswith(forbidden) for n in names), f"{path.name} imports {names}"
