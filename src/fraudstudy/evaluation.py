"""The one-time test-set evaluation and the significance tests: numbers only.

evaluate_test_set() is the ONLY code that touches the test rows: it scores every finished cell
once, behind a guard file. significance_tests() compares the techniques on the repeated-CV fold
scores. Reads artifacts/metrics/cells/*.json and artifacts/models/*.joblib; writes
test_results.json and stats.json in artifacts/metrics/ (artifacts/sample/metrics/ for the fake
sample). notebooks/03_results.ipynb reads these files and draws the tables and figures.

Run it: python -m fraudstudy.evaluation [--source sample] [--test [--rerun-reason "..."]] [--stats]
"""

import argparse
import json
from datetime import date

import joblib
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from statsmodels.stats.multitest import multipletests

from fraudstudy.data import SEED, load_data, remove_duplicates, split_data
from fraudstudy.experiments import MODELS, TECHNIQUES, artifacts_dir

# ---- SETTINGS ------------------------------------------------------------------------------------
N_BOOTSTRAP = 1000  # resamples for the PR-AUC confidence interval
ALPHA = 0.05  # significance level, and 1 - ALPHA = 95% confidence intervals
GUARD_FILE = "TEST_SET_OPENED.json"  # written in artifacts/metrics/ the first time the test is used
CURVE_POINTS = 101  # at most this many points per saved PR or ROC curve
# --------------------------------------------------------------------------------------------------


def compute_metrics(y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict:
    """Every test metric; a payment counts as flagged when its score >= threshold.

    PR-AUC (average precision) and ROC-AUC use the scores and ignore the threshold. A ratio
    with nothing to divide (e.g. precision when nothing is flagged) is reported as 0.
    """
    flagged = (scores >= threshold).astype(int)
    tn, fp, fn, tp = (int(n) for n in confusion_matrix(y_true, flagged, labels=[0, 1]).ravel())
    return {
        "precision": float(precision_score(y_true, flagged, zero_division=0.0)),
        "recall": float(recall_score(y_true, flagged, zero_division=0.0)),
        "f1": float(f1_score(y_true, flagged, zero_division=0.0)),
        "f2": float(fbeta_score(y_true, flagged, beta=2.0, zero_division=0.0)),
        "fpr": fp / (fp + tn) if fp + tn else 0.0,  # share of legit payments flagged
        "fnr": fn / (fn + tp) if fn + tp else 0.0,  # share of frauds missed
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, flagged)),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def bootstrap_ci(y_true: pd.Series, scores: np.ndarray) -> list[float]:
    """Stratified bootstrap 95% confidence interval of PR-AUC: [low, high].

    Each resample redraws, with replacement, as many frauds from the frauds and as many legit
    rows from the legit rows as the test set has, and computes PR-AUC on it. The interval is
    the middle 95% of the N_BOOTSTRAP values.
    """
    y = np.asarray(y_true)
    fraud_rows, legit_rows = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(SEED)
    values = []
    # Rows are drawn for 32 resamples at a time, as the earlier version of this code did; drawing
    # them one by one would use the random numbers in another order and move the interval.
    for start in range(0, N_BOOTSTRAP, 32):
        block = min(32, N_BOOTSTRAP - start)
        fraud_draws = rng.integers(0, len(fraud_rows), size=(block, len(fraud_rows)))
        legit_draws = rng.integers(0, len(legit_rows), size=(block, len(legit_rows)))
        for fraud_pick, legit_pick in zip(fraud_draws, legit_draws, strict=True):
            rows = np.concatenate([fraud_rows[fraud_pick], legit_rows[legit_pick]])
            values.append(average_precision_score(y[rows], scores[rows]))
    low, high = np.quantile(values, [ALPHA / 2, 1 - ALPHA / 2])
    return [float(low), float(high)]


def thin(points: np.ndarray) -> list[float]:
    """At most CURVE_POINTS evenly spaced points of a curve, always keeping both ends."""
    keep = np.unique(np.round(np.linspace(0, len(points) - 1, CURVE_POINTS)).astype(int))
    return points[keep].tolist()


def load_cells(source: str) -> dict:
    """{cell name: the results dict experiments.py saved}, for every finished cell."""
    cells_dir = artifacts_dir(source) / "metrics" / "cells"
    cells = {path.stem: json.loads(path.read_text()) for path in sorted(cells_dir.glob("*.json"))}
    if not cells:
        raise FileNotFoundError(f"No finished cells in {cells_dir}; run "
                                f"python -m fraudstudy.experiments --source {source} first.")
    return cells


def open_guard(source: str, cells: list[str], rerun_reason: str | None) -> None:
    """Record this opening of the test set; refuse a second one that gives no reason."""
    guard = artifacts_dir(source) / "metrics" / GUARD_FILE
    if guard.is_file() and not rerun_reason:
        raise RuntimeError(
            f"The test set was already opened (see {guard}). Using it again biases the final "
            'result; if you really must, add --rerun-reason "<why>". Never change a tuned value '
            "because of test results."
        )
    log = json.loads(guard.read_text()) if guard.is_file() else {"openings": []}
    log["openings"].append({"date": date.today().isoformat(), "reason": rerun_reason,
                            "cells": cells})
    guard.parent.mkdir(parents=True, exist_ok=True)
    guard.write_text(json.dumps(log, indent=2), encoding="utf-8")


def evaluate_test_set(rerun_reason: str | None = None, source: str = "raw") -> dict:
    """Score every finished cell on the locked test set, once; write test_results.json.
    The ONLY function that reads X_test and y_test.

    Each cell is judged at the F1 threshold it chose on out-of-fold training scores (read from
    its cell file, never recomputed here).
    """
    cells = load_cells(source)
    open_guard(source, sorted(cells), rerun_reason)

    df, _, _ = remove_duplicates(load_data(source))
    _, X_test, _, y_test = split_data(df)
    print(f"test set: {len(y_test):,} rows, {int(y_test.sum())} frauds")
    results = {}
    for name, cell in cells.items():
        pipeline = joblib.load(artifacts_dir(source) / "models" / f"{name}.joblib")
        scores = pipeline.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, scores)
        fpr, tpr, _ = roc_curve(y_test, scores)
        results[name] = {
            "model": cell["model"], "technique": cell["technique"],
            "threshold_f1": cell["threshold_f1"], "cv_pr_auc_mean": cell["cv_pr_auc_mean"],
            "test": compute_metrics(y_test, scores, cell["threshold_f1"]),
            "pr_auc_ci": bootstrap_ci(y_test, scores),
            # reversed so recall goes up from 0 to 1
            "pr_curve": {"recall": thin(recall[::-1]), "precision": thin(precision[::-1])},
            "roc_curve": {"fpr": thin(fpr), "tpr": thin(tpr)},
        }
        low, high = results[name]["pr_auc_ci"]
        print(f"{name:<28} test PR-AUC {results[name]['test']['pr_auc']:.4f} "
              f"[{low:.3f}, {high:.3f}]")

    metrics_dir = artifacts_dir(source) / "metrics"
    (metrics_dir / "test_results.json").write_text(json.dumps(results, indent=2))
    return results


def significance_tests(source: str = "raw") -> dict:
    """Per model: Wilcoxon signed-rank test of each technique against the baseline, and a
    Friedman test across the techniques, on the repeated-CV fold PR-AUCs. Writes stats.json.

    Every cell was scored on the same folds, so fold i of one technique is paired with fold i
    of the baseline. Holm's correction (statsmodels) adjusts the 5 p-values of each model for
    testing 5 times. Fold scores share training rows, so the p-values are only approximate.
    """
    cells = load_cells(source)
    stats = {"alpha": ALPHA, "note": "Fold scores are not independent; p-values are approximate.",
             "models": {}}
    for model in MODELS:
        folds = {t: np.array(cells[f"{model}__{t}"]["fold_pr_auc"])
                 for t in TECHNIQUES if f"{model}__{t}" in cells}
        if "baseline" not in folds or len(folds) < 2:
            continue
        others = [t for t in folds if t != "baseline"]
        tests = []
        for technique in others:
            difference = folds[technique] - folds["baseline"]
            difference[np.abs(difference) <= 1e-12] = 0.0  # rounding noise counts as a tie
            if np.all(difference == 0):
                statistic, p_value = 0.0, 1.0  # identical on every fold: no evidence at all
            else:
                statistic, p_value = wilcoxon(difference, zero_method="wilcox")
            tests.append({"technique": technique, "median_difference": float(np.median(difference)),
                          "statistic": float(statistic), "p_value": float(p_value)})
        _, p_holm, _, _ = multipletests([t["p_value"] for t in tests], method="holm")
        for test, p in zip(tests, p_holm, strict=True):
            test["p_value_holm"] = float(p)
            test["significant"] = bool(p < ALPHA)
        stats["models"][model] = {"wilcoxon_vs_baseline": tests}
        if len(folds) >= 3:  # the Friedman test needs at least 3 techniques
            statistic, p_value = friedmanchisquare(*folds.values())
            stats["models"][model]["friedman"] = {"techniques": list(folds),
                                                  "statistic": float(statistic),
                                                  "p_value": float(p_value)}
    path = artifacts_dir(source) / "metrics" / "stats.json"
    path.write_text(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    """Run the steps asked for on the command line, in the order test, stats."""
    parser = argparse.ArgumentParser(description="Test-set evaluation and statistics.")
    parser.add_argument("--source", choices=["raw", "sample"], default="raw")
    parser.add_argument("--test", action="store_true", help="the one-time test-set evaluation")
    parser.add_argument("--rerun-reason", help="why the test set must be opened again")
    parser.add_argument("--stats", action="store_true", help="significance tests")
    args = parser.parse_args()
    if not (args.test or args.stats):
        parser.error("choose at least one of --test, --stats")

    if args.test:
        try:
            evaluate_test_set(args.rerun_reason, args.source)
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            raise SystemExit(f"error: {error}") from None  # one clear line, no traceback
    if args.stats:
        for model, result in significance_tests(args.source)["models"].items():
            for test in result["wilcoxon_vs_baseline"]:
                print(f"{model:<14} {test['technique']:<13} vs baseline: p = {test['p_value']:.3g}"
                      f", Holm p = {test['p_value_holm']:.3g}")


if __name__ == "__main__":
    main()
