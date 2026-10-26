"""The one-time test-set evaluation, the significance tests and the report figures.

evaluate_test_set() is the ONLY code that touches the test rows: it scores every finished cell
once, behind a guard file, and also picks 5 example test payments for the app.
significance_tests() compares the techniques on the repeated-CV fold scores, and make_figures()
draws the report figures. Reads artifacts/metrics/cells/*.json and artifacts/models/*.joblib;
writes test_results.json, presets.json and stats.json in artifacts/metrics/ and PNGs in
artifacts/figures/ (all under artifacts/sample/ for the fake sample).

Run it: python -m fraudstudy.evaluation [--source sample] --test --default-cell <cell>
    [--rerun-reason "..."] | --stats | --figures
"""

import argparse
import json
from datetime import date

import joblib
import matplotlib.pyplot as plt
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
LABELS = {"baseline": "A baseline", "ros": "B ROS", "rus": "C RUS", "smote": "D SMOTE",
          "smote_rus": "E SMOTE+RUS", "class_weight": "F class weight"}
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


def pick_presets(X_test: pd.DataFrame, y_test: pd.Series, scores: np.ndarray,
                 threshold: float) -> list[dict]:
    """5 example test payments for the app, scored by the team's default cell.

    Each rule takes the row with the middle score among the rows that match it (the borderline
    one: the score closest to the threshold), and no row is used twice.
    """
    y, amount, flagged = np.asarray(y_test), X_test["Amount"].to_numpy(), scores >= threshold
    large = np.quantile(amount[y == 0], 0.99)  # "large" = top 1% of legit amounts
    rules = [
        ("caught_fraud", "A fraud the model catches", (y == 1) & flagged),
        ("missed_fraud", "A fraud the model misses", (y == 1) & ~flagged),
        ("everyday_legit", "An everyday legitimate payment",
         (y == 0) & (amount >= 5) & (amount <= 50) & ~flagged),
        ("large_legit", "A large legitimate payment", (y == 0) & (amount >= large) & ~flagged),
        ("borderline", "A payment scored close to the threshold", np.ones(len(y), dtype=bool)),
    ]
    used = np.zeros(len(y), dtype=bool)
    presets = []
    for preset_id, label, matches in rules:
        rows = np.flatnonzero(matches & ~used)
        if len(rows) == 0:
            print(f"no test row matches preset {preset_id}; it is left out")
            continue
        if preset_id == "borderline":
            row = rows[np.argmin(np.abs(scores[rows] - threshold))]
        else:
            row = rows[np.argsort(scores[rows], kind="stable")][(len(rows) - 1) // 2]
        used[row] = True
        presets.append({"id": preset_id, "label": label, "true_class": int(y[row]),
                        "score": float(scores[row]), "threshold": threshold,
                        "features": X_test.iloc[row].astype(float).to_dict()})
    return presets


def evaluate_test_set(default_cell: str, rerun_reason: str | None = None,
                      source: str = "raw") -> dict:
    """Score every finished cell on the locked test set, once; write test_results.json and
    presets.json. The ONLY function that reads X_test and y_test.

    Each cell is judged at the F1 threshold it chose on out-of-fold training scores (read from
    its cell file, never recomputed here). In the same single opening, the default cell's scores
    pick the 5 example payments for the app.
    """
    cells = load_cells(source)
    if default_cell not in cells:
        raise ValueError(f"--default-cell {default_cell} is not a finished cell; "
                         f"choose one of {sorted(cells)}.")
    open_guard(source, sorted(cells), rerun_reason)

    df, _, _ = remove_duplicates(load_data(source))
    _, X_test, _, y_test = split_data(df)
    print(f"test set: {len(y_test):,} rows, {int(y_test.sum())} frauds")
    results, presets = {}, []
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
        if name == default_cell:
            presets = pick_presets(X_test, y_test, scores, cell["threshold_f1"])

    metrics_dir = artifacts_dir(source) / "metrics"
    (metrics_dir / "test_results.json").write_text(json.dumps(results, indent=2))
    (metrics_dir / "presets.json").write_text(
        json.dumps({"default_cell": default_cell, "presets": presets}, indent=2))
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


def make_figures(source: str = "raw") -> None:
    """PR curves per model, test PR-AUC bars with CI whiskers, and the training class counts
    before and after each technique (log scale). PNGs in artifacts/figures/."""
    results_file = artifacts_dir(source) / "metrics" / "test_results.json"
    if not results_file.is_file():
        raise FileNotFoundError(f"{results_file} not found; run --test first.")
    results = json.loads(results_file.read_text())
    cells = load_cells(source)
    figures_dir = artifacts_dir(source) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    models = [m for m in MODELS if any(r["model"] == m for r in results.values())]

    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4), squeeze=False)
    for ax, model in zip(axes[0], models, strict=True):
        for technique in TECHNIQUES:
            result = results.get(f"{model}__{technique}")
            if result:
                curve = result["pr_curve"]
                ax.plot(curve["recall"], curve["precision"], label=LABELS[technique])
        ax.set(title=model, xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Precision-recall curves on the test set")
    fig.tight_layout()
    fig.savefig(figures_dir / "pr_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.8 / len(TECHNIQUES)
    for j, technique in enumerate(TECHNIQUES):
        for i, model in enumerate(models):
            result = results.get(f"{model}__{technique}")
            if result:
                value, (low, high) = result["test"]["pr_auc"], result["pr_auc_ci"]
                ax.bar(i + (j - 2.5) * width, value, width, color=f"C{j}",
                       yerr=[[value - low], [high - value]], capsize=2,
                       label=LABELS[technique] if i == 0 else None)
    ax.set_xticks(range(len(models)), models)
    ax.set(ylabel="Test PR-AUC", ylim=(0, 1.05), title="Test PR-AUC with 95% bootstrap CI")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(figures_dir / "pr_auc_ci.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4), squeeze=False)
    for ax, model in zip(axes[0], models, strict=True):
        names = [t for t in TECHNIQUES if f"{model}__{t}" in cells]
        after = [cells[f"{model}__{t}"]["train_counts_after"] for t in names]
        before = cells[f"{model}__{names[0]}"]["train_counts_before"]
        x = np.arange(len(names))
        ax.bar(x - 0.2, [a["legit"] for a in after], 0.4, label="legit after")
        ax.bar(x + 0.2, [a["fraud"] for a in after], 0.4, label="fraud after")
        ax.axhline(before["legit"], color="C0", linestyle="--", label="legit before")
        ax.axhline(before["fraud"], color="C1", linestyle="--", label="fraud before")
        ax.set_xticks(x, [LABELS[t] for t in names], rotation=45, ha="right")
        ax.set(yscale="log", title=model, ylabel="Training rows (log scale)")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Training class counts before and after each technique")
    fig.tight_layout()
    fig.savefig(figures_dir / "training_counts.png", dpi=150)
    plt.close(fig)
    print(f"figures written to {figures_dir}")


def main() -> None:
    """Run the steps asked for on the command line, in the order test, stats, figures."""
    parser = argparse.ArgumentParser(description="Test-set evaluation, statistics, figures.")
    parser.add_argument("--source", choices=["raw", "sample"], default="raw")
    parser.add_argument("--test", action="store_true", help="the one-time test-set evaluation")
    parser.add_argument("--default-cell", help="the team's chosen cell, e.g. lightgbm__smote")
    parser.add_argument("--rerun-reason", help="why the test set must be opened again")
    parser.add_argument("--stats", action="store_true", help="significance tests")
    parser.add_argument("--figures", action="store_true", help="report figures")
    args = parser.parse_args()
    if not (args.test or args.stats or args.figures):
        parser.error("choose at least one of --test, --stats, --figures")
    if args.test and not args.default_cell:
        parser.error("--test needs --default-cell <model>__<technique> (the team's choice)")

    if args.test:
        try:
            evaluate_test_set(args.default_cell, args.rerun_reason, args.source)
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            raise SystemExit(f"error: {error}") from None  # one clear line, no traceback
    if args.stats:
        for model, result in significance_tests(args.source)["models"].items():
            for test in result["wilcoxon_vs_baseline"]:
                print(f"{model:<14} {test['technique']:<13} vs baseline: p = {test['p_value']:.3g}"
                      f", Holm p = {test['p_value_holm']:.3g}")
    if args.figures:
        make_figures(args.source)


if __name__ == "__main__":
    main()
