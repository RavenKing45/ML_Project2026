"""Statistical comparison of the techniques on the repeated-CV fold scores, and docs/findings.md.

scripts/run_stats.py calls compute_stats() and findings_markdown(). Per model, on the PR-AUC of
each (repeat, fold) block: a Wilcoxon signed-rank test of every technique against the baseline
with Holm's correction, and a Friedman test with the Nemenyi critical difference. Reads nothing
itself; the script writes stats.json and the findings file.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from fraudstudy.experiment import MODEL_LABELS, TECHNIQUE_LABELS

CONTRACT_VERSION = "1.6"  # docs/contracts.md; written into stats.json
INTERPRETATION_HEADING = "## Interpretation (written by the team)"
DEPENDENCE_NOTE = (
    "Fold scores are not independent (the CV folds share training rows and the repeats reuse "
    "the same data), so every p-value is approximate. Report effect sizes (median differences, "
    "wins/ties/losses) and the test-set bootstrap confidence intervals alongside p-values."
)
# Nemenyi critical values q_alpha by number of techniques k, from Demsar (2006), "Statistical
# comparisons of classifiers over multiple data sets", JMLR 7, Table 5(a) (two-tailed).
NEMENYI_Q = {
    0.05: {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102,
           10: 3.164},
    0.10: {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.459, 6: 2.589, 7: 2.693, 8: 2.780, 9: 2.855,
           10: 2.920},
}


def paired_scores(fold_scores: pd.DataFrame, model: str, techniques: list) -> pd.DataFrame:
    """One model's fold scores as a table: one row per (repeat, fold), one column per technique.

    The tests compare techniques block by block, so every technique must have been scored on
    exactly the same folds (same fold_id) with no block missing or repeated.
    """
    rows = fold_scores[fold_scores["model"] == model]
    present = [t for t in techniques if t in set(rows["technique"])]
    if not present:
        raise ValueError(f"No fold scores for model '{model}'; run its cells first.")
    rows = rows[rows["technique"].isin(present)]
    if rows.duplicated(["technique", "repeat", "fold"]).any():
        raise ValueError(f"Duplicate (technique, repeat, fold) rows for model '{model}' in fold "
                         "scores; re-run scripts/merge_results.py.")
    if (rows.groupby(["repeat", "fold"])["fold_id"].nunique() > 1).any():
        raise ValueError(f"Techniques of model '{model}' were scored on different folds "
                         "(fold_id differs); the paired tests need identical folds.")
    table = rows.pivot(index=["repeat", "fold"], columns="technique", values="pr_auc")[present]
    if table.isna().any().any():
        missing = table.columns[table.isna().any()].tolist()
        raise ValueError(f"Model '{model}': techniques {missing} lack scores for some (repeat, "
                         "fold) blocks; every cell must use the same repeated-CV protocol.")
    return table.sort_index()


def holm_adjust(p_values: list) -> np.ndarray:
    """Holm's step-down adjusted p-values, in the input order.

    With the m p-values sorted ascending, p_adj_(j) = max over i <= j of min(1, (m - i + 1) p_(i)).
    """
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p, kind="stable")
    scaled = np.minimum(1.0, (len(p) - np.arange(len(p))) * p[order])
    adjusted = np.empty(len(p))
    adjusted[order] = np.maximum.accumulate(scaled)  # never below an earlier adjusted p
    return adjusted


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sided Wilcoxon signed-rank test of a vs b (same blocks), plus effect sizes.

    Differences of 1e-12 or less count as ties and are dropped by the test; if every block is a
    tie there is no evidence of a difference, so statistic = 0 and p = 1.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError(f"Paired test needs two equal-length, non-empty score lists; got {a.size} "
                         f"and {b.size}.")
    difference = a - b
    difference[np.abs(difference) <= 1e-12] = 0.0
    wins, losses = int((difference > 0).sum()), int((difference < 0).sum())
    if wins + losses == 0:
        statistic, p_value = 0.0, 1.0
    else:
        result = stats.wilcoxon(difference, zero_method="wilcox", alternative="two-sided")
        statistic, p_value = float(result.statistic), float(min(1.0, result.pvalue))
    return {"statistic": statistic, "p_value": p_value,
            "mean_difference": float(difference.mean()),
            "median_difference": float(np.median(difference)), "wins": wins,
            "ties": int(difference.size - wins - losses), "losses": losses,
            "n_pairs": int(difference.size)}


def compare_to_baseline(table: pd.DataFrame, model: str, alpha: float) -> list:
    """Wilcoxon test of every technique against the baseline, Holm-corrected over the model."""
    if "baseline" not in table.columns:
        raise ValueError(f"Model '{model}' has no 'baseline' fold scores; the baseline cell must "
                         "be run.")
    others = [t for t in table.columns if t != "baseline"]
    tests = [wilcoxon_paired(table[t].to_numpy(), table["baseline"].to_numpy()) for t in others]
    adjusted = holm_adjust([t["p_value"] for t in tests])
    return [{"model": model, "technique_a": technique, "technique_b": "baseline",
             "test": "wilcoxon", "metric": "average_precision",
             "difference": t["mean_difference"], "difference_ci": None,
             "statistic": t["statistic"], "p_value": t["p_value"],
             "p_value_adjusted": float(p_adj), "significant": bool(p_adj < alpha),
             "n_pairs": t["n_pairs"], "median_difference": t["median_difference"],
             "wins": t["wins"], "ties": t["ties"], "losses": t["losses"]}
            for technique, t, p_adj in zip(others, tests, adjusted, strict=True)]


def average_ranks(table: pd.DataFrame) -> pd.Series:
    """Mean rank of each technique over the blocks (1 = highest score; ties share the mean rank)."""
    ranks = stats.rankdata(-table.to_numpy(dtype=float), axis=1, method="average")
    return pd.Series(ranks.mean(axis=0), index=table.columns)


def friedman(table: pd.DataFrame) -> tuple[float, float]:
    """Friedman test over the techniques (columns), blocks = folds (rows): (chi-square, p).

    If every block is a complete tie, scipy would divide by zero; there is no difference to
    find, so the answer is (0, 1).
    """
    if (table.nunique(axis=1) == 1).all():
        return 0.0, 1.0
    result = stats.friedmanchisquare(*table.to_numpy(dtype=float).T)
    return float(result.statistic), float(result.pvalue)


def critical_difference(k: int, n_blocks: int, alpha: float) -> float:
    """Nemenyi critical difference CD = q_alpha * sqrt(k (k + 1) / (6 N)) (Demsar 2006).

    Two techniques whose average ranks differ by less than CD are not significantly different.
    """
    if k not in NEMENYI_Q.get(alpha, {}):
        raise ValueError(f"No Nemenyi q-value for k={k} techniques at alpha={alpha}; the table "
                         "covers k = 2..10 and alpha 0.05 or 0.10 (stats.alpha in "
                         "experiment.yaml).")
    return NEMENYI_Q[alpha][k] * float(np.sqrt(k * (k + 1) / (6.0 * n_blocks)))


def friedman_summary(table: pd.DataFrame, scope: str, alpha: float) -> dict:
    """Friedman test, average ranks and Nemenyi CD of one model (or "all" models pooled)."""
    statistic, p_value = friedman(table)
    n_blocks, k = table.shape
    return {"scope": scope, "metric": "average_precision", "n_blocks": n_blocks,
            "statistic": statistic, "p_value": min(1.0, p_value),
            "average_ranks": {str(t): float(r) for t, r in average_ranks(table).items()},
            "critical_difference": critical_difference(k, n_blocks, alpha)}


def compute_stats(fold_scores: pd.DataFrame, models: list, techniques: list, alpha: float,
                  n_bootstrap: int, confidence: float, seed: int,
                  dataset_source: str | None = None) -> dict:
    """Every test for stats.json: per model, Wilcoxon vs baseline (Holm) and Friedman + CD;
    plus a Friedman test pooling all models when they all have the same techniques.

    Args:
        fold_scores: Rows of fold_scores.csv.
        models: Models in report order (those without fold scores are skipped).
        techniques: Techniques in report order (the baseline must be one of them).
        alpha: Significance level (stats.alpha in experiment.yaml).
        n_bootstrap, confidence, seed: From experiment.yaml, recorded in the file (the
            intervals themselves are computed on the test set by final_test.py).
        dataset_source: "real" or "synthetic".

    Returns:
        The stats.json contents.
    """
    present = [m for m in models if (fold_scores["model"] == m).any()]
    if not present:
        raise ValueError("fold_scores.csv has no rows for the configured models; run "
                         "scripts/run_experiments.py.")
    pairwise, friedmans, tables = [], [], {}
    for model in present:
        tables[model] = paired_scores(fold_scores, model, techniques)
        pairwise += compare_to_baseline(tables[model], model, alpha)
        if tables[model].shape[1] >= 3:  # Friedman needs at least 3 techniques
            friedmans.append(friedman_summary(tables[model], model, alpha))
    notes = [DEPENDENCE_NOTE]
    columns = {tuple(t.columns) for t in tables.values()}
    if len(tables) > 1 and len(columns) == 1 and len(next(iter(columns))) >= 3:
        friedmans.append(friedman_summary(pd.concat(tables, names=["model"]), "all", alpha))
        notes.append("The 'all' Friedman test pools every model x fold block; blocks of the same "
                     "fold share data across models, which makes its p-value even more "
                     "approximate.")
    return {"contract_version": CONTRACT_VERSION, "is_example": False, "alpha": alpha,
            "correction": "holm", "n_bootstrap": n_bootstrap, "confidence": confidence,
            "seed": seed, "friedman": friedmans,
            "pairwise": pairwise, "dataset_source": dataset_source, "notes": notes}


# --- findings.md --------------------------------------------------------------------------------


def number(value: float | None, digits: int = 4) -> str:
    """A fixed-point number, or an em dash for None."""
    return "—" if value is None else f"{value:.{digits}f}"


def p_text(value: float) -> str:
    """A p-value with 3 significant digits ("< 0.0001" below that)."""
    return "< 0.0001" if value < 1e-4 else f"{value:.3g}"


def markdown_table(header: list, rows: list) -> list:
    """The lines of a GitHub-flavoured markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def existing_interpretation(path: Path) -> str:
    """The team's text after the interpretation heading of an existing findings file ("" if
    none), so re-running the script never erases it."""
    if not Path(path).is_file():
        return ""
    _, found, rest = Path(path).read_text(encoding="utf-8").partition(INTERPRETATION_HEADING)
    return rest.strip("\n") if found else ""


def test_results_rows(comparison: dict) -> list:
    """Rows of the findings' test-set table, in model then technique order."""
    order = [(m, t) for m in MODEL_LABELS for t in TECHNIQUE_LABELS]
    rows = []
    for r in sorted(comparison["records"], key=lambda r: order.index((r["model"], r["technique"]))):
        t = r["test_at_tuned"]
        rows.append([MODEL_LABELS[r["model"]], TECHNIQUE_LABELS[r["technique"]],
                     number(r["chosen_ratio"], 3),
                     f"{r['cv_pr_auc_mean']:.4f} ± {r['cv_pr_auc_std']:.4f}",
                     f"{t['pr_auc']:.4f} ({r['pr_auc_ci'][0]:.3f}–{r['pr_auc_ci'][1]:.3f})",
                     number(t["roc_auc"]), number(r["threshold_f1"], 3), number(t["precision"], 3),
                     number(t["recall"], 3), number(t["f1"], 3), number(t["f2"], 3),
                     number(r["test_at_0_5"]["f1"], 3)])
    return rows


def findings_markdown(result: dict, comparison: dict | None, figures: list, findings_path: Path,
                      source: str, interpretation: str = "") -> str:
    """The text of findings.md: tables only, no conclusions; the team's last section is kept.

    Args:
        result: The stats.json contents.
        comparison: The comparison.json contents, or None before the test set is evaluated.
        figures: Figure files to link (written relative to the findings file).
        findings_path: Where the findings file will be written.
        source: "raw" or "sample" (shown in the header).
        interpretation: The team's existing text for the last section.

    Returns:
        Markdown text ending in a newline.
    """
    alpha = result["alpha"]
    lines = [
        "# Findings", "",
        f"> Auto-generated by `scripts/run_stats.py --source {source}`; re-run it instead of "
        "editing the tables. Only the last section is written by people and it is kept between "
        "runs. The script reports numbers; it draws no conclusions.", "",
        f"Dataset source: **{result['dataset_source'] or 'unknown'}**. Primary metric: PR-AUC "
        f"(average precision). Significance level α = {alpha:g}; correction: "
        f"{result['correction']}.",
    ]
    if result["dataset_source"] == "synthetic":
        lines += ["", "**These numbers come from the synthetic development sample, not the real "
                  "dataset.**"]

    lines += ["", "## Test-set results", ""]
    if comparison is None:
        lines.append("The test set has not been evaluated yet (no `comparison.json`).")
    else:
        lines += [f"Metrics at the F1 threshold chosen on out-of-fold training scores; PR-AUC "
                  f"with {result['confidence']:.0%} stratified bootstrap interval "
                  f"({result['n_bootstrap']} resamples). CV PR-AUC is the repeated-CV mean ± "
                  "standard deviation.", ""]
        lines += markdown_table(["Model", "Technique", "Ratio", "CV PR-AUC", "Test PR-AUC (CI)",
                                 "ROC-AUC", "Thr F1", "Precision", "Recall", "F1", "F2",
                                 "F1 at 0.5"], test_results_rows(comparison))

    lines += ["", "## Wilcoxon signed-rank tests against the baseline", "",
              "Two-sided, on CV PR-AUC paired by (repeat, fold); difference = technique − "
              "baseline; W/T/L = folds where the technique scored higher / equal / lower; Holm "
              "correction over the tests of each model.", ""]
    lines += markdown_table(
        ["Model", "Technique", "Median diff", "Mean diff", "W/T/L", "Statistic", "p (raw)",
         "p (Holm)", f"Adjusted p < {alpha:g}"],
        [[MODEL_LABELS[t["model"]], TECHNIQUE_LABELS[t["technique_a"]],
          number(t["median_difference"]), number(t["difference"]),
          f"{t['wins']}/{t['ties']}/{t['losses']}", f"{t['statistic']:g}", p_text(t["p_value"]),
          p_text(t["p_value_adjusted"]), "yes" if t["significant"] else "no"]
         for t in result["pairwise"]])

    techniques = [t for t in TECHNIQUE_LABELS
                  if any(t in f["average_ranks"] for f in result["friedman"])]
    lines += ["", "## Friedman tests and average ranks", "",
              "Blocks are CV folds (rank 1 = highest PR-AUC in a fold). CD is the Nemenyi critical "
              f"difference at α = {alpha:g}: average ranks closer than CD are not significantly "
              "different.", ""]
    lines += markdown_table(
        ["Scope", "N", "χ²_F", "p", "CD", *[TECHNIQUE_LABELS[t].split(".")[0] for t in techniques]],
        [[MODEL_LABELS.get(f["scope"], "All models pooled"), str(f["n_blocks"]),
          f"{f['statistic']:.3f}", p_text(f["p_value"]), number(f["critical_difference"], 3),
          *[number(f["average_ranks"].get(t), 2) for t in techniques]]
         for f in result["friedman"]])
    lines += ["", "Rank columns: " + "; ".join(TECHNIQUE_LABELS[t] for t in techniques) + "."]

    if figures:
        folder = Path(findings_path).parent
        lines += ["", "## Figures", ""]
        lines += [f"* [{Path(f).name}]({Path(os.path.relpath(f, folder)).as_posix()})"
                  for f in figures]
    lines += ["", "## Notes", ""] + [f"* {note}" for note in result["notes"]]
    lines += ["", INTERPRETATION_HEADING, ""]
    if interpretation:
        lines += [interpretation, ""]
    return "\n".join(lines).rstrip("\n") + "\n"
