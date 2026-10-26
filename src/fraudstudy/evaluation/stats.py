"""Statistical comparison of techniques on repeated-CV fold scores (master guide Part 6,
"Measuring uncertainty").

Per model, on the repeated-CV PR-AUC of each cell, paired by ``(repeat, fold)``:

* **Wilcoxon signed-rank test** of each technique against the baseline, on the per-fold
  differences ``d_i = score_technique_i - score_baseline_i``. Folds with ``d_i = 0`` are dropped
  (Wilcoxon's zero method); if every difference is zero there is no evidence of a difference, so
  the test reports ``statistic = 0`` and ``p = 1``. **Holm** correction over the tests of a
  model: sort the ``m`` raw p-values ascending, ``p_adj_(j) = max_{i <= j} min(1, (m - i + 1)
  p_(i))``.
* **Friedman test** across all techniques, blocks = folds. Within each block the techniques are
  ranked (1 = highest score, ties get the mean rank); with average ranks ``R_j`` over ``N``
  blocks and ``k`` techniques, ``chi2_F = 12N / (k(k+1)) * sum_j (R_j - (k+1)/2)^2``, divided by
  the tie correction ``1 - sum(t^3 - t) / (N k (k^2 - 1))``; ``p`` from chi-squared with
  ``k - 1`` degrees of freedom.
* **Nemenyi critical difference** (Demsar 2006): ``CD = q_alpha * sqrt(k(k+1) / (6N))`` with
  ``q_alpha = studentized_range.ppf(1 - alpha, k, inf) / sqrt(2)``. Two techniques whose average
  ranks differ by less than CD are not significantly different.

Fold scores share training rows, so they are not independent and every p-value is approximate
(:data:`DEPENDENCE_NOTE`, written into ``stats.json``).
"""

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from fraudstudy.contracts import (
    MODEL_LABELS,
    TECHNIQUE_LABELS,
    ComparisonTable,
    FriedmanTest,
    PairwiseTest,
    StatsResult,
)

BASELINE = "baseline"
METRIC_COLUMN = "pr_auc"
METRIC_NAME = "average_precision"
BLOCK_KEY: list[str] = ["repeat", "fold"]
TIE_TOLERANCE = 1e-12  # |difference| at or below this counts as a tie
MIN_TECHNIQUES_FRIEDMAN = 3
INTERPRETATION_HEADING = "## Interpretation (written by the team)"
DEPENDENCE_NOTE = (
    "Fold scores are not independent (the CV folds share training rows and the repeats reuse "
    "the same data), so every p-value is approximate. Report effect sizes (median differences, "
    "wins/ties/losses) and the test-set bootstrap confidence intervals alongside p-values."
)


def paired_scores(
    fold_scores: pd.DataFrame,
    model: str,
    techniques: Sequence[str],
    metric: str = METRIC_COLUMN,
) -> pd.DataFrame:
    """Pivot one model's fold scores to a blocks x techniques table, checking the pairing.

    Args:
        fold_scores: Rows of ``fold_scores.csv`` (``model, technique, repeat, fold, fold_id`` and
            the metric column).
        model: Model id.
        techniques: Techniques to include, in output column order (missing ones are skipped).
        metric: Score column.

    Returns:
        A DataFrame indexed by ``(repeat, fold)`` with one column per technique present.

    Raises:
        ValueError: If the model has no rows, a block is missing for some technique, a
            ``(repeat, fold)`` appears twice, or the techniques were scored on different folds
            (different ``fold_id``), which would break the pairing.
    """
    rows = fold_scores[fold_scores["model"] == model]
    present = [t for t in techniques if t in set(rows["technique"])]
    if not present:
        raise ValueError(f"No fold scores for model '{model}'; run its cells first.")
    rows = rows[rows["technique"].isin(present)]
    if rows.duplicated(["technique", *BLOCK_KEY]).any():
        raise ValueError(
            f"Duplicate (technique, repeat, fold) rows for model '{model}' in fold scores; "
            "re-run scripts/merge_results.py."
        )
    if "fold_id" in rows.columns:
        ids = rows.groupby(BLOCK_KEY)["fold_id"].nunique()
        if (ids > 1).any():
            raise ValueError(
                f"Techniques of model '{model}' were scored on different folds (fold_id differs "
                f"at {list(ids[ids > 1].index)[:3]}); the paired tests need identical folds."
            )
    table = rows.pivot(index=BLOCK_KEY, columns="technique", values=metric)[present]
    if table.isna().any().any():
        missing = table.columns[table.isna().any()].tolist()
        raise ValueError(
            f"Model '{model}': techniques {missing} lack scores for some (repeat, fold) blocks; "
            "every cell must use the same repeated-CV protocol."
        )
    return table.sort_index()


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (controls the family-wise error rate).

    ``p_adj_(j) = max_{i <= j} min(1, (m - i + 1) p_(i))`` over the ascending p-values
    ``p_(1) <= ... <= p_(m)``, returned in the input order.

    Args:
        p_values: Raw p-values.

    Returns:
        Adjusted p-values, same order and length.
    """
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p, kind="stable")
    scaled = np.minimum(1.0, (m - np.arange(m)) * p[order])
    adjusted = np.empty(m)
    adjusted[order] = np.maximum.accumulate(scaled)
    return adjusted


def wilcoxon_paired(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    """Two-sided Wilcoxon signed-rank test of ``a`` vs ``b`` plus effect-size summaries.

    Args:
        a: Scores of the technique, one per block.
        b: Scores of the reference (baseline), same blocks in the same order.

    Returns:
        ``statistic`` (scipy's two-sided W = min(W+, W-)), ``p_value``, ``mean_difference``,
        ``median_difference``, ``wins`` / ``ties`` / ``losses`` (a higher / equal / lower) and
        ``n_pairs``. If every difference is zero: ``statistic = 0``, ``p_value = 1``.

    Raises:
        ValueError: If the inputs differ in length or are empty.
    """
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape or a_arr.size == 0:
        raise ValueError(
            f"Paired test needs two equal-length, non-empty score lists; got {a_arr.size} and "
            f"{b_arr.size}."
        )
    diff = a_arr - b_arr
    diff[np.abs(diff) <= TIE_TOLERANCE] = 0.0
    wins, losses = int((diff > 0).sum()), int((diff < 0).sum())
    if wins + losses == 0:
        statistic, p_value = 0.0, 1.0
    else:
        result = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        statistic, p_value = float(result.statistic), float(min(1.0, result.pvalue))
    return {
        "statistic": statistic,
        "p_value": p_value,
        "mean_difference": float(diff.mean()),
        "median_difference": float(np.median(diff)),
        "wins": wins,
        "ties": int(diff.size - wins - losses),
        "losses": losses,
        "n_pairs": int(diff.size),
    }


def compare_to_baseline(
    table: pd.DataFrame,
    model: str,
    alpha: float,
    baseline: str = BASELINE,
    metric_name: str = METRIC_NAME,
) -> list[PairwiseTest]:
    """Wilcoxon test of every technique in ``table`` against ``baseline``, Holm-corrected.

    Args:
        table: Blocks x techniques scores from :func:`paired_scores`.
        model: Model id written into each result.
        alpha: Significance level for ``significant`` (adjusted p < alpha).
        baseline: Reference technique (a column of ``table``).
        metric_name: Metric label for the results.

    Returns:
        One :class:`PairwiseTest` per non-baseline technique, in column order.

    Raises:
        ValueError: If ``baseline`` is not a column of ``table``.
    """
    if baseline not in table.columns:
        raise ValueError(
            f"Model '{model}' has no '{baseline}' fold scores; the baseline cell must be run."
        )
    others = [t for t in table.columns if t != baseline]
    raw = [wilcoxon_paired(table[t].to_numpy(), table[baseline].to_numpy()) for t in others]
    adjusted = holm_adjust([r["p_value"] for r in raw])
    return [
        PairwiseTest(
            model=model,
            technique_a=technique,
            technique_b=baseline,
            test="wilcoxon",
            metric=metric_name,
            difference=r["mean_difference"],
            difference_ci=None,
            statistic=r["statistic"],
            p_value=r["p_value"],
            p_value_adjusted=float(p_adj),
            significant=bool(p_adj < alpha),
            n_pairs=r["n_pairs"],
            median_difference=r["median_difference"],
            wins=r["wins"],
            ties=r["ties"],
            losses=r["losses"],
        )
        for technique, r, p_adj in zip(others, raw, adjusted, strict=True)
    ]


def average_ranks(table: pd.DataFrame) -> pd.Series:
    """Mean within-block rank of each technique (1 = highest score; ties share the mean rank).

    Args:
        table: Blocks x techniques scores.

    Returns:
        A Series indexed by technique.
    """
    ranks = stats.rankdata(-table.to_numpy(dtype=float), axis=1, method="average")
    return pd.Series(ranks.mean(axis=0), index=table.columns)


def friedman(table: pd.DataFrame) -> tuple[float, float]:
    """Friedman chi-squared test with tie correction (formula in the module docstring).

    Args:
        table: Blocks x techniques scores, at least 2 blocks and 3 techniques.

    Returns:
        ``(statistic, p_value)``; ``(0.0, 1.0)`` when every block is a complete tie.

    Raises:
        ValueError: If there are fewer than 2 blocks or 3 techniques.
    """
    n, k = table.shape
    if n < 2 or k < MIN_TECHNIQUES_FRIEDMAN:
        raise ValueError(
            f"Friedman test needs >= 2 blocks and >= {MIN_TECHNIQUES_FRIEDMAN} techniques; "
            f"got {n} blocks and {k} techniques."
        )
    ranks = stats.rankdata(-table.to_numpy(dtype=float), axis=1, method="average")
    tie_sum = 0.0
    for row in ranks:
        _, counts = np.unique(row, return_counts=True)
        tie_sum += float((counts**3 - counts).sum())
    correction = 1.0 - tie_sum / (n * k * (k * k - 1))
    if correction <= TIE_TOLERANCE:
        return 0.0, 1.0
    r_mean = ranks.mean(axis=0)
    chi2 = 12.0 * n / (k * (k + 1)) * float(((r_mean - (k + 1) / 2.0) ** 2).sum()) / correction
    return chi2, float(stats.chi2.sf(chi2, k - 1))


def nemenyi_q(k: int, alpha: float) -> float:
    """Nemenyi critical value ``q_alpha = studentized_range.ppf(1 - alpha, k, inf) / sqrt(2)``.

    Args:
        k: Number of techniques compared.
        alpha: Significance level.

    Returns:
        ``q_alpha`` (2.850 for k = 6, alpha = 0.05; Demsar 2006, Table 5a).
    """
    return float(stats.studentized_range.ppf(1.0 - alpha, k, np.inf) / np.sqrt(2.0))


def critical_difference(k: int, n_blocks: int, alpha: float) -> float:
    """Nemenyi critical difference ``CD = q_alpha * sqrt(k(k+1) / (6N))``.

    Args:
        k: Number of techniques.
        n_blocks: Number of blocks ``N`` (folds).
        alpha: Significance level.

    Returns:
        The CD in average-rank units (about 1.95 for k = 6, N = 15, alpha = 0.05).

    Raises:
        ValueError: If ``k < 2`` or ``n_blocks < 1``.
    """
    if k < 2 or n_blocks < 1:
        raise ValueError(f"CD needs k >= 2 and N >= 1; got k={k}, N={n_blocks}.")
    return nemenyi_q(k, alpha) * float(np.sqrt(k * (k + 1) / (6.0 * n_blocks)))


def friedman_summary(table: pd.DataFrame, scope: str, alpha: float) -> FriedmanTest:
    """Friedman test, average ranks and Nemenyi CD for one scope.

    Args:
        table: Blocks x techniques scores.
        scope: Model id, or ``"all"``.
        alpha: Significance level for the CD.

    Returns:
        A :class:`FriedmanTest`.
    """
    statistic, p_value = friedman(table)
    n, k = table.shape
    return FriedmanTest(
        scope=scope,
        metric=METRIC_NAME,
        n_blocks=n,
        statistic=statistic,
        p_value=min(1.0, p_value),
        average_ranks={str(t): float(r) for t, r in average_ranks(table).items()},
        critical_difference=critical_difference(k, n, alpha),
    )


def compute_stats(
    fold_scores: pd.DataFrame,
    models: Sequence[str],
    techniques: Sequence[str],
    alpha: float,
    n_bootstrap: int,
    confidence: float,
    seed: int,
    dataset_source: str | None = None,
) -> StatsResult:
    """All tests for ``stats.json``: per model Wilcoxon vs baseline (Holm) and Friedman + CD,
    plus a pooled Friedman over every model x fold block when all models share the techniques.

    Args:
        fold_scores: Rows of ``fold_scores.csv``.
        models: Model ids in report order (models without fold scores are skipped).
        techniques: Technique ids in report order (the baseline must be among them).
        alpha: Significance level.
        n_bootstrap, confidence, seed: Recorded from the config (the bootstrap CIs themselves
            are computed on the test set by the test evaluation).
        dataset_source: ``"real"`` or ``"synthetic"``.

    Returns:
        A validated :class:`StatsResult`.

    Raises:
        ValueError: If no model has fold scores, or a model's pairing is broken.
    """
    present = [m for m in models if (fold_scores["model"] == m).any()]
    if not present:
        raise ValueError(
            "fold_scores.csv has no rows for the configured models; run scripts/run_experiments.py."
        )
    pairwise: list[PairwiseTest] = []
    friedmans: list[FriedmanTest] = []
    tables: dict[str, pd.DataFrame] = {}
    for model in present:
        table = paired_scores(fold_scores, model, techniques)
        tables[model] = table
        pairwise.extend(compare_to_baseline(table, model, alpha))
        if table.shape[1] >= MIN_TECHNIQUES_FRIEDMAN:
            friedmans.append(friedman_summary(table, model, alpha))
    columns = {tuple(t.columns) for t in tables.values()}
    notes = [DEPENDENCE_NOTE]
    if len(tables) > 1 and len(columns) == 1 and len(next(iter(columns))) >= 3:
        pooled = pd.concat(tables, names=["model"])
        friedmans.append(friedman_summary(pooled, "all", alpha))
        notes.append(
            "The 'all' Friedman test pools every model x fold block; blocks of the same fold "
            "share data across models, which makes its p-value even more approximate."
        )
    return StatsResult(
        alpha=alpha,
        correction="holm",
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
        friedman=friedmans,
        pairwise=pairwise,
        dataset_source=dataset_source,
        notes=notes,
    )


# --- findings.md --------------------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 4) -> str:
    """Fixed-point number, or an em dash for ``None``."""
    return "—" if value is None else f"{value:.{digits}f}"


def _p(value: float) -> str:
    """p-value with 3 significant digits (``< 0.0001`` below that)."""
    return "< 0.0001" if value < 1e-4 else f"{value:.3g}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """A GitHub-flavoured markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def existing_interpretation(path: Path) -> str:
    """Everything after the interpretation heading of an existing findings file (``""`` if the
    file or the heading is absent), so re-running the script never erases the team's text."""
    path = Path(path)
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8")
    _, found, rest = text.partition(INTERPRETATION_HEADING)
    return rest.strip("\n") if found else ""


def _comparison_rows(comparison: ComparisonTable) -> list[list[str]]:
    """Rows of the test-set table, in model then technique order."""
    order = {
        (m, t): i for i, (m, t) in enumerate((m, t) for m in MODEL_LABELS for t in TECHNIQUE_LABELS)
    }
    rows = []
    for r in sorted(comparison.records, key=lambda r: order[(r.model, r.technique)]):
        t = r.test_at_tuned
        rows.append(
            [
                MODEL_LABELS[r.model],
                TECHNIQUE_LABELS[r.technique],
                _fmt(r.chosen_ratio, 3),
                f"{r.cv_pr_auc_mean:.4f} ± {r.cv_pr_auc_std:.4f}",
                f"{t.pr_auc:.4f} ({r.pr_auc_ci[0]:.3f}–{r.pr_auc_ci[1]:.3f})",
                _fmt(t.roc_auc),
                _fmt(r.threshold_f1, 3),
                _fmt(t.precision, 3),
                _fmt(t.recall, 3),
                _fmt(t.f1, 3),
                _fmt(t.f2, 3),
                _fmt(r.test_at_0_5.f1, 3),
            ]
        )
    return rows


def findings_markdown(
    result: StatsResult,
    comparison: ComparisonTable | None,
    figures: Sequence[Path],
    findings_path: Path,
    source: str,
    interpretation: str = "",
) -> str:
    """The text of ``findings.md``: auto-generated tables only, no conclusions.

    Sections: test-set results (if ``comparison`` is given), Wilcoxon tests against the
    baseline, Friedman tests with average ranks and CD, figure links, the caveats from
    ``result.notes``, and the team's interpretation section (``interpretation`` is carried over
    verbatim; empty on the first run).

    Args:
        result: The statistics.
        comparison: Test results, or ``None`` if the test set has not been evaluated.
        figures: Figure files to link (relative to the findings file).
        findings_path: Where the file will be written (for relative links).
        source: ``"raw"`` or ``"sample"`` (shown in the header).
        interpretation: Existing team text for the last section.

    Returns:
        Markdown text ending in a newline.
    """
    lines = [
        "# Findings",
        "",
        f"> Auto-generated by `scripts/run_stats.py --source {source}`; re-run it instead of "
        "editing the tables. Only the last section is written by people and it is kept between "
        "runs. The script reports numbers; it draws no conclusions.",
        "",
        f"Dataset source: **{result.dataset_source or 'unknown'}**. Primary metric: PR-AUC "
        f"(average precision). Significance level α = {result.alpha:g}; correction: "
        f"{result.correction}.",
    ]
    if result.dataset_source == "synthetic":
        lines += [
            "",
            "**These numbers come from the synthetic development sample, not the real dataset.**",
        ]

    lines += ["", "## Test-set results", ""]
    if comparison is None:
        lines.append("The test set has not been evaluated yet (no `comparison.json`).")
    else:
        lines += [
            f"Metrics at the F1 threshold chosen on out-of-fold training scores; PR-AUC with "
            f"{result.confidence:.0%} stratified bootstrap interval ({result.n_bootstrap} "
            "resamples). CV PR-AUC is the repeated-CV mean ± standard deviation.",
            "",
        ]
        lines += _table(
            [
                "Model",
                "Technique",
                "Ratio",
                "CV PR-AUC",
                "Test PR-AUC (CI)",
                "ROC-AUC",
                "Thr F1",
                "Precision",
                "Recall",
                "F1",
                "F2",
                "F1 at 0.5",
            ],
            _comparison_rows(comparison),
        )

    lines += [
        "",
        "## Wilcoxon signed-rank tests against the baseline",
        "",
        "Two-sided, on CV PR-AUC paired by (repeat, fold); difference = technique − baseline; "
        "W/T/L = folds where the technique scored higher / equal / lower; Holm correction over "
        "the tests of each model.",
        "",
    ]
    lines += _table(
        [
            "Model",
            "Technique",
            "Median diff",
            "Mean diff",
            "W/T/L",
            "Statistic",
            "p (raw)",
            "p (Holm)",
            f"Adjusted p < {result.alpha:g}",
        ],
        [
            [
                MODEL_LABELS[t.model],
                TECHNIQUE_LABELS[t.technique_a],
                _fmt(t.median_difference),
                _fmt(t.difference),
                f"{t.wins}/{t.ties}/{t.losses}",
                f"{t.statistic:g}",
                _p(t.p_value),
                _p(t.p_value_adjusted),
                "yes" if t.significant else "no",
            ]
            for t in result.pairwise
            if t.test == "wilcoxon"
        ],
    )

    techniques = [t for t in TECHNIQUE_LABELS if any(t in f.average_ranks for f in result.friedman)]
    lines += [
        "",
        "## Friedman tests and average ranks",
        "",
        "Blocks are CV folds (rank 1 = highest PR-AUC in a fold). CD is the Nemenyi critical "
        f"difference at α = {result.alpha:g}: average ranks closer than CD are not significantly "
        "different.",
        "",
    ]
    lines += _table(
        ["Scope", "N", "χ²_F", "p", "CD", *[TECHNIQUE_LABELS[t].split(".")[0] for t in techniques]],
        [
            [
                MODEL_LABELS.get(f.scope, "All models pooled"),
                str(f.n_blocks),
                f"{f.statistic:.3f}",
                _p(f.p_value),
                _fmt(f.critical_difference, 3),
                *[_fmt(f.average_ranks.get(t), 2) for t in techniques],
            ]
            for f in result.friedman
        ],
    )
    lines += ["", "Rank columns: " + "; ".join(TECHNIQUE_LABELS[t] for t in techniques) + "."]

    if figures:
        base = Path(findings_path).parent
        lines += ["", "## Figures", ""]
        lines += [
            f"* [{Path(f).name}]({Path(os.path.relpath(f, base)).as_posix()})" for f in figures
        ]

    lines += ["", "## Notes", ""]
    lines += [f"* {note}" for note in result.notes]
    lines += ["", INTERPRETATION_HEADING, ""]
    if interpretation:
        lines += [interpretation, ""]
    return "\n".join(lines).rstrip("\n") + "\n"
