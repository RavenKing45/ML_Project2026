"""Report figures for the experiment results (master guide Part 6, "What the experiment
produces").

Inputs are the saved artifacts only (``comparison.json``, ``curves.json``, ``stats.json``,
``fold_scores.csv``, ``resampling_audit.json``); nothing here fits or scores a model. Figures are
drawn with ``matplotlib.figure.Figure`` (Agg canvas, no display needed) at 300 dpi in the EDA
style. Techniques use Okabe-Ito colours (colour-blind safe) and also differ by line style /
marker, so they stay distinguishable in greyscale. Titles are sentence case, axes carry units.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from fraudstudy.contracts import (
    MODEL_LABELS,
    TECHNIQUE_LABELS,
    ComparisonRecord,
    ComparisonTable,
    CurvesFile,
    FriedmanTest,
    Metrics,
    ResamplingAudit,
    StatsResult,
)
from fraudstudy.eda import FIGURE_DPI, FIGURE_STYLE
from fraudstudy.errors import DataValidationError
from fraudstudy.paths import resolve

TECHNIQUE_ORDER: tuple[str, ...] = tuple(TECHNIQUE_LABELS)
MODEL_ORDER: tuple[str, ...] = tuple(MODEL_LABELS)
TECHNIQUE_COLOURS: dict[str, str] = {  # Okabe-Ito
    "baseline": "#000000",
    "ros": "#E69F00",
    "rus": "#56B4E9",
    "smote": "#009E73",
    "smote_rus": "#CC79A7",
    "class_weight": "#D55E00",
}
TECHNIQUE_LINESTYLES: dict[str, str | tuple[int, tuple[int, ...]]] = {
    "baseline": "-",
    "ros": "--",
    "rus": "-.",
    "smote": ":",
    "smote_rus": (0, (5, 1, 1, 1, 1, 1)),
    "class_weight": (0, (3, 1)),
}
TECHNIQUE_MARKERS: dict[str, str] = {
    "baseline": "o",
    "ros": "s",
    "rus": "^",
    "smote": "D",
    "smote_rus": "v",
    "class_weight": "P",
}
SHORT_LABELS: dict[str, str] = {
    "baseline": "A baseline",
    "ros": "B ROS",
    "rus": "C RUS",
    "smote": "D SMOTE",
    "smote_rus": "E SMOTE+RUS",
    "class_weight": "F weights",
}
FIGURES_DIR = "artifacts/figures"
SAMPLE_FIGURES_SUBDIR = "sample"
RESAMPLERS: tuple[str, ...] = ("ros", "rus", "smote", "smote_rus")
REFERENCE_COLOUR = "#808080"
LEGIT_COLOUR = "#0072B2"
FRAUD_COLOUR = "#D55E00"
FRAUD_HATCH = "////"
CONFUSION_CMAP = "cividis"
CONFUSION_TEXT_SWITCH = 0.6  # cividis is light above this value: use dark text
CURVE_LINEWIDTH = 1.3
PR_AUC_UNIT = "PR-AUC (average precision, 0-1)"

FIGURE_NAMES: tuple[str, ...] = (
    "pr_curves_by_model.png",
    "roc_curves_by_model.png",
    "pr_auc_with_ci.png",
    "confusion_matrices_tuned.png",
    "threshold_effect.png",
    "resampling_counts.png",
    "fold_score_boxplots.png",
)
"""Fixed figure names; :func:`make_all_figures` also writes ``cd_diagram_<scope>.png`` per
Friedman scope (each model, and ``all`` when present)."""


# --- helpers ------------------------------------------------------------------------------------


def figures_dir_for(source: str) -> Path:
    """Default figure folder: ``artifacts/figures`` for raw, ``artifacts/figures/sample`` for the
    sample (so synthetic figures never overwrite the report's)."""
    base = resolve(FIGURES_DIR)
    return base if source == "raw" else base / SAMPLE_FIGURES_SUBDIR


def _new_figure(width: float, height: float) -> Figure:
    """A blank figure with constrained layout (call inside ``rc_context(FIGURE_STYLE)``)."""
    return Figure(figsize=(width, height), layout="constrained")


def _save(fig: Figure, path: Path) -> Path:
    """Save ``fig`` as a 300-dpi PNG, creating the parent folder."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI)
    return path


def _models(records: Sequence[ComparisonRecord]) -> list[str]:
    """Models present in ``records``, in report order."""
    present = {r.model for r in records}
    return [m for m in MODEL_ORDER if m in present]


def _cells(records: Sequence[ComparisonRecord], model: str) -> list[ComparisonRecord]:
    """One model's records, in technique order."""
    by_technique = {r.technique: r for r in records if r.model == model}
    return [by_technique[t] for t in TECHNIQUE_ORDER if t in by_technique]


def _require(records: Sequence[ComparisonRecord]) -> None:
    """Raise if there is nothing to plot."""
    if not records:
        raise DataValidationError(
            "comparison.json has no records; run scripts/evaluate_test.py before the figures."
        )


def fraud_rate(metrics: Metrics) -> float:
    """Share of fraud rows in the scored set, ``(tp + fn) / (tp + fp + tn + fn)``: the precision
    of a no-skill classifier, i.e. the PR-curve baseline.

    Args:
        metrics: Metrics of any threshold on that set (the counts sum to its size).

    Returns:
        The fraud rate in [0, 1].
    """
    total = metrics.tp + metrics.fp + metrics.tn + metrics.fn
    return (metrics.tp + metrics.fn) / total if total else 0.0


def _technique_legend(fig: Figure, axes: Axes, ncol: int = 6) -> None:
    """One shared legend below the panels, built from ``axes``' labelled artists."""
    handles, labels = axes.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=ncol)


# --- curves -------------------------------------------------------------------------------------


def plot_pr_curves(comparison: ComparisonTable, curves: CurvesFile, path: Path) -> Path:
    """1 x 3 panels of test-set precision-recall curves, one line per technique, with the
    no-skill line at the test fraud rate (precision of flagging everyone).

    Args:
        comparison: Test results (legend PR-AUC and the fraud rate).
        curves: Test-set curves.
        path: Output PNG.

    Returns:
        ``path``.
    """
    _require(comparison.records)
    curve = {(c.model, c.technique): c for c in curves.cells}
    models = _models(comparison.records)
    with matplotlib.rc_context(FIGURE_STYLE):
        fig = _new_figure(3.3 * len(models), 3.8)
        axes = np.atleast_1d(fig.subplots(1, len(models), sharey=True))
        for ax, model in zip(axes, models, strict=True):
            cells = _cells(comparison.records, model)
            for r in cells:
                c = curve.get((r.model, r.technique))
                if c is None:
                    continue
                ax.plot(
                    c.pr.recall,
                    c.pr.precision,
                    color=TECHNIQUE_COLOURS[r.technique],
                    linestyle=TECHNIQUE_LINESTYLES[r.technique],
                    linewidth=CURVE_LINEWIDTH,
                    label=SHORT_LABELS[r.technique],
                )
            rate = fraud_rate(cells[0].test_at_0_5)
            ax.axhline(rate, color=REFERENCE_COLOUR, linewidth=0.9, linestyle="--")
            ax.annotate(
                f"no skill ({rate:.2%} fraud)",
                (0.02, rate),
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=7,
                color=REFERENCE_COLOUR,
            )
            ax.set(
                title=MODEL_LABELS[model],
                xlim=(0, 1),
                ylim=(0, 1.02),
                xlabel="Recall (share of frauds caught)",
            )
        axes[0].set_ylabel("Precision (share of flags that are fraud)")
        fig.suptitle("Precision-recall curves on the test set")
        _technique_legend(fig, axes[0])
        return _save(fig, path)


def plot_roc_curves(comparison: ComparisonTable, curves: CurvesFile, path: Path) -> Path:
    """1 x 3 panels of test-set ROC curves, one line per technique, with the chance diagonal.

    Args:
        comparison: Test results (which cells exist).
        curves: Test-set curves.
        path: Output PNG.

    Returns:
        ``path``.
    """
    _require(comparison.records)
    curve = {(c.model, c.technique): c for c in curves.cells}
    models = _models(comparison.records)
    with matplotlib.rc_context(FIGURE_STYLE):
        fig = _new_figure(3.3 * len(models), 3.8)
        axes = np.atleast_1d(fig.subplots(1, len(models), sharey=True))
        for ax, model in zip(axes, models, strict=True):
            for r in _cells(comparison.records, model):
                c = curve.get((r.model, r.technique))
                if c is None:
                    continue
                ax.plot(
                    c.roc.fpr,
                    c.roc.tpr,
                    color=TECHNIQUE_COLOURS[r.technique],
                    linestyle=TECHNIQUE_LINESTYLES[r.technique],
                    linewidth=CURVE_LINEWIDTH,
                    label=SHORT_LABELS[r.technique],
                )
            ax.plot([0, 1], [0, 1], color=REFERENCE_COLOUR, linewidth=0.9, linestyle="--")
            ax.set(
                title=MODEL_LABELS[model],
                xlim=(0, 1),
                ylim=(0, 1.02),
                xlabel="False positive rate (share of legitimate flagged)",
            )
        axes[0].set_ylabel("True positive rate (recall)")
        fig.suptitle("ROC curves on the test set (dashed: chance)")
        _technique_legend(fig, axes[0])
        return _save(fig, path)


# --- bars, matrices, thresholds -----------------------------------------------------------------


def plot_pr_auc_with_ci(comparison: ComparisonTable, confidence: float, path: Path) -> Path:
    """Grouped bars of test PR-AUC per model and technique with bootstrap CI error bars; a
    dashed segment per model marks the no-skill PR-AUC (the test fraud rate).

    Args:
        comparison: Test results.
        confidence: Coverage of ``pr_auc_ci`` (``bootstrap.confidence``), for the title.
        path: Output PNG.

    Returns:
        ``path``.
    """
    _require(comparison.records)
    models = _models(comparison.records)
    width = 0.8 / len(TECHNIQUE_ORDER)
    with matplotlib.rc_context(FIGURE_STYLE):
        fig = _new_figure(7.5, 3.8)
        ax = fig.subplots()
        for i, model in enumerate(models):
            cells = _cells(comparison.records, model)
            for r in cells:
                j = TECHNIQUE_ORDER.index(r.technique)
                x = i - 0.4 + width * (j + 0.5)
                value = r.test_at_tuned.pr_auc
                err = [[value - r.pr_auc_ci[0]], [r.pr_auc_ci[1] - value]]
                ax.bar(
                    x,
                    value,
                    width * 0.9,
                    color=TECHNIQUE_COLOURS[r.technique],
                    hatch=FRAUD_HATCH if j % 2 else None,
                    edgecolor="white",
                    linewidth=0.3,
                    label=SHORT_LABELS[r.technique] if i == 0 else None,
                )
                ax.errorbar(
                    x, value, yerr=np.clip(err, 0, None), color="#333333", capsize=2, linewidth=0.8
                )
            rate = fraud_rate(cells[0].test_at_0_5)
            ax.hlines(
                rate,
                i - 0.42,
                i + 0.42,
                colors=REFERENCE_COLOUR,
                linestyles="--",
                linewidth=0.9,
                label="no skill (fraud rate)" if i == 0 else None,
            )
        ax.set_xticks(range(len(models)), [MODEL_LABELS[m] for m in models])
        ax.set(ylim=(0, 1.05), ylabel=f"Test {PR_AUC_UNIT}")
        ax.set_title(f"Test PR-AUC with {confidence:.0%} bootstrap confidence intervals")
        fig.legend(loc="outside lower center", ncol=4)
        return _save(fig, path)


def plot_confusion_matrices(comparison: ComparisonTable, path: Path) -> Path:
    """Grid (models x techniques) of test confusion matrices at the tuned F1 threshold.

    Colour is the row-normalised share (so the small fraud row is as visible as the legitimate
    one); each square shows the count and the row share.

    Args:
        comparison: Test results.
        path: Output PNG.

    Returns:
        ``path``.
    """
    _require(comparison.records)
    models = _models(comparison.records)
    techniques = [t for t in TECHNIQUE_ORDER if any(r.technique == t for r in comparison.records)]
    cell = {(r.model, r.technique): r for r in comparison.records}
    with matplotlib.rc_context(FIGURE_STYLE):
        fig = _new_figure(1.75 * len(techniques) + 0.8, 1.75 * len(models) + 0.9)
        axes = np.atleast_2d(fig.subplots(len(models), len(techniques), squeeze=False))
        image = None
        for i, model in enumerate(models):
            for j, technique in enumerate(techniques):
                ax = axes[i, j]
                r = cell.get((model, technique))
                if r is None:
                    ax.set_axis_off()
                    continue
                m = r.test_at_tuned
                counts = np.array([[m.tn, m.fp], [m.fn, m.tp]], dtype=float)
                share = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
                image = ax.imshow(share, cmap=CONFUSION_CMAP, vmin=0.0, vmax=1.0)
                for (a, b), n in np.ndenumerate(counts):
                    colour = "black" if share[a, b] > CONFUSION_TEXT_SWITCH else "white"
                    ax.text(
                        b,
                        a,
                        f"{int(n):,}\n{share[a, b]:.1%}",
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color=colour,
                    )
                ax.set_xticks([0, 1], ["legit", "fraud"] if i == len(models) - 1 else ["", ""])
                ax.set_yticks([0, 1], ["legit", "fraud"] if j == 0 else ["", ""])
                ax.tick_params(length=0, labelsize=7)
                if i == 0:
                    ax.set_title(SHORT_LABELS[technique], fontsize=8)
                if j == 0:
                    ax.set_ylabel(f"{MODEL_LABELS[model]}\nTrue class", fontsize=8)
                if i == len(models) - 1:
                    ax.set_xlabel(f"Predicted (thr {r.threshold_f1:.2f})", fontsize=7)
                else:
                    ax.set_xlabel(f"thr {r.threshold_f1:.2f}", fontsize=7)
        if image is not None:
            bar = fig.colorbar(image, ax=axes, shrink=0.6, location="right")
            bar.set_label("Share of the true class (row)")
        fig.suptitle("Test confusion matrices at the tuned F1 threshold (counts and row share)")
        return _save(fig, path)


def plot_threshold_effect(comparison: ComparisonTable, path: Path) -> Path:
    """Per model, test F1 at the default 0.5 threshold vs at the tuned F1 threshold, for each
    technique (joined markers, so the size and direction of the change is visible).

    Args:
        comparison: Test results.
        path: Output PNG.

    Returns:
        ``path``.
    """
    _require(comparison.records)
    models = _models(comparison.records)
    with matplotlib.rc_context(FIGURE_STYLE):
        fig = _new_figure(3.3 * len(models), 3.4)
        axes = np.atleast_1d(fig.subplots(1, len(models), sharey=True))
        for ax, model in zip(axes, models, strict=True):
            cells = _cells(comparison.records, model)
            x = np.arange(len(cells))
            f1_default = [r.test_at_0_5.f1 for r in cells]
            f1_tuned = [r.test_at_tuned.f1 for r in cells]
            ax.vlines(x, f1_default, f1_tuned, colors=REFERENCE_COLOUR, linewidth=1.0)
            ax.scatter(
                x,
                f1_default,
                marker="o",
                facecolors="white",
                edgecolors="#000000",
                zorder=3,
                label="threshold 0.5",
            )
            ax.scatter(
                x,
                f1_tuned,
                marker="s",
                color="#0072B2",
                zorder=3,
                label="tuned threshold (F1 on out-of-fold training scores)",
            )
            ax.set_xticks(x, [SHORT_LABELS[r.technique] for r in cells], rotation=40, ha="right")
            ax.set(title=MODEL_LABELS[model], ylim=(0, 1.02))
        axes[0].set_ylabel("Test F1 score (0-1)")
        fig.suptitle("Effect of threshold tuning on test F1")
        _technique_legend(fig, axes[0], ncol=2)
        return _save(fig, path)


# --- statistics ---------------------------------------------------------------------------------


def cliques(ranks: Sequence[float], cd: float) -> list[tuple[int, int]]:
    """Maximal runs of rank-sorted techniques whose ranks span less than ``cd``.

    Args:
        ranks: Average ranks sorted ascending.
        cd: Critical difference.

    Returns:
        ``(first, last)`` index pairs (``first < last``) of each maximal group; techniques in a
        group are not significantly different by the Nemenyi test.
    """
    groups: list[tuple[int, int]] = []
    for i in range(len(ranks)):
        j = i
        while j + 1 < len(ranks) and ranks[j + 1] - ranks[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in groups):
            groups.append((i, j))
    return groups


def plot_cd_diagram(test: FriedmanTest, path: Path) -> Path:
    """Critical-difference diagram (Demsar 2006) for one Friedman scope.

    Techniques sit at their average rank on an axis from 1 (best) to k; the CD bar shows the
    Nemenyi critical difference and thick bars join techniques that are not significantly
    different (ranks within one CD).

    Args:
        test: Friedman test with ``average_ranks`` and ``critical_difference``.
        path: Output PNG.

    Returns:
        ``path``.
    """
    items = sorted(test.average_ranks.items(), key=lambda kv: kv[1])
    names = [t for t, _ in items]
    ranks = [r for _, r in items]
    k = len(items)
    cd = test.critical_difference
    half = (k + 1) // 2
    scope = MODEL_LABELS.get(test.scope, "all models pooled")
    with matplotlib.rc_context(FIGURE_STYLE):
        fig = _new_figure(6.5, 1.0 + 0.32 * half + 0.6)
        ax = fig.subplots()
        ax.set_axis_off()
        axis_y = 0.0
        ax.hlines(axis_y, 1, k, color="black", linewidth=1.0)
        for tick in range(1, k + 1):
            ax.vlines(tick, axis_y, axis_y + 0.08, color="black", linewidth=1.0)
            ax.text(tick, axis_y + 0.14, str(tick), ha="center", va="bottom", fontsize=8)
        ax.text(
            (1 + k) / 2,
            axis_y + 0.42,
            "Average rank (1 = highest CV PR-AUC)",
            ha="center",
            fontsize=8,
        )
        if cd is not None:
            ax.hlines(axis_y + 0.7, 1, 1 + cd, color="black", linewidth=1.5)
            for end in (1, 1 + cd):
                ax.vlines(end, axis_y + 0.65, axis_y + 0.75, color="black", linewidth=1.0)
            ax.text(1 + cd / 2, axis_y + 0.8, f"CD = {cd:.2f}", ha="center", fontsize=8)
        step = 0.32
        for idx, (name, rank) in enumerate(zip(names, ranks, strict=True)):
            left = idx < half
            row = idx if left else k - 1 - idx
            y = axis_y - 0.35 - step * row
            x_end = 0.6 if left else k + 0.4
            ax.vlines(rank, y, axis_y, color=TECHNIQUE_COLOURS[name], linewidth=1.0)
            ax.hlines(
                y, min(rank, x_end), max(rank, x_end), color=TECHNIQUE_COLOURS[name], linewidth=1.0
            )
            ax.text(
                x_end + (-0.05 if left else 0.05),
                y,
                f"{SHORT_LABELS[name]} ({rank:.2f})",
                ha="right" if left else "left",
                va="center",
                fontsize=8,
            )
        if cd is not None:
            for g, (a, b) in enumerate(cliques(ranks, cd)):
                y = axis_y - 0.12 - 0.07 * g
                ax.hlines(y, ranks[a] - 0.03, ranks[b] + 0.03, color="black", linewidth=3.0)
        ax.set_xlim(-1.4, k + 2.4)
        ax.set_ylim(axis_y - 0.45 - step * half, axis_y + 1.05)
        ax.set_title(
            f"Critical-difference diagram, {scope} "
            f"(Friedman p = {test.p_value:.3g}, N = {test.n_blocks} blocks)"
        )
        return _save(fig, path)


def plot_fold_boxplots(fold_scores: pd.DataFrame, path: Path) -> Path:
    """1 x 3 panels of repeated-CV PR-AUC per technique (box plus the individual folds).

    Args:
        fold_scores: Rows of ``fold_scores.csv``.
        path: Output PNG.

    Returns:
        ``path``.

    Raises:
        DataValidationError: If there are no fold scores.
    """
    if fold_scores.empty:
        raise DataValidationError("fold_scores.csv is empty; run scripts/run_experiments.py.")
    models = [m for m in MODEL_ORDER if (fold_scores["model"] == m).any()]
    rng = np.random.default_rng(0)  # jitter only, not part of any result
    with matplotlib.rc_context(FIGURE_STYLE):
        fig = _new_figure(3.3 * len(models), 3.6)
        axes = np.atleast_1d(fig.subplots(1, len(models), sharey=True))
        for ax, model in zip(axes, models, strict=True):
            rows = fold_scores[fold_scores["model"] == model]
            techniques = [t for t in TECHNIQUE_ORDER if (rows["technique"] == t).any()]
            data = [rows.loc[rows["technique"] == t, "pr_auc"].to_numpy() for t in techniques]
            box = ax.boxplot(
                data,
                positions=range(len(data)),
                widths=0.6,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black"},
            )
            for patch, t in zip(box["boxes"], techniques, strict=True):
                patch.set(facecolor=TECHNIQUE_COLOURS[t], alpha=0.35)
            for x, (t, values) in enumerate(zip(techniques, data, strict=True)):
                ax.scatter(
                    x + rng.uniform(-0.15, 0.15, len(values)),
                    values,
                    s=8,
                    color=TECHNIQUE_COLOURS[t],
                    marker=TECHNIQUE_MARKERS[t],
                    zorder=3,
                )
            ax.set_xticks(
                range(len(techniques)),
                [SHORT_LABELS[t] for t in techniques],
                rotation=40,
                ha="right",
            )
            ax.set_title(MODEL_LABELS[model])
        n_folds = int(fold_scores.groupby(["model", "technique"]).size().max())
        axes[0].set_ylabel(f"CV {PR_AUC_UNIT}")
        fig.suptitle(f"Repeated cross-validation PR-AUC per technique ({n_folds} folds per cell)")
        return _save(fig, path)


def resampling_counts_table(audit: ResamplingAudit) -> pd.DataFrame:
    """Mean training-fold class counts per resampler and ratio, from the resampling audit.

    Args:
        audit: The resampling audit.

    Returns:
        Columns ``technique, ratio, smote_ratio, legit_before, fraud_before, legit_after,
        fraud_after`` (fold means), one row per configuration.
    """
    rows = [
        {
            "technique": e.technique,
            "ratio": e.ratio,
            "smote_ratio": e.smote_ratio,
            "legit_before": e.train_before.legit,
            "fraud_before": e.train_before.fraud,
            "legit_after": e.train_after.legit,
            "fraud_after": e.train_after.fraud,
        }
        for e in audit.entries
        if e.technique in RESAMPLERS
    ]
    if not rows:
        raise DataValidationError(
            "resampling_audit.json has no resampler entries; run scripts/run_resampling_audit.py."
        )
    df = pd.DataFrame(rows)
    df["smote_ratio"] = df["smote_ratio"].fillna(0.0)
    keys = ["technique", "ratio", "smote_ratio"]
    return df.groupby(keys, as_index=False).mean().sort_values(keys, ignore_index=True)


def plot_resampling_counts(audit: ResamplingAudit, path: Path) -> Path:
    """Training-fold class counts after each resampler at each ratio, log scale; dashed lines
    mark the counts before resampling. Baseline and class weighting leave the data unchanged.

    Args:
        audit: The resampling audit.
        path: Output PNG.

    Returns:
        ``path``.
    """
    table = resampling_counts_table(audit)
    techniques = [t for t in RESAMPLERS if (table["technique"] == t).any()]
    with matplotlib.rc_context(FIGURE_STYLE):
        fig = _new_figure(2.7 * len(techniques) + 0.4, 3.6)
        axes = np.atleast_1d(fig.subplots(1, len(techniques), sharey=True))
        for ax, technique in zip(axes, techniques, strict=True):
            rows = table[table["technique"] == technique]
            x = np.arange(len(rows))
            ax.bar(x - 0.2, rows["legit_after"], 0.4, color=LEGIT_COLOUR, label="Legitimate")
            ax.bar(
                x + 0.2,
                rows["fraud_after"],
                0.4,
                color=FRAUD_COLOUR,
                hatch=FRAUD_HATCH,
                edgecolor="white",
                linewidth=0.3,
                label="Fraud",
            )
            ax.axhline(
                rows["legit_before"].iloc[0],
                color=LEGIT_COLOUR,
                linestyle="--",
                linewidth=0.8,
                label="Legitimate before",
            )
            ax.axhline(
                rows["fraud_before"].iloc[0],
                color=FRAUD_COLOUR,
                linestyle=":",
                linewidth=0.9,
                label="Fraud before",
            )
            if technique == "smote_rus":
                labels = [
                    f"{a:g} / {b:g}"
                    for a, b in zip(rows["smote_ratio"], rows["ratio"], strict=True)
                ]
                ax.set_xlabel("SMOTE ratio a / then RUS ratio b")
            else:
                labels = [f"{r:g}" for r in rows["ratio"]]
                ax.set_xlabel("Target ratio r (fraud / legitimate)")
            if technique == "smote_rus":
                ax.set_xticks(x, labels, fontsize=7, rotation=60, ha="right")
            else:
                ax.set_xticks(x, labels, fontsize=8)
            ax.set_yscale("log")
            ax.set_title(SHORT_LABELS[technique])
        axes[0].set_ylabel("Rows in a training fold (log scale, fold mean)")
        fig.suptitle(
            "Training-fold class counts after resampling (A and F leave the data unchanged)"
        )
        _technique_legend(fig, axes[0], ncol=4)
        return _save(fig, path)


# --- everything ---------------------------------------------------------------------------------


def make_all_figures(
    comparison: ComparisonTable,
    curves: CurvesFile,
    stats: StatsResult,
    fold_scores: pd.DataFrame,
    audit: ResamplingAudit | None,
    out_dir: Path,
) -> dict[str, Path]:
    """Write every report figure into ``out_dir``.

    Args:
        comparison, curves: Test-set results.
        stats: Friedman / Wilcoxon results (one CD diagram per Friedman scope).
        fold_scores: Rows of ``fold_scores.csv``.
        audit: Resampling audit, or ``None`` to skip ``resampling_counts.png``.
        out_dir: Output folder.

    Returns:
        Figure file name -> path written.
    """
    out_dir = Path(out_dir)
    written = {
        "pr_curves_by_model.png": plot_pr_curves(
            comparison, curves, out_dir / "pr_curves_by_model.png"
        ),
        "roc_curves_by_model.png": plot_roc_curves(
            comparison, curves, out_dir / "roc_curves_by_model.png"
        ),
        "pr_auc_with_ci.png": plot_pr_auc_with_ci(
            comparison, stats.confidence, out_dir / "pr_auc_with_ci.png"
        ),
        "confusion_matrices_tuned.png": plot_confusion_matrices(
            comparison, out_dir / "confusion_matrices_tuned.png"
        ),
        "threshold_effect.png": plot_threshold_effect(comparison, out_dir / "threshold_effect.png"),
        "fold_score_boxplots.png": plot_fold_boxplots(
            fold_scores, out_dir / "fold_score_boxplots.png"
        ),
    }
    if audit is not None:
        written["resampling_counts.png"] = plot_resampling_counts(
            audit, out_dir / "resampling_counts.png"
        )
    for test in stats.friedman:
        name = f"cd_diagram_{test.scope}.png"
        written[name] = plot_cd_diagram(test, out_dir / name)
    return written
