"""Every report figure, drawn from the saved result files (nothing is fitted or scored here).

scripts/make_figures.py calls make_all_figures() with comparison.json, curves.json, stats.json,
fold_scores.csv and resampling_audit.json; it writes 300-dpi PNGs. Techniques use Okabe-Ito
colours (colour-blind safe) AND differ by line style or marker, and fraud bars are hatched, so
everything still reads in greyscale. Axes carry units.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from fraudstudy.config import resolve
from fraudstudy.eda import FIGURE_DPI, FIGURE_STYLE
from fraudstudy.experiment import MODEL_LABELS, TECHNIQUE_LABELS

# Per technique: Okabe-Ito colour, line style, marker and a short legend label.
STYLE = {
    "baseline": ("#000000", "-", "o", "A baseline"),
    "ros": ("#E69F00", "--", "s", "B ROS"),
    "rus": ("#56B4E9", "-.", "^", "C RUS"),
    "smote": ("#009E73", ":", "D", "D SMOTE"),
    "smote_rus": ("#CC79A7", (0, (5, 1, 1, 1, 1, 1)), "v", "E SMOTE+RUS"),
    "class_weight": ("#D55E00", (0, (3, 1)), "P", "F weights"),
}
GREY, LEGIT_COLOUR, FRAUD_COLOUR, FRAUD_HATCH = "#808080", "#0072B2", "#D55E00", "////"
PR_AUC_UNIT = "PR-AUC (average precision, 0-1)"
FIGURE_NAMES = [
    "pr_curves_by_model.png", "roc_curves_by_model.png", "pr_auc_with_ci.png",
    "confusion_matrices_tuned.png", "threshold_effect.png", "resampling_counts.png",
    "fold_score_boxplots.png",
]  # plus cd_diagram_<scope>.png for each Friedman scope


def figures_dir_for(source: str) -> Path:
    """artifacts/figures for the real data, artifacts/figures/sample for the sample."""
    folder = resolve("artifacts/figures")
    return folder if source == "raw" else folder / "sample"


def save(fig: Figure, path: Path) -> Path:
    """Save a figure as a 300-dpi PNG, creating the folder."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI)
    return Path(path)


def cells_by_model(comparison: dict) -> dict:
    """{model: that model's comparison records in technique order}, models in report order."""
    if not comparison["records"]:
        raise ValueError("comparison.json has no records; run scripts/evaluate_test.py before "
                         "the figures.")
    out = {}
    for model in MODEL_LABELS:
        by_technique = {r["technique"]: r for r in comparison["records"] if r["model"] == model}
        if by_technique:
            out[model] = [by_technique[t] for t in TECHNIQUE_LABELS if t in by_technique]
    return out


def fraud_rate(metrics: dict) -> float:
    """Share of fraud in the scored set: the precision of flagging everyone (PR no-skill line)."""
    total = metrics["tp"] + metrics["fp"] + metrics["tn"] + metrics["fn"]
    return (metrics["tp"] + metrics["fn"]) / total if total else 0.0


def plot_curves(comparison: dict, curves: dict, kind: str, path: Path) -> Path:
    """One panel per model of test-set PR curves (kind "pr") or ROC curves (kind "roc").

    The PR panels get the no-skill line at the test fraud rate, the ROC panels the chance
    diagonal.
    """
    by_model = cells_by_model(comparison)
    curve = {(c["model"], c["technique"]): c[kind] for c in curves["cells"]}
    with matplotlib.rc_context(FIGURE_STYLE):
        fig = Figure(figsize=(3.3 * len(by_model), 3.8), layout="constrained")
        axes = np.atleast_1d(fig.subplots(1, len(by_model), sharey=True))
        for ax, (model, records) in zip(axes, by_model.items(), strict=True):
            for r in records:
                points = curve.get((model, r["technique"]))
                if points is None:
                    continue
                x, y = (points["recall"], points["precision"]) if kind == "pr" else \
                    (points["fpr"], points["tpr"])
                colour, line, _, label = STYLE[r["technique"]]
                ax.plot(x, y, color=colour, linestyle=line, linewidth=1.3, label=label)
            if kind == "pr":
                rate = fraud_rate(records[0]["test_at_0_5"])
                ax.axhline(rate, color=GREY, linewidth=0.9, linestyle="--")
                ax.annotate(f"no skill ({rate:.2%} fraud)", (0.02, rate), xytext=(0, 3),
                            textcoords="offset points", fontsize=7, color=GREY)
                ax.set_xlabel("Recall (share of frauds caught)")
            else:
                ax.plot([0, 1], [0, 1], color=GREY, linewidth=0.9, linestyle="--")
                ax.set_xlabel("False positive rate (share of legitimate flagged)")
            ax.set(title=MODEL_LABELS[model], xlim=(0, 1), ylim=(0, 1.02))
        if kind == "pr":
            axes[0].set_ylabel("Precision (share of flags that are fraud)")
            fig.suptitle("Precision-recall curves on the test set")
        else:
            axes[0].set_ylabel("True positive rate (recall)")
            fig.suptitle("ROC curves on the test set (dashed: chance)")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=6)
        return save(fig, path)


def plot_pr_auc_with_ci(comparison: dict, confidence: float, path: Path) -> Path:
    """Bars of test PR-AUC per model and technique with bootstrap CI error bars; a dashed
    segment per model marks the no-skill PR-AUC (the test fraud rate)."""
    by_model = cells_by_model(comparison)
    width = 0.8 / len(TECHNIQUE_LABELS)
    with matplotlib.rc_context(FIGURE_STYLE):
        fig = Figure(figsize=(7.5, 3.8), layout="constrained")
        ax = fig.subplots()
        for i, records in enumerate(by_model.values()):
            for r in records:
                j = list(TECHNIQUE_LABELS).index(r["technique"])
                x, value = i - 0.4 + width * (j + 0.5), r["test_at_tuned"]["pr_auc"]
                colour, _, _, label = STYLE[r["technique"]]
                # Every other technique is hatched so neighbouring bars differ without colour.
                ax.bar(x, value, width * 0.9, color=colour, hatch=FRAUD_HATCH if j % 2 else None,
                       edgecolor="white", linewidth=0.3, label=label if i == 0 else None)
                error = [[max(value - r["pr_auc_ci"][0], 0)], [max(r["pr_auc_ci"][1] - value, 0)]]
                ax.errorbar(x, value, yerr=error, color="#333333", capsize=2, linewidth=0.8)
            ax.hlines(fraud_rate(records[0]["test_at_0_5"]), i - 0.42, i + 0.42, colors=GREY,
                      linestyles="--", linewidth=0.9,
                      label="no skill (fraud rate)" if i == 0 else None)
        ax.set_xticks(range(len(by_model)), [MODEL_LABELS[m] for m in by_model])
        ax.set(ylim=(0, 1.05), ylabel=f"Test {PR_AUC_UNIT}")
        ax.set_title(f"Test PR-AUC with {confidence:.0%} bootstrap confidence intervals")
        fig.legend(loc="outside lower center", ncol=4)
        return save(fig, path)


def plot_confusion_matrices(comparison: dict, path: Path) -> Path:
    """Grid (models x techniques) of test confusion matrices at the tuned F1 threshold.

    Colour is the share of the true class (row), so the small fraud row is as visible as the
    legitimate one; each square also prints the count and the share.
    """
    by_model = cells_by_model(comparison)
    techniques = [t for t in TECHNIQUE_LABELS
                  if any(r["technique"] == t for r in comparison["records"])]
    with matplotlib.rc_context(FIGURE_STYLE):
        fig = Figure(figsize=(1.75 * len(techniques) + 0.8, 1.75 * len(by_model) + 0.9),
                     layout="constrained")
        axes = fig.subplots(len(by_model), len(techniques), squeeze=False)
        image, last_row = None, len(by_model) - 1
        for i, (model, records) in enumerate(by_model.items()):
            by_technique = {r["technique"]: r for r in records}
            for j, technique in enumerate(techniques):
                ax, r = axes[i, j], by_technique.get(technique)
                if r is None:
                    ax.set_axis_off()
                    continue
                m = r["test_at_tuned"]
                counts = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], dtype=float)
                share = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
                image = ax.imshow(share, cmap="cividis", vmin=0.0, vmax=1.0)
                for (a, b), n in np.ndenumerate(counts):
                    # cividis is light above 0.6, so the text turns dark there.
                    ax.text(b, a, f"{int(n):,}\n{share[a, b]:.1%}", ha="center", va="center",
                            fontsize=6.5, color="black" if share[a, b] > 0.6 else "white")
                ax.set_xticks([0, 1], ["legit", "fraud"] if i == last_row else ["", ""])
                ax.set_yticks([0, 1], ["legit", "fraud"] if j == 0 else ["", ""])
                ax.tick_params(length=0, labelsize=7)
                if i == 0:
                    ax.set_title(STYLE[technique][3], fontsize=8)
                if j == 0:
                    ax.set_ylabel(f"{MODEL_LABELS[model]}\nTrue class", fontsize=8)
                prefix = "Predicted (" if i == last_row else ""
                suffix = ")" if i == last_row else ""
                ax.set_xlabel(f"{prefix}thr {r['threshold_f1']:.2f}{suffix}", fontsize=7)
        if image is not None:
            fig.colorbar(image, ax=axes, shrink=0.6, location="right").set_label(
                "Share of the true class (row)")
        fig.suptitle("Test confusion matrices at the tuned F1 threshold (counts and row share)")
        return save(fig, path)


def plot_threshold_effect(comparison: dict, path: Path) -> Path:
    """Per model, test F1 at the default 0.5 threshold vs at the tuned F1 threshold."""
    by_model = cells_by_model(comparison)
    with matplotlib.rc_context(FIGURE_STYLE):
        fig = Figure(figsize=(3.3 * len(by_model), 3.4), layout="constrained")
        axes = np.atleast_1d(fig.subplots(1, len(by_model), sharey=True))
        for ax, (model, records) in zip(axes, by_model.items(), strict=True):
            x = np.arange(len(records))
            f1_default = [r["test_at_0_5"]["f1"] for r in records]
            f1_tuned = [r["test_at_tuned"]["f1"] for r in records]
            ax.vlines(x, f1_default, f1_tuned, colors=GREY, linewidth=1.0)
            ax.scatter(x, f1_default, marker="o", facecolors="white", edgecolors="#000000",
                       zorder=3, label="threshold 0.5")
            ax.scatter(x, f1_tuned, marker="s", color="#0072B2", zorder=3,
                       label="tuned threshold (F1 on out-of-fold training scores)")
            ax.set_xticks(x, [STYLE[r["technique"]][3] for r in records], rotation=40, ha="right")
            ax.set(title=MODEL_LABELS[model], ylim=(0, 1.02))
        axes[0].set_ylabel("Test F1 score (0-1)")
        fig.suptitle("Effect of threshold tuning on test F1")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2)
        return save(fig, path)


def cliques(ranks: list, cd: float) -> list:
    """(first, last) index pairs of the longest runs of sorted ranks that span less than cd:
    the techniques the Nemenyi test cannot tell apart."""
    groups = []
    for i in range(len(ranks)):
        j = i
        while j + 1 < len(ranks) and ranks[j + 1] - ranks[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in groups):
            groups.append((i, j))
    return groups


def plot_cd_diagram(test: dict, path: Path) -> Path:
    """Critical-difference diagram (Demsar 2006) for one Friedman scope.

    Techniques sit at their average rank on an axis from 1 (best) to k; the top bar is one
    critical difference long, and thick bars join techniques within one CD of each other.
    """
    items = sorted(test["average_ranks"].items(), key=lambda kv: kv[1])
    names, ranks = [t for t, _ in items], [r for _, r in items]
    k, cd, half = len(items), test["critical_difference"], (len(items) + 1) // 2
    with matplotlib.rc_context(FIGURE_STYLE):
        fig = Figure(figsize=(6.5, 1.0 + 0.32 * half + 0.6), layout="constrained")
        ax = fig.subplots()
        ax.set_axis_off()
        ax.hlines(0.0, 1, k, color="black", linewidth=1.0)
        for tick in range(1, k + 1):
            ax.vlines(tick, 0.0, 0.08, color="black", linewidth=1.0)
            ax.text(tick, 0.14, str(tick), ha="center", va="bottom", fontsize=8)
        ax.text((1 + k) / 2, 0.42, "Average rank (1 = highest CV PR-AUC)", ha="center",
                fontsize=8)
        ax.hlines(0.7, 1, 1 + cd, color="black", linewidth=1.5)
        ax.vlines([1, 1 + cd], 0.65, 0.75, color="black", linewidth=1.0)
        ax.text(1 + cd / 2, 0.8, f"CD = {cd:.2f}", ha="center", fontsize=8)
        # The best half is labelled on the left, the rest on the right.
        for i, (name, rank) in enumerate(zip(names, ranks, strict=True)):
            left = i < half
            y = -0.35 - 0.32 * (i if left else k - 1 - i)
            x_end = 0.6 if left else k + 0.4
            colour = STYLE[name][0]
            ax.vlines(rank, y, 0.0, color=colour, linewidth=1.0)
            ax.hlines(y, min(rank, x_end), max(rank, x_end), color=colour, linewidth=1.0)
            ax.text(x_end + (-0.05 if left else 0.05), y, f"{STYLE[name][3]} ({rank:.2f})",
                    ha="right" if left else "left", va="center", fontsize=8)
        for g, (a, b) in enumerate(cliques(ranks, cd)):
            ax.hlines(-0.12 - 0.07 * g, ranks[a] - 0.03, ranks[b] + 0.03, color="black",
                      linewidth=3.0)
        ax.set_xlim(-1.4, k + 2.4)
        ax.set_ylim(-0.45 - 0.32 * half, 1.05)
        scope = MODEL_LABELS.get(test["scope"], "all models pooled")
        ax.set_title(f"Critical-difference diagram, {scope} (Friedman p = {test['p_value']:.3g}, "
                     f"N = {test['n_blocks']} blocks)")
        return save(fig, path)


def plot_fold_boxplots(fold_scores: pd.DataFrame, path: Path) -> Path:
    """One panel per model: repeated-CV PR-AUC per technique (box plus the individual folds)."""
    if fold_scores.empty:
        raise ValueError("fold_scores.csv is empty; run scripts/run_experiments.py.")
    models = [m for m in MODEL_LABELS if (fold_scores["model"] == m).any()]
    rng = np.random.default_rng(0)  # sideways jitter of the dots only, not part of any result
    with matplotlib.rc_context(FIGURE_STYLE):
        fig = Figure(figsize=(3.3 * len(models), 3.6), layout="constrained")
        axes = np.atleast_1d(fig.subplots(1, len(models), sharey=True))
        for ax, model in zip(axes, models, strict=True):
            rows = fold_scores[fold_scores["model"] == model]
            techniques = [t for t in TECHNIQUE_LABELS if (rows["technique"] == t).any()]
            data = [rows.loc[rows["technique"] == t, "pr_auc"].to_numpy() for t in techniques]
            box = ax.boxplot(data, positions=range(len(data)), widths=0.6, patch_artist=True,
                             showfliers=False, medianprops={"color": "black"})
            for x, (patch, technique, values) in enumerate(zip(box["boxes"], techniques, data,
                                                                strict=True)):
                colour, _, marker, _ = STYLE[technique]
                patch.set(facecolor=colour, alpha=0.35)
                ax.scatter(x + rng.uniform(-0.15, 0.15, len(values)), values, s=8, color=colour,
                           marker=marker, zorder=3)
            ax.set_xticks(range(len(techniques)), [STYLE[t][3] for t in techniques], rotation=40,
                          ha="right")
            ax.set_title(MODEL_LABELS[model])
        n_folds = int(fold_scores.groupby(["model", "technique"]).size().max())
        axes[0].set_ylabel(f"CV {PR_AUC_UNIT}")
        fig.suptitle(f"Repeated cross-validation PR-AUC per technique ({n_folds} folds per cell)")
        return save(fig, path)


def resampling_counts_table(audit: dict) -> pd.DataFrame:
    """Fold-mean training class counts before and after each resampler, per ratio."""
    # smote_ratio is optional in the contract (older audit files lack it).
    rows = [{"technique": e["technique"], "ratio": e["ratio"], "smote_ratio": e.get("smote_ratio"),
             "legit_before": e["train_before"]["legit"], "fraud_before": e["train_before"]["fraud"],
             "legit_after": e["train_after"]["legit"], "fraud_after": e["train_after"]["fraud"]}
            for e in audit["entries"] if e["technique"] in ("ros", "rus", "smote", "smote_rus")]
    if not rows:
        raise ValueError("resampling_audit.json has no resampler entries; run "
                         "scripts/run_resampling_audit.py.")
    df = pd.DataFrame(rows)
    df["smote_ratio"] = df["smote_ratio"].fillna(0.0)
    keys = ["technique", "ratio", "smote_ratio"]
    return df.groupby(keys, as_index=False).mean().sort_values(keys, ignore_index=True)


def plot_resampling_counts(audit: dict, path: Path) -> Path:
    """Training-fold class counts after each resampler at each ratio (log scale); dashed and
    dotted lines mark the counts before. Baseline and class weighting leave the data unchanged."""
    table = resampling_counts_table(audit)
    techniques = [t for t in ("ros", "rus", "smote", "smote_rus")
                  if (table["technique"] == t).any()]
    with matplotlib.rc_context(FIGURE_STYLE):
        fig = Figure(figsize=(2.7 * len(techniques) + 0.4, 3.6), layout="constrained")
        axes = np.atleast_1d(fig.subplots(1, len(techniques), sharey=True))
        for ax, technique in zip(axes, techniques, strict=True):
            rows = table[table["technique"] == technique]
            x = np.arange(len(rows))
            ax.bar(x - 0.2, rows["legit_after"], 0.4, color=LEGIT_COLOUR, label="Legitimate")
            ax.bar(x + 0.2, rows["fraud_after"], 0.4, color=FRAUD_COLOUR, hatch=FRAUD_HATCH,
                   edgecolor="white", linewidth=0.3, label="Fraud")
            ax.axhline(rows["legit_before"].iloc[0], color=LEGIT_COLOUR, linestyle="--",
                       linewidth=0.8, label="Legitimate before")
            ax.axhline(rows["fraud_before"].iloc[0], color=FRAUD_COLOUR, linestyle=":",
                       linewidth=0.9, label="Fraud before")
            if technique == "smote_rus":
                labels = [f"{a:g} / {b:g}" for a, b in zip(rows["smote_ratio"], rows["ratio"],
                                                            strict=True)]
                ax.set_xticks(x, labels, fontsize=7, rotation=60, ha="right")
                ax.set_xlabel("SMOTE ratio a / then RUS ratio b")
            else:
                ax.set_xticks(x, [f"{r:g}" for r in rows["ratio"]], fontsize=8)
                ax.set_xlabel("Target ratio r (fraud / legitimate)")
            ax.set_yscale("log")
            ax.set_title(STYLE[technique][3])
        axes[0].set_ylabel("Rows in a training fold (log scale, fold mean)")
        fig.suptitle("Training-fold class counts after resampling (A and F leave the data "
                     "unchanged)")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4)
        return save(fig, path)


def make_all_figures(comparison: dict, curves: dict, stats: dict, fold_scores: pd.DataFrame,
                     audit: dict | None, out_dir: Path) -> dict:
    """Write every report figure into out_dir; returns {file name: path}.

    resampling_counts.png is skipped when audit is None; one cd_diagram_<scope>.png is drawn
    per Friedman test in stats.
    """
    out_dir = Path(out_dir)
    written = {
        "pr_curves_by_model.png": plot_curves(comparison, curves, "pr",
                                              out_dir / "pr_curves_by_model.png"),
        "roc_curves_by_model.png": plot_curves(comparison, curves, "roc",
                                               out_dir / "roc_curves_by_model.png"),
        "pr_auc_with_ci.png": plot_pr_auc_with_ci(comparison, stats["confidence"],
                                                  out_dir / "pr_auc_with_ci.png"),
        "confusion_matrices_tuned.png": plot_confusion_matrices(
            comparison, out_dir / "confusion_matrices_tuned.png"),
        "threshold_effect.png": plot_threshold_effect(comparison, out_dir / "threshold_effect.png"),
        "fold_score_boxplots.png": plot_fold_boxplots(fold_scores,
                                                      out_dir / "fold_score_boxplots.png"),
    }
    if audit is not None:
        written["resampling_counts.png"] = plot_resampling_counts(
            audit, out_dir / "resampling_counts.png")
    for test in stats["friedman"]:
        name = f"cd_diagram_{test['scope']}.png"
        written[name] = plot_cd_diagram(test, out_dir / name)
    return written
