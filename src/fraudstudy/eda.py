"""Descriptive EDA: the ``eda.json`` summary (``EdaSummary``) and the report figures.

Everything here is descriptive. It runs on the full deduplicated dataset and nothing it computes
feeds model, ratio or threshold choices (those use training folds only).

Figures are drawn with ``matplotlib.figure.Figure`` (no pyplot), which always renders with the
non-interactive Agg canvas, so no display is needed. Colours are Okabe-Ito (colour-blind safe), and
the classes also differ by line style and hatch, so they can be told apart in greyscale.
"""

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from scipy import stats

from fraudstudy.contracts import (
    HOUR_ASSUMPTION,
    HOURS_PER_DAY,
    AmountStats,
    AmountSummary,
    ClassCounts,
    ClassSplit,
    DatasetSource,
    Distribution,
    EdaSummary,
    EffectSize,
    FeatureHistogram,
    FeatureSeparation,
    Histogram,
    HourOfDay,
)
from fraudstudy.errors import DataValidationError

N_BINS = 20
N_TOP_FEATURES = 6
FEATURE_BIN_QUANTILES = (0.005, 0.995)  # V features: bins span each class's 0.5-99.5th pct.
FULL_RANGE = (0.0, 1.0)
SECONDS_PER_HOUR = 3600
LOG_AMOUNT = "log1p(Amount)"

FIGURE_DPI = 300
LEGIT_COLOUR = "#0072B2"  # Okabe-Ito blue
FRAUD_COLOUR = "#D55E00"  # Okabe-Ito vermillion
FRAUD_HATCH = "////"
LEGIT_STYLE = {"color": LEGIT_COLOUR, "linestyle": "-", "label": "Legitimate"}
FRAUD_STYLE = {"color": FRAUD_COLOUR, "linestyle": "--", "label": "Fraud"}
AMOUNT_UNIT = "Amount (dataset units, believed EUR)"
PCA_UNIT = "(PCA component, unitless)"
FIGURE_STYLE: dict[str, Any] = {
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
}
FIGURE_NAMES: tuple[str, ...] = (
    "eda_class_counts_linear_vs_log.png",
    "eda_amount_raw_vs_log.png",
    "eda_top_features_by_class.png",
    "eda_effect_sizes.png",
)


# --- numbers ------------------------------------------------------------------------------------


def _finite(x: float) -> float:
    """Return ``x`` as a float, or 0.0 if it is NaN/inf (e.g. skewness of a constant column)."""
    x = float(x)
    return x if np.isfinite(x) else 0.0


def cohens_d(fraud: np.ndarray, legit: np.ndarray) -> float:
    """Cohen's d between two samples, using the pooled standard deviation.

    ``d = (mean_f - mean_l) / s_p`` with
    ``s_p = sqrt(((n_f - 1) var_f + (n_l - 1) var_l) / (n_f + n_l - 2))`` (``ddof=1`` variances).

    Args:
        fraud: Values of the fraud class.
        legit: Values of the legitimate class.

    Returns:
        The effect size; 0.0 if the pooled standard deviation is 0 or undefined.
    """
    n_f, n_l = len(fraud), len(legit)
    if n_f + n_l <= 2:
        return 0.0
    var_f = float(np.var(fraud, ddof=1)) if n_f > 1 else 0.0
    var_l = float(np.var(legit, ddof=1)) if n_l > 1 else 0.0
    pooled = np.sqrt(((n_f - 1) * var_f + (n_l - 1) * var_l) / (n_f + n_l - 2))
    if not pooled > 0:
        return 0.0
    return _finite((float(np.mean(fraud)) - float(np.mean(legit))) / pooled)


def imbalance_ratio(class_counts: ClassCounts) -> str:
    """Format the class balance as ``"1 : N"`` with ``N = round(legit / fraud)``.

    Args:
        class_counts: Rows per class; must contain at least one fraud.

    Returns:
        e.g. ``"1 : 578"`` for the real dataset.
    """
    if class_counts.fraud == 0:
        raise DataValidationError("Cannot compute an imbalance ratio with zero fraud rows.")
    return f"1 : {round(class_counts.legit / class_counts.fraud)}"


def _class_counts(y: pd.Series) -> ClassCounts:
    """Count legitimate (0) and fraud (1) labels."""
    return ClassCounts(legit=int((y == 0).sum()), fraud=int((y == 1).sum()))


def _distribution(x: pd.Series) -> Distribution:
    """Five-number summary, mean, std (``ddof=1``) and zero count of one column."""
    q1, median, q3 = (float(v) for v in x.quantile([0.25, 0.5, 0.75]))
    return Distribution(
        count=int(x.size),
        mean=float(x.mean()),
        std=_finite(x.std(ddof=1)) if x.size > 1 else 0.0,
        min=float(x.min()),
        q1=q1,
        median=median,
        q3=q3,
        max=float(x.max()),
        n_zero=int((x == 0).sum()),
    )


def _amount_stats(x: pd.Series) -> AmountStats:
    """Min, median, mean, max and bias-corrected skewness (pandas ``skew``) of ``x``."""
    return AmountStats(
        min=float(x.min()),
        median=float(x.median()),
        mean=float(x.mean()),
        max=float(x.max()),
        skewness=_finite(x.skew()),
    )


def shared_bin_edges(
    groups: list[np.ndarray], n_bins: int = N_BINS, quantiles: tuple[float, float] = FULL_RANGE
) -> np.ndarray:
    """Equal-width bin edges spanning the union of each group's ``quantiles`` range.

    Quantiles are taken per group (class), not pooled: pooled quantiles are dominated by the
    majority class and would cut off much of the minority class's tail.

    Args:
        groups: Values of each class (empty groups are ignored).
        n_bins: Number of bins.
        quantiles: Lower and upper quantile of each group's range; ``(0, 1)`` means min to max.

    Returns:
        ``n_bins + 1`` increasing edges (a constant column gets a range of width 1).
    """
    ranges = np.array([np.quantile(g, quantiles) for g in groups if len(g)])
    low, high = float(ranges[:, 0].min()), float(ranges[:, 1].max())
    if not high > low:
        low, high = low - 0.5, high + 0.5
    return np.linspace(low, high, n_bins + 1)


def _clipped_counts(values: np.ndarray, edges: np.ndarray) -> list[int]:
    """Histogram counts where values beyond the edges fall into the outermost bins."""
    counts, _ = np.histogram(np.clip(values, edges[0], edges[-1]), bins=edges)
    return [int(c) for c in counts]


def _fractions(counts: list[int]) -> list[float]:
    """Counts divided by their total (all zeros if the total is 0)."""
    total = sum(counts)
    return [c / total if total else 0.0 for c in counts]


def feature_histogram(
    feature: str,
    legit: np.ndarray,
    fraud: np.ndarray,
    n_bins: int = N_BINS,
    quantiles: tuple[float, float] = FULL_RANGE,
) -> FeatureHistogram:
    """Per-class histogram of one feature over shared bins, raw counts and per-class fractions.

    Args:
        feature: Name stored in the result.
        legit, fraud: Values of each class.
        n_bins: Number of bins.
        quantiles: Per-class quantile range the bins span (see :func:`shared_bin_edges`).

    Returns:
        A :class:`FeatureHistogram`.
    """
    edges = shared_bin_edges([legit, fraud], n_bins, quantiles)
    legit_counts, fraud_counts = _clipped_counts(legit, edges), _clipped_counts(fraud, edges)
    return FeatureHistogram(
        feature=feature,
        bin_edges=[float(e) for e in edges],
        legit_counts=legit_counts,
        fraud_counts=fraud_counts,
        legit_fraction=_fractions(legit_counts),
        fraud_fraction=_fractions(fraud_counts),
        bin_range_quantiles=quantiles,
    )


def hour_of_day(time: pd.Series, y: pd.Series) -> HourOfDay:
    """Count transactions per class per hour, ``hour = floor(Time / 3600) mod 24``.

    This assumes ``Time = 0`` is midnight, which the data provider never documented; the result
    carries that caveat in ``hour_assumption``.

    Args:
        time: Seconds since the first transaction.
        y: Labels aligned with ``time``.

    Returns:
        A :class:`HourOfDay` with 24 counts per class.
    """
    hours = (np.floor(time.to_numpy(dtype=float) / SECONDS_PER_HOUR) % HOURS_PER_DAY).astype(int)
    labels = y.to_numpy()
    per_class = [np.bincount(hours[labels == label], minlength=HOURS_PER_DAY) for label in (1, 0)]
    return HourOfDay(
        hour_assumption=HOUR_ASSUMPTION,
        fraud_counts=[int(c) for c in per_class[0]],
        legit_counts=[int(c) for c in per_class[1]],
    )


def compute_eda_summary(
    df_raw: pd.DataFrame,
    df_dedup: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    dataset_source: DatasetSource,
    dataset_sha256: str,
) -> EdaSummary:
    """Compute every dataset fact the report and the dashboard need.

    Raw-file facts (shape, class counts, fraud rate, imbalance ratio, missing values) come from
    ``df_raw``; duplicates are ``len(df_raw) - len(df_dedup)``; every distributional statistic
    (``Amount``, ``Time``, effect sizes, KS statistics, histograms, hour of day) comes from
    ``df_dedup``. ``training_counts`` is left empty: it needs tuned ratios from the experiment.

    Maths:

    * ``fraud_rate = frauds / rows``; ``imbalance_ratio = "1 : round(legit / fraud)"``.
    * Cohen's d per model feature (``V1..V28, Amount``): see :func:`cohens_d`; sorted by ``|d|``.
    * KS statistic: ``max |F_fraud(x) - F_legit(x)|`` (two-sample Kolmogorov-Smirnov).
    * Histograms: 20 equal-width shared bins for the 6 PCA features with the largest ``|d|``
      (range = union of each class's 0.5-99.5th percentile, tails counted in the outer bins) and for
      ``log1p(Amount)`` (full range). ``amount_histogram`` uses the same bins mapped back to the
      raw scale with ``expm1``.
    * ``time_span_hours = (max Time - min Time) / 3600``; ``time_histogram`` has one bin per hour.

    Args:
        df_raw: The validated file as loaded (before deduplication).
        df_dedup: ``df_raw`` after :func:`fraudstudy.data.deduplicate`.
        cfg: Contents of ``configs/dataset.yaml``.
        dataset_source: ``"real"`` or ``"synthetic"``.
        dataset_sha256: SHA-256 of the source file.

    Returns:
        The validated :class:`EdaSummary`.

    Raises:
        DataValidationError: If ``df_dedup`` has more rows than ``df_raw`` or either class is
            missing from ``df_dedup``.
    """
    target, amount_col, time_col = cfg["target"], cfg["amount_column"], cfg["time_column"]
    pca = list(cfg["pca_features"])
    if len(df_dedup) > len(df_raw):
        raise DataValidationError(
            f"df_dedup has {len(df_dedup)} rows but df_raw only {len(df_raw)}; pass the "
            "deduplicated frame second."
        )
    y = df_dedup[target]
    is_fraud = (y == 1).to_numpy()
    if is_fraud.all() or not is_fraud.any():
        raise DataValidationError("EDA needs both classes in the deduplicated data.")
    legit_df, fraud_df = df_dedup.loc[~is_fraud], df_dedup.loc[is_fraud]

    raw_counts = _class_counts(df_raw[target])
    dedup_counts = _class_counts(y)
    raw_fraud_rows = int((df_raw[target] == 1).sum())

    amount = df_dedup[amount_col].astype(float)
    log_amount = np.log1p(amount)

    features = [*pca, amount_col]
    effects = [
        EffectSize(
            feature=f,
            cohens_d=cohens_d(fraud_df[f].to_numpy(float), legit_df[f].to_numpy(float)),
            legit_mean=float(legit_df[f].mean()),
            fraud_mean=float(fraud_df[f].mean()),
        )
        for f in features
    ]
    effects.sort(key=lambda e: abs(e.cohens_d), reverse=True)

    separation = [
        FeatureSeparation(
            feature=f,
            legit_mean=float(legit_df[f].mean()),
            fraud_mean=float(fraud_df[f].mean()),
            legit_std=_finite(legit_df[f].std(ddof=1)) if len(legit_df) > 1 else 0.0,
            fraud_std=_finite(fraud_df[f].std(ddof=1)) if len(fraud_df) > 1 else 0.0,
            ks_statistic=float(
                stats.ks_2samp(fraud_df[f].to_numpy(float), legit_df[f].to_numpy(float)).statistic
            ),
        )
        for f in features
    ]
    separation.sort(key=lambda s: s.ks_statistic, reverse=True)

    top_pca = [e.feature for e in effects if e.feature in pca][:N_TOP_FEATURES]
    histograms = [
        feature_histogram(
            f,
            legit_df[f].to_numpy(float),
            fraud_df[f].to_numpy(float),
            quantiles=FEATURE_BIN_QUANTILES,
        )
        for f in top_pca
    ]
    log_hist = feature_histogram(
        LOG_AMOUNT, log_amount[~is_fraud].to_numpy(), log_amount[is_fraud].to_numpy()
    )
    histograms.append(log_hist)
    raw_edges = np.expm1(np.asarray(log_hist.bin_edges))
    amount_histogram = Histogram(
        bin_edges=[float(e) for e in raw_edges],
        legit_counts=log_hist.legit_counts,
        fraud_counts=log_hist.fraud_counts,
    )

    time = df_dedup[time_col].astype(float)
    t_min, t_max = float(time.min()), float(time.max())
    n_time_bins = max(1, int(np.ceil((t_max - t_min) / SECONDS_PER_HOUR)))
    time_edges = t_min + SECONDS_PER_HOUR * np.arange(n_time_bins + 1)
    time_histogram = Histogram(
        bin_edges=[float(e) for e in time_edges],
        legit_counts=_clipped_counts(time[~is_fraud].to_numpy(), time_edges),
        fraud_counts=_clipped_counts(time[is_fraud].to_numpy(), time_edges),
    )

    return EdaSummary(
        dataset_source=dataset_source,
        dataset_sha256=dataset_sha256,
        n_rows=len(df_raw),
        n_columns=df_raw.shape[1],
        class_counts=raw_counts,
        fraud_rate=raw_fraud_rows / len(df_raw),
        n_duplicate_rows=len(df_raw) - len(df_dedup),
        n_duplicate_frauds=raw_fraud_rows - dedup_counts.fraud,
        missing_values={str(c): int(n) for c, n in df_raw.isna().sum().items()},
        amount=ClassSplit(
            legit=_distribution(legit_df[amount_col].astype(float)),
            fraud=_distribution(fraud_df[amount_col].astype(float)),
        ),
        amount_histogram=amount_histogram,
        time_histogram=time_histogram,
        feature_separation=separation,
        imbalance_ratio=imbalance_ratio(raw_counts),
        class_counts_dedup=dedup_counts,
        amount_summary=AmountSummary(raw=_amount_stats(amount), log1p=_amount_stats(log_amount)),
        time_span_hours=(t_max - t_min) / SECONDS_PER_HOUR,
        effect_sizes=effects,
        feature_histograms=histograms,
        hour_of_day=hour_of_day(time, y),
    )


# --- figures ------------------------------------------------------------------------------------


def _new_figure(width: float, height: float) -> Figure:
    """A blank figure with constrained layout (call inside ``rc_context(FIGURE_STYLE)``)."""
    return Figure(figsize=(width, height), layout="constrained")


def _save(fig: Figure, path: Path) -> Path:
    """Save ``fig`` as a 300-dpi PNG, creating the parent folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI)
    return path


def _step_hist(ax: Axes, hist: FeatureHistogram) -> None:
    """Draw both classes' bin fractions as outlined steps (fraud dashed and hatched)."""
    edges = np.asarray(hist.bin_edges)
    ax.stairs(hist.legit_fraction, edges, fill=True, alpha=0.15, color=LEGIT_COLOUR, linewidth=0)
    ax.stairs(
        hist.fraud_fraction,
        edges,
        fill=True,
        facecolor="none",
        edgecolor=FRAUD_COLOUR,
        hatch=FRAUD_HATCH,
        linewidth=0,
    )
    ax.stairs(hist.legit_fraction, edges, linewidth=1.4, **LEGIT_STYLE)
    ax.stairs(hist.fraud_fraction, edges, linewidth=1.4, **FRAUD_STYLE)


def plot_class_counts(summary: EdaSummary, path: Path) -> Path:
    """Bar chart of class counts after deduplication, on a linear and a log axis side by side.

    On the linear axis the fraud bar is almost invisible, which is the point of the figure.
    """
    counts = summary.class_counts_dedup or summary.class_counts
    values = [counts.legit, counts.fraud]
    with matplotlib.rc_context(FIGURE_STYLE):
        fig = _new_figure(7.0, 3.0)
        axes = fig.subplots(1, 2)
        for ax, scale in zip(axes, ("linear", "log"), strict=True):
            bars = ax.bar(
                ["Legitimate", "Fraud"],
                values,
                color=[LEGIT_COLOUR, FRAUD_COLOUR],
                edgecolor="black",
                linewidth=0.6,
            )
            bars[1].set_hatch(FRAUD_HATCH)
            ax.set_yscale(scale)
            ax.set_title(f"{scale.capitalize()} scale")
            ax.set_ylabel("Number of transactions" + (" (log scale)" if scale == "log" else ""))
            ax.set_xlabel("Class")
            for bar, v in zip(bars, values, strict=True):
                ax.annotate(
                    f"{v:,}",
                    (bar.get_x() + bar.get_width() / 2, v),
                    ha="center",
                    va="bottom",
                    xytext=(0, 2),
                    textcoords="offset points",
                )
            ax.margins(y=0.15)
        fig.suptitle(
            f"Class counts after removing duplicates (fraud rate "
            f"{counts.fraud / (counts.fraud + counts.legit):.3%}, {imbalance_ratio(counts)})"
        )
        return _save(fig, path)


def plot_amount(df_dedup: pd.DataFrame, cfg: dict[str, Any], path: Path) -> Path:
    """Per-class ``Amount`` distribution on the raw scale and after ``log1p``.

    Each class is normalised to fractions of its own rows, so the shapes are comparable.
    """
    y = df_dedup[cfg["target"]].to_numpy()
    amount = df_dedup[cfg["amount_column"]].to_numpy(float)
    with matplotlib.rc_context(FIGURE_STYLE):
        fig = _new_figure(7.5, 3.0)
        axes = fig.subplots(1, 2)
        panels = (
            (amount, AMOUNT_UNIT, "Raw amount (axis to 99.5th pct.)", FEATURE_BIN_QUANTILES),
            (np.log1p(amount), f"log(1 + {AMOUNT_UNIT})", "After log(1 + Amount)", FULL_RANGE),
        )
        for ax, (values, xlabel, title, q) in zip(axes, panels, strict=True):
            hist = feature_histogram("Amount", values[y == 0], values[y == 1], quantiles=q)
            _step_hist(ax, hist)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Fraction of class's transactions")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncols=2)
        fig.suptitle("Transaction amount by class, raw and log-transformed")
        return _save(fig, path)


def plot_top_features(summary: EdaSummary, path: Path) -> Path:
    """2 x 3 grid of per-class histograms for the six PCA features with the largest ``|d|``."""
    effect = {e.feature: e.cohens_d for e in summary.effect_sizes}
    hists = [h for h in summary.feature_histograms if h.feature != LOG_AMOUNT][:N_TOP_FEATURES]
    with matplotlib.rc_context(FIGURE_STYLE):
        fig = _new_figure(8.0, 5.0)
        axes = fig.subplots(2, 3).ravel()
        for ax, hist in zip(axes, hists, strict=False):
            _step_hist(ax, hist)
            ax.set_title(f"{hist.feature} (Cohen's d = {effect.get(hist.feature, 0.0):+.2f})")
            ax.set_xlabel(f"{hist.feature} {PCA_UNIT}")
            ax.set_ylabel("Fraction of class")
        for ax in axes[len(hists) :]:
            ax.set_visible(False)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncols=2)
        fig.suptitle("Six features that differ most between classes (largest |Cohen's d|)")
        return _save(fig, path)


def plot_effect_sizes(summary: EdaSummary, path: Path) -> Path:
    """Horizontal bars of Cohen's d per feature, largest ``|d|`` at the top.

    Negative bars (fraud lower than legit) are hatched, so sign does not rely on colour.
    """
    effects = summary.effect_sizes
    names = [e.feature for e in effects][::-1]
    values = [e.cohens_d for e in effects][::-1]
    with matplotlib.rc_context(FIGURE_STYLE):
        fig = _new_figure(6.0, max(3.0, 0.2 * len(effects) + 1.0))
        ax = fig.subplots()
        bars = ax.barh(
            names,
            values,
            color=[FRAUD_COLOUR if v > 0 else LEGIT_COLOUR for v in values],
            edgecolor="black",
            linewidth=0.5,
        )
        for bar, v in zip(bars, values, strict=True):
            if v < 0:
                bar.set_hatch(FRAUD_HATCH)
        ax.axvline(0.0, color="black", linewidth=0.8)
        for ref in (-0.8, 0.8):
            ax.axvline(ref, color="grey", linewidth=0.6, linestyle=":")
        ax.legend(
            handles=[
                Patch(facecolor=FRAUD_COLOUR, edgecolor="black", label="Fraud mean higher"),
                Patch(
                    facecolor=LEGIT_COLOUR,
                    edgecolor="black",
                    hatch=FRAUD_HATCH,
                    label="Fraud mean lower (hatched)",
                ),
            ],
            loc="lower right",
        )
        ax.set_xlabel("Cohen's d, (fraud mean - legit mean) / pooled SD (unitless)")
        ax.set_ylabel("Feature")
        ax.set_title("Per-feature effect size between classes (dotted lines: |d| = 0.8, large)")
        ax.margins(y=0.01)
        return _save(fig, path)


def make_eda_figures(
    summary: EdaSummary, df_dedup: pd.DataFrame, cfg: dict[str, Any], out_dir: Path
) -> list[Path]:
    """Write the four EDA figures (:data:`FIGURE_NAMES`) to ``out_dir``.

    Args:
        summary: Output of :func:`compute_eda_summary`.
        df_dedup: The deduplicated dataset (for the ``Amount`` panels).
        cfg: Contents of ``configs/dataset.yaml``.
        out_dir: Destination folder (created if missing).

    Returns:
        The paths written, in :data:`FIGURE_NAMES` order.
    """
    out_dir = Path(out_dir)
    paths = [out_dir / name for name in FIGURE_NAMES]
    plot_class_counts(summary, paths[0])
    plot_amount(df_dedup, cfg, paths[1])
    plot_top_features(summary, paths[2])
    plot_effect_sizes(summary, paths[3])
    return paths
