"""Feature selection and the train/test split. No scaling or resampling happens here."""

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
from sklearn.model_selection import train_test_split

from fraudstudy.errors import ConfigError, DataValidationError

Strategy = Literal["stratified", "temporal"]
STRATEGIES: tuple[str, ...] = ("stratified", "temporal")


@dataclass(frozen=True)
class Split:
    """A train/test split. Index labels are those of the input frame (rows stay traceable).

    Attributes:
        X_train, X_test: Feature columns only, in :func:`feature_columns` order.
        y_train, y_test: Target labels.
        strategy: ``"stratified"`` or ``"temporal"``.
        summary: ``{"all" | "train" | "test": {"rows", "frauds", "fraud_rate"}}``.
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    strategy: str
    summary: dict[str, dict[str, float]]


def feature_columns(cfg: dict[str, Any], use_hour_feature: bool) -> list[str]:
    """Return the model input columns in a fixed order.

    Raw ``Time`` is excluded (docs/decisions.md) unless the optional hour-of-day side test is on,
    in which case ``Time`` is passed through so the in-pipeline hour transformer can use it.

    Args:
        cfg: Contents of ``configs/dataset.yaml``.
        use_hour_feature: ``use_hour_feature`` from ``configs/experiment.yaml``.

    Returns:
        ``[V1..V28, Amount]``, plus ``Time`` last when ``use_hour_feature`` is true.
    """
    cols = [*cfg["pca_features"], cfg["amount_column"]]
    if use_hour_feature:
        cols.append(cfg["time_column"])
    return cols


def _part_summary(y: pd.Series) -> dict[str, float]:
    """Rows, frauds and fraud rate of one label vector."""
    rows, frauds = len(y), int((y == 1).sum())
    return {"rows": rows, "frauds": frauds, "fraud_rate": frauds / rows if rows else 0.0}


def split_dataset(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    exp_cfg: dict[str, Any],
    strategy: Strategy = "stratified",
) -> Split:
    """Split ``df`` into train and test parts with ``test_size`` from the experiment config.

    * ``stratified``: ``train_test_split(stratify=y, test_size, random_state=seed)``, so both
      parts keep the overall fraud rate.
    * ``temporal``: rows are stably sorted by ``Time``; the first ``1 - test_size`` (70%) form
      the training set and the rest the test set, without shuffling, so
      ``max(train Time) <= min(test Time)``.

    Args:
        df: Dataset (typically deduplicated). Not modified.
        cfg: Contents of ``configs/dataset.yaml``.
        exp_cfg: Contents of ``configs/experiment.yaml`` (``seed``, ``test_size``,
            ``use_hour_feature``).
        strategy: ``"stratified"`` or ``"temporal"``.

    Returns:
        A :class:`Split`.

    Raises:
        ConfigError: If ``strategy`` is unknown or ``test_size`` is not in (0, 1).
        DataValidationError: If a part would be empty.
    """
    if strategy not in STRATEGIES:
        raise ConfigError(f"strategy must be one of {', '.join(STRATEGIES)}; got {strategy!r}.")
    test_size = float(exp_cfg["test_size"])
    if not 0.0 < test_size < 1.0:
        raise ConfigError(
            f"test_size in configs/experiment.yaml must be in (0, 1); got {test_size}."
        )

    target = cfg["target"]
    cols = feature_columns(cfg, bool(exp_cfg["use_hour_feature"]))

    if strategy == "stratified":
        X_train, X_test, y_train, y_test = train_test_split(
            df[cols],
            df[target],
            test_size=test_size,
            stratify=df[target],
            random_state=exp_cfg["seed"],
        )
    else:
        ordered = df.sort_values(cfg["time_column"], kind="stable")
        n_train = int(round(len(ordered) * (1.0 - test_size)))
        if not 0 < n_train < len(ordered):
            raise DataValidationError(
                f"Temporal split of {len(ordered)} rows with test_size={test_size} leaves an "
                "empty part; use more rows."
            )
        train, test = ordered.iloc[:n_train], ordered.iloc[n_train:]
        X_train, X_test = train[cols].copy(), test[cols].copy()
        y_train, y_test = train[target].copy(), test[target].copy()

    summary = {
        "all": _part_summary(df[target]),
        "train": _part_summary(y_train),
        "test": _part_summary(y_test),
    }
    return Split(X_train, X_test, y_train, y_test, strategy, summary)
