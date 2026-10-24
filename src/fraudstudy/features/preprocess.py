"""The shared, leakage-proof preprocessor (master guide Part 5).

Every statistic (imputation medians, scaler medians and IQRs) is learned in ``fit``. The
preprocessor is always the first step of the model pipeline, so during cross-validation it
only ever sees the training folds.
"""

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from fraudstudy.features.transformers import CyclicalHourTransformer, Log1pTransformer

IMPUTE_STRATEGY = "median"


def build_preprocessor(cfg: dict[str, Any], exp_cfg: dict[str, Any]) -> ColumnTransformer:
    """Build the unfitted preprocessing ``ColumnTransformer``.

    Branches (each starts with a median imputer so a blank cell in an upload cannot crash it):

    * ``pca``: ``V1..V28`` -> median impute -> ``RobustScaler``, i.e. ``(x - median) / IQR``.
    * ``amount``: ``Amount`` -> median impute -> ``log1p(max(x, 0))`` -> ``RobustScaler``;
      output column ``log1p_Amount``.
    * ``hour`` (only if ``use_hour_feature``): ``Time`` -> median impute ->
      ``hour_sin, hour_cos`` with ``hour = (Time / 3600) mod 24``.

    Every other column, including raw ``Time``, is dropped. Output columns:
    ``V1..V28, log1p_Amount`` (29) by default; with the hour feature, raw ``Time`` is replaced
    by ``hour_sin, hour_cos`` (31).

    Args:
        cfg: Contents of ``configs/dataset.yaml`` (column names).
        exp_cfg: Contents of ``configs/experiment.yaml`` (``use_hour_feature``).

    Returns:
        An unfitted ``ColumnTransformer`` with ``set_output(transform="pandas")``.
    """
    transformers: list[tuple[str, Any, list[str]]] = [
        (
            "pca",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy=IMPUTE_STRATEGY)),
                    ("scale", RobustScaler()),
                ]
            ),
            list(cfg["pca_features"]),
        ),
        (
            "amount",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy=IMPUTE_STRATEGY)),
                    ("log1p", Log1pTransformer()),
                    ("scale", RobustScaler()),
                ]
            ),
            [cfg["amount_column"]],
        ),
    ]
    if exp_cfg["use_hour_feature"]:
        transformers.append(
            (
                "hour",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy=IMPUTE_STRATEGY)),
                        ("cyclical", CyclicalHourTransformer()),
                    ]
                ),
                [cfg["time_column"]],
            )
        )
    preprocessor = ColumnTransformer(
        transformers, remainder="drop", verbose_feature_names_out=False
    )
    return preprocessor.set_output(transform="pandas")
