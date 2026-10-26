"""The preprocessing step that starts every model pipeline.

pipeline.build_pipeline() puts build_preprocessor() first in every pipeline, so the imputer
medians and the scaler medians and IQRs are learned in fit, from training rows (or CV training
folds) only, and never from validation or test rows. Reads and writes no files.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler


def log1p_clipped(X: object) -> object:
    """log(1 + x), with negative values clipped to 0 first, so the result is finite and 0 -> 0.

    Amount is never negative in the real data, but a bad upload could be.
    """
    return np.log1p(np.clip(X, 0.0, None))


def log1p_names(transformer: object, input_features: list) -> list[str]:
    """Output column names of the log step: Amount becomes log1p_Amount."""
    return [f"log1p_{name}" for name in input_features]


def build_preprocessor(cfg: dict) -> ColumnTransformer:
    """Build the unfitted preprocessor; its output columns are V1..V28 and log1p_Amount (29).

    V1..V28: median imputer, then RobustScaler, i.e. (x - median) / IQR. Amount: median imputer,
    then log1p_clipped, then RobustScaler. Every other column, including raw Time, is dropped.
    Each branch starts with an imputer so a blank cell in an uploaded file cannot crash it.
    """
    pca_branch = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
    ])
    # Amount is heavy-tailed: the log brings it to a scale like the V columns before scaling.
    amount_branch = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(log1p_clipped, feature_names_out=log1p_names)),
        ("scale", RobustScaler()),
    ])
    preprocessor = ColumnTransformer(
        [("pca", pca_branch, list(cfg["pca_features"])),
         ("amount", amount_branch, [cfg["amount_column"]])],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    # pandas output keeps the row index, which the leakage tests use to track rows.
    return preprocessor.set_output(transform="pandas")
