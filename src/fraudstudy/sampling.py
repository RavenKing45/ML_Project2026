"""Imbalance techniques A-F: their samplers, the shared ratio search space, the class weights
and the resampling audit.

pipeline.build_pipeline() calls build_sampler_steps() and class_weight_model_params(); the
tuning and scripts/run_resampling_audit.py call technique_search_space() and
resampling_audit(). A ratio r is imblearn's float sampling_strategy: after resampling,
frauds = r * legit. Reads and writes no files.
"""

import logging

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import clone

logger = logging.getLogger(__name__)

# Every technique and the technique_params keys it accepts (anything else is a typo).
TECHNIQUE_PARAMS = {
    "baseline": [],
    "ros": ["sampling_strategy"],
    "rus": ["sampling_strategy"],
    "smote": ["sampling_strategy"],
    "smote_rus": ["smote_sampling_strategy", "rus_sampling_strategy"],
    "class_weight": ["fraud_weight"],
}
TECHNIQUES = list(TECHNIQUE_PARAMS)
# The model setting that carries technique F's fraud weight.
CLASS_WEIGHT_PARAM = {"logreg": "class_weight", "random_forest": "class_weight",
                      "lightgbm": "scale_pos_weight"}
# imblearn rejects a target ratio at or below the current one. The 5% headroom also covers CV
# training folds whose fraud share is slightly above that of the whole training set.
RATIO_MARGIN = 1.05


def base_ratio(y: pd.Series | np.ndarray) -> float:
    """frauds / legit of a label vector: the ratio before any resampling."""
    labels = np.asarray(y)
    frauds, legit = int((labels == 1).sum()), int((labels == 0).sum())
    if frauds == 0 or legit == 0:
        raise ValueError(
            f"Need both classes to compute the base ratio; got {frauds} frauds and {legit} "
            "legitimate rows. Check the training labels."
        )
    return frauds / legit


def valid_ratios(ratios: list, base: float) -> list[float]:
    """Keep the ratios above base * RATIO_MARGIN (imblearn rejects the others); log the rest."""
    threshold = base * RATIO_MARGIN
    kept = [float(r) for r in ratios if float(r) > threshold]
    dropped = [float(r) for r in ratios if float(r) <= threshold]
    if dropped:
        logger.info(
            "Dropped ratios %s: not above base ratio %.5f x %.2f = %.5f (imblearn rejects them).",
            dropped, base, RATIO_MARGIN, threshold,
        )
    if not kept:
        raise ValueError(
            f"No ratio in {list(ratios)} exceeds base ratio {base:.5f} x {RATIO_MARGIN}; add "
            "larger values to ratio_grid in configs/experiment.yaml."
        )
    return kept


def fraud_weight(ratio: float, base: float) -> float:
    """The fraud class weight that matches resampling ratio `ratio`: w = ratio / base.

    With n_f frauds and n_l legit rows, the weighted fraud mass w * n_f = ratio * n_l, the same
    class balance a resampler at that ratio produces.
    """
    return float(ratio) / float(base)


def class_weight_model_params(model_name: str, weight: float | str) -> dict:
    """Model settings that give fraud rows `weight` (a number, or "balanced") for technique F.

    sklearn's class_weight takes a dict or "balanced", not a bare number; LightGBM takes
    scale_pos_weight and has no "balanced" option.
    """
    if model_name not in CLASS_WEIGHT_PARAM:
        raise ValueError(f"No class-weight setting known for model {model_name!r}.")
    param = CLASS_WEIGHT_PARAM[model_name]
    if weight == "balanced":
        if model_name == "lightgbm":
            raise ValueError("lightgbm has no 'balanced' option; pass a numeric fraud_weight.")
        return {param: "balanced"}
    if isinstance(weight, bool) or not isinstance(weight, int | float) or weight <= 0:
        raise ValueError(f"fraud_weight must be a positive number or 'balanced'; got {weight!r}.")
    if param == "class_weight":
        return {param: {0: 1.0, 1: float(weight)}}
    return {param: float(weight)}


def build_sampler_steps(technique: str, params: dict, seed: int, k_neighbors: int) -> list:
    """The (name, sampler) pipeline steps of a technique; none for baseline and class_weight.

    A missing ratio means imblearn's "auto" (fully balanced); the search sets the real ratio
    later through <step>__sampling_strategy. imblearn itself rejects impossible ratios.
    """
    def smote(ratio: object) -> SMOTE:
        return SMOTE(sampling_strategy=ratio, k_neighbors=k_neighbors, random_state=seed)

    ratio = params.get("sampling_strategy", "auto")
    if technique == "ros":
        return [("ros", RandomOverSampler(sampling_strategy=ratio, random_state=seed))]
    if technique == "rus":
        return [("rus", RandomUnderSampler(sampling_strategy=ratio, random_state=seed))]
    if technique == "smote":
        return [("smote", smote(ratio))]
    if technique == "smote_rus":
        # SMOTE adds frauds up to a * legit; RUS then drops legit rows until fraud = b * legit.
        a = params.get("smote_sampling_strategy", "auto")
        b = params.get("rus_sampling_strategy", "auto")
        return [("smote", smote(a)),
                ("rus", RandomUnderSampler(sampling_strategy=b, random_state=seed))]
    return []


def technique_search_space(technique: str, exp_cfg: dict, base: float,
                           model_name: str = "logreg") -> dict | list:
    """The tuning grid of a technique, built from the SAME ratio grid for every technique.

    ros / rus / smote: {"<step>__sampling_strategy": valid ratios}. smote_rus: a list with one
    dict per valid pair a < b from `hybrid`. class_weight: the weights r / base for the same valid
    ratios (plus "balanced" for logreg and random forest). baseline: {}.
    """
    if technique not in TECHNIQUE_PARAMS:
        raise ValueError(f"Unknown technique {technique!r}; available: {', '.join(TECHNIQUES)}.")
    if technique == "baseline":
        return {}
    if technique in ("ros", "rus", "smote"):
        return {f"{technique}__sampling_strategy": valid_ratios(exp_cfg["ratio_grid"], base)}
    if technique == "smote_rus":
        smote_ratios = valid_ratios(exp_cfg["hybrid"]["smote_ratios"], base)
        rus_ratios = valid_ratios(exp_cfg["hybrid"]["rus_ratios"], base)
        pairs = [(a, b) for a in smote_ratios for b in rus_ratios if a < b]
        if not pairs:
            raise ValueError("No hybrid pair with a < b survives; check hybrid in "
                             "configs/experiment.yaml.")
        return [{"smote__sampling_strategy": [a], "rus__sampling_strategy": [b]} for a, b in pairs]
    param = CLASS_WEIGHT_PARAM[model_name]
    values = [class_weight_model_params(model_name, fraud_weight(r, base))[param]
              for r in valid_ratios(exp_cfg["ratio_grid"], base)]
    if model_name != "lightgbm":
        values.append("balanced")
    return {f"model__{param}": values}


def row_hashes(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """One 64-bit hash per row of values (the index is ignored); equal rows get equal hashes."""
    return pd.util.hash_pandas_object(pd.DataFrame(np.asarray(X)), index=False).to_numpy()


def resampling_audit(pipe: object, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Run every step of `pipe` except the model on TRAINING rows and count what came out.

    Pipeline.fit hides the resampled rows, so the steps are run one by one on a copy of the
    pipeline. That is allowed here only because the audit gets training rows and never feeds a
    model. Rows are compared in the preprocessed space, the space the samplers see.
    Returns legit/fraud counts before and after, ratio_after, fraud_unique_after,
    fraud_duplicates_after (exact copies, from ROS) and n_synthetic (new rows, from SMOTE).
    """
    y_before = np.asarray(y_train)
    X, y = X_train, y_before
    sampler_input = None
    for _, step in clone(pipe).steps[:-1]:  # the last step is the model
        if hasattr(step, "fit_resample"):
            if sampler_input is None:
                sampler_input = X
            X, y = step.fit_resample(X, y)
            y = np.asarray(y)
        else:
            X = step.fit_transform(X, y)
    originals = set(row_hashes(X if sampler_input is None else sampler_input).tolist())
    fraud_hashes = row_hashes(X)[y == 1].tolist()

    legit_after, fraud_after = int((y == 0).sum()), int((y == 1).sum())
    unique_after = len(set(fraud_hashes))
    return {
        "legit_before": int((y_before == 0).sum()),
        "fraud_before": int((y_before == 1).sum()),
        "legit_after": legit_after,
        "fraud_after": fraud_after,
        "ratio_after": fraud_after / legit_after,
        "fraud_unique_after": unique_after,
        "fraud_duplicates_after": fraud_after - unique_after,
        "n_synthetic": sum(h not in originals for h in fraud_hashes),
    }
