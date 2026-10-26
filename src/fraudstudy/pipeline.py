"""Builds every model pipeline: preprocess -> the technique's sampler steps -> model.

Scripts and tests call build_pipeline(); models.get_model() builds the model at its end. The result
is always an imblearn Pipeline, so sampler steps run only inside fit, on training rows, and are
skipped by predict and predict_proba (docs/decisions.md). Reads the configs; writes nothing.
"""

from imblearn.pipeline import Pipeline

from fraudstudy.config import get_dataset_config, get_experiment_config
from fraudstudy.models import get_model
from fraudstudy.preprocess import build_preprocessor
from fraudstudy.sampling import (
    CLASS_WEIGHT_PARAM,
    TECHNIQUE_PARAMS,
    TECHNIQUES,
    build_sampler_steps,
    class_weight_model_params,
)


def build_pipeline(
    model_name: str,
    technique: str,
    model_params: dict | None = None,
    technique_params: dict | None = None,
    cfg: dict | None = None,
    exp_cfg: dict | None = None,
) -> Pipeline:
    """Build the unfitted pipeline for one (model, technique) cell.

    Args:
        model_name: "logreg", "random_forest" or "lightgbm" (models.MODEL_NAMES).
        technique: One of sampling.TECHNIQUES.
        model_params: Model hyper-parameters, e.g. {"C": 0.1, "max_iter": 2000}.
        technique_params: The technique's settings, e.g. {"sampling_strategy": 0.1};
            {"fraud_weight": w} for class_weight (default "balanced").
        cfg: configs/dataset.yaml; loaded if omitted.
        exp_cfg: configs/experiment.yaml (seed, smote_k_neighbors); loaded if omitted.

    Returns:
        Pipeline with steps "preprocess", the sampler steps ("ros", "rus", "smote", or "smote"
        then "rus"), and "model".
    """
    cfg = cfg or get_dataset_config()
    exp_cfg = exp_cfg or get_experiment_config()
    model_params = dict(model_params or {})
    technique_params = technique_params or {}
    if technique not in TECHNIQUE_PARAMS:
        raise ValueError(f"Unknown or not yet implemented technique {technique!r}; available: "
                         f"{', '.join(TECHNIQUES)}.")
    unknown = sorted(set(technique_params) - set(TECHNIQUE_PARAMS[technique]))
    if unknown:
        allowed = ", ".join(TECHNIQUE_PARAMS[technique]) or "no parameters"
        raise ValueError(f"technique {technique!r} got unknown parameters {', '.join(unknown)}; "
                         f"it takes {allowed}.")
    # The class weight belongs to technique F only, so it cannot leak into other techniques.
    weight_param = CLASS_WEIGHT_PARAM.get(model_name)
    if weight_param in model_params:
        raise ValueError(f"Do not pass {weight_param!r} in model_params; class weighting is "
                         "technique 'class_weight' (use technique_params={'fraud_weight': ...}).")
    if technique == "class_weight":
        weight = technique_params.get("fraud_weight", "balanced")
        model_params.update(class_weight_model_params(model_name, weight))

    seed = int(exp_cfg["seed"])
    samplers = build_sampler_steps(technique, technique_params, seed,
                                   int(exp_cfg["smote_k_neighbors"]))
    model = get_model(model_name, seed, **model_params)
    return Pipeline([("preprocess", build_preprocessor(cfg)), *samplers, ("model", model)])
