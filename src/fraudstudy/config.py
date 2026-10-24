"""Typed access to the YAML files in ``configs/``.

The configs folder defaults to ``<project root>/configs`` and can be overridden with the
environment variable ``FRAUDSTUDY_CONFIG_DIR`` (relative values resolve from the project root).
"""

import os
from pathlib import Path
from typing import Any

import yaml

from fraudstudy.errors import ConfigError
from fraudstudy.paths import resolve

CONFIG_DIR_ENV = "FRAUDSTUDY_CONFIG_DIR"
DEFAULT_CONFIG_DIR = "configs"

REQUIRED_KEYS: dict[str, tuple[str, ...]] = {
    "dataset": (
        "name",
        "raw_path",
        "sample_path",
        "sha256",
        "target",
        "time_column",
        "amount_column",
        "pca_features",
        "excluded_from_features",
    ),
    "experiment": (
        "seed",
        "test_size",
        "cv_folds",
        "cv_repeats",
        "primary_metric",
        "techniques",
        "models",
        "ratio_grid",
        "hybrid",
        "smote_k_neighbors",
        "tuning",
        "quick",
        "threshold",
        "bootstrap",
        "use_hour_feature",
    ),
    "models": ("logreg", "random_forest", "lightgbm"),
}


def config_dir() -> Path:
    """Return the configs folder, honouring ``FRAUDSTUDY_CONFIG_DIR`` if set.

    Returns:
        Absolute path of the configs folder.
    """
    return resolve(os.environ.get(CONFIG_DIR_ENV) or DEFAULT_CONFIG_DIR)


def load_config(name: str) -> dict[str, Any]:
    """Load ``<config dir>/<name>.yaml`` and check that its required keys are present.

    Args:
        name: Config name without extension, e.g. ``"experiment"``.

    Returns:
        The parsed YAML mapping.

    Raises:
        ConfigError: If the file is missing, is not a YAML mapping, or lacks required keys.
    """
    path = config_dir() / f"{name}.yaml"
    if not path.is_file():
        raise ConfigError(
            f"Config file not found: {path}. Create it or set {CONFIG_DIR_ENV} "
            "to the folder that contains it."
        )
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Config file {path} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {path} must contain a YAML mapping at the top level.")
    missing = [key for key in REQUIRED_KEYS.get(name, ()) if key not in data]
    if missing:
        raise ConfigError(f"Config file {path} is missing required keys: {', '.join(missing)}.")
    return data


def get_dataset_config() -> dict[str, Any]:
    """Return the validated contents of ``dataset.yaml``."""
    return load_config("dataset")


def get_experiment_config() -> dict[str, Any]:
    """Return the validated contents of ``experiment.yaml``."""
    return load_config("experiment")


def get_models_config() -> dict[str, Any]:
    """Return the validated contents of ``models.yaml`` (search space per model)."""
    return load_config("models")
