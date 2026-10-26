"""Settings, paths and the seed for the whole project.

Every module and script calls get_dataset_config(), get_experiment_config() or get_models_config()
to read the YAML files in configs/, and resolve() to turn a path written relative to the project
root into an absolute one. set_seed() seeds Python and NumPy with the one seed from
experiment.yaml.
"""

import os
import random
from pathlib import Path

import numpy as np
import yaml

# This file is src/fraudstudy/config.py, so the project root is three folders up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"

# Keys the code reads; a missing one is almost always a typo in the YAML file.
REQUIRED_KEYS = {
    "dataset": ["name", "raw_path", "sample_path", "sha256", "target", "time_column",
                "amount_column", "pca_features", "excluded_from_features"],
    "experiment": ["seed", "test_size", "cv_folds", "cv_repeats", "primary_metric", "techniques",
                   "models", "ratio_grid", "hybrid", "smote_k_neighbors", "tuning", "quick",
                   "threshold", "bootstrap"],
    "models": ["logreg", "random_forest", "lightgbm"],
}


def resolve(path: str | Path) -> Path:
    """Return `path` as an absolute path; relative paths are taken from the project root."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(name: str) -> dict:
    """Read configs/<name>.yaml and check that every required key is there."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}. Restore it from git.")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_KEYS.get(name, []) if key not in config]
    if missing:
        raise ValueError(f"Config file {path} is missing required keys: {', '.join(missing)}.")
    return config


def get_dataset_config() -> dict:
    """Return configs/dataset.yaml (file paths, column names, the pinned SHA-256)."""
    return load_config("dataset")


def get_experiment_config() -> dict:
    """Return configs/experiment.yaml (seed, split, CV, ratio grid, tuning budget)."""
    return load_config("experiment")


def get_models_config() -> dict:
    """Return configs/models.yaml (the hyper-parameter search space of each model)."""
    return load_config("models")


def set_seed(seed: int) -> None:
    """Seed Python's random module and NumPy's global generator.

    Models and samplers still get random_state=seed directly; this covers any code that draws
    from the global generators. PYTHONHASHSEED only affects processes started after this call.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
