"""Tests for the config loader and the core utilities (paths, seed, errors, io)."""

import hashlib
import random
from pathlib import Path

import numpy as np
import pytest
import yaml

from fraudstudy import config
from fraudstudy.errors import ConfigError, FraudStudyError
from fraudstudy.io import git_commit_hash, load_json, save_json, sha256_file, to_builtin
from fraudstudy.paths import PROJECT_ROOT, find_project_root, resolve
from fraudstudy.seed import set_global_seed


@pytest.fixture
def clean_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure tests use the repository configs unless they override the env var."""
    monkeypatch.delenv(config.CONFIG_DIR_ENV, raising=False)


# --- config -----------------------------------------------------------------


def test_load_all_three_configs(clean_config_env: None) -> None:
    dataset = config.get_dataset_config()
    experiment = config.get_experiment_config()
    models = config.get_models_config()

    assert dataset["target"] == "Class"
    assert dataset["pca_features"] == [f"V{i}" for i in range(1, 29)]
    assert dataset["excluded_from_features"] == ["Time"]
    assert experiment["seed"] == 42
    assert experiment["primary_metric"] == "average_precision"
    assert set(experiment["models"]) == set(models)


def test_config_dir_defaults_to_project_configs(clean_config_env: None) -> None:
    assert config.config_dir() == PROJECT_ROOT / "configs"


def test_config_dir_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config.CONFIG_DIR_ENV, str(tmp_path))
    assert config.config_dir() == tmp_path


def test_missing_key_raises_config_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = yaml.safe_load((PROJECT_ROOT / "configs" / "experiment.yaml").read_text())
    del experiment["seed"]
    (tmp_path / "experiment.yaml").write_text(yaml.safe_dump(experiment))
    monkeypatch.setenv(config.CONFIG_DIR_ENV, str(tmp_path))

    with pytest.raises(ConfigError, match="seed"):
        config.get_experiment_config()


def test_missing_file_raises_config_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(config.CONFIG_DIR_ENV, str(tmp_path))
    with pytest.raises(ConfigError, match="not found"):
        config.load_config("dataset")


def test_non_mapping_config_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "models.yaml").write_text("- just\n- a list\n")
    monkeypatch.setenv(config.CONFIG_DIR_ENV, str(tmp_path))
    with pytest.raises(ConfigError, match="mapping"):
        config.get_models_config()


def test_config_error_is_fraudstudy_error() -> None:
    assert issubclass(ConfigError, FraudStudyError)


# --- paths ------------------------------------------------------------------


def test_project_root_contains_pyproject() -> None:
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
    assert find_project_root(Path(__file__)) == PROJECT_ROOT


def test_find_project_root_raises_without_marker(tmp_path: Path) -> None:
    # tmp_path normally lives outside the repo; skip if it happens not to.
    if any((p / "pyproject.toml").is_file() for p in [tmp_path, *tmp_path.parents]):
        pytest.skip("tmp_path is inside a directory tree containing pyproject.toml")
    with pytest.raises(ConfigError):
        find_project_root(tmp_path)


def test_resolve_relative_and_absolute(tmp_path: Path) -> None:
    assert resolve("configs/dataset.yaml") == PROJECT_ROOT / "configs" / "dataset.yaml"
    assert resolve(tmp_path) == tmp_path


# --- seed -------------------------------------------------------------------


def test_set_global_seed_makes_draws_repeatable() -> None:
    assert set_global_seed(123) == 123
    first_np, first_py = np.random.rand(5), random.random()
    set_global_seed(123)
    np.testing.assert_array_equal(np.random.rand(5), first_np)
    assert random.random() == first_py


def test_set_global_seed_sets_pythonhashseed(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    set_global_seed(7)
    assert os.environ["PYTHONHASHSEED"] == "7"


# --- io ---------------------------------------------------------------------


def test_to_builtin_converts_numpy_types() -> None:
    converted = to_builtin({"a": np.int64(3), "b": (np.float32(0.5),), "c": np.array([1, 2])})
    assert converted == {"a": 3, "b": [0.5], "c": [1, 2]}
    assert type(converted["a"]) is int


def test_save_json_round_trips_numpy_types(tmp_path: Path) -> None:
    payload = {
        "int": np.int64(5),
        "float": np.float64(0.25),
        "bool": np.bool_(True),
        "array": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "nested": {"z_first": 1, "a_second": np.int32(2)},
    }
    path = save_json(payload, tmp_path / "sub" / "out.json")
    loaded = load_json(path)

    assert loaded == {
        "int": 5,
        "float": 0.25,
        "bool": True,
        "array": [[1.0, 2.0], [3.0, 4.0]],
        "nested": {"z_first": 1, "a_second": 2},
    }
    assert list(loaded["nested"]) == ["z_first", "a_second"]  # sort_keys=False
    assert '\n  "int"' in path.read_text(encoding="utf-8")  # indent=2


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    content = b"fraudstudy" * 1000
    path.write_bytes(content)
    assert sha256_file(path) == hashlib.sha256(content).hexdigest()


def test_git_commit_hash_is_none_or_hex() -> None:
    commit = git_commit_hash()
    assert commit is None or (len(commit) == 40 and int(commit, 16) >= 0)


# --- CI ---------------------------------------------------------------------


def test_ci_workflow_is_valid_yaml() -> None:
    workflow = yaml.safe_load((PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text())
    assert "jobs" in workflow
