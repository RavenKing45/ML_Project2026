"""Tests for fraudstudy.config (settings, paths, seed) and fraudstudy.io."""

import hashlib
import os
import random
from pathlib import Path

import numpy as np
import pytest
import yaml

from fraudstudy import config
from fraudstudy.config import PROJECT_ROOT, resolve, set_seed
from fraudstudy.io import git_hash, load_json, save_json, sha256_file

# --- config -----------------------------------------------------------------


def test_load_all_three_configs() -> None:
    dataset = config.get_dataset_config()
    experiment = config.get_experiment_config()
    models = config.get_models_config()

    assert dataset["target"] == "Class"
    assert dataset["pca_features"] == [f"V{i}" for i in range(1, 29)]
    assert dataset["excluded_from_features"] == ["Time"]
    assert experiment["seed"] == 42
    assert experiment["primary_metric"] == "average_precision"
    assert set(experiment["models"]) == set(models)
    assert "use_hour_feature" not in experiment  # removed in October 2026


def test_missing_key_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    experiment = yaml.safe_load((PROJECT_ROOT / "configs" / "experiment.yaml").read_text())
    del experiment["seed"]
    (tmp_path / "experiment.yaml").write_text(yaml.safe_dump(experiment))
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)

    with pytest.raises(ValueError, match="seed"):
        config.get_experiment_config()


def test_missing_file_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="not found"):
        config.load_config("dataset")


# --- paths ------------------------------------------------------------------


def test_project_root_contains_pyproject() -> None:
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
    assert config.CONFIG_DIR == PROJECT_ROOT / "configs"


def test_resolve_relative_and_absolute(tmp_path: Path) -> None:
    assert resolve("configs/dataset.yaml") == PROJECT_ROOT / "configs" / "dataset.yaml"
    assert resolve(tmp_path) == tmp_path


# --- seed -------------------------------------------------------------------


def test_set_seed_makes_draws_repeatable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    set_seed(123)
    first_np, first_py = np.random.rand(5), random.random()
    set_seed(123)
    np.testing.assert_array_equal(np.random.rand(5), first_np)
    assert random.random() == first_py
    assert os.environ["PYTHONHASHSEED"] == "123"


# --- io ---------------------------------------------------------------------


def test_save_json_round_trips_numpy_types(tmp_path: Path) -> None:
    payload = {
        "int": np.int64(5),
        "float": np.float64(0.25),
        "bool": np.bool_(True),
        "array": np.array([[1.0, 2.0], [3.0, 4.0]]),
        "tuple": (np.float32(0.5),),
        "path": Path("a") / "b.json",
        "nested": {"z_first": 1, "a_second": np.int32(2)},
    }
    path = save_json(payload, tmp_path / "sub" / "out.json")
    loaded = load_json(path)

    assert loaded == {
        "int": 5,
        "float": 0.25,
        "bool": True,
        "array": [[1.0, 2.0], [3.0, 4.0]],
        "tuple": [0.5],
        "path": "a/b.json",
        "nested": {"z_first": 1, "a_second": 2},
    }
    assert list(loaded["nested"]) == ["z_first", "a_second"]  # key order kept
    assert '\n  "int"' in path.read_text(encoding="utf-8")  # indent=2


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    content = b"fraudstudy" * 1000
    path.write_bytes(content)
    assert sha256_file(path) == hashlib.sha256(content).hexdigest()


def test_git_hash_is_none_or_hex() -> None:
    commit = git_hash()
    assert commit is None or (len(commit) == 40 and int(commit, 16) >= 0)


# --- CI ---------------------------------------------------------------------


def test_ci_workflow_is_valid_yaml() -> None:
    workflow = yaml.safe_load((PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text())
    assert "jobs" in workflow
