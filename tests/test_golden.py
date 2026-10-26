"""Tests for the comparison in scripts/golden.py (a full save/check run is too slow for pytest)."""

import importlib.util
from pathlib import Path

import pandas as pd

from fraudstudy.config import PROJECT_ROOT

_spec = importlib.util.spec_from_file_location("golden", PROJECT_ROOT / "scripts" / "golden.py")
assert _spec is not None and _spec.loader is not None
golden = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(golden)


def differences(old: object, new: object) -> list:
    diffs: list = []
    golden.compare(old, new, "f", diffs)
    return diffs


def test_numbers_within_tolerance_are_equal() -> None:
    old = {"a": 0.5, "b": [1, 2.0], "c": {"d": "x", "e": None, "f": float("nan")}}
    new = {"a": 0.5 + 1e-12, "b": [1, 2.0], "c": {"d": "x", "e": None, "f": float("nan")}}
    assert differences(old, new) == []


def test_changed_number_text_or_length_is_reported() -> None:
    assert differences({"a": 0.5}, {"a": 0.5 + 1e-6}) == [("f.a", "0.5", "0.500001")]
    assert differences({"a": "x"}, {"a": "y"}) == [("f.a", "'x'", "'y'")]
    assert differences([1, 2], [1]) == [("f", "2 items", "1 items")]
    assert differences(True, 1) != []  # a bool is not the number 1


def test_missing_and_extra_keys_are_reported() -> None:
    diffs = differences({"a": 1, "b": 2}, {"a": 1, "c": 2})
    assert ("f.b", "key", "missing") in diffs and ("f.c", "missing", "key") in diffs


def test_skip_keys_ignore_values_only() -> None:
    assert "git_commit" in golden.SKIP_KEYS
    assert differences({"git_commit": "abc"}, {"git_commit": "def"}) == []
    assert differences({"git_commit": "abc"}, {}) == [("f.git_commit", "key", "missing")]


def test_output_files_and_read(tmp_path: Path) -> None:
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "a.json").write_text('{"x": 1}')
    pd.DataFrame({"x": [1.5]}).to_csv(tmp_path / "b.csv", index=False)
    (tmp_path / "c.png").write_bytes(b"not compared")
    assert golden.output_files(tmp_path) == ["b.csv", "results/a.json"]
    assert golden.read(tmp_path / "results" / "a.json") == {"x": 1}
    assert golden.read(tmp_path / "b.csv") == [{"x": 1.5}]
