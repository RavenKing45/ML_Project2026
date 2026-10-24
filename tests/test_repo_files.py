"""Checks on repository files: the requirement sets and the model ignore rules.

The backend and CI install ``requirements.txt`` / ``requirements-dev.txt``; the notebook's
packages live in ``requirements-notebook.txt`` (docs/decisions.md). Only deployed models may be
committed.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from fraudstudy.paths import PROJECT_ROOT, resolve

NOTEBOOK_PACKAGES = ("jupyter", "jupyterlab", "notebook", "ipykernel")
MODEL_BUILD = "-e ./model_build"


def _lines(path: Path) -> list[str]:
    """Requirement lines of ``path``, following ``-r`` includes; comments and blanks dropped."""
    out: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.startswith("-r "):
            out += _lines(path.parent / line[3:].strip())
        elif line:
            out.append(line)
    return out


def _name(line: str) -> str:
    """Package name of a requirement line (``jupyter==1.1.1`` -> ``jupyter``)."""
    for sep in ("==", ">=", "<=", "~=", ">", "<", "["):
        line = line.split(sep, 1)[0]
    return line.strip().lower()


@pytest.mark.parametrize("name", ["requirements.txt", "requirements-dev.txt"])
def test_backend_and_ci_requirements_have_no_notebook_packages(name: str) -> None:
    lines = _lines(resolve(name))
    assert not [line for line in lines if _name(line) in NOTEBOOK_PACKAGES]
    assert not [line for line in lines if "model_build" in line]


def test_notebook_requirements_extend_the_project_set() -> None:
    lines = _lines(resolve("requirements-notebook.txt"))
    assert set(_lines(resolve("requirements.txt"))) <= set(lines)
    assert MODEL_BUILD in lines and "jupyter" in {_name(line) for line in lines}


@pytest.mark.parametrize(
    ("path", "ignored"),
    [
        ("artifacts/models/deploy/lightgbm__smote_rus.joblib", False),
        ("artifacts/models/deploy/lightgbm__smote_rus.calibrator.joblib", False),
        ("artifacts/models/cells/lightgbm__smote_rus.joblib", True),
        ("artifacts/models/sample/deploy/lightgbm__smote_rus.joblib", True),
        ("artifacts/models/lightgbm__smote_rus.joblib", True),
        ("data/raw/creditcard.csv", True),
        ("data/raw/.gitkeep", False),
    ],
)
def test_only_deployed_models_are_committable(path: str, ignored: bool) -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", path], cwd=PROJECT_ROOT, check=False
    )
    assert result.returncode in (0, 1), "git check-ignore failed"
    assert (result.returncode == 0) == ignored
