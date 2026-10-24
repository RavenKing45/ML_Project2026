"""JSON persistence, file hashing and provenance helpers."""

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from fraudstudy.paths import PROJECT_ROOT

_HASH_CHUNK_BYTES = 1 << 20


def to_builtin(obj: Any) -> Any:
    """Recursively convert NumPy scalars/arrays (and tuples) to plain Python types.

    Args:
        obj: Any JSON-like structure possibly containing NumPy values.

    Returns:
        An equivalent structure containing only built-in Python types.
    """
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_builtin(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return obj.as_posix()
    return obj


def save_json(data: Any, path: str | Path) -> Path:
    """Write ``data`` as UTF-8 JSON (indent=2, key order preserved), creating parent dirs.

    Args:
        data: JSON-like structure; NumPy types are converted to plain Python.
        path: Destination file.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_builtin(data), indent=2, sort_keys=False, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def load_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON file.

    Args:
        path: Source file.

    Returns:
        The parsed JSON value.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: str | Path) -> str:
    """Return the hex SHA-256 digest of a file, read in 1 MiB chunks.

    Args:
        path: File to hash.

    Returns:
        64-character lowercase hex digest.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit_hash() -> str | None:
    """Return the current git HEAD commit hash, or ``None`` if unavailable.

    Returns:
        The 40-character commit hash, or ``None`` outside a git repo or without git.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None
