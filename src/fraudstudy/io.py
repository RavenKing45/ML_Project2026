"""Reading and writing JSON results, hashing files and reading the git commit.

Scripts call save_json() for every result file in artifacts/, load_json() to read them back,
sha256_file() to pin a dataset, and git_hash() to record which code produced a result.
"""

import hashlib
import json
import subprocess
from pathlib import Path

from fraudstudy.config import PROJECT_ROOT


def to_plain(value: object) -> object:
    """Convert a value json cannot write itself: a path, or a NumPy number or array."""
    if isinstance(value, Path):
        return value.as_posix()
    # NumPy scalars and arrays both have tolist(), which returns plain Python numbers and lists.
    return value.tolist()


def save_json(data: object, path: str | Path) -> Path:
    """Write `data` as indented UTF-8 JSON (keys kept in order), creating folders as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, ensure_ascii=False, default=to_plain)
    path.write_text(text, encoding="utf-8")
    return path


def load_json(path: str | Path) -> object:
    """Read a UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of a file as 64 hex characters."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        # Read 1 MB at a time so the 150 MB dataset never sits in memory at once.
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_hash() -> str | None:
    """Return the current git commit hash, or None when git or the repository is missing."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None
