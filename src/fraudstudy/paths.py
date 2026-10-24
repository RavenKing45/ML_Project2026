"""Project-root discovery and path resolution. Never hard-code absolute paths elsewhere."""

from pathlib import Path

from fraudstudy.errors import ConfigError

_MARKER = "pyproject.toml"


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` until a directory containing ``pyproject.toml`` is found.

    Args:
        start: Directory or file to start from. Defaults to this module's location.

    Returns:
        The absolute path of the project root.

    Raises:
        ConfigError: If no ancestor contains ``pyproject.toml``.
    """
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / _MARKER).is_file():
            return candidate
    raise ConfigError(
        f"Could not find {_MARKER} above {here}. Install the package with "
        "`pip install -e .` from the repository root."
    )


PROJECT_ROOT: Path = find_project_root()


def resolve(rel_path: str | Path) -> Path:
    """Resolve a path relative to the project root.

    Args:
        rel_path: A path relative to the project root (absolute paths are returned unchanged).

    Returns:
        The absolute path.
    """
    path = Path(rel_path)
    return path if path.is_absolute() else PROJECT_ROOT / path
