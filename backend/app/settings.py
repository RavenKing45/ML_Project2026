"""Backend settings, read from environment variables or a ``.env`` file at the project root.

Each variable is its field name in upper case (master guide Part 11, "Configuration"):
``ARTIFACTS_DIR``, ``MODELS_DIR``, ``USE_EXAMPLE_ARTIFACTS``, ``EXAMPLES_DIR``,
``ALLOWED_ORIGINS``, ``ENVIRONMENT``, ``DEFAULT_MODEL_ID``, ``MAX_UPLOAD_MB``, ``MAX_BATCH_ROWS``,
``LOG_LEVEL``, ``RATE_LIMIT_ENABLED``. Relative paths resolve from the project root
(``fraudstudy.config.resolve``), not from the working directory.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import PositiveFloat, PositiveInt, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from fraudstudy.config import PROJECT_ROOT, resolve

ENV_FILE = ".env"
# The Vite dev server; allowed automatically in development (master guide Part 11).
DEV_ORIGINS: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")
WILDCARD = "*"

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
Environment = Literal["development", "production"]


class Settings(BaseSettings):
    """Every backend setting, with its default.

    Attributes:
        artifacts_dir: Holds ``results/`` (comparison, curves, stats, eda, resampling audit,
            temporal check) and ``demo/presets.json``.
        models_dir: Holds ``manifest.json`` and the model files it lists.
        use_example_artifacts: Serve the fake fixtures in ``examples_dir`` instead (frontend
            work without real results); no model file is loaded in this mode.
        examples_dir: The fixtures folder (``docs/contracts/examples``).
        allowed_origins: Comma-separated CORS origins, e.g. ``https://app.vercel.app``.
        environment: ``development`` also allows the Vite dev server; ``production`` refuses
            ``*`` and example artifacts.
        default_model_id: Overrides the manifest's ``default_model_id`` (a manifest id).
        max_upload_mb: Largest CSV upload of ``POST /predict/batch``, in MB.
        max_batch_rows: Most data rows in one CSV upload.
        log_level: Python logging level of the ``backend`` loggers.
        rate_limit_enabled: Turns every slowapi limit on or off.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    artifacts_dir: Path = Path("artifacts")
    models_dir: Path = Path("artifacts/models")
    use_example_artifacts: bool = False
    examples_dir: Path = Path("docs/contracts/examples")
    allowed_origins: str = ""
    environment: Environment = "development"
    default_model_id: str | None = None
    max_upload_mb: PositiveFloat = 5
    max_batch_rows: PositiveInt = 10_000
    log_level: LogLevel = "INFO"
    rate_limit_enabled: bool = True

    @field_validator("artifacts_dir", "models_dir", "examples_dir", mode="after")
    @classmethod
    def _from_project_root(cls, value: Path) -> Path:
        """Resolve a relative path from the project root."""
        return resolve(value)

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper(cls, value: Any) -> Any:
        """Accept ``info`` as well as ``INFO``."""
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("environment", mode="before")
    @classmethod
    def _lower(cls, value: Any) -> Any:
        """Accept ``Production`` as well as ``production``."""
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("default_model_id", mode="before")
    @classmethod
    def _blank_is_none(cls, value: Any) -> Any:
        """Treat ``DEFAULT_MODEL_ID=`` (empty) as unset."""
        return None if isinstance(value, str) and not value.strip() else value

    def listed_origins(self) -> list[str]:
        """``ALLOWED_ORIGINS`` split on commas, stripped, without trailing ``/``, de-duplicated.

        Returns:
            The origins in the order given.
        """
        parts = (part.strip().rstrip("/") for part in self.allowed_origins.split(","))
        return list(dict.fromkeys(part for part in parts if part))

    def cors_origins(self) -> list[str]:
        """The origins the CORS middleware allows.

        Development: the listed origins plus :data:`DEV_ORIGINS`. Production: the listed
        origins without ``*`` (a wildcard would let any site call the API).

        Returns:
            Allowed origins.
        """
        origins = self.listed_origins()
        if self.environment == "production":
            return [origin for origin in origins if origin != WILDCARD]
        return origins + [origin for origin in DEV_ORIGINS if origin not in origins]

    def config_problems(self) -> list[str]:
        """Settings that are unsafe or useless in production, as messages saying what to fix.

        Returns:
            Problems (empty in development). ``/health`` reports them and turns "degraded".
        """
        if self.environment != "production":
            return []
        problems: list[str] = []
        if WILDCARD in self.listed_origins():
            problems.append(
                "ALLOWED_ORIGINS contains '*', which is not allowed in production, so it was "
                "ignored. List the dashboard's address instead (e.g. https://<app>.vercel.app)."
            )
        if not self.cors_origins():
            problems.append(
                "ALLOWED_ORIGINS is empty in production, so browsers cannot call the API. Set it "
                "to the dashboard's address."
            )
        if self.use_example_artifacts:
            problems.append(
                "USE_EXAMPLE_ARTIFACTS is true in production, so the API serves fake example "
                "data. Set it to false and bake the real artifacts into the image."
            )
        return problems


@lru_cache
def get_settings() -> Settings:
    """The settings from the environment, read once per process.

    Returns:
        The cached :class:`Settings`.
    """
    return Settings()
