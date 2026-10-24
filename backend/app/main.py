"""The FastAPI application (master guide Parts 9 and 11).

Run from the repository root::

    uvicorn backend.app.main:app --reload

Start-up loads and validates every artifact once (:func:`load_artifacts`); it never crashes,
it reports problems in ``GET /api/v1/health``. Endpoints live under ``/api/v1``; the interactive
docs are at ``/docs``. Middleware, outermost first: CORS, then request id / timing / access log
(:class:`RequestContextMiddleware`). The backend reads artifacts only; it never trains models.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app import __version__
from backend.app.errors import install_error_handlers
from backend.app.logging import LOGGER_NAME, configure_logging
from backend.app.middleware import REQUEST_ID_HEADER, TIMING_HEADER, RequestContextMiddleware
from backend.app.ratelimit import configure_rate_limiting
from backend.app.routers import eda, health, models, presets, results
from backend.app.services.artifacts import load_artifacts
from backend.app.settings import Settings, get_settings

API_PREFIX = "/api/v1"
API_TITLE = "Fraud imbalance study API"
API_DESCRIPTION = (
    "Read-only results of the class-imbalance study on the ULB credit-card dataset. Research "
    "demo on anonymised public data; not for real fraud decisions."
)
CORS_METHODS = ["GET", "POST", "OPTIONS"]
CORS_HEADERS = ["Content-Type", REQUEST_ID_HEADER]
CORS_MAX_AGE_S = 600

logger = logging.getLogger(LOGGER_NAME)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the artifacts into ``app.state.store`` at start-up and log the outcome.

    Args:
        app: The application (its ``state.settings`` is set by :func:`create_app`).

    Yields:
        Nothing; the app serves requests while suspended here.
    """
    settings: Settings = app.state.settings
    store = load_artifacts(settings)
    app.state.store = store
    logger.info(
        "start-up: source=%s status=%s models_loaded=%d default_model=%s problems=%d",
        store.source,
        store.status,
        len(store.models),
        store.default_model_id,
        len(store.problems),
    )
    for problem in store.problems:
        logger.warning("problem: %s", problem)
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application: settings, logging, error handlers, rate limiter, middleware and
    routers. Artifacts load later, in :func:`lifespan`.

    Args:
        settings: Defaults to :func:`get_settings` (environment and ``.env``).

    Returns:
        The FastAPI application.
    """
    settings = settings if settings is not None else get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(
        title=API_TITLE, description=API_DESCRIPTION, version=__version__, lifespan=lifespan
    )
    app.state.settings = settings
    install_error_handlers(app)
    configure_rate_limiting(app, settings)
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(  # added last, so outermost: error responses get CORS headers too
        CORSMiddleware,
        allow_origins=settings.cors_origins(),
        allow_methods=CORS_METHODS,
        allow_headers=CORS_HEADERS,
        expose_headers=[REQUEST_ID_HEADER, TIMING_HEADER],
        allow_credentials=False,
        max_age=CORS_MAX_AGE_S,
    )
    for module in (health, models, results, eda, presets):
        app.include_router(module.router, prefix=API_PREFIX)
    return app


app = create_app()
