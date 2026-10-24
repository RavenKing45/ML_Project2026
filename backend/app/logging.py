"""Logging set-up and the one-line access record of each request.

The access record has the time, request id, method, path (never the query string), status,
latency and model id. Request and response bodies, feature values and uploaded files are never
logged (CLAUDE.md; master guide Part 9, "Logging"). Paths are sanitised so a crafted URL cannot
forge extra log lines.
"""

import logging
import time

LOGGER_NAME = "backend"
ACCESS_LOGGER_NAME = f"{LOGGER_NAME}.access"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
NO_MODEL = "-"
MAX_LOGGED_PATH = 200
_HANDLER_MARK = "_fraudstudy_backend_handler"


class UtcFormatter(logging.Formatter):
    """A formatter that writes ``asctime`` in UTC (``DATE_FORMAT`` ends in ``Z``)."""

    converter = time.gmtime


def configure_logging(level: str) -> logging.Logger:
    """Give the ``backend`` logger one stderr handler and set its level (safe to call again).

    Records still propagate to the root logger, so test tools can capture them.

    Args:
        level: A logging level name, e.g. ``"INFO"``.

    Returns:
        The ``backend`` logger.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    if not any(getattr(handler, _HANDLER_MARK, False) for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(UtcFormatter(LOG_FORMAT, DATE_FORMAT))
        setattr(handler, _HANDLER_MARK, True)
        logger.addHandler(handler)
    return logger


def safe_path(path: str) -> str:
    """``path`` with non-printable characters replaced by ``?`` and cut to ``MAX_LOGGED_PATH``.

    Args:
        path: The decoded request path.

    Returns:
        A single-line path, safe to log.
    """
    cleaned = "".join(ch if ch.isprintable() and ch != " " else "?" for ch in path)
    return cleaned[:MAX_LOGGED_PATH]


def access_message(
    request_id: str,
    method: str,
    path: str,
    status: int,
    latency_ms: float,
    model_id: str | None,
) -> str:
    """Format the access record as ``key=value`` pairs.

    Args:
        request_id: The request id.
        method: HTTP method.
        path: Request path (sanitised here).
        status: Response status.
        latency_ms: Time from receiving the request to the end of the response, in ms.
        model_id: Model that served the request, or ``None``.

    Returns:
        E.g. ``request_id=ab12 method=GET path=/api/v1/health status=200 latency_ms=1.2
        model_id=-``.
    """
    return (
        f"request_id={request_id} method={method} path={safe_path(path)} status={status} "
        f"latency_ms={latency_ms:.1f} model_id={model_id or NO_MODEL}"
    )


def log_access(
    request_id: str,
    method: str,
    path: str,
    status: int,
    latency_ms: float,
    model_id: str | None,
) -> None:
    """Write the access record at INFO on the ``backend.access`` logger (args as
    :func:`access_message`)."""
    logging.getLogger(ACCESS_LOGGER_NAME).info(
        access_message(request_id, method, path, status, latency_ms, model_id)
    )
