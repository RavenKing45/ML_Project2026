"""Request id, timing and access logging for every HTTP request (a pure ASGI middleware).

* Request id: the incoming ``X-Request-ID`` when it is a short safe token, else a new UUID4
  hex. Stored in ``request.state.request_id`` and returned in the ``X-Request-ID`` header.
* Timing: ``Server-Timing: app;dur=<ms>`` on the response (time until its headers).
* Access log: one record per request (:func:`backend.app.logging.log_access`). A route that
  scores with a model sets ``request.state.model_id`` so the record names it. The middleware
  never reads the request body.
* An exception that escapes the app becomes a 500 in the error shape carrying the request id,
  so the browser still gets JSON (and CORS headers, since CORS wraps this middleware). The log
  names the exception type and its stack, not its message, which could contain input values.
"""

import logging
import re
import time
import traceback
import uuid

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from backend.app.errors import STATUS_CODES, error_response
from backend.app.logging import LOGGER_NAME, log_access, safe_path

REQUEST_ID_HEADER = "X-Request-ID"
TIMING_HEADER = "Server-Timing"
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
MS_PER_S = 1000.0
INTERNAL_ERROR_STATUS = 500

logger = logging.getLogger(f"{LOGGER_NAME}.errors")


def request_id_from(headers: Headers) -> str:
    """The client's ``X-Request-ID`` if it matches ``REQUEST_ID_PATTERN``, else a new id.

    Args:
        headers: Request headers.

    Returns:
        The request id (a new one is 32 lowercase hex characters).
    """
    incoming = headers.get(REQUEST_ID_HEADER, "")
    return incoming if REQUEST_ID_PATTERN.fullmatch(incoming) else uuid.uuid4().hex


class RequestContextMiddleware:
    """Adds the request id and timing headers, logs one access record, and turns an escaped
    exception into a 500 error response.

    Args:
        app: The wrapped ASGI application.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI connection (non-HTTP scopes pass through untouched)."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        start = time.perf_counter()
        request_id = request_id_from(Headers(scope=scope))
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        status = INTERNAL_ERROR_STATUS
        started = False

        async def send_with_headers(message: Message) -> None:
            nonlocal status, started
            if message["type"] == "http.response.start":
                started = True
                status = int(message["status"])
                headers = MutableHeaders(scope=message)
                headers[REQUEST_ID_HEADER] = request_id
                elapsed_ms = (time.perf_counter() - start) * MS_PER_S
                headers[TIMING_HEADER] = f"app;dur={elapsed_ms:.1f}"
            await send(message)

        try:
            await self.app(scope, receive, send_with_headers)
        except Exception as exc:
            stack = "".join(traceback.format_tb(exc.__traceback__))
            logger.error(
                "unhandled %s request_id=%s path=%s\n%s",
                type(exc).__name__,
                request_id,
                safe_path(str(scope.get("path", ""))),
                stack,
            )
            if started:
                raise
            response = error_response(
                INTERNAL_ERROR_STATUS,
                STATUS_CODES[INTERNAL_ERROR_STATUS],
                f"Internal server error. Quote request id {request_id} when reporting it.",
                {"request_id": request_id},
            )
            await response(scope, receive, send_with_headers)
        finally:
            log_access(
                request_id,
                str(scope.get("method", "")),
                str(scope.get("path", "")),
                status,
                (time.perf_counter() - start) * MS_PER_S,
                state.get("model_id"),
            )
