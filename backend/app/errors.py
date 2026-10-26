"""One body for every non-2xx response: ``{"error": {"code", "message", "details"}}``.

``code`` is a stable snake_case string the dashboard may switch on; ``message`` says what went
wrong and what to do; ``details`` is any JSON value, or ``null``. Bodies are built through the
``ErrorResponse`` contract, so they always validate. FastAPI's own 404 / 405 bodies and its 422
validation body are replaced by this shape. Messages and details never contain request bodies
or feature values: a 422 lists only where and why each field failed, not the value sent.

Unhandled exceptions become a 500 in :mod:`backend.app.middleware`, which knows the request id.
"""

from collections.abc import Sequence
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from fraudstudy.contracts import ErrorDetail, ErrorResponse

STATUS_CODES: dict[int, str] = {
    400: "bad_request",
    404: "not_found",
    405: "method_not_allowed",
    413: "payload_too_large",
    422: "validation_error",
    429: "rate_limited",
    500: "internal_error",
    503: "service_unavailable",
}
ARTIFACT_UNAVAILABLE = "artifact_unavailable"
DOCS_PATH = "/docs"


class ArtifactError(Exception):
    """An artifact file is missing, unreadable, or fails a checksum or version check."""


class ApiError(Exception):
    """An error the API returns on purpose, with its HTTP status and error-shape fields.

    Args:
        status_code: HTTP status (400, 404, 413, 422, 429, 500 or 503).
        code: Stable snake_case code, e.g. ``"artifact_unavailable"``.
        message: What went wrong and what to do; never request bodies or feature values.
        details: Optional JSON-serialisable context.
        headers: Optional response headers (e.g. ``Retry-After``).
    """

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: Any = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details
        self.headers = headers


def error_code_for_status(status_code: int) -> str:
    """The error code for an HTTP status (``http_<status>`` for one without a named code).

    Args:
        status_code: HTTP status.

    Returns:
        A snake_case code, e.g. ``404 -> "not_found"``.
    """
    return STATUS_CODES.get(status_code, f"http_{status_code}")


def error_body(code: str, message: str, details: Any = None) -> dict[str, Any]:
    """Build an error body through the contract (``details`` is always present).

    Args:
        code: Error code.
        message: Human message.
        details: Context, or ``None``.

    Returns:
        ``{"error": {"code": ..., "message": ..., "details": ...}}`` as plain JSON types.
    """
    detail = ErrorDetail(code=code, message=message, details=details)
    return ErrorResponse(error=detail).model_dump(mode="json")


def error_response(
    status_code: int,
    code: str,
    message: str,
    details: Any = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """A ``JSONResponse`` with the error body.

    Args:
        status_code: HTTP status.
        code: Error code.
        message: Human message.
        details: Context, or ``None``.
        headers: Extra response headers.

    Returns:
        The response.
    """
    return JSONResponse(error_body(code, message, details), status_code, headers=headers)


def validation_details(errors: Sequence[Any]) -> list[dict[str, Any]]:
    """Field errors of a 422 as ``{"loc", "msg", "type"}``, without the submitted values.

    Pydantic's errors also carry ``input`` (the value sent) and ``ctx``; both are dropped so a
    422 never echoes feature values back or into logs.

    Args:
        errors: ``RequestValidationError.errors()``.

    Returns:
        One dict per error.
    """
    return [
        {
            "loc": [str(part) for part in err.get("loc", ())],
            "msg": str(err.get("msg", "")),
            "type": str(err.get("type", "")),
        }
        for err in errors
    ]


def _phrase(status_code: int) -> str:
    """The standard reason phrase of a status (``""`` for a non-standard one)."""
    try:
        return HTTPStatus(status_code).phrase
    except ValueError:
        return ""


def _http_message(request: Request, status_code: int, detail: Any) -> str:
    """The message for an ``HTTPException``: its own detail, or a default per status."""
    default_phrase = _phrase(status_code)
    if isinstance(detail, str) and detail and detail != default_phrase:
        if status_code == HTTPStatus.TOO_MANY_REQUESTS:
            return f"Too many requests (limit: {detail}). Wait a minute and try again."
        return detail
    method, path = request.method, request.url.path
    if status_code == HTTPStatus.NOT_FOUND:
        return f"No endpoint at {method} {path}. Endpoints live under /api/v1; see {DOCS_PATH}."
    if status_code == HTTPStatus.METHOD_NOT_ALLOWED:
        return f"{method} is not allowed on {path}; see {DOCS_PATH} for the allowed methods."
    if status_code == HTTPStatus.TOO_MANY_REQUESTS:
        return "Too many requests. Wait a minute and try again."
    return default_phrase or f"HTTP error {status_code}."


async def api_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Turn an :class:`ApiError` into its error response."""
    if not isinstance(exc, ApiError):  # pragma: no cover - registered for ApiError only
        raise exc
    return error_response(exc.status_code, exc.code, exc.message, exc.details, exc.headers)


async def http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Turn Starlette/FastAPI ``HTTPException`` (404, 405, slowapi's 429, ...) into the shape."""
    if not isinstance(exc, StarletteHTTPException):  # pragma: no cover
        raise exc
    status = exc.status_code
    details = {"path": request.url.path} if status == HTTPStatus.NOT_FOUND else None
    message = _http_message(request, status, exc.detail)
    headers = dict(exc.headers) if exc.headers else None
    return error_response(status, error_code_for_status(status), message, details, headers)


async def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Replace FastAPI's default 422 body with the error shape (field errors in ``details``)."""
    if not isinstance(exc, RequestValidationError):  # pragma: no cover
        raise exc
    details = validation_details(exc.errors())
    noun = "problem" if len(details) == 1 else "problems"
    message = f"The request is invalid ({len(details)} {noun}); see details for each field."
    return error_response(HTTPStatus.UNPROCESSABLE_ENTITY, STATUS_CODES[422], message, details)


def install_error_handlers(app: FastAPI) -> None:
    """Register the handlers above on ``app``.

    Args:
        app: The FastAPI application.
    """
    app.add_exception_handler(ApiError, api_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
