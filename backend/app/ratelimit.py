"""Per-client rate limiting with slowapi (master guide Part 11, "Security basics").

``limiter`` is module-level so routers can decorate endpoints at import time, e.g.
``@limiter.limit("60/minute")`` (the endpoint must take ``request: Request``). No endpoint is
limited yet; the prediction endpoints will be. ``RATE_LIMIT_ENABLED=false`` switches every limit
off. A client over its limit gets slowapi's ``RateLimitExceeded`` (an HTTP 429), which
:mod:`backend.app.errors` turns into the error shape. Clients are keyed by IP address; behind a
proxy run uvicorn with ``--proxy-headers`` so that is the visitor's address, not the proxy's.
"""

from fastapi import FastAPI
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.app.settings import Settings

limiter = Limiter(key_func=get_remote_address)


def configure_rate_limiting(app: FastAPI, settings: Settings) -> Limiter:
    """Switch the shared limiter on or off and attach it to ``app`` (slowapi reads it there).

    Args:
        app: The FastAPI application.
        settings: Uses ``rate_limit_enabled``.

    Returns:
        The shared :data:`limiter`.
    """
    limiter.enabled = settings.rate_limit_enabled
    app.state.limiter = limiter
    return limiter
