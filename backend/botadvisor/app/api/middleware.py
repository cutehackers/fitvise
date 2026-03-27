"""API middleware for request correlation."""

from __future__ import annotations

from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware

from botadvisor.app.observability.logging import get_logger


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Bind request IDs for the lifetime of an API request."""

    async def dispatch(self, request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid4())
        request.state.request_id = request_id
        logger = get_logger("api.request")

        with logger.with_request_id(request_id):
            response = await call_next(request)

        response.headers["x-request-id"] = request_id
        return response
