#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

"""HTTP Basic Authentication middleware for NeuralOps Web UI."""

from __future__ import annotations

import base64
import secrets
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class BasicAuthMiddleware(BaseHTTPMiddleware):
    """Enforce HTTP Basic Authentication across all UI routes except /health."""

    def __init__(self, app, username: str, password: str):
        super().__init__(app)
        self.username = username
        self.password = password

    async def dispatch(self, request: Request, call_next):
        # Exclude /health check from authentication
        if request.url.path == "/health":
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Basic "):
            return Response(
                content="Unauthorized: Access to NeuralOps Web UI requires authentication.",
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="NeuralOps UI"'},
            )

        try:
            encoded_credentials = auth_header.split(" ", 1)[1]
            decoded_bytes = base64.b64decode(encoded_credentials)
            decoded_str = decoded_bytes.decode("utf-8")
            user, pwd = decoded_str.split(":", 1)
        except Exception:
            return Response(
                content="Invalid credentials format.",
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="NeuralOps UI"'},
            )

        is_user_correct = secrets.compare_digest(user, self.username)
        is_pwd_correct = secrets.compare_digest(pwd, self.password)

        if not (is_user_correct and is_pwd_correct):
            return Response(
                content="Invalid username or password.",
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="NeuralOps UI"'},
            )

        return await call_next(request)
