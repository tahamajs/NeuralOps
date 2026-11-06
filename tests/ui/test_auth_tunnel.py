#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

"""Unit tests for Phase 9 Share mode, Cloudflare tunnel, and Basic Authentication."""

import base64
from unittest.mock import MagicMock, patch

from starlette.requests import Request
from starlette.responses import PlainTextResponse

from neuralops.ui.auth import BasicAuthMiddleware
from neuralops.ui.server import create_app
from neuralops.ui.tunnel import CloudflareTunnel


def _make_auth_header(user: str, pwd: str) -> str:
    encoded = base64.b64encode(f"{user}:{pwd}".encode("utf-8")).decode("utf-8")
    return f"Basic {encoded}"


def test_basic_auth_middleware_blocks_unauthorized():
    """Verify BasicAuthMiddleware rejects unauthenticated requests with 401."""
    async def dummy_app(scope, receive, send):
        response = PlainTextResponse("Secret")
        await response(scope, receive, send)

    middleware = BasicAuthMiddleware(dummy_app, username="admin", password="password123")

    # No header
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [],
    }
    req = Request(scope)

    async def call_next(r):
        return PlainTextResponse("OK")

    import asyncio
    resp = asyncio.run(middleware.dispatch(req, call_next))
    assert resp.status_code == 401
    assert "Basic" in resp.headers.get("www-authenticate", "")


def test_basic_auth_middleware_permits_valid_credentials():
    """Verify BasicAuthMiddleware accepts valid credentials."""
    async def dummy_app(scope, receive, send):
        response = PlainTextResponse("Secret")
        await response(scope, receive, send)

    middleware = BasicAuthMiddleware(dummy_app, username="admin", password="password123")
    header_val = _make_auth_header("admin", "password123")

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [(b"authorization", header_val.encode("utf-8"))],
    }
    req = Request(scope)

    async def call_next(r):
        return PlainTextResponse("Authorized OK")

    import asyncio
    resp = asyncio.run(middleware.dispatch(req, call_next))
    assert resp.status_code == 200


def test_basic_auth_exempts_health_check():
    """Verify /health endpoint remains unauthenticated for monitoring probes."""
    middleware = BasicAuthMiddleware(None, username="admin", password="password123")
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/health",
        "headers": [],
    }
    req = Request(scope)

    async def call_next(r):
        return PlainTextResponse("Health OK")

    import asyncio
    resp = asyncio.run(middleware.dispatch(req, call_next))
    assert resp.status_code == 200


def test_server_with_auth_credentials():
    """Verify create_app instantiates cleanly with auth_credentials."""
    app = create_app(auth_credentials=("taha", "neuralops2026"))
    assert app is not None


def test_cloudflare_tunnel_missing_binary():
    """Verify CloudflareTunnel raises RuntimeError when binary is not in PATH."""
    tunnel = CloudflareTunnel(port=8080)
    with patch("shutil.which", return_value=None):
        try:
            tunnel.start()
            raise AssertionError("Should raise RuntimeError")
        except RuntimeError as e:
            assert "cloudflared executable not found" in str(e)


def test_cloudflare_tunnel_lifecycle():
    """Verify CloudflareTunnel parses trycloudflare.com URL and stops."""
    tunnel = CloudflareTunnel(port=8080)

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.stderr.readline.side_effect = [
        "2026-11-06 INF Starting tunnel\n",
        "2026-11-06 INF Your quick Tunnel has been created: https://test-sample-domain.trycloudflare.com\n",
    ]

    with patch("shutil.which", return_value="/usr/local/bin/cloudflared"), \
         patch("subprocess.Popen", return_value=mock_proc):
        url = tunnel.start(timeout_sec=2)
        assert url == "https://test-sample-domain.trycloudflare.com"
        assert tunnel.tunnel_url == url

        tunnel.stop()
        mock_proc.terminate.assert_called_once()
        assert tunnel.process is None
