#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
CLI Web UI Command Module.

Implements argparse parser configuration and command handler for the NeuralOps Web UI.
"""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from typing import Optional


def _add_ui_arguments(parser: argparse.ArgumentParser) -> None:
    """Add standard UI runtime arguments to parser."""
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Server port (default: 8080)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server bind host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Create a public Cloudflare tunnel to share the UI",
    )
    parser.add_argument(
        "--static",
        type=str,
        default=None,
        metavar="DIR",
        help="Generate a static HTML bundle for HuggingFace Spaces deployment",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        dest="no_browser",
        default=False,
        help="Do not automatically launch web browser on startup",
    )
    parser.add_argument(
        "--auth",
        type=str,
        default=None,
        metavar="USERNAME",
        help="Basic authentication username (required with --share). Password is read from NEURALOPS_UI_AUTH or stdin via getpass.",
    )


def build_ui_parser(subparser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    """Construct argument parser for neuralops ui subcommand."""
    parser = subparser or argparse.ArgumentParser(
        prog="neuralops ui",
        description="Launch NeuralOps publication-grade Web UI dashboard.",
    )
    _add_ui_arguments(parser)

    # Support optional `start` sub-action for parity with start workflows
    ui_subparsers = parser.add_subparsers(dest="ui_action", help="UI sub-actions")
    start_p = ui_subparsers.add_parser("start", help="Start the Web UI server")
    _add_ui_arguments(start_p)
    start_p.set_defaults(handler=cmd_ui)

    return parser


def cmd_ui(args: argparse.Namespace) -> int:
    """Execute the UI CLI command starting the FastAPI server."""
    port = getattr(args, "port", 8080)
    host = getattr(args, "host", "127.0.0.1")
    share = getattr(args, "share", False)
    static_path = getattr(args, "static", None)
    no_browser = getattr(args, "no_browser", False)
    auth = getattr(args, "auth", None)

    if share and not auth:
        raise ValueError("--auth is required when sharing the Web UI via --share.")

    auth_creds = None
    if auth:
        import getpass
        import os

        if ":" in auth:
            raise ValueError(
                "Passing passwords via '--auth USER:PASS' is no longer supported for security. "
                "Pass '--auth USERNAME' and set NEURALOPS_UI_AUTH or enter password when prompted."
            )
        user = auth
        pwd = os.environ.get("NEURALOPS_UI_AUTH")
        if not pwd:
            pwd = getpass.getpass(f"Enter password for Web UI user '{user}': ")
        if not pwd:
            raise ValueError("Web UI authentication password cannot be empty.")
        auth_creds = (user, pwd)
        print(f"[SECURITY] Basic authentication enabled for user: {auth_creds[0]}")

    tunnel = None
    if share:
        from neuralops.ui.tunnel import CloudflareTunnel

        tunnel = CloudflareTunnel(port=port, host=host)
        try:
            tunnel_url = tunnel.start()
            print(f"[SHARE] Public Cloudflare tunnel active at: {tunnel_url}")
        except Exception as exc:
            print(f"[ERROR] Failed to establish Cloudflare tunnel: {exc}", file=sys.stderr)
            return 1

    if static_path:
        from neuralops.ui.static_builder import build_static_site

        print(f"Building static HTML distribution bundle into {static_path} ...")
        try:
            summary = build_static_site(static_path)
            print(
                f"[SUCCESS] Static site generated successfully: "
                f"{summary['total_pages']} pages created at {summary['output_directory']}"
            )
            return 0
        except Exception as exc:
            print(f"[ERROR] Failed to build static site: {exc}", file=sys.stderr)
            return 1

    url = f"http://{host}:{port}"
    print(f"Starting NeuralOps Web UI server on {url} ...")

    if not no_browser and not share:

        def _open_browser() -> None:
            try:
                webbrowser.open(url)
            except Exception:
                pass

        timer = threading.Timer(0.8, _open_browser)
        timer.daemon = True
        timer.start()

    from neuralops.ui.server import create_app

    try:
        import uvicorn
    except ImportError:
        print(
            "Error: 'uvicorn' is required to launch the Web UI. Install with: pip install 'neuralops[ui]'",
            file=sys.stderr,
        )
        return 1

    app = create_app(auth_credentials=auth_creds)

    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
        )
    except KeyboardInterrupt:
        print("\nNeuralOps Web UI stopped cleanly.")
    except Exception as exc:
        print(f"Error starting NeuralOps Web UI: {exc}", file=sys.stderr)
        return 1
    finally:
        if tunnel:
            tunnel.stop()

    return 0


__all__ = ["build_ui_parser", "cmd_ui"]
