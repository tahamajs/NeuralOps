#!/usr/bin/env python3
"""Unified NeuralOps Web Monitor & Dashboard Runner Script.

Usage:
    python scripts/web/monitor_server.py --port 8080
    python scripts/web/monitor_server.py --daemon
    python scripts/web/monitor_server.py --standard-only

Author: Taha Majlesi
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = str(Path(__file__).resolve().parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from neuralops.runners.monitor.server import ComprehensiveStudioHandler, run_server


def parse_args() -> argparse.Namespace:
    """Execute parse args routine."""
    parser = argparse.ArgumentParser(description="NeuralOps Telemetry & Web Dashboard Server")
    parser.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Binding host (default: 0.0.0.0)")
    parser.add_argument("--standard-only", action="store_true", help="Force standard HTTP server instead of FastAPI")
    parser.add_argument("--daemon", action="store_true", help="Run server as background daemon process")
    return parser.parse_args()


def daemonize() -> None:
    """Daemonize process into background."""
    if os.fork() > 0:
        sys.exit(0)
    os.setsid()
    if os.fork() > 0:
        sys.exit(0)
    sys.stdout.flush()
    sys.stderr.flush()
    with open("/dev/null", "r") as devnull_r:
        os.dup2(devnull_r.fileno(), sys.stdin.fileno())
    log_dir = os.path.join(root_dir, "results")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "monitor_server.log")
    devnull_w = open(log_file, "a+")
    os.dup2(devnull_w.fileno(), sys.stdout.fileno())
    os.dup2(devnull_w.fileno(), sys.stderr.fileno())


def main() -> None:
    """Execute main routine."""
    args = parse_args()
    if args.daemon:
        daemonize()

    ComprehensiveStudioHandler.project_dir = root_dir
    ComprehensiveStudioHandler.web_dir = os.path.join(root_dir, "docs")
    ComprehensiveStudioHandler.docs_dir = os.path.join(root_dir, "docs")
    ComprehensiveStudioHandler.harvested_json = os.path.join(root_dir, "results", "harvested_benchmark_data.json")

    print(f"Starting NeuralOps Web Monitor Server on http://{args.host}:{args.port}...")
    run_server(
        port=args.port,
        host=args.host,
        use_fastapi=not args.standard_only,
        project_dir=root_dir,
        docs_dir=os.path.join(root_dir, "docs"),
    )


if __name__ == "__main__":
    main()
