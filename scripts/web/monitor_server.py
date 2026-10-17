#!/usr/bin/env python3
"""Unified NeuralOps Web Monitor & Dashboard Runner Script.

Usage:
    python scripts/web/monitor_server.py --port 8080
    python scripts/web/monitor_server.py --standard-only
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
    parser = argparse.ArgumentParser(description="NeuralOps Telemetry & Web Dashboard Server")
    parser.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Binding host")
    parser.add_argument("--standard-only", action="store_true", help="Force standard HTTP server instead of FastAPI")
    parser.add_argument("--daemon", action="store_true", help="Run server as background daemon process")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ComprehensiveStudioHandler.project_dir = root_dir
    ComprehensiveStudioHandler.web_dir = os.path.join(root_dir, "docs")
    ComprehensiveStudioHandler.docs_dir = os.path.join(root_dir, "docs")
    ComprehensiveStudioHandler.harvested_json = os.path.join(root_dir, "results", "harvested_benchmark_data.json")

    if args.daemon:
        import subprocess
        log_path = os.path.join(root_dir, "logs", "monitor_server.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as log_f:
            cmd = [sys.executable, __file__, "--port", str(args.port), "--host", args.host]
            if args.standard_only:
                cmd.append("--standard-only")
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=log_f, start_new_session=True)
            print(f"NeuralOps Web Monitor running in background daemon mode (PID: {proc.pid}, Log: {log_path})")
            return

    print(f"Starting NeuralOps Web Monitor Server on port {args.port}...")
    run_server(port=args.port, use_fastapi=not args.standard_only)


if __name__ == "__main__":
    main()
