#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

"""Cloudflare quick-tunnel manager for NeuralOps Web UI share mode."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import time
from typing import Optional

logger = logging.getLogger("neuralops.ui.tunnel")


class CloudflareTunnel:
    """Manages an ephemeral cloudflared tunnel subprocess for remote sharing."""

    def __init__(self, port: int, host: str = "127.0.0.1"):
        self.port = port
        self.host = host
        self.process: Optional[subprocess.Popen] = None
        self.tunnel_url: Optional[str] = None

    def start(self, timeout_sec: int = 15) -> str:
        """Start cloudflared tunnel subprocess and extract trycloudflare.com URL.

        Raises:
            RuntimeError: If cloudflared binary is not found or fails to establish tunnel.
        """
        bin_path = shutil.which("cloudflared")
        if not bin_path:
            raise RuntimeError(
                "cloudflared executable not found in PATH. Install cloudflared to use --share mode."
            )

        cmd = [
            bin_path,
            "tunnel",
            "--url",
            f"http://{self.host}:{self.port}",
            "--no-autoupdate",
        ]

        logger.info("Starting Cloudflare tunnel: %s", " ".join(cmd))
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        url_pattern = re.compile(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com")
        start_time = time.time()

        while time.time() - start_time < timeout_sec:
            if self.process.poll() is not None:
                err = self.process.stderr.read() if self.process.stderr else ""
                raise RuntimeError(f"cloudflared exited unexpectedly: {err}")

            line = self.process.stderr.readline() if self.process.stderr else ""
            if not line:
                time.sleep(0.1)
                continue

            match = url_pattern.search(line)
            if match:
                self.tunnel_url = match.group(0)
                logger.info("Tunnel established: %s", self.tunnel_url)
                return self.tunnel_url

        self.stop()
        raise TimeoutError("Timed out waiting for trycloudflare.com tunnel URL.")

    def stop(self) -> None:
        """Terminate the cloudflared subprocess."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
            logger.info("Cloudflare tunnel stopped.")
        self.process = None
        self.tunnel_url = None
