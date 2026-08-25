"""
RemoteWorkerAgent – runs as a standalone process on each GPU machine.
Connects to the central Django server and waits for training jobs.
"""

import json
import logging
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


class WorkerAPIHandler(BaseHTTPRequestHandler):
    """Tiny REST API that accepts launch commands from the coordinator."""

    agent = None  # set by RemoteWorkerAgent

    def do_POST(self) -> None:
        """Do post."""
        if self.path == "/launch":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)
            # Verify auth token
            if data.get("auth_token") != self.agent.auth_token:
                self.send_response(403)
                self.end_headers()
                self.wfile.write(b"Forbidden")
                return
            # Launch training in a background thread
            threading.Thread(
                target=self.agent.execute_job,
                args=(data,),
                daemon=True,
            ).start()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")


class RemoteWorkerAgent:
    """
    Runs on each GPU machine.  Registers with the central server and
    listens for launch commands.

    Usage:
        agent = RemoteWorkerAgent(
            server_url="http://my-django-server.com",
            worker_id="germany-gpu-01",
            auth_token="shared-secret",
            num_gpus=1,
        )
        agent.start()
    """

    def __init__(
        self,
        server_url: str,
        worker_id: str,
        auth_token: str,
        num_gpus: int = 1,
        listen_port: int = 8765,
        metadata: Dict = None,
    ) -> None:
        self.server_url = server_url
        self.worker_id = worker_id
        self.auth_token = auth_token
        self.num_gpus = num_gpus
        self.listen_port = listen_port
        self.metadata = metadata or {}
        self._running = False
        self._http_server: Optional[HTTPServer] = None
        # Determine public IP (optional)
        self.public_ip = self._get_public_ip()

    def start(self) -> None:
        """Start the agent (registers + starts HTTP listener)."""
        self._register()
        self._running = True

        # Start heartbeat thread
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()

        # Start HTTP server for coordinator commands
        WorkerAPIHandler.agent = self
        self._http_server = HTTPServer(("0.0.0.0", self.listen_port), WorkerAPIHandler)
        logger.info("Worker agent listening on port %d", self.listen_port)
        self._http_server.serve_forever()

    def stop(self) -> None:
        """Stop."""
        self._running = False
        self._unregister()
        if self._http_server:
            self._http_server.shutdown()

    def execute_job(self, launch_data: Dict[str, Any]) -> None:
        """
        Called by the coordinator via HTTP.  Sets up distributed environment
        and runs the training function.
        """
        job_id = launch_data["job_id"]
        config = launch_data["config"]
        rank = launch_data["rank"]
        world_size = launch_data["world_size"]
        all_workers = launch_data["all_workers"]

        logger.info("Starting job %s (rank %d/%d)", job_id, rank, world_size)

        # Set environment variables for torch distributed
        os.environ["MASTER_ADDR"] = all_workers[0]["host"]
        os.environ["MASTER_PORT"] = str(
            config.get("distributed", {}).get("master_port", 29500)
        )
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = "0"
        os.environ["NODE_RANK"] = str(rank)

        # If workers are behind NAT, use the InternetRendezvous instead
        # (see integration.py)

        try:
            # Here we call the actual training function (same as single‑node)
            from framework.experiment_utils.mega_orchestrator import MegaOrchestrator

            orchestrator = MegaOrchestrator(
                config=config,
                run_id=f"{job_id}-{rank}",
            )
            orchestrator.run()
        except Exception:
            logger.exception("Job %s failed on worker %s", job_id, self.worker_id)

    def _register(self) -> None:
        """ register."""
        try:
            resp = requests.post(
                f"{self.server_url}/api/remote/register/",
                json={
                    "worker_id": self.worker_id,
                    "host": self.public_ip,
                    "port": self.listen_port,
                    "auth_token": self.auth_token,
                    "num_gpus": self.num_gpus,
                    "metadata": self.metadata,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                logger.info("Registered with server")
            else:
                logger.error("Registration failed: %s", resp.text)
        except Exception as e:
            logger.error("Unable to reach server: %s", e)

    def _heartbeat_loop(self) -> None:
        """ heartbeat loop."""
        while self._running:
            time.sleep(30)
            try:
                requests.post(
                    f"{self.server_url}/api/remote/heartbeat/",
                    json={"worker_id": self.worker_id, "auth_token": self.auth_token},
                    timeout=5,
                )
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, OSError, ImportError):
                pass

    def _unregister(self) -> None:
        """ unregister."""
        try:
            requests.post(
                f"{self.server_url}/api/remote/unregister/",
                json={"worker_id": self.worker_id, "auth_token": self.auth_token},
                timeout=5,
            )
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, OSError, ImportError):
            pass

    def _get_public_ip(self) -> Any:
        """ get public ip."""
        try:
            return requests.get("https://api.ipify.org", timeout=5).text.strip()
        except Exception:
            return socket.gethostbyname(socket.gethostname())
