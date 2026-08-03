"""
RemoteCoordinator – runs inside Django.  Dispatches training jobs to
remote worker agents that have registered themselves.
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class RemoteWorker:
    """Information about a registered remote GPU machine."""

    worker_id: str
    host: str  # public IP or hostname
    port: int  # agent API port
    auth_token: str  # shared secret
    num_gpus: int
    status: str = "idle"  # idle | busy | offline
    current_job: Optional[str] = None
    last_heartbeat: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)  # country, GPU model, etc.


class RemoteCoordinator:
    """
    Manages a pool of remote GPU workers.

    Workers call `POST /api/remote/register/` to join the pool.
    The coordinator selects workers for distributed jobs and instructs them
    via their agent API.

    Usage (inside Django view):
        coord = RemoteCoordinator()
        job_id = coord.launch_job(config, workers_needed=2)
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._workers: Dict[str, RemoteWorker] = {}

    def register_worker(
        self,
        worker_id: str,
        host: str,
        port: int,
        auth_token: str,
        num_gpus: int = 1,
        metadata: Dict = None,
    ) -> RemoteWorker:
        """Called when a new remote agent comes online."""
        with self._lock:
            worker = RemoteWorker(
                worker_id=worker_id,
                host=host,
                port=port,
                auth_token=auth_token,
                num_gpus=num_gpus,
                metadata=metadata or {},
            )
            self._workers[worker_id] = worker
            logger.info(
                "Worker registered: %s (%s:%d, %d GPUs)",
                worker_id,
                host,
                port,
                num_gpus,
            )
            return worker

    def heartbeat(self, worker_id: str) -> None:
        """Update last‑seen timestamp."""
        with self._lock:
            if worker_id in self._workers:
                self._workers[worker_id].last_heartbeat = datetime.now().isoformat()
                self._workers[worker_id].status = "idle"

    def unregister(self, worker_id: str) -> None:
        """Unregister."""
        with self._lock:
            self._workers.pop(worker_id, None)
            logger.info("Worker unregistered: %s", worker_id)

    @property
    def available_workers(self) -> List[RemoteWorker]:
        """Workers that are currently idle and reachable."""
        now = datetime.now()
        with self._lock:
            idle = []
            for w in self._workers.values():
                last = datetime.fromisoformat(w.last_heartbeat)
                if (now - last).total_seconds() < 120 and w.status == "idle":
                    idle.append(w)
            return idle

    @property
    def total_available_gpus(self) -> int:
        """Total available gpus."""
        return sum(w.num_gpus for w in self.available_workers)

    def select_workers(self, num_gpus_needed: int) -> Optional[List[RemoteWorker]]:
        """
        Pick the best set of idle workers that collectively have
        at least `num_gpus_needed` GPUs.  Prefers fewer workers.
        """
        idle = sorted(self.available_workers, key=lambda w: -w.num_gpus)
        selected = []
        gpus = 0
        for w in idle:
            selected.append(w)
            gpus += w.num_gpus
            if gpus >= num_gpus_needed:
                return selected
        return None

    def launch_job(
        self, config: Dict[str, Any], num_gpus_needed: int = 2
    ) -> Optional[str]:
        """
        Find suitable workers and instruct them to form a training ring.

        Returns the job_id if successful, or None.
        """
        workers = self.select_workers(num_gpus_needed)
        if workers is None:
            logger.warning(
                "Not enough GPUs. Need %d, available %d",
                num_gpus_needed,
                self.total_available_gpus,
            )
            return None

        # Build the rendezvous info so all workers can find each other
        worker_list = [
            {"host": w.host, "port": w.port, "auth_token": w.auth_token}
            for w in workers
        ]

        job_id = f"remote-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Mark workers as busy
        for w in workers:
            w.status = "busy"
            w.current_job = job_id

        # Send launch command to each worker (fire & forget)
        for idx, w in enumerate(workers):
            self._send_launch_command(w, job_id, config, idx, len(workers), worker_list)

        logger.info("Job %s launched on %d workers", job_id, len(workers))
        return job_id

    def _send_launch_command(
        self,
        worker: RemoteWorker,
        job_id: str,
        config: Dict,
        rank: int,
        world_size: int,
        all_workers: List[Dict],
    ) -> None:
        """POST to the worker agent to start training."""
        url = f"http://{worker.host}:{worker.port}/launch"
        payload = {
            "job_id": job_id,
            "config": config,
            "rank": rank,
            "world_size": world_size,
            "all_workers": all_workers,
            "auth_token": worker.auth_token,
        }
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code != 200:
                logger.error("Worker %s launch failed: %s", worker.worker_id, resp.text)
                worker.status = "idle"
        except requests.RequestException as e:
            logger.error("Worker %s unreachable: %s", worker.worker_id, e)
            worker.status = "offline"
