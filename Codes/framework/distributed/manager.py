"""GPU Manager – discovers devices and allocates GPUs to jobs."""

import logging
import threading
from typing import Dict, List

import torch

logger = logging.getLogger(__name__)


class GPUManager:
    """
    Singleton that tracks GPU availability across multiple jobs.

    Usage:
        mgr = GPUManager()
        gpu_ids = mgr.allocate(2)   # get two free GPUs
        # … run training …
        mgr.release(gpu_ids)
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialised = False
        return cls._instance

    def __init__(self):
        if self._initialised:
            return
        self.total_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self._allocated: Dict[int, str] = {}  # gpu_id → job_id
        self._initialised = True
        logger.info("GPUManager initialised: %d GPU(s) available", self.total_gpus)

    @property
    def available_gpus(self) -> int:
        with self._lock:
            return self.total_gpus - len(self._allocated)

    def allocate(self, num: int, job_id: str) -> List[int]:
        """Reserve `num` GPUs for `job_id`.  Returns list of GPU indices."""
        with self._lock:
            free = [i for i in range(self.total_gpus) if i not in self._allocated]
            if len(free) < num:
                raise RuntimeError(
                    f"Requested {num} GPU(s), only {len(free)} available "
                    f"(total: {self.total_gpus}, allocated: {len(self._allocated)})"
                )
            chosen = free[:num]
            for gpu_id in chosen:
                self._allocated[gpu_id] = job_id
            logger.info("Allocated GPUs %s to job %s", chosen, job_id)
            return chosen

    def release(self, job_id: str) -> None:
        """Free all GPUs allocated to `job_id`."""
        with self._lock:
            to_release = [k for k, v in self._allocated.items() if v == job_id]
            for gpu_id in to_release:
                del self._allocated[gpu_id]
            logger.info("Released GPUs %s from job %s", to_release, job_id)

    def status(self) -> Dict:
        """Return current allocation status."""
        with self._lock:
            return {
                "total": self.total_gpus,
                "allocated": len(self._allocated),
                "available": self.total_gpus - len(self._allocated),
                "jobs": list(set(self._allocated.values())),
            }
