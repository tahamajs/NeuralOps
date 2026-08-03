"""
Distributed Job Scheduler – inspired by AdaptDL and TorchSubmit.

Queues multiple training jobs and schedules them across available GPUs
with priority, preemption, and fair‑sharing.

References:
  - AdaptDL: https://github.com/petuum/adaptdl
  - KAI Scheduler (NVIDIA): https://github.com/NVIDIA/kai-scheduler
  - Blox scheduler toolkit: https://github.com/yue2388253/blox
"""

import enum
import heapq
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class SchedulingPolicy(enum.Enum):
    FIFO = "fifo"  # first‑in, first‑out
    PRIORITY = "priority"  # highest priority first
    FAIR_SHARE = "fair"  # equal GPU time per user
    SHORTEST_JOB = "sjf"  # estimate‑based shortest first


class JobState(enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PREEMPTED = "preempted"


@dataclass(order=True)
class JobSpec:
    """A training job waiting to be scheduled."""

    priority: int
    submit_time: float = field(compare=False)
    job_id: str = field(compare=False)
    num_gpus: int = field(compare=False)
    estimated_duration: float = 3600.0
    state: JobState = JobState.PENDING
    config: Dict[str, Any] = field(default_factory=dict)
    user: str = "anonymous"
    start_time: Optional[float] = None
    gpu_ids: List[int] = field(default_factory=list)


class DistributedJobScheduler:
    """
    Manages a queue of training jobs and dispatches them to available GPUs.

    Usage:
        scheduler = DistributedJobScheduler()
        job_id = scheduler.submit(config, num_gpus=4, priority=1)
        scheduler.run_loop()   # blocking
    """

    def __init__(self, policy: SchedulingPolicy = SchedulingPolicy.PRIORITY) -> None:
        self.policy = policy
        self._lock = threading.Lock()
        self._queue: List[JobSpec] = []  # heap for priority queue
        self._running: Dict[str, JobSpec] = {}
        self._history: List[JobSpec] = []
        self._stop = threading.Event()
        from .gpu_allocator import GPUAllocator

        self.allocator = GPUAllocator()

    def submit(
        self,
        config: Dict[str, Any],
        num_gpus: int = 1,
        priority: int = 5,
        estimated_duration: float = 3600.0,
        user: str = "anonymous",
    ) -> str:
        """Submit a job. Returns job_id."""
        job_id = f"job-{int(time.time() * 1000)}"
        job = JobSpec(
            priority=priority,
            submit_time=time.time(),
            job_id=job_id,
            num_gpus=num_gpus,
            estimated_duration=estimated_duration,
            config=config,
            user=user,
        )
        with self._lock:
            heapq.heappush(self._queue, job)
            logger.info(
                "Job %s submitted (priority %d, %d GPUs)", job_id, priority, num_gpus
            )
        return job_id

    def cancel(self, job_id: str) -> bool:
        """Cancel a pending or running job."""
        with self._lock:
            if job_id in self._running:
                self.allocator.release(job_id)
                self._running.pop(job_id)
                return True
            for i, job in enumerate(self._queue):
                if job.job_id == job_id:
                    self._queue.pop(i)
                    heapq.heapify(self._queue)
                    return True
        return False

    def status(self) -> Dict[str, Any]:
        """Return current queue status."""
        with self._lock:
            return {
                "queue_size": len(self._queue),
                "running": len(self._running),
                "available_gpus": self.allocator.available_gpus,
                "total_gpus": self.allocator.total_gpus,
                "running_jobs": [
                    {"job_id": j.job_id, "gpus": j.gpu_ids, "started": j.start_time}
                    for j in self._running.values()
                ],
            }

    def run_loop(self, train_fn: Callable, tick_interval: float = 2.0) -> None:
        """
        Main scheduling loop.  Blocks until stopped.

        Args:
            train_fn: callable (job_spec, gpu_list) that runs the training
            tick_interval: seconds between scheduling attempts
        """
        logger.info("Scheduler started with policy %s", self.policy.value)
        while not self._stop.is_set():
            self._tick(train_fn)
            time.sleep(tick_interval)

    def stop(self) -> None:
        """Stop."""
        self._stop.set()

    def _tick(self, train_fn: Callable) -> None:
        """One scheduling cycle."""
        with self._lock:
            # 1. Check for completed jobs
            for jid, job in self._running.items():
                # In a real system, you'd poll the job's subprocess
                pass

            # 2. Try to dispatch pending jobs
            if not self._queue:
                return

            available = self.allocator.available_gpus
            if available == 0:
                return

            # Pop the highest‑priority job
            job = heapq.heappop(self._queue)

            # Try to allocate GPUs
            try:
                gpus = self.allocator.allocate(
                    num_gpus=job.num_gpus,
                    job_id=job.job_id,
                )
                job.gpu_ids = [(g.node, g.index) for g in gpus]
                job.start_time = time.time()
                job.state = JobState.RUNNING
                self._running[job.job_id] = job

                # Launch in background thread
                threading.Thread(
                    target=self._launch_job,
                    args=(job, train_fn),
                    daemon=True,
                ).start()
                logger.info("Job %s started on GPUs %s", job.job_id, job.gpu_ids)
            except RuntimeError:
                # Not enough GPUs – re‑queue
                heapq.heappush(self._queue, job)

    def _launch_job(self, job: JobSpec, train_fn: Callable) -> None:
        """Execute the job (called in a background thread)."""
        try:
            train_fn(job)
            job.state = JobState.COMPLETED
        except Exception as e:
            logger.exception("Job %s failed: %s", job.job_id, e)
            job.state = JobState.FAILED
        finally:
            with self._lock:
                self.allocator.release(job.job_id)
                if job.job_id in self._running:
                    del self._running[job.job_id]
                self._history.append(job)
