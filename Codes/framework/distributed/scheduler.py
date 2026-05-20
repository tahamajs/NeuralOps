"""Job scheduler — priority queue with GPU‑aware dispatch."""

import time, threading, heapq, logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from .allocator import GPUAllocator

logger = logging.getLogger(__name__)


class SchedulingPolicy(Enum):
    FIFO = "fifo"; PRIORITY = "priority"; FAIR_SHARE = "fair"; SHORTEST_JOB = "sjf"

class JobState(Enum):
    PENDING = "pending"; RUNNING = "running"; COMPLETED = "completed"; FAILED = "failed"


@dataclass(order=True)
class JobSpec:
    priority: int; submit_time: float = field(compare=False)
    job_id: str = field(compare=False); num_gpus: int = field(compare=False)
    estimated_duration: float = 3600.0; state: JobState = JobState.PENDING
    config: Dict[str, Any] = field(default_factory=dict); user: str = "anonymous"
    gpu_ids: List = field(default_factory=list)


class DistributedJobScheduler:
    """Priority queue that dispatches jobs to available GPUs."""

    def __init__(self, policy=SchedulingPolicy.PRIORITY):
        self.policy = policy; self._lock = threading.Lock()
        self._queue: List[JobSpec] = []; self._running: Dict[str, JobSpec] = {}
        self._history: List[JobSpec] = []; self._stop = threading.Event()
        self.allocator = GPUAllocator()

    def submit(self, config: Dict, num_gpus: int = 1, priority: int = 5,
               duration: float = 3600.0, user: str = "anonymous") -> str:
        jid = f"job-{int(time.time()*1000)}"
        job = JobSpec(priority=priority, submit_time=time.time(), job_id=jid,
                      num_gpus=num_gpus, estimated_duration=duration, config=config, user=user)
        with self._lock: heapq.heappush(self._queue, job)
        logger.info("Job %s submitted (priority %d, %d GPUs)", jid, priority, num_gpus)
        return jid

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            if job_id in self._running:
                self.allocator.release(job_id); del self._running[job_id]; return True
            for i, j in enumerate(self._queue):
                if j.job_id == job_id: self._queue.pop(i); heapq.heapify(self._queue); return True
        return False

    def status(self) -> Dict:
        with self._lock:
            return {"queue_size": len(self._queue), "running": len(self._running),
                    "available_gpus": self.allocator.available_gpus}

    def run_loop(self, train_fn: Callable, tick: float = 2.0):
        logger.info("Scheduler started with %s", self.policy.value)
        while not self._stop.is_set():
            self._tick(train_fn); time.sleep(tick)

    def stop(self): self._stop.set()

    def _tick(self, train_fn):
        with self._lock:
            free = self.allocator.available_gpus
            while self._queue and free > 0:
                job = heapq.heappop(self._queue)
                if job.num_gpus > free: heapq.heappush(self._queue, job); break
                try:
                    gpus = self.allocator.allocate(job.num_gpus, job_id=job.job_id)
                    job.gpu_ids = [(g.node, g.index) for g in gpus]; job.state = JobState.RUNNING
                    self._running[job.job_id] = job
                    threading.Thread(target=self._exec, args=(job, train_fn), daemon=True).start()
                    free -= job.num_gpus
                except RuntimeError: heapq.heappush(self._queue, job); break

    def _exec(self, job: JobSpec, train_fn):
        try:
            train_fn(job); job.state = JobState.COMPLETED
        except Exception as e:
            logger.exception("Job %s failed: %s", job.job_id, e); job.state = JobState.FAILED
        finally:
            with self._lock:
                self.allocator.release(job.job_id)
                if job.job_id in self._running: del self._running[job.job_id]
                self._history.append(job)