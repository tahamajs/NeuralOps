# jobscheduler/batching.py
import threading
import time
from .models import Job

class ShortJobBatcher:
    def __init__(self, scheduler, threshold_seconds=60, window_seconds=10, max_batch_size=4):
        self.scheduler = scheduler
        self.threshold = threshold_seconds
        self.window = window_seconds
        self.max_batch = max_batch_size
        self.batch = []   # list of (job_id, priority)
        self.lock = threading.Lock()
        self._timer = None

    def add_job(self, job_id, priority, estimated_duration):
        if estimated_duration and estimated_duration <= self.threshold:
            with self.lock:
                self.batch.append((job_id, priority))
                if len(self.batch) >= self.max_batch:
                    self._flush()
                elif self._timer is None:
                    self._timer = threading.Timer(self.window, self._flush)
                    self._timer.start()
        else:
            # Long job: submit directly
            self.scheduler.add_job(job_id, priority, estimated_duration)

    def _flush(self):
        with self.lock:
            batch_copy = self.batch.copy()
            self.batch.clear()
            if self._timer:
                self._timer.cancel()
                self._timer = None
        if not batch_copy:
            return
        # Submit the whole batch as a single "meta‑job"
        self.scheduler._submit_batch(batch_copy)

    def _submit_batch(self, batch):
        # The meta‑job has the highest priority among the batch members
        max_prio = min(p for _, p in batch)  # smaller = higher priority
        # Launch all jobs sequentially (or in parallel, depending on available GPUs)
        # For simplicity, we launch them one after another on the same GPU.
        for job_id, _ in batch:
            self.scheduler._launch_job(job_id)