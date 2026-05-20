import threading, time

class ShortJobBatcher:
    def __init__(self, scheduler, threshold_seconds=60, window_seconds=10, max_batch_size=4):
        self.scheduler = scheduler
        self.threshold = threshold_seconds
        self.window = window_seconds
        self.max_batch = max_batch_size
        self.batch = []
        self.lock = threading.Lock()
        self._timer = None

    def add_job(self, job_id, priority, estimated_duration, memory_gb=None):
        if estimated_duration and estimated_duration <= self.threshold:
            with self.lock:
                self.batch.append((job_id, priority))
                if len(self.batch) >= self.max_batch:
                    self._flush()
                elif self._timer is None:
                    self._timer = threading.Timer(self.window, self._flush)
                    self._timer.start()
        else:
            self.scheduler._submit_job(job_id, priority)

    def _flush(self):
        with self.lock:
            batch_copy = self.batch.copy()
            self.batch.clear()
            if self._timer:
                self._timer.cancel()
                self._timer = None
        if not batch_copy:
            return
        max_priority = min(p for _, p in batch_copy)
        for job_id, _ in batch_copy:
            self.scheduler._submit_job(job_id, max_priority)