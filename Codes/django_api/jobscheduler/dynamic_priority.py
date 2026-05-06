import threading
import time
from django.utils import timezone
from django.db import connection
from .models import Job

class DynamicPriorityManager:
    """Adjusts priorities of queued and running jobs periodically."""

    def __init__(self, scheduler, boost_wait_threshold=600, boost_amount=1,
                 reduce_runtime_threshold=3600, reduce_amount=1,
                 check_interval=30):
        self.scheduler = scheduler
        self.boost_wait_threshold = boost_wait_threshold
        self.boost_amount = boost_amount
        self.reduce_runtime_threshold = reduce_runtime_threshold
        self.reduce_amount = reduce_amount
        self.check_interval = check_interval
        self._stop_event = threading.Event()
        self.thread = None

    def start(self):
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def _run(self):
        while not self._stop_event.is_set():
            time.sleep(self.check_interval)

            # Guard against running after the database connection has been closed
            # (e.g., during test teardown)
            if connection.connection is None:
                continue

            try:
                # Boost queued jobs that waited too long
                for job in Job.objects.filter(status='queued'):
                    wait = (timezone.now() - job.created_at).total_seconds()
                    if wait > self.boost_wait_threshold:
                        new_prio = max(1, job.priority - self.boost_amount)
                        if new_prio != job.priority:
                            job.priority = new_prio
                            job.save(update_fields=['priority'])
                            if hasattr(self.scheduler, 'update_priority'):
                                self.scheduler.update_priority(str(job.id), new_prio)

                # Reduce priority of long‑running jobs (encourage preemption)
                for job in Job.objects.filter(status='running'):
                    if job.started_at is None:
                        continue
                    runtime = (timezone.now() - job.started_at).total_seconds()
                    if runtime > self.reduce_runtime_threshold and job.priority < 10:
                        new_prio = min(10, job.priority + self.reduce_amount)
                        if new_prio != job.priority:
                            job.priority = new_prio
                            job.save(update_fields=['priority'])
            except Exception:
                # Silently ignore any DB errors during shutdown or tests
                pass