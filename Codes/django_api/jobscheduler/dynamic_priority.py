# jobscheduler/dynamic_priority.py
import threading
import time
from .models import Job
from .scheduler import scheduler

class DynamicPriorityManager:
    def __init__(self, check_interval=30, boost_wait_threshold=600, boost_amount=1):
        self.check_interval = check_interval
        self.boost_wait_threshold = boost_wait_threshold  # seconds
        self.boost_amount = boost_amount
        self.thread = None
        self.active = True

    def _adjust_priorities(self):
        while self.active:
            time.sleep(self.check_interval)
            for job in Job.objects.filter(status='queued'):
                wait_time = (timezone.now() - job.created_at).total_seconds()
                if wait_time > self.boost_wait_threshold:
                    new_priority = max(1, job.priority - self.boost_amount)
                    job.priority = new_priority
                    job.save(update_fields=['priority'])
                    # Also notify the scheduler to update the priority in the queue
                    scheduler.update_priority(str(job.id), new_priority)
            # Optional: reduce priority of long-running tasks to free resources
            for job in Job.objects.filter(status='running'):
                runtime = (timezone.now() - job.started_at).total_seconds()
                if runtime > 3600 and job.priority < 10:
                    new_priority = min(10, job.priority + 1)
                    job.priority = new_priority
                    job.save()

    def start(self):
        self.thread = threading.Thread(target=self._adjust_priorities, daemon=True)
        self.thread.start()