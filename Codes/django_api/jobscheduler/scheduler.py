import queue
import threading
import time
import subprocess
import torch
from django.utils import timezone
from .models import Job

class GPUScheduler:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self.task_queue = queue.PriorityQueue()
        self.lock = threading.Lock()
        self.total_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.available_gpus = self.total_gpus
        self.running_processes = {}  # job_id -> subprocess.Popen
        self._stop_event = threading.Event()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def add_job(self, job_id, priority):
        self.task_queue.put((priority, str(job_id)))
        Job.objects.filter(id=job_id).update(status='queued')

    def _run(self):
        while not self._stop_event.is_set():
            try:
                priority, job_id = self.task_queue.get(timeout=2)
                with self.lock:
                    if self.available_gpus <= 0:
                        self.task_queue.put((priority, job_id))
                        time.sleep(5)
                        continue
                    self.available_gpus -= 1

                job = Job.objects.get(id=job_id)
                job.status = 'running'
                job.started_at = timezone.now()
                job.save()

                # Run training in a subprocess
                cmd = [
                    'python', 'manage.py', 'run_training_job',
                    '--job-id', str(job_id)
                ]
                proc = subprocess.Popen(cmd)
                self.running_processes[job_id] = proc
                proc.wait()

                with self.lock:
                    self.available_gpus += 1
                    self.running_processes.pop(job_id, None)

            except queue.Empty:
                continue

    def pause_job(self, job_id):
        proc = self.running_processes.get(str(job_id))
        if proc:
            proc.send_signal(subprocess.signal.SIGSTOP)
            Job.objects.filter(id=job_id).update(status='paused')

    def resume_job(self, job_id):
        proc = self.running_processes.get(str(job_id))
        if proc:
            proc.send_signal(subprocess.signal.SIGCONT)
            Job.objects.filter(id=job_id).update(status='running')
        else:
            # Not currently running → re‑queue
            job = Job.objects.get(id=job_id)
            self.add_job(job_id, job.priority)

    def stop_job(self, job_id):
        proc = self.running_processes.get(str(job_id))
        if proc:
            proc.terminate()
        Job.objects.filter(id=job_id).update(status='stopped')
        with self.lock:
            self.available_gpus += 1

# Global instance (singleton)
scheduler = GPUScheduler()