# jobscheduler/preemptible_scheduler.py
import os
import queue
import signal
import subprocess
import threading
import time

import torch
from django.utils import timezone

from .models import Job


class PreemptibleScheduler:
    """
    A priority-based GPU scheduler with preemption support.
    Lower numeric priority = higher priority.
    """

    _instance = None

    def __new__(cls, total_gpus=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, total_gpus=None):
        if self._initialized:
            return
        if total_gpus is None:
            total_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.total_gpus = total_gpus
        self.available_gpus = total_gpus
        self.task_queue = queue.PriorityQueue()
        self.running_jobs = {}  # job_id -> subprocess.Popen
        self.job_gpu_map = {}  # job_id -> allocated GPUs
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self._preempt_flag_dir = "/tmp/preempt_flags"
        os.makedirs(self._preempt_flag_dir, exist_ok=True)
        self.worker_thread = None
        self._initialized = True

    def add_job(self, job_id, priority, estimated_duration=None):
        """
        Add a job to the queue.
        - job_id: UUID string
        - priority: int (1-10, lower = higher priority)
        - estimated_duration: optional (not used for scheduling logic here)
        """
        # Optionally refine priority using a predictor (can be added later)
        final_prio = priority
        self.task_queue.put((final_prio, time.time(), str(job_id)))
        Job.objects.filter(id=job_id).update(status="queued")

    def _try_preempt(self, incoming_prio):
        """
        Try to preempt a running job with lower priority than the incoming one.
        Returns True if preemption succeeded, False otherwise.
        """
        with self.lock:
            for job_id, proc in list(self.running_jobs.items()):
                job = Job.objects.get(id=job_id)
                if job.priority > incoming_prio:  # lower numeric = higher priority
                    # Signal the job to save checkpoint and exit
                    flag_file = os.path.join(self._preempt_flag_dir, str(job_id))
                    with open(flag_file, "w") as f:
                        f.write("preempt")
                    # Wait up to 10 seconds for graceful termination
                    start = time.time()
                    while proc.poll() is None and (time.time() - start) < 10:
                        time.sleep(0.5)
                    if proc.poll() is None:
                        proc.terminate()
                    # Free GPUs
                    self.available_gpus += self.job_gpu_map.pop(job_id, 1)
                    self.running_jobs.pop(job_id, None)
                    Job.objects.filter(id=job_id).update(status="preempted")
                    return True
        return False

    def _run_job(self, job_id):
        """Launch a training job as a subprocess."""
        job = Job.objects.get(id=job_id)
        job.status = "running"
        job.started_at = timezone.now()
        job.save()
        # Clear any pre-existing preempt flag
        flag_file = os.path.join(self._preempt_flag_dir, str(job_id))
        if os.path.exists(flag_file):
            os.remove(flag_file)
        # Build command
        cmd = [
            "python",
            "manage.py",
            "run_training_job",
            "--job-id",
            str(job_id),
            "--checkpoint-dir",
            job.checkpoint_path or "",
        ]
        proc = subprocess.Popen(cmd)
        with self.lock:
            self.running_jobs[job_id] = proc
            self.job_gpu_map[job_id] = job.gpu_requirements
        proc.wait()
        # After completion, free resources
        with self.lock:
            self.available_gpus += self.job_gpu_map.pop(job_id, 1)
            self.running_jobs.pop(job_id, None)
        # If job was preempted, re-queue it
        job.refresh_from_db()
        if job.status == "preempted":
            self.add_job(job_id, job.priority, job.estimated_duration)

    def _worker(self):
        """Background worker that picks jobs from the queue and runs them."""
        while not self._stop_event.is_set():
            try:
                priority, _, job_id = self.task_queue.get(timeout=2)
                with self.lock:
                    if self.available_gpus <= 0:
                        # Try to preempt a lower-priority job
                        preempted = self._try_preempt(priority)
                        if not preempted:
                            # Re-queue and wait
                            self.task_queue.put((priority, time.time(), job_id))
                            time.sleep(5)
                            continue
                        self.available_gpus -= 1
                    else:
                        self.available_gpus -= 1
                self._run_job(job_id)
            except queue.Empty:
                continue

    def start(self):
        """Start the scheduler's worker thread."""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self._worker, daemon=True)
            self.worker_thread.start()
            print(f"PreemptibleScheduler started with {self.total_gpus} GPU(s)")

    def stop(self):
        """Stop the scheduler."""
        self._stop_event.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=5)

    def pause_job(self, job_id):
        """Pause a running job (send SIGSTOP)."""
        proc = self.running_jobs.get(str(job_id))
        if proc:
            proc.send_signal(signal.SIGSTOP)
            Job.objects.filter(id=job_id).update(status="paused")

    def resume_job(self, job_id):
        """Resume a paused job (send SIGCONT) or re‑queue if not running."""
        proc = self.running_jobs.get(str(job_id))
        if proc:
            proc.send_signal(signal.SIGCONT)
            Job.objects.filter(id=job_id).update(status="running")
        else:
            job = Job.objects.get(id=job_id)
            self.add_job(job_id, job.priority, job.estimated_duration)

    def stop_job(self, job_id):
        """Terminate a running job (no auto-resume)."""
        proc = self.running_jobs.get(str(job_id))
        if proc:
            proc.terminate()
        Job.objects.filter(id=job_id).update(status="stopped")
        with self.lock:
            self.available_gpus += self.job_gpu_map.pop(job_id, 1)

    def update_priority(self, job_id, new_priority):
        """Update priority of a queued job (only affects DB; re‑queue not implemented)."""
        Job.objects.filter(id=job_id).update(priority=new_priority)


# Singleton instance
scheduler = PreemptibleScheduler()
