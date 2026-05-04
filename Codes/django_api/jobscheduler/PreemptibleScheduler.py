import os
import time
import signal
import threading
import queue
from django.utils import timezone
from .models import Job

class PreemptibleScheduler:
    def __init__(self, total_gpus):
        self.task_queue = queue.PriorityQueue()
        self.available_gpus = total_gpus
        self.running_jobs = {}          # job_id -> subprocess.Popen
        self.job_gpu_map = {}           # job_id -> allocated GPUs
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self._preempt_flag_dir = "/tmp/preempt_flags"
        os.makedirs(self._preempt_flag_dir, exist_ok=True)

    def add_job(self, job_id, priority, estimated_duration=3600):
        # 1. Use predictive model (see Section 3) to refine priority
        predicted = self.predictor.predict_priority(job_id)
        final_prio = min(priority, predicted) if predicted else priority
        self.task_queue.put((final_prio, time.time(), job_id))

    def _run_worker(self):
        while not self._stop_event.is_set():
            try:
                # Wait up to 2 seconds for a job
                prio, _, job_id = self.task_queue.get(timeout=2)
                with self.lock:
                    # If no GPU available, re‑queue the job after a short sleep
                    if self.available_gpus <= 0:
                        # Optional: preempt a lower‑priority job if this one has higher priority
                        self._try_preempt(job_id, prio)
                        # If still no resources, re‑queue and wait
                        if self.available_gpus <= 0:
                            self.task_queue.put((prio, time.time(), job_id))
                            time.sleep(5)
                            continue
                    self.available_gpus -= 1

                # Launch the job
                self._launch_job(job_id)
            except queue.Empty:
                continue

    def _try_preempt(self, incoming_id, incoming_prio):
        """Find a running job with lower priority and preempt it."""
        for r_id, proc in list(self.running_jobs.items()):
            job = Job.objects.get(id=r_id)
            if job.priority > incoming_prio:   # lower numeric = higher priority
                # Signal the job to save checkpoint and exit
                flag_file = os.path.join(self._preempt_flag_dir, str(r_id))
                with open(flag_file, 'w') as f:
                    f.write('preempt')
                # Wait a few seconds for graceful termination
                timeout = 10
                start = time.time()
                while proc.poll() is None and (time.time() - start) < timeout:
                    time.sleep(0.5)
                if proc.poll() is None:
                    proc.terminate()
                # Free GPUs
                with self.lock:
                    self.available_gpus += self.job_gpu_map.pop(r_id, 1)
                self.running_jobs.pop(r_id, None)
                # Mark job as paused
                Job.objects.filter(id=r_id).update(status='paused')
                break

    def _launch_job(self, job_id):
        # Spawn subprocess or thread to run the training
        # The training script must check the preempt flag periodically.
        cmd = ['python', 'manage.py', 'run_training_job', '--job-id', job_id]
        proc = subprocess.Popen(cmd)
        with self.lock:
            self.running_jobs[job_id] = proc
            self.job_gpu_map[job_id] = 1   # assume 1 GPU per job for simplicity
        proc.wait()
        # After completion, update database and free GPUs
        with self.lock:
            self.available_gpus += self.job_gpu_map.pop(job_id, 1)
            self.running_jobs.pop(job_id, None)




# preempt_flag = f"/tmp/preempt_flags/{job_id}"
# if os.path.exists(preempt_flag):
#     # Save checkpoint and exit
#     trainer.save_checkpoint()
#     sys.exit(0)
