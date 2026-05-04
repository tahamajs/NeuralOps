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





# jobscheduler/scheduler.py
import queue
import threading
import time
from .models import Job
from .dynamic_priority import DynamicPriorityManager
from .predictive import PriorityPredictor

class GPUScheduler:
    def __init__(self, total_gpus):
        self.task_queue = queue.PriorityQueue()
        self.available_gpus = total_gpus
        self.lock = threading.Lock()
        self.running_jobs = {}
        self.predictor = PriorityPredictor()
        self.dynamic_mgr = DynamicPriorityManager()
        self.short_job_batch = []
        self.batch_lock = threading.Lock()
        self._batch_timer = None
        # ... (rest of existing code)

    def add_job(self, job_id, priority, estimated_duration=None):
        # Use predictive model to get a refined priority
        job = Job.objects.get(id=job_id)
        predicted_priority = self.predictor.predict_priority(job) if self.predictor.model else priority
        final_priority = min(priority, predicted_priority)  # or a weighted combination

        self.task_queue.put((final_priority, time.time(), job_id))
        # ... rest

    def _collect_short_jobs(self):
        # Try to batch very short jobs
        with self.batch_lock:
            if self.short_job_batch and self.available_gpus >= len(self.short_job_batch):
                # Execute batch as a single meta-job
                self._execute_batch(self.short_job_batch)
                self.short_job_batch = []

    def _execute_batch(self, batch_jobs):
        # Co‑scheduling: launch all at once on separate GPUs or sequentially
        # For now, submit them individually
        for job_id in batch_jobs:
            self._run_job(job_id)

    def _run_job(self, job_id):
        # Existing job launching logic
        # ...
        pass

    def start(self):
        # Start dynamic priority manager
        self.dynamic_mgr.start()
        # Periodically retrain predictor
        threading.Timer(3600, self._retrain_predictor).start()
        # ... rest of scheduler loop

    def _retrain_predictor(self):
        self.predictor.train()
        threading.Timer(3600, self._retrain_predictor).start()





class GPUScheduler:
    def __init__(self, total_gpus):
        self.task_queue = queue.PriorityQueue()
        self.available_gpus = total_gpus
        self.running_jobs = {}
        self.lock = threading.Lock()
        self.predictor = PriorityPredictor()
        self.batcher = ShortJobBatcher(self)
        self.dynamic_mgr = DynamicPriorityManager()

    def add_job(self, job_id, priority, estimated_duration=None):
        # Use predictive model to refine priority
        predicted = self.predictor.predict_priority(job_id)
        final_prio = min(priority, predicted) if predicted else priority
        # Let the batcher decide whether to batch or queue directly
        self.batcher.add_job(job_id, final_prio, estimated_duration)

    def _submit_batch(self, batch):
        # Called by the batcher to submit a batch of jobs
        # We submit them as a single meta‑item to the priority queue
        # (the queue will treat them as a single element)
        meta_prio = min(p for _, p in batch)
        self.task_queue.put((meta_prio, time.time(), batch))

    def _run_worker(self):
        # ... existing worker loop ...
        # When a batch is taken from the queue, execute its jobs in sequence
        # (or in parallel if enough GPUs).

    def start(self):
        # Start the dynamic manager
        self.dynamic_mgr.start()
        # Start the worker thread
        threading.Thread(target=self._run_worker, daemon=True).start()



import os
import time
import signal
import threading
import queue
import subprocess
import torch
from django.utils import timezone
from .models import Job
from .dynamic_priority import DynamicPriorityManager
from .predictive import PriorityPredictor
from .batching import ShortJobBatcher

class GPUScheduler:
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
            import torch
            total_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.total_gpus = total_gpus
        self.available_gpus = total_gpus
        self.task_queue = queue.PriorityQueue()
        self.running_jobs = {}          # job_id -> subprocess.Popen
        self.job_gpu_map = {}           # job_id -> allocated GPUs
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self.worker_thread = None
        self.predictor = PriorityPredictor()
        self.dynamic_mgr = DynamicPriorityManager(self)
        self.batcher = ShortJobBatcher(self)
        self.preempt_flag_dir = "/tmp/preempt_flags"
        os.makedirs(self.preempt_flag_dir, exist_ok=True)
        self._initialized = True

    def add_job(self, job_id, priority, estimated_duration=None, memory_gb=None):
        """Add a new job to the scheduler."""
        job = Job.objects.get(id=job_id)
        # Use AI predictor to refine priority (if available)
        predicted = self.predictor.predict_priority(job)
        if predicted:
            # Weighted combination: 70% user priority, 30% predicted (or vice versa)
            final_priority = int(round(0.3 * priority + 0.7 * predicted))
            final_priority = max(1, min(10, final_priority))
        else:
            final_priority = priority
        # Pass to batcher (may batch short jobs)
        self.batcher.add_job(job_id, final_priority, estimated_duration, memory_gb)
        # Optionally update job's priority in DB
        if job.priority != final_priority:
            job.priority = final_priority
            job.save(update_fields=['priority'])

    def _submit_job(self, job_id, priority):
        """Internal: put job directly into priority queue."""
        self.task_queue.put((priority, time.time(), job_id))

    def _try_preempt(self, incoming_prio):
        """Preempt a lower-priority job if needed."""
        with self.lock:
            # Find a running job with lower priority (higher numeric value)
            for job_id, proc in list(self.running_jobs.items()):
                job = Job.objects.get(id=job_id)
                if job.priority > incoming_prio:
                    # Signal the job to stop
                    flag_file = os.path.join(self.preempt_flag_dir, str(job_id))
                    with open(flag_file, 'w') as f:
                        f.write('preempt')
                    # Wait for graceful termination (max 10 seconds)
                    start = time.time()
                    while proc.poll() is None and (time.time() - start) < 10:
                        time.sleep(0.5)
                    if proc.poll() is None:
                        proc.terminate()
                    # Free resources
                    self.available_gpus += self.job_gpu_map.pop(job_id, 1)
                    self.running_jobs.pop(job_id, None)
                    Job.objects.filter(id=job_id).update(status='preempted')
                    return True
        return False

    def _run_job(self, job_id):
        """Launch a training job as a subprocess."""
        job = Job.objects.get(id=job_id)
        job.status = 'running'
        job.started_at = timezone.now()
        job.save()
        # Clear any preempt flag
        flag_file = os.path.join(self.preempt_flag_dir, str(job_id))
        if os.path.exists(flag_file):
            os.remove(flag_file)
        # Launch subprocess
        cmd = [
            'python', 'manage.py', 'run_training_job',
            '--job-id', str(job_id),
            '--checkpoint-dir', job.checkpoint_path or ''
        ]
        proc = subprocess.Popen(cmd)
        with self.lock:
            self.running_jobs[job_id] = proc
            self.job_gpu_map[job_id] = job.gpu_requirements
        proc.wait()
        # After completion, check status (will be updated by the subprocess)
        with self.lock:
            self.available_gpus += self.job_gpu_map.pop(job_id, 1)
            self.running_jobs.pop(job_id, None)
        # If job was preempted, re-queue it with same priority
        job.refresh_from_db()
        if job.status == 'preempted':
            self.add_job(job_id, job.priority, job.estimated_duration, job.memory_requirements)

    def _worker(self):
        """Background worker that picks jobs from the queue and runs them."""
        while not self._stop_event.is_set():
            try:
                priority, _, job_id = self.task_queue.get(timeout=2)
                # Check if we have GPUs
                with self.lock:
                    if self.available_gpus <= 0:
                        # Try preemption
                        preempted = self._try_preempt(priority)
                        if not preempted:
                            # Re-queue and wait
                            self.task_queue.put((priority, time.time(), job_id))
                            time.sleep(5)
                            continue
                        # Preemption succeeded, we now have free GPU
                        self.available_gpus -= 1
                    else:
                        self.available_gpus -= 1
                # Launch the job
                self._run_job(job_id)
            except queue.Empty:
                continue

    def start(self):
        """Start the scheduler background threads."""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self._worker, daemon=True)
            self.worker_thread.start()
        self.dynamic_mgr.start()
        print(f"Scheduler started with {self.total_gpus} GPUs")

    def stop(self):
        self._stop_event.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=5)

    def update_priority(self, job_id, new_priority):
        """Update the priority of a queued job in the priority queue."""
        # This is non-trivial because PriorityQueue doesn't support removal.
        # We'll simply mark the job as updated and rely on the dynamic manager
        # to use the new priority when the job is re-inserted (if it gets preempted).
        # For now, we only update the DB.
        Job.objects.filter(id=job_id).update(priority=new_priority)

# Singleton instance
scheduler = GPUScheduler()