# jobscheduler/scheduler.py
import os
import time
import signal
import threading
import queue
import subprocess
import logging
import torch
from django.utils import timezone
from django.conf import settings
from .models import Job
from .dynamic_priority import DynamicPriorityManager
from .predictive import PriorityPredictor
from .batching import ShortJobBatcher

logger = logging.getLogger(__name__)

class GPUScheduler:
    """
    Singleton GPU scheduler with priority queue, preemption, batching,
    AI priority prediction, dynamic adjustment, and job timeout.
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
        self.running_jobs = {}          # job_id -> subprocess.Popen
        self.job_gpu_map = {}           # job_id -> allocated GPUs
        self.job_start_time = {}        # job_id -> start timestamp (for timeout)
        self.lock = threading.RLock()   # reentrant lock for safety
        self._stop_event = threading.Event()
        self.worker_thread = None
        self.monitor_thread = None
        self.predictor = PriorityPredictor()
        self.dynamic_mgr = DynamicPriorityManager(self)
        self.batcher = ShortJobBatcher(self)
        self.preempt_flag_dir = "/tmp/preempt_flags"
        os.makedirs(self.preempt_flag_dir, exist_ok=True)
        # Timeout for jobs (seconds); 0 = disabled
        self.job_timeout = getattr(settings, 'JOB_TIMEOUT_SECONDS', 0)
        self._initialized = True
        logger.info(f"GPUScheduler initialized with {self.total_gpus} GPU(s)")

    def add_job(self, job_id, priority, estimated_duration=None, memory_gb=None):
        """
        Add a new job to the scheduler.
        - job_id: UUID string
        - priority: user-assigned priority (1-10, lower = higher)
        - estimated_duration: optional estimated runtime in seconds (for batching)
        - memory_gb: optional memory requirement (for future resource allocation)
        """
        job = Job.objects.get(id=job_id)
        # Use AI predictor to refine priority (if model exists)
        predicted = self.predictor.predict_priority(job)
        if predicted:
            final_priority = int(round(0.3 * priority + 0.7 * predicted))
            final_priority = max(1, min(10, final_priority))
        else:
            final_priority = priority
        # Pass to batcher (which may batch short jobs)
        self.batcher.add_job(job_id, final_priority, estimated_duration, memory_gb)
        # Update job's priority in DB if changed
        if job.priority != final_priority:
            job.priority = final_priority
            job.save(update_fields=['priority'])
        logger.info(f"Job {job_id} added with priority {final_priority}")

    def _submit_job(self, job_id, priority):
        """Internal: put a job directly into the priority queue (bypass batching)."""
        self.task_queue.put((priority, time.time(), job_id))

    def _try_preempt(self, incoming_prio):
        """
        Attempt to preempt a lower-priority running job to free a GPU.
        Returns True if preemption succeeded (GPU freed), False otherwise.
        """
        with self.lock:
            # Sort running jobs by priority (lowest priority first)
            sorted_jobs = sorted(self.running_jobs.items(), key=lambda x: Job.objects.get(id=x[0]).priority, reverse=True)
            for job_id, proc in sorted_jobs:
                job = Job.objects.get(id=job_id)
                if job.priority > incoming_prio:   # lower numeric = higher priority
                    # Signal the job to save checkpoint and exit
                    flag_file = os.path.join(self.preempt_flag_dir, str(job_id))
                    with open(flag_file, 'w') as f:
                        f.write('preempt')
                    logger.info(f"Preempt signal sent to job {job_id} (priority {job.priority} > {incoming_prio})")
                    # Wait up to 15 seconds for graceful termination
                    start = time.time()
                    while proc.poll() is None and (time.time() - start) < 15:
                        time.sleep(0.5)
                    if proc.poll() is None:
                        proc.terminate()
                        logger.warning(f"Job {job_id} did not exit gracefully, terminated")
                    # Free resources
                    self.available_gpus += self.job_gpu_map.pop(job_id, 1)
                    self.running_jobs.pop(job_id, None)
                    self.job_start_time.pop(job_id, None)
                    Job.objects.filter(id=job_id).update(status='preempted')
                    return True
        return False

    def _run_job(self, job_id):
        """Launch a training job as a subprocess."""
        job = Job.objects.get(id=job_id)
        job.status = 'running'
        job.started_at = timezone.now()
        job.save()
        self.job_start_time[job_id] = time.time()
        # Clear any preempt flag for this job
        flag_file = os.path.join(self.preempt_flag_dir, str(job_id))
        if os.path.exists(flag_file):
            os.remove(flag_file)
        # Build command
        cmd = [
            'python', 'manage.py', 'run_training_job',
            '--job-id', str(job_id),
            '--checkpoint-dir', job.checkpoint_path or ''
        ]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with self.lock:
                self.running_jobs[job_id] = proc
                self.job_gpu_map[job_id] = job.gpu_requirements
            # Wait for completion
            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                logger.error(f"Job {job_id} failed with return code {proc.returncode}")
                job.status = 'failed'
                job.logs += f"\nProcess error: {stderr.decode()}"
                job.save()
        except Exception as e:
            logger.exception(f"Exception while running job {job_id}: {e}")
            job.status = 'failed'
            job.logs += f"\nException: {str(e)}"
            job.save()
        finally:
            with self.lock:
                self.available_gpus += self.job_gpu_map.pop(job_id, 1)
                self.running_jobs.pop(job_id, None)
                self.job_start_time.pop(job_id, None)
            # If job was preempted, re-queue it
            job.refresh_from_db()
            if job.status == 'preempted':
                logger.info(f"Re-queueing preempted job {job_id}")
                self.add_job(job_id, job.priority, job.estimated_duration, job.memory_requirements)

    def _monitor_jobs(self):
        """Background thread to check for timed-out jobs and update metrics."""
        while not self._stop_event.is_set():
            time.sleep(10)
            if self.job_timeout > 0:
                with self.lock:
                    now = time.time()
                    for job_id, start in list(self.job_start_time.items()):
                        if now - start > self.job_timeout:
                            logger.warning(f"Job {job_id} timed out after {self.job_timeout} seconds")
                            proc = self.running_jobs.get(job_id)
                            if proc:
                                proc.terminate()
                            # Update DB
                            Job.objects.filter(id=job_id).update(status='failed', logs=models.F('logs') + f"\nJob timed out after {self.job_timeout} seconds")
                            # Free resources
                            self.available_gpus += self.job_gpu_map.pop(job_id, 1)
                            self.running_jobs.pop(job_id, None)
                            self.job_start_time.pop(job_id, None)
            # Log queue size periodically
            queue_size = self.task_queue.qsize()
            if queue_size > 0:
                logger.debug(f"Queue size: {queue_size}, available GPUs: {self.available_gpus}")

    def _worker(self):
        """Background worker that picks jobs from the queue and runs them."""
        while not self._stop_event.is_set():
            try:
                priority, _, job_id = self.task_queue.get(timeout=2)
                logger.debug(f"Worker picked job {job_id} with priority {priority}")
                # Check if we have available GPUs
                with self.lock:
                    if self.available_gpus <= 0:
                        # Try preemption
                        preempted = self._try_preempt(priority)
                        if not preempted:
                            # No preemption possible -> re-queue and wait
                            self.task_queue.put((priority, time.time(), job_id))
                            logger.debug(f"No GPU for job {job_id}, re-queued")
                            time.sleep(5)
                            continue
                        # Preemption succeeded, we now have a free GPU
                        self.available_gpus -= 1
                    else:
                        self.available_gpus -= 1
                # Launch the job
                self._run_job(job_id)
            except queue.Empty:
                continue
            except Exception as e:
                logger.exception(f"Unexpected error in scheduler worker: {e}")

    def start(self):
        """Start the scheduler background threads."""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self._worker, daemon=True, name="SchedulerWorker")
            self.worker_thread.start()
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.monitor_thread = threading.Thread(target=self._monitor_jobs, daemon=True, name="SchedulerMonitor")
            self.monitor_thread.start()
        self.dynamic_mgr.start()
        logger.info(f"Scheduler started with {self.total_gpus} GPU(s) (timeout={self.job_timeout}s)")

    def stop(self, graceful=True):
        """Stop the scheduler. If graceful, wait for running jobs to finish."""
        self._stop_event.set()
        if graceful:
            # Wait for running jobs to complete (max 60 seconds)
            start = time.time()
            while self.running_jobs and (time.time() - start) < 60:
                time.sleep(1)
            # Terminate remaining jobs
            for job_id, proc in list(self.running_jobs.items()):
                proc.terminate()
                logger.warning(f"Force-terminated job {job_id} on scheduler stop")
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Scheduler stopped")

    def update_priority(self, job_id, new_priority):
        """
        Update the priority of a queued job. 
        For queued jobs, we cannot modify the priority in the PriorityQueue easily.
        Instead, we mark the job as updated and rely on the dynamic manager or
        the job being re-inserted after preemption to use the new priority.
        """
        Job.objects.filter(id=job_id).update(priority=new_priority)
        logger.info(f"Updated priority of job {job_id} to {new_priority}")

    def pause_job(self, job_id):
        """Pause a currently running job (send SIGSTOP)."""
        with self.lock:
            proc = self.running_jobs.get(str(job_id))
            if proc:
                proc.send_signal(signal.SIGSTOP)
                Job.objects.filter(id=job_id).update(status='paused')
                logger.info(f"Job {job_id} paused")
            else:
                logger.warning(f"Cannot pause job {job_id}: not running")

    def resume_job(self, job_id):
        """Resume a paused job (send SIGCONT) or re‑queue if not running."""
        with self.lock:
            proc = self.running_jobs.get(str(job_id))
            if proc:
                proc.send_signal(signal.SIGCONT)
                Job.objects.filter(id=job_id).update(status='running')
                logger.info(f"Job {job_id} resumed")
            else:
                # Not running, re‑queue
                job = Job.objects.get(id=job_id)
                self.add_job(job_id, job.priority, job.estimated_duration, job.memory_requirements)
                logger.info(f"Job {job_id} re-queued (was not running)")

    def stop_job(self, job_id):
        """Terminate a running job (no auto‑resume)."""
        with self.lock:
            proc = self.running_jobs.get(str(job_id))
            if proc:
                proc.terminate()
                Job.objects.filter(id=job_id).update(status='stopped')
                logger.info(f"Job {job_id} stopped")
                # Free resources
                self.available_gpus += self.job_gpu_map.pop(job_id, 1)
                self.running_jobs.pop(job_id, None)
                self.job_start_time.pop(job_id, None)
            else:
                logger.warning(f"Cannot stop job {job_id}: not running")

    def get_status(self):
        """Return current scheduler status (for API)."""
        return {
            'total_gpus': self.total_gpus,
            'available_gpus': self.available_gpus,
            'queued_jobs': self.task_queue.qsize(),
            'running_jobs': len(self.running_jobs),
            'preempted_jobs': Job.objects.filter(status='preempted').count(),
        }

# Singleton instance
scheduler = GPUScheduler()