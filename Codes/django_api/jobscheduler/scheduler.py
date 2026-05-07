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
        self.job_start_time = {}        # job_id -> start timestamp
        self.lock = threading.RLock()
        self._stop_event = threading.Event()
        self.worker_thread = None
        self.monitor_thread = None
        self.predictor = PriorityPredictor()
        self.dynamic_mgr = DynamicPriorityManager(self)
        self.batcher = ShortJobBatcher(self)
        self.preempt_flag_dir = "/tmp/preempt_flags"
        os.makedirs(self.preempt_flag_dir, exist_ok=True)
        self.job_timeout = getattr(settings, 'JOB_TIMEOUT_SECONDS', 0)
        self._initialized = True
        logger.info(f"GPUScheduler initialized with {self.total_gpus} GPU(s)")

    def add_job(self, job_id, priority, estimated_duration=None, memory_gb=None):
        job = Job.objects.get(id=job_id)
        predicted = self.predictor.predict_priority(job)
        if predicted:
            final_priority = int(round(0.3 * priority + 0.7 * predicted))
            final_priority = max(1, min(10, final_priority))
        else:
            final_priority = priority
        self.batcher.add_job(job_id, final_priority, estimated_duration, memory_gb)
        if job.priority != final_priority:
            job.priority = final_priority
            job.save(update_fields=['priority'])
        logger.info(f"Job {job_id} added with priority {final_priority}")

    def _submit_job(self, job_id, priority):
        self.task_queue.put((priority, time.time(), job_id))

    def _try_preempt(self, incoming_prio):
        with self.lock:
            sorted_jobs = sorted(self.running_jobs.items(),
                                 key=lambda x: Job.objects.get(id=x[0]).priority, reverse=True)
            for job_id, proc in sorted_jobs:
                job = Job.objects.get(id=job_id)
                if job.priority > incoming_prio:
                    flag_file = os.path.join(self.preempt_flag_dir, str(job_id))
                    with open(flag_file, 'w') as f:
                        f.write('preempt')
                    logger.info(f"Preempt signal sent to job {job_id}")
                    start = time.time()
                    while proc.poll() is None and (time.time() - start) < 15:
                        time.sleep(0.5)
                    if proc.poll() is None:
                        proc.terminate()
                        logger.warning(f"Job {job_id} did not exit, terminated")
                    self.available_gpus += self.job_gpu_map.pop(job_id, 1)
                    self.running_jobs.pop(job_id, None)
                    self.job_start_time.pop(job_id, None)
                    Job.objects.filter(id=job_id).update(status='preempted')
                    return True
        return False

    def _run_job(self, job_id):
        job = Job.objects.get(id=job_id)
        job.status = 'running'
        job.started_at = timezone.now()
        job.save()
        self.job_start_time[job_id] = time.time()
        flag_file = os.path.join(self.preempt_flag_dir, str(job_id))
        if os.path.exists(flag_file):
            os.remove(flag_file)
        cmd = ['python', 'manage.py', 'run_training_job',
               '--job-id', str(job_id),
               '--checkpoint-dir', job.checkpoint_path or '']
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with self.lock:
                self.running_jobs[job_id] = proc
                self.job_gpu_map[job_id] = job.gpu_requirements
            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                logger.error(f"Job {job_id} failed: {stderr.decode()}")
                job.status = 'failed'
                job.logs += f"\nProcess error: {stderr.decode()}"
                job.save()
        except Exception as e:
            logger.exception(f"Exception running job {job_id}: {e}")
            job.status = 'failed'
            job.logs += f"\nException: {str(e)}"
            job.save()
        finally:
            with self.lock:
                self.available_gpus += self.job_gpu_map.pop(job_id, 1)
                self.running_jobs.pop(job_id, None)
                self.job_start_time.pop(job_id, None)
            job.refresh_from_db()
            if job.status == 'preempted':
                self.add_job(job_id, job.priority, job.estimated_duration, job.memory_requirements)

    def _monitor_jobs(self):
        while not self._stop_event.is_set():
            time.sleep(10)
            if self.job_timeout > 0:
                with self.lock:
                    now = time.time()
                    for job_id, start in list(self.job_start_time.items()):
                        if now - start > self.job_timeout:
                            proc = self.running_jobs.get(job_id)
                            if proc:
                                proc.terminate()
                            Job.objects.filter(id=job_id).update(status='failed')
                            self.available_gpus += self.job_gpu_map.pop(job_id, 1)
                            self.running_jobs.pop(job_id, None)
                            self.job_start_time.pop(job_id, None)

    def _worker(self):
        while not self._stop_event.is_set():
            try:
                priority, _, job_id = self.task_queue.get(timeout=2)
                with self.lock:
                    if self.available_gpus <= 0:
                        if not self._try_preempt(priority):
                            self.task_queue.put((priority, time.time(), job_id))
                            time.sleep(5)
                            continue
                    self.available_gpus -= 1
                self._run_job(job_id)
            except queue.Empty:
                continue
            except Exception as e:
                logger.exception(f"Error in scheduler worker: {e}")

    def start(self):
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self._worker, daemon=True, name="SchedulerWorker")
            self.worker_thread.start()
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.monitor_thread = threading.Thread(target=self._monitor_jobs, daemon=True, name="SchedulerMonitor")
            self.monitor_thread.start()
        self.dynamic_mgr.start()
        logger.info(f"Scheduler started (timeout={self.job_timeout}s)")

    def stop(self, graceful=True):
        self._stop_event.set()
        if graceful:
            start = time.time()
            while self.running_jobs and (time.time() - start) < 60:
                time.sleep(1)
        for job_id, proc in list(self.running_jobs.items()):
            proc.terminate()
            logger.warning(f"Terminated job {job_id}")
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Scheduler stopped")

    def update_priority(self, job_id, new_priority):
        Job.objects.filter(id=job_id).update(priority=new_priority)
        logger.info(f"Priority of {job_id} changed to {new_priority}")

    def pause_job(self, job_id):
        with self.lock:
            proc = self.running_jobs.get(str(job_id))
            if proc:
                proc.send_signal(signal.SIGSTOP)
                Job.objects.filter(id=job_id).update(status='paused')
                logger.info(f"Job {job_id} paused")

    def resume_job(self, job_id):
        with self.lock:
            proc = self.running_jobs.get(str(job_id))
            if proc:
                proc.send_signal(signal.SIGCONT)
                Job.objects.filter(id=job_id).update(status='running')
                logger.info(f"Job {job_id} resumed")
            else:
                job = Job.objects.get(id=job_id)
                self.add_job(job_id, job.priority, job.estimated_duration, job.memory_requirements)

    def stop_job(self, job_id):
        with self.lock:
            proc = self.running_jobs.get(str(job_id))
            if proc:
                proc.terminate()
                Job.objects.filter(id=job_id).update(status='stopped')
                self.available_gpus += self.job_gpu_map.pop(job_id, 1)
                self.running_jobs.pop(job_id, None)
                self.job_start_time.pop(job_id, None)
                logger.info(f"Job {job_id} stopped")

    def get_status(self):
        return {
            'total_gpus': self.total_gpus,
            'available_gpus': self.available_gpus,
            'queued_jobs': self.task_queue.qsize(),
            'running_jobs': len(self.running_jobs),
            'preempted_jobs': Job.objects.filter(status='preempted').count(),
        }


# Singleton instance
scheduler = GPUScheduler()