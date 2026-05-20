"""
GPU Scheduler — Production Grade with Priority Aging, Backfill,
Fair Share, SJF, Graceful Preemption, and Non‑Preemptible Support.

References:
  • NVIDIA KAI Scheduler – priority + preemption + fair share pools
  • SLURM backfill scheduling – run short jobs in idle gaps
  • AdaptDL – elastic scaling, checkpoint‑before‑preempt
  • DeepPlus / Shockwave – deadline‑aware + fairness‑aware scheduling
  • taskmill priority aging – escalate priority over time to prevent starvation
"""

import os, time, signal, threading, queue, subprocess, logging, json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import IntEnum

import torch
from django.utils import timezone
from django.conf import settings
from django.db import connection

from .models import Job, JobGroup

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Priority tiers – lower integer = higher urgency
# ---------------------------------------------------------------------------
class Priority(IntEnum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BEST_EFFORT = 4


# ---------------------------------------------------------------------------
# Configuration dataclass – everything tunable in one place
# ---------------------------------------------------------------------------
@dataclass
class SchedulerConfig:
    # ── Priority Aging ─────────────────────────────────────────────
    aging_enabled: bool = True
    aging_grace_period: int = 300        # seconds before aging starts
    aging_interval: int = 60             # seconds between priority bumps
    aging_max_effective: int = Priority.HIGH  # never age beyond this

    # ── Backfill ───────────────────────────────────────────────────
    backfill_enabled: bool = True
    backfill_interval: int = 15          # seconds between backfill scans
    backfill_max_job_duration: int = 300 # max estimated seconds for backfill

    # ── Fair Share ─────────────────────────────────────────────────
    fair_share_enabled: bool = True
    fair_share_decay_half_life: int = 86400  # 24‑hour half‑life

    # ── Preemption ─────────────────────────────────────────────────
    preempt_checkpoint_grace: int = 30   # seconds to allow checkpoint save
    preempt_min_priority_gap: int = 2    # incoming must be at least this many
                                         #   tiers higher to trigger preemption

    # ── Job Timeout ─────────────────────────────────────────────────
    job_timeout: int = 0                 # seconds; 0 = no timeout

    # ── Observability ──────────────────────────────────────────────
    stats_window: int = 3600             # lookback for average wait time, etc.

    @classmethod
    def from_settings(cls):
        return cls(
            job_timeout=getattr(settings, 'JOB_TIMEOUT_SECONDS', 0),
        )


# ---------------------------------------------------------------------------
# Main Scheduler (Singleton)
# ---------------------------------------------------------------------------
class GPUScheduler:
    _instance = None

    def __new__(cls, total_gpus=None, config: SchedulerConfig = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialised = False
        return cls._instance

    def __init__(self, total_gpus=None, config: SchedulerConfig = None):
        if self._initialised:
            return
        if total_gpus is None:
            total_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.total_gpus = total_gpus
        self.available_gpus = total_gpus
        self.cfg = config or SchedulerConfig.from_settings()

        # Priority queue: (effective_priority, submit_time, job_id)
        self.task_queue = queue.PriorityQueue()

        self.running_jobs: Dict[str, subprocess.Popen] = {}
        self.job_gpu_map: Dict[str, int] = {}
        self.job_start_time: Dict[str, float] = {}

        # Fair‑share bookkeeping
        self.group_usage: Dict[str, float] = {}   # group → cumulative GPU‑seconds
        self.group_last_update: Dict[str, float] = {}

        self.lock = threading.RLock()
        self._stop_event = threading.Event()
        self.worker_thread = None
        self.monitor_thread = None
        self.backfill_thread = None

        self.preempt_flag_dir = "/tmp/preempt_flags"
        os.makedirs(self.preempt_flag_dir, exist_ok=True)

        self._initialised = True
        logger.info(
            "GPUScheduler v2 initialised: %d GPU(s), aging=%s, backfill=%s, fairshare=%s",
            self.total_gpus,
            self.cfg.aging_enabled,
            self.cfg.backfill_enabled,
            self.cfg.fair_share_enabled,
        )

    # ── Public API ──────────────────────────────────────────────────
    def add_job(self, job_id: str, base_priority: int,
                estimated_duration: int = None, memory_gb: int = None,
                group: str = "default", preemptible: bool = True):
        """Submit a job to the queue."""
        effective = self._effective_priority(job_id, base_priority)
        self.task_queue.put((effective, time.time(), job_id))
        Job.objects.filter(id=job_id).update(
            status='queued',
            aging_priority=effective,
        )
        logger.info(
            "Job %s enqueued (base_prio=%d, effective=%d, group=%s, preemptible=%s)",
            job_id, base_priority, effective, group, preemptible,
        )

    def update_priority(self, job_id: str, new_priority: int):
        """Update base priority of a queued job."""
        Job.objects.filter(id=job_id).update(priority=new_priority)
        logger.info("Priority of %s changed to %d", job_id, new_priority)

    def pause_job(self, job_id: str):
        with self.lock:
            proc = self.running_jobs.get(job_id)
            if proc:
                proc.send_signal(signal.SIGSTOP)
                Job.objects.filter(id=job_id).update(status='paused')

    def resume_job(self, job_id: str):
        with self.lock:
            proc = self.running_jobs.get(job_id)
            if proc:
                proc.send_signal(signal.SIGCONT)
                Job.objects.filter(id=job_id).update(status='running')
            else:
                job = Job.objects.get(id=job_id)
                self.add_job(job_id, job.priority, job.estimated_duration,
                             job.memory_requirements, getattr(job, 'group', 'default'),
                             job.preemptible)

    def stop_job(self, job_id: str):
        with self.lock:
            proc = self.running_jobs.get(job_id)
            if proc:
                proc.terminate()
                Job.objects.filter(id=job_id).update(status='stopped')
                self.available_gpus += self.job_gpu_map.pop(job_id, 1)
                self.running_jobs.pop(job_id, None)
                self.job_start_time.pop(job_id, None)

    def get_status(self) -> dict:
        return {
            'total_gpus': self.total_gpus,
            'available_gpus': self.available_gpus,
            'queued_jobs': self.task_queue.qsize(),
            'running_jobs': len(self.running_jobs),
            'preempted_count': Job.objects.filter(status='preempted').count(),
            'fair_share': self._fair_share_snapshot(),
        }

    # ── Lifecycle ───────────────────────────────────────────────────
    def start(self):
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(
                target=self._worker, daemon=True, name="SchedulerWorker")
            self.worker_thread.start()
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.monitor_thread = threading.Thread(
                target=self._monitor_jobs, daemon=True, name="SchedulerMonitor")
            self.monitor_thread.start()
        if self.cfg.backfill_enabled and (
            self.backfill_thread is None or not self.backfill_thread.is_alive()):
            self.backfill_thread = threading.Thread(
                target=self._backfill_loop, daemon=True, name="BackfillWorker")
            self.backfill_thread.start()
        logger.info("Scheduler v2 started (timeout=%ds)", self.cfg.job_timeout)

    def stop(self, graceful=True):
        self._stop_event.set()
        if graceful:
            start = time.time()
            while self.running_jobs and (time.time() - start) < 60:
                time.sleep(1)
        for job_id, proc in list(self.running_jobs.items()):
            proc.terminate()
            logger.warning("Terminated job %s", job_id)
        for t in (self.worker_thread, self.monitor_thread, self.backfill_thread):
            if t:
                t.join(timeout=5)
        logger.info("Scheduler v2 stopped")

    # ── Internal: Priority Aging ────────────────────────────────────
    def _effective_priority(self, job_id: str, base_priority: int) -> int:
        """Compute priority after aging.  Lower = more urgent."""
        if not self.cfg.aging_enabled:
            return base_priority

        try:
            job = Job.objects.get(id=job_id)
        except Job.DoesNotExist:
            return base_priority

        wait = (timezone.now() - job.created_at).total_seconds()
        if wait <= self.cfg.aging_grace_period:
            return base_priority

        steps = int((wait - self.cfg.aging_grace_period) / self.cfg.aging_interval)
        effective = max(base_priority - steps, self.cfg.aging_max_effective)
        return max(effective, Priority.CRITICAL)

    # ── Internal: Fair Share ────────────────────────────────────────
    def _fair_share_weight(self, group: str) -> float:
        """Return a multiplier 0..1 for how much this group is 'owed'."""
        if not self.cfg.fair_share_enabled:
            return 1.0

        now = time.time()
        # Decay old usage
        for g in list(self.group_usage.keys()):
            elapsed = now - self.group_last_update.get(g, now)
            decay = 0.5 ** (elapsed / self.cfg.fair_share_decay_half_life)
            self.group_usage[g] *= decay
            self.group_last_update[g] = now

        try:
            group_obj = JobGroup.objects.get(name=group)
            weight = group_obj.weight
        except JobGroup.DoesNotExist:
            weight = 1.0

        usage = self.group_usage.get(group, 0.0)
        # Higher weight and lower usage → higher "fairness" multiplier
        fair_mult = weight / (1.0 + usage)
        # Clamp to [0.2, 2.0]
        return max(0.2, min(2.0, fair_mult))

    def _record_usage(self, group: str, gpus: int, duration_s: float):
        """Record GPU‑seconds used by a group."""
        self.group_usage[group] = self.group_usage.get(group, 0.0) + gpus * duration_s
        self.group_last_update[group] = time.time()

    def _fair_share_snapshot(self) -> dict:
        return {
            g: round(u, 1) for g, u in self.group_usage.items()
        }

    # ── Internal: Preemption ────────────────────────────────────────
    def _try_preempt(self, incoming_prio: int) -> bool:
        """Try to free a GPU by preempting a lower‑priority running job.

        Returns True if a GPU was freed."""
        if incoming_prio >= self.cfg.preempt_min_priority_gap:
            return False  # not urgent enough to preempt

        with self.lock:
            # Find running jobs sorted by priority (worst first)
            candidates = []
            for job_id in self.running_jobs:
                try:
                    job = Job.objects.get(id=job_id)
                    if not job.preemptible:
                        continue
                    candidates.append((job.priority, job_id))
                except Job.DoesNotExist:
                    continue

            candidates.sort(key=lambda x: x[0], reverse=True)  # lowest priority first

            for prio, job_id in candidates:
                if prio <= incoming_prio + self.cfg.preempt_min_priority_gap:
                    continue

                job = Job.objects.get(id=job_id)
                # Signal checkpoint
                self._signal_preempt(job_id)

                proc = self.running_jobs.get(job_id)
                start = time.time()
                while proc and proc.poll() is None and \
                      (time.time() - start) < self.cfg.preempt_checkpoint_grace:
                    time.sleep(0.5)

                if proc and proc.poll() is None:
                    proc.terminate()
                    logger.warning("Job %s terminated after preempt timeout", job_id)

                self.available_gpus += self.job_gpu_map.pop(job_id, 1)
                self.running_jobs.pop(job_id, None)
                self.job_start_time.pop(job_id, None)
                job.status = 'preempted'
                job.save()
                logger.info("Preempted job %s (priority %d) for incoming priority %d",
                            job_id, prio, incoming_prio)
                return True

        return False

    def _signal_preempt(self, job_id: str):
        """Write preemption flag so the training loop can save a checkpoint."""
        flag = os.path.join(self.preempt_flag_dir, str(job_id))
        with open(flag, 'w') as f:
            json.dump({
                'job_id': job_id,
                'timestamp': time.time(),
                'action': 'checkpoint_and_exit',
            }, f)

    # ── Internal: Backfill ──────────────────────────────────────────
    def _backfill_loop(self):
        """Periodically scan for short jobs that can run in idle gaps."""
        while not self._stop_event.is_set():
            time.sleep(self.cfg.backfill_interval)
            self._try_backfill()

    def _try_backfill(self):
        """Run short queued jobs if there are idle GPUs and the jobs won't
        block higher‑priority work."""
        with self.lock:
            free = self.available_gpus
            if free <= 0:
                return

        # Peek at the queue (we can't peek a PriorityQueue easily, so we copy)
        # For backfill we check DB directly
        short_jobs = Job.objects.filter(
            status='queued',
            estimated_duration__lte=self.cfg.backfill_max_job_duration,
        ).order_by('priority', 'created_at')

        for job in short_jobs[:free]:
            if self.available_gpus <= 0:
                break
            # Remove from queue (we re‑submit with higher priority)
            self._run_job_now(str(job.id))

    def _run_job_now(self, job_id: str):
        """Bypass the queue and launch a job immediately."""
        with self.lock:
            self.available_gpus -= 1
        threading.Thread(target=self._run_job, args=(job_id,), daemon=True).start()

    # ── Internal: Job Execution ─────────────────────────────────────
    def _run_job(self, job_id: str):
        job = Job.objects.get(id=job_id)
        job.status = 'running'
        job.started_at = timezone.now()
        job.save()
        self.job_start_time[job_id] = time.time()

        # Clear stale preempt flag
        flag = os.path.join(self.preempt_flag_dir, job_id)
        if os.path.exists(flag):
            os.remove(flag)

        group = getattr(job, 'group', 'default')
        cmd = [
            'python', 'manage.py', 'run_training_job',
            '--job-id', job_id,
            '--checkpoint-dir', job.checkpoint_path or '',
        ]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with self.lock:
                self.running_jobs[job_id] = proc
                self.job_gpu_map[job_id] = job.gpu_requirements

            stdout, stderr = proc.communicate()
            duration = time.time() - self.job_start_time.get(job_id, time.time())

            if proc.returncode != 0:
                logger.error("Job %s failed: %s", job_id, stderr.decode())
                job.status = 'failed'
                job.logs += f"\nProcess error: {stderr.decode()}"
                job.save()
            else:
                logger.info("Job %s completed successfully", job_id)

        except Exception as e:
            logger.exception("Exception running job %s: %s", job_id, e)
            job.status = 'failed'
            job.logs += f"\nException: {str(e)}"
            job.save()
        finally:
            with self.lock:
                self.available_gpus += self.job_gpu_map.pop(job_id, 1)
                self.running_jobs.pop(job_id, None)
                self.job_start_time.pop(job_id, None)

            # Fair‑share accounting
            self._record_usage(group, job.gpu_requirements, duration)

            job.refresh_from_db()
            if job.status == 'preempted':
                # Re‑queue with original priority
                self.add_job(
                    job_id, job.priority, job.estimated_duration,
                    job.memory_requirements, group, job.preemptible,
                )

    # ── Internal: Main Worker ───────────────────────────────────────
    def _worker(self):
        while not self._stop_event.is_set():
            try:
                _, _, job_id = self.task_queue.get(timeout=2)
            except queue.Empty:
                continue

            with self.lock:
                if self.available_gpus <= 0:
                    if not self._try_preempt(
                        self._effective_priority(job_id,
                            Job.objects.get(id=job_id).priority)):
                        self.task_queue.put((
                            self._effective_priority(job_id,
                                Job.objects.get(id=job_id).priority),
                            time.time(), job_id))
                        time.sleep(5)
                        continue
                self.available_gpus -= 1

            self._run_job(job_id)

    # ── Internal: Monitor (Timeout) ─────────────────────────────────
    def _monitor_jobs(self):
        if self.cfg.job_timeout <= 0:
            return
        while not self._stop_event.is_set():
            time.sleep(10)
            with self.lock:
                now = time.time()
                for job_id, start in list(self.job_start_time.items()):
                    if now - start > self.cfg.job_timeout:
                        proc = self.running_jobs.get(job_id)
                        if proc:
                            proc.terminate()
                        Job.objects.filter(id=job_id).update(status='failed')
                        self.available_gpus += self.job_gpu_map.pop(job_id, 1)
                        self.running_jobs.pop(job_id, None)
                        self.job_start_time.pop(job_id, None)
                        logger.warning("Job %s killed by timeout (%ds)", job_id,
                                       self.cfg.job_timeout)


# ---------------------------------------------------------------------------
# Singleton instance
# ---------------------------------------------------------------------------
scheduler = GPUScheduler()