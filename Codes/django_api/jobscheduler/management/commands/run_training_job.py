# jobscheduler/management/commands/run_training_job.py
import json
import os
import sys

from django.core.management.base import BaseCommand
from django.utils import timezone
from framework.experiment import run_training  # your existing training function
from jobscheduler.models import Job
from jobscheduler.version_utils import capture_environment


class Command(BaseCommand):
    help = "Run a training job for a given Job ID with preemption and logging support"

    def add_arguments(self, parser):
        parser.add_argument("--job-id", type=str, required=True, help="UUID of the job")
        parser.add_argument(
            "--checkpoint-dir", type=str, default="", help="Directory for checkpoints"
        )

    def handle(self, *args, **options):
        job_id = options["job_id"]
        checkpoint_dir = options["checkpoint_dir"]
        job = Job.objects.get(id=job_id)

        # Create log directory and file
        log_dir = "/tmp/job_logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = f"{log_dir}/{job_id}.txt"

        # Preemption flag file (used by scheduler)
        preempt_flag = f"/tmp/preempt_flags/{job_id}"

        # Ensure checkpoint directory exists
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        def check_preempt():
            """Check if scheduler has signalled preemption."""
            if os.path.exists(preempt_flag):
                self.stdout.write(
                    self.style.WARNING(
                        f"Preemption signal received for job {job_id}, saving checkpoint..."
                    )
                )
                # Save checkpoint (the training function should do this via callback)
                # We'll just update status and exit; training loop must call this callback.
                job.status = "preempted"
                job.save(update_fields=["status"])
                sys.exit(0)

        # Callback for logging training steps
        def log_callback(step, logs):
            with open(log_file, "a") as f:
                f.write(f"Step {step}: {json.dumps(logs)}\n")
            job.logs += f"Step {step}: {json.dumps(logs)}\n"
            job.save(update_fields=["logs"])

        # Callback for progress (step / total steps)
        def progress_callback(step, total, pct):
            job.current_step = step
            job.total_steps = total
            job.progress = pct
            job.save(update_fields=["current_step", "total_steps", "progress"])

        # Callback for final metrics
        def final_callback(metrics, history):
            job.final_metrics = metrics
            job.training_history = history
            job.status = "completed"
            job.completed_at = timezone.now()
            job.actual_duration = (
                int((job.completed_at - job.started_at).total_seconds())
                if job.started_at
                else None
            )
            job.save()

        # Build configuration for training function
        config = {
            "model_name": job.model_name,
            "task_name": job.task_name,
            "attention_type": job.attention_type,
            "method": job.method,
            "epochs": job.epochs,
            "batch_size": job.batch_size,
            "learning_rate": job.learning_rate,
            "max_seq_length": job.max_seq_length,
            "rank": job.compression_rank,
            "output_dir": checkpoint_dir or f"/models/{job_id}",
            "resume_from_checkpoint": (
                job.checkpoint_path if job.checkpoint_path else None
            ),
            "job_id": str(job.id),
        }

        try:
            # Capture environment at the start of training (pip freeze, python version)
            capture_environment(job)
            self.stdout.write(f"Starting training for job {job_id}")

            # Run training with callbacks
            run_training(
                config,
                progress_callback=progress_callback,
                log_callback=log_callback,
                final_callback=final_callback,
                preempt_callback=check_preempt,  # training function must call this periodically
                stop_event=None,
            )
        except Exception as e:
            job.status = "failed"
            job.logs += f"\nERROR: {str(e)}\n"
            job.save()
            self.stdout.write(
                self.style.ERROR(f"Training failed for job {job_id}: {e}")
            )
            sys.exit(1)
        finally:
            # Remove preempt flag if present
            if os.path.exists(preempt_flag):
                os.remove(preempt_flag)
