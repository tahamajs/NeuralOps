import sys
import os
import json
import signal
from django.core.management.base import BaseCommand
from django.utils import timezone
from jobscheduler.models import Job
from framework.experiment import run_training   # your existing training function

class Command(BaseCommand):
    help = 'Run a training job for a given Job ID'

    def add_arguments(self, parser):
        parser.add_argument('--job-id', type=str, required=True)

    def handle(self, *args, **options):
        job_id = options['job_id']
        job = Job.objects.get(id=job_id)

        # Ensure log directory exists
        log_dir = '/tmp/job_logs'
        os.makedirs(log_dir, exist_ok=True)
        log_file = f'{log_dir}/{job_id}.txt'

        def log_callback(step, logs):
            with open(log_file, 'a') as f:
                f.write(f"Step {step}: {logs}\n")
            job.logs += f"Step {step}: {logs}\n"
            job.save(update_fields=['logs'])

        def progress_callback(step, total, pct):
            job.current_step = step
            job.total_steps = total
            job.progress = pct
            job.save(update_fields=['current_step', 'total_steps', 'progress'])

        def final_callback(metrics, history):
            job.final_metrics = metrics
            job.training_history = history
            job.status = 'completed'
            job.completed_at = timezone.now()
            job.save()

        config = {
            'model_name': job.model_name,
            'task_name': job.task_name,
            'attention_type': job.attention_type,
            'method': job.method,
            'epochs': job.epochs,
            'batch_size': job.batch_size,
            'learning_rate': job.learning_rate,
            'max_seq_length': job.max_seq_length,
            'rank': job.compression_rank,
            'output_dir': f'/models/{job_id}',
            'resume_from_checkpoint': job.checkpoint_path if job.checkpoint_path else None,
        }

        try:
            run_training(
                config,
                progress_callback=progress_callback,
                log_callback=log_callback,
                final_callback=final_callback,
                stop_event=None
            )
        except Exception as e:
            job.status = 'failed'
            job.logs += f"\nERROR: {str(e)}"
            job.save()
            sys.exit(1)

from jobscheduler.version_utils import capture_environment

class Command(BaseCommand):
    def handle(self, *args, **options):
        job = Job.objects.get(id=job_id)
        # Capture environment at training start
        capture_environment(job)
        
        # Then run training (with logging integration)
        ... (existing code)




import sys
import os
import json
import signal
from django.core.management.base import BaseCommand
from django.utils import timezone
from jobscheduler.models import Job
from framework.experiment import run_training

class Command(BaseCommand):
    help = 'Run a training job for a given Job ID with preemption support'

    def add_arguments(self, parser):
        parser.add_argument('--job-id', type=str, required=True)
        parser.add_argument('--checkpoint-dir', type=str, default='')

    def handle(self, *args, **options):
        job_id = options['job_id']
        job = Job.objects.get(id=job_id)
        checkpoint_dir = options['checkpoint_dir']
        preempt_flag = f"/tmp/preempt_flags/{job_id}"

        def check_preempt():
            if os.path.exists(preempt_flag):
                self.stdout.write("Preemption signal received, saving checkpoint...")
                # Save checkpoint logic (framework should support)
                # For simplicity, we'll just exit
                job.status = 'preempted'
                job.save()
                sys.exit(0)

        # Hook into training loop (periodically call check_preempt)
        # For brevity, assume the training function accepts a callback
        try:
            run_training(
                config={...},
                preempt_callback=check_preempt,
            )
            job.status = 'completed'
            job.completed_at = timezone.now()
            job.save()
        except Exception as e:
            job.status = 'failed'
            job.logs += f"\nERROR: {str(e)}"
            job.save()
            sys.exit(1)