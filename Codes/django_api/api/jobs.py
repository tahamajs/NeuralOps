import threading, logging
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from .models import Experiment

logger = logging.getLogger(__name__)

# Store running threads to allow cancellation
running_jobs = {}

def start_job(experiment: Experiment, runner_func):
    def _run():
        running_jobs[str(experiment.id)] = threading.current_thread()
        try:
            experiment.status = "running"
            experiment.started_at = timezone.now()
            experiment.save()
            push_update(experiment, {"type": "status", "status": "running"})
            push_update(experiment, {"type": "log", "data": "Job started."})
            runner_func(experiment)
            if experiment.status != "cancelled":
                experiment.status = "completed"
                experiment.completed_at = timezone.now()
                experiment.save()
                push_update(experiment, {"type": "status", "status": "completed"})
                push_update(experiment, {"type": "log", "data": "Job completed successfully."})
        except Exception as e:
            experiment.status = "failed"
            experiment.logs += f"\nError: {str(e)}"
            experiment.save()
            push_update(experiment, {"type": "status", "status": "failed"})
            push_update(experiment, {"type": "log", "data": f"Job failed: {str(e)}"})
            logger.exception("Job failed")
        finally:
            running_jobs.pop(str(experiment.id), None)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

def cancel_job(experiment: Experiment):
    """Signal cancellation to a running job."""
    experiment.status = "cancelled"
    experiment.save()
    # The runner should periodically check if status changed to cancelled.
    # In the runner callback, we'll raise an exception to stop training.
    push_update(experiment, {"type": "status", "status": "cancelled"})





_running_jobs = {}   # job_id -> threading.Event for stopping

def stop_job(job_id):
    event = _running_jobs.pop(str(job_id), None)
    if event:
        event.set()

def start_job(experiment: Experiment, runner_func):
    stop_event = threading.Event()
    _running_jobs[str(experiment.id)] = stop_event

    def _run():
        try:
            experiment.status = "running"
            experiment.save()
            push_update(experiment, {"type": "status", "status": "running"})
            # Pass stop_event to runner so it can check during training
            runner_func(experiment, stop_event)
            if stop_event.is_set():
                experiment.status = "stopped"
                experiment.save()
                push_update(experiment, {"type": "status", "status": "stopped"})
            else:
                experiment.status = "completed"
                experiment.save()
                push_update(experiment, {"type": "status", "status": "completed"})
        except Exception as e:
            experiment.status = "failed"
            experiment.logs += f"\nError: {str(e)}"
            experiment.save()
            push_update(experiment, {"type": "status", "status": "failed"})
            logger.exception("Job failed")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()