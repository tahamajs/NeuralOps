import threading
import logging
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from django.utils import timezone
from .models import TrainingRun

logger = logging.getLogger(__name__)
_running_jobs = {}   # run_id -> threading.Event

def stop_job(run_id):
    event = _running_jobs.pop(str(run_id), None)
    if event:
        event.set()

def cancel_job(run_id):
    """Cancel a running job by setting its stop event."""
    stop_job(run_id)

def start_job(run: TrainingRun, runner_func):
    """
    Start a background job for a training run.
    :param run: TrainingRun instance
    :param runner_func: function that takes (run, stop_event)
    """
    stop_event = threading.Event()
    _running_jobs[str(run.id)] = stop_event

    def _run():
        try:
            run.status = "running"
            run.started_at = timezone.now()
            run.save()
            push_update(run, {"type": "status", "status": "running"})
            runner_func(run, stop_event)

            if stop_event.is_set():
                run.status = "stopped"
                run.save()
                push_update(run, {"type": "status", "status": "stopped"})
            else:
                run.status = "completed"
                run.completed_at = timezone.now()
                run.save()
                push_update(run, {"type": "status", "status": "completed"})
        except Exception as e:
            run.status = "failed"
            run.logs = (run.logs or "") + f"\nError: {str(e)}"
            run.save()
            push_update(run, {"type": "status", "status": "failed"})
            logger.exception(f"Job for run {run.id} failed")
        finally:
            _running_jobs.pop(str(run.id), None)

    threading.Thread(target=_run, daemon=True).start()

def push_update(run: TrainingRun, message: dict):
    try:
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            f"job_{run.id}", {"type": "send_message", "message": message}
        )
    except Exception as e:
        logger.warning(f"WebSocket push failed for run {run.id}: {e}")