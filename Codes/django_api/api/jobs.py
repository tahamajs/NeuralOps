import threading, logging
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from .models import Experiment

logger = logging.getLogger(__name__)
_running_jobs = {}   # job_id -> threading.Event

def stop_job(job_id):
    event = _running_jobs.pop(str(job_id), None)
    if event:
        event.set()
def cancel_job(job_id):
    """Cancel a running job by setting its stop event."""
    stop_job(job_id)
def start_job(experiment: Experiment, runner_func):
    stop_event = threading.Event()
    _running_jobs[str(experiment.id)] = stop_event

    def _run():
        try:
            experiment.status = "running"
            experiment.save()
            push_update(experiment, {"type": "status", "status": "running"})
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
        finally:
            _running_jobs.pop(str(experiment.id), None)

    threading.Thread(target=_run, daemon=True).start()

def push_update(experiment: Experiment, message: dict):
    try:
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            f"job_{experiment.id}", {"type": "send_message", "message": message}
        )
    except Exception as e:
        logger.warning(f"WebSocket push failed: {e}")