import threading, logging
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from .models import Experiment

logger = logging.getLogger(__name__)

def start_job(experiment: Experiment, runner_func):
    """Run the given runner as a background thread, pushing updates via WS."""
    def _run():
        try:
            experiment.status = "running"
            experiment.save()
            push_update(experiment, {"type": "status", "status": "running"})
            push_update(experiment, {"type": "log", "data": "Job started."})

            runner_func(experiment)

            experiment.status = "completed"
            experiment.save()
            push_update(experiment, {"type": "status", "status": "completed"})
            push_update(experiment, {"type": "log", "data": "Job completed successfully."})
        except Exception as e:
            experiment.status = "failed"
            experiment.logs += f"\nError: {str(e)}"
            experiment.save()
            push_update(experiment, {"type": "status", "status": "failed"})
            push_update(experiment, {"type": "log", "data": f"Job failed: {str(e)}"})
            logger.exception("Job execution failed")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

def push_update(experiment: Experiment, message: dict):
    """Send a message to the job's WebSocket group."""
    try:
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            f"job_{experiment.id}",
            {"type": "send_message", "message": message}
        )
    except Exception as e:
        logger.warning(f"WebSocket push failed: {e}")