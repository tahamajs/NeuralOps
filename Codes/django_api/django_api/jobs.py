import threading
import logging
from .models import Experiment
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer

logger = logging.getLogger(__name__)

def start_job(experiment: Experiment, runner_func):
    """Launch a background thread to run the experiment and update status."""
    def _run():
        try:
            experiment.status = "running"
            experiment.save()
            push_update(experiment, {"type": "log", "data": "Job started."})

            # The runner function receives the experiment object and can update logs
            runner_func(experiment)

            experiment.status = "completed"
            experiment.save()
            push_update(experiment, {"type": "log", "data": "Job completed."})
            push_update(experiment, {"type": "status", "status": "completed"})
        except Exception as e:
            experiment.status = "failed"
            experiment.logs += f"\nError: {str(e)}"
            experiment.save()
            push_update(experiment, {"type": "log", "data": f"Job failed: {str(e)}"})
            push_update(experiment, {"type": "status", "status": "failed"})
            logger.exception("Job failed")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()


def push_update(experiment: Experiment, message: dict):
    """Send a WebSocket message to the job's group."""
    try:
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            f"job_{experiment.id}",
            {
                "type": "send_message",
                "message": message,
            }
        )
    except Exception as e:
        logger.warning(f"Failed to push update: {e}")