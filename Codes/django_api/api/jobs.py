import threading
from .models import TrainingRun
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

_stop_events = {}

def stop_job(run_id: str):
    """Signal a running job to stop."""
    if run_id in _stop_events:
        _stop_events[run_id].set()
        return True
    return False

def cancel_job(run_id: str):
    return stop_job(run_id)

def start_job(run: TrainingRun, target_func):
    """Run a job in a background thread."""
    run_id = str(run.id)
    _stop_events[run_id] = threading.Event()

    def worker():
        try:
            target_func(run, _stop_events[run_id])
        except Exception as e:
            run.status = "failed"
            run.save()
            raise
        finally:
            _stop_events.pop(run_id, None)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

def push_update(run: TrainingRun, data: dict):
    """Send WebSocket update to job group."""
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        f"job_{run.id}",
        {"type": "job_update", "data": data}
    )

def stop_all_jobs():
    for event in _stop_events.values():
        event.set()