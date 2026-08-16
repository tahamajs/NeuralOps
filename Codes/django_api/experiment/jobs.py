import threading

from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer

from .models import TrainingRun

_stop_events = {}


def stop_job(run_id: str):
    if run_id in _stop_events:
        _stop_events[run_id].set()
        return True
    return False


def start_job(run: TrainingRun, target_func):
    run_id = str(run.id)
    _stop_events[run_id] = threading.Event()

    def worker():
        from django.db import connection
        connection.close()
        try:
            target_func(run, _stop_events[run_id])
        except Exception:
            try:
                connection.close()
                run.status = "failed"
                run.save()
            except Exception:
                pass
            raise
        finally:
            _stop_events.pop(run_id, None)
            try:
                connection.close()
            except Exception:
                pass

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()


def push_update(run: TrainingRun, data: dict):
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        f"job_{run.id}", {"type": "job_update", "data": data}
    )


def stop_all_jobs():
    for event in _stop_events.values():
        event.set()
