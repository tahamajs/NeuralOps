import threading
from .models import Experiment

_job_threads = {}
_stop_events = {}

def start_job(experiment: Experiment, target_func):
    """Run a job in a background thread."""
    experiment_id = str(experiment.id)
    print(f"🚀 start_job called for {experiment_id}")   # debug
    
    # Create a stop event for this job
    _stop_events[experiment_id] = threading.Event()
    
    def worker():
        print(f"🧵 Worker thread started for {experiment_id}")   # debug
        try:
            target_func(experiment, _stop_events[experiment_id])
        except Exception as e:
            print(f"❌ Job {experiment_id} failed: {e}")
            experiment.status = "failed"
            experiment.save()
        finally:
            _job_threads.pop(experiment_id, None)
            _stop_events.pop(experiment_id, None)
            print(f"🏁 Worker thread finished for {experiment_id}")
    
    thread = threading.Thread(target=worker, daemon=True)
    _job_threads[experiment_id] = thread
    thread.start()
    print(f"✅ Thread started for {experiment_id}")

def cancel_job(experiment_id):
    """Signal a job to stop."""
    if experiment_id in _stop_events:
        _stop_events[experiment_id].set()
        return True
    return False