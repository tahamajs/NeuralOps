# experiments/runners.py
import os
import json
import traceback
import psutil
from pathlib import Path
from django.utils import timezone
from django.conf import settings
from .models import TrainingRun, TrainingSnapshot
from .jobs import push_update

# ----------------------------------------------------------------------
# Lazy import of real framework functions – avoids any startup issues
# ----------------------------------------------------------------------
def _get_framework():
    """Return the three main functions from the framework."""
    from framework.experiment import run_training as fw_train
    from framework.experiment import run_evaluation as fw_eval
    from framework.experiment import run_compression as fw_compress
    return fw_train, fw_eval, fw_compress


# ----------------------------------------------------------------------
# Helpers (unchanged except minor improvements)
# ----------------------------------------------------------------------
def _log(run: TrainingRun, msg: str):
    run.logs = (run.logs or "") + msg + "\n"
    run.save(update_fields=['logs'])
    push_update(run, {"type": "log", "data": msg})
    from channels.layers import get_channel_layer
    from asgiref.sync import async_to_sync
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)("terminal", {"type": "terminal_message", "data": msg})


def _update_status(run: TrainingRun, status: str):
    run.status = status
    now = timezone.now()
    if status == "running" and not run.started_at:
        run.started_at = now
    elif status in ("completed", "failed", "stopped"):
        run.completed_at = now
    run.save(update_fields=['status', 'started_at', 'completed_at'])
    push_update(run, {"type": "status", "status": status})


# ----------------------------------------------------------------------
# Training Runner
# ----------------------------------------------------------------------
def run_training(run: TrainingRun, stop_event=None):
    """
    Execute a training job using the framework.
    `stop_event` is an optional threading.Event to cancel the run.
    """
    _update_status(run, "running")
    config = run.config.copy()

    # Prepare output directory and add to config for the framework
    output_dir = Path(settings.MEDIA_ROOT) / str(run.id) / "model"
    output_dir.mkdir(parents=True, exist_ok=True)
    config["output_dir"] = str(output_dir)

    # Progress callback – updates the DB and pushes to WebSocket
    def progress_callback(step, total, pct):
        run.current_step = step
        run.total_steps = total
        run.estimated_remaining_seconds = (total - step) * 0.1   # crude estimate
        run.cpu_usage = psutil.cpu_percent()
        run.ram_usage_mb = psutil.virtual_memory().used / 1024**2
        run.save(update_fields=[
            'current_step', 'total_steps', 'estimated_remaining_seconds',
            'cpu_usage', 'ram_usage_mb'
        ])
        push_update(run, {
            "type": "progress",
            "progress": pct,
            "current_step": step,
            "total_steps": total
        })

    # Logging/metrics callback – stores snapshots and sends live updates
    def log_callback(step, logs):
        metrics = {}
        if "loss" in logs:
            metrics["loss"] = logs["loss"]
        if "eval_loss" in logs:
            metrics["eval_loss"] = logs["eval_loss"]
        if "eval_accuracy" in logs:
            metrics["eval_accuracy"] = logs["eval_accuracy"]
        if metrics:
            push_update(run, {"type": "metrics", "data": metrics})

        # Create a training snapshot record
        TrainingSnapshot.objects.create(
            run=run,
            step=step,
            train_loss=logs.get("loss"),
            eval_loss=logs.get("eval_loss"),
            eval_accuracy=logs.get("eval_accuracy"),
            learning_rate=logs.get("learning_rate"),
            cpu_percent=psutil.cpu_percent(),
            ram_mb=psutil.virtual_memory().used / 1024**2
        )
        _log(run, f"Step {step}: {logs}")

    # Final callback – saves final metrics and training history
    def final_callback(final_metrics, history):
        run.final_metrics = final_metrics
        run.training_history = history
        run.save(update_fields=['final_metrics', 'training_history'])
        push_update(run, {"type": "final", "data": final_metrics})

    try:
        # Import the real framework function
        fw_train, _, _ = _get_framework()

        # Run the actual training
        fw_train(
            config,
            stop_event=stop_event,
            progress_callback=progress_callback,
            log_callback=log_callback,
            final_callback=final_callback
        )

        if stop_event and stop_event.is_set():
            _update_status(run, "stopped")
            _log(run, "Training stopped by user.")
        else:
            _update_status(run, "completed")
            _log(run, "Training completed successfully.")

    except Exception as e:
        _log(run, f"Training failed: {str(e)}\n{traceback.format_exc()}")
        _update_status(run, "failed")


# ----------------------------------------------------------------------
# Evaluation Runner
# ----------------------------------------------------------------------
def run_evaluation(run: TrainingRun, stop_event=None):
    """
    Execute an evaluation job using the framework.
    The framework's run_evaluation expects a config dict and a model path.
    """
    _update_status(run, "running")
    model_path = Path(settings.MEDIA_ROOT) / str(run.id) / "model"
    if not model_path.exists():
        _log(run, "No trained model found, using base model")
        model_path = run.config.get("model_name", "distilbert-base-uncased")

    try:
        _, fw_eval, _ = _get_framework()
        metrics = fw_eval(run.config, str(model_path))
        run.final_metrics = metrics
        run.save(update_fields=['final_metrics'])
        _log(run, f"Evaluation results: {json.dumps(metrics)}")
        _update_status(run, "completed")
    except Exception as e:
        _log(run, f"Evaluation failed: {e}")
        _update_status(run, "failed")


# ----------------------------------------------------------------------
# Compression Runner
# ----------------------------------------------------------------------
def run_compression(run: TrainingRun, stop_event=None):
    """
    Execute a compression job using the framework.
    The framework's run_compression expects a config dict and an output directory.
    """
    _update_status(run, "running")
    output_dir = Path(settings.MEDIA_ROOT) / str(run.id) / "model"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        _, _, fw_compress = _get_framework()
        stats = fw_compress(run.config, str(output_dir))
        run.final_metrics = stats
        run.save(update_fields=['final_metrics'])
        _log(run, f"Compression stats: {json.dumps(stats)}")
        _update_status(run, "completed")
    except Exception as e:
        _log(run, f"Compression failed: {e}")
        _update_status(run, "failed")