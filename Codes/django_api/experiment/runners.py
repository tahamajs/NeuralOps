# experiment/runners.py
import json
import traceback
from functools import partial
from pathlib import Path

import psutil
from django.conf import settings
from django.utils import timezone

from .jobs import push_update
from .models import TrainingRun, TrainingSnapshot

# ----------------------------------------------------------------------
# Helpers: logging, status updates, and channel notifications
# ----------------------------------------------------------------------
def _log(run: TrainingRun, msg: str):
    """Append a log line to the run and send it via channels."""
    run.logs = (run.logs or "") + msg + "\n"
    run.save(update_fields=['logs'])
    push_update(run, {"type": "log", "data": msg})
    try:
        from channels.layers import get_channel_layer
        from asgiref.sync import async_to_sync
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            "terminal", {"type": "terminal_message", "data": msg}
        )
    except Exception:
        pass  # channels not available


def _update_status(run: TrainingRun, status: str):
    """Update run status and timestamps, then broadcast."""
    run.status = status
    now = timezone.now()
    if status == "running" and not run.started_at:
        run.started_at = now
    elif status in ("completed", "failed", "stopped"):
        run.completed_at = now
    run.save(update_fields=['status', 'started_at', 'completed_at'])
    push_update(run, {"type": "status", "status": status})


# ----------------------------------------------------------------------
# Training (using MegaOrchestrator)
# ----------------------------------------------------------------------
def run_training(run: TrainingRun, stop_event=None):
    """Execute training using MegaOrchestrator or fallback."""
    _update_status(run, "running")
    _log(run, "Starting training...")

    # Try to import the orchestrator
    try:
        from framework.experiment_utils.mega_orchestrator import MegaOrchestrator
    except ImportError:
        _log(run, "MegaOrchestrator not found – falling back to legacy runner")
        _run_training_legacy(run, stop_event)
        return

    base_dir = Path(settings.MEDIA_ROOT) / str(run.id)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Define callbacks that close over the run
    def progress_callback(step: int, total: int, pct: float):
        run.current_step = step
        run.total_steps = total
        run.estimated_remaining_seconds = (total - step) * 0.1 if total > step else 0
        run.cpu_usage = psutil.cpu_percent()
        run.ram_usage_mb = psutil.virtual_memory().used / (1024 ** 2)
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

    def log_callback(step: int, logs: dict):
        # Save snapshot
        TrainingSnapshot.objects.create(
            run=run,
            step=step,
            train_loss=logs.get("loss"),
            eval_loss=logs.get("eval_loss"),
            eval_accuracy=logs.get("eval_accuracy"),
            learning_rate=logs.get("learning_rate"),
            cpu_percent=psutil.cpu_percent(),
            ram_mb=psutil.virtual_memory().used / (1024 ** 2)
        )
        # Broadcast metrics
        metrics = {
            k: v for k, v in logs.items()
            if k in ("loss", "eval_loss", "eval_accuracy", "learning_rate")
        }
        if metrics:
            push_update(run, {"type": "metrics", "data": metrics})
        _log(run, f"Step {step}: {logs}")

    def final_callback(final_metrics: dict, history: dict):
        run.final_metrics = final_metrics
        run.training_history = history
        run.save(update_fields=['final_metrics', 'training_history'])
        push_update(run, {"type": "final", "data": final_metrics})

        # Optional persistent storage
        try:
            from framework.utils.results_saver import ResultsSaver
            saver = ResultsSaver(run.id, Path(settings.MEDIA_ROOT), run.config.get("save_config", {}))
            saver.save_config(run.config)
            saver.save_history(history)
            saver.save_final_metrics(final_metrics)
            saver.generate_plots(history)
            # Model saving would require the actual model object – not available here
            # Usually the orchestrator handles it; this is just a placeholder.
            saver.handle_storage_backend()
            _log(run, "Results saved to disk.")
        except ImportError:
            _log(run, "ResultsSaver not available – skipping persistent storage.")

    orchestrator = MegaOrchestrator(
        config=run.config,
        run_id=str(run.id),
        base_dir=base_dir,
        progress_callback=progress_callback,
        log_callback=log_callback,
        final_callback=final_callback,
        stop_event=stop_event,
    )

    try:
        orchestrator.run()
        if stop_event and stop_event.is_set():
            _update_status(run, "stopped")
            _log(run, "Training stopped by user.")
        else:
            _update_status(run, "completed")
            _log(run, "Training completed successfully.")
    except Exception as e:
        _log(run, f"Training failed: {str(e)}\n{traceback.format_exc()}")
        _update_status(run, "failed")


def _run_training_legacy(run: TrainingRun, stop_event=None):
    """Fallback that uses the older direct framework functions (if available)."""
    _update_status(run, "running")
    try:
        from framework.experiment_utils import run_training as fw_train
        fw_train(
            run.config,
            stop_event=stop_event,
            progress_callback=partial(_legacy_progress, run),
            log_callback=partial(_legacy_log, run),
            final_callback=partial(_legacy_final, run),
        )
        if stop_event and stop_event.is_set():
            _update_status(run, "stopped")
        else:
            _update_status(run, "completed")
    except Exception as e:
        _log(run, f"Legacy training failed: {e}")
        _update_status(run, "failed")


def _legacy_progress(run: TrainingRun, step: int, total: int, pct: float):
    run.current_step = step
    run.total_steps = total
    run.save(update_fields=['current_step', 'total_steps'])
    push_update(run, {"type": "progress", "progress": pct, "step": step, "total": total})


def _legacy_log(run: TrainingRun, step: int, logs: dict):
    TrainingSnapshot.objects.create(
        run=run,
        step=step,
        train_loss=logs.get("loss"),
        eval_loss=logs.get("eval_loss"),
        eval_accuracy=logs.get("eval_accuracy"),
        learning_rate=logs.get("learning_rate"),
        cpu_percent=psutil.cpu_percent(),
        ram_mb=psutil.virtual_memory().used / (1024 ** 2)
    )
    push_update(run, {"type": "metrics", "data": logs})
    _log(run, f"Step {step}: {logs}")


def _legacy_final(run: TrainingRun, final_metrics: dict, history: dict):
    run.final_metrics = final_metrics
    run.training_history = history
    run.save(update_fields=['final_metrics', 'training_history'])
    push_update(run, {"type": "final", "data": final_metrics})


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def run_evaluation(run: TrainingRun, stop_event=None):
    """Evaluate a model using the framework."""
    _update_status(run, "running")
    _log(run, "Starting evaluation...")

    model_path = Path(settings.MEDIA_ROOT) / str(run.id) / "model"
    if not model_path.exists():
        _log(run, "No trained model found, using base model from config")
        model_path = run.config.get("model_name", "distilbert-base-uncased")

    try:
        _, fw_eval, _ = _get_framework()
        metrics = fw_eval(run.config, str(model_path))
        run.final_metrics = metrics
        run.save(update_fields=['final_metrics'])
        _log(run, f"Evaluation results: {json.dumps(metrics)}")
        _update_status(run, "completed")
    except Exception as e:
        _log(run, f"Evaluation failed: {str(e)}\n{traceback.format_exc()}")
        _update_status(run, "failed")


# ----------------------------------------------------------------------
# Compression
# ----------------------------------------------------------------------
def run_compression(run: TrainingRun, stop_event=None):
    """Compress a model using the framework."""
    _update_status(run, "running")
    _log(run, "Starting compression...")

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
        _log(run, f"Compression failed: {str(e)}\n{traceback.format_exc()}")
        _update_status(run, "failed")


# ----------------------------------------------------------------------
# Internal lazy loader for framework functions
# ----------------------------------------------------------------------
def _get_framework():
    """Lazy import of framework functions to avoid early import issues."""
    from framework.experiment_utils import run_training as fw_train
    from framework.experiment_utils import run_evaluation as fw_eval
    from framework.experiment_utils import run_compression as fw_compress
    return fw_train, fw_eval, fw_compress