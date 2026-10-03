"""
Background training task (Django integration) with WebSocket broadcasting
and AI assistant.
"""

import json
import threading
import traceback
from pathlib import Path

import psutil
from django.conf import settings

try:
    from asgiref.sync import async_to_sync
    from channels.layers import get_channel_layer

    HAS_CHANNELS = True
except ImportError:
    async_to_sync = None
    get_channel_layer = None
    HAS_CHANNELS = False

from framework.experiment_utils.orchestrator import MegaOrchestrator


# -------------------------------------------------------------------------
# WebSocket broadcaster
# -------------------------------------------------------------------------
def broadcast_terminal(msg: str) -> None:
    """Send a log message to the terminal WebSocket group."""
    if not HAS_CHANNELS:
        return
    try:
        channel_layer = get_channel_layer()
        if channel_layer and async_to_sync:
            async_to_sync(channel_layer.group_send)(
                "terminal", {"type": "terminal_message", "data": msg}
            )
    except Exception as e:
        print(f"WS broadcast error: {e}")


class WSMessageQueue:
    """Thread‑safe WebSocket message sender."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
    def send(experiment_id: str, data: dict) -> None:
        """Send."""
        try:
            channel_layer = get_channel_layer()
            async_to_sync(channel_layer.group_send)(
                f"job_{experiment_id}", {"type": "job_update", "data": data}
            )
        except Exception as e:
            print(f"WS send error: {e}")


# -------------------------------------------------------------------------
# Background training runner
# -------------------------------------------------------------------------
def run_training_async(experiment, stop_event: threading.Event) -> None:
    """
    Main entry point called by Django in a background thread.

    Args:
        experiment: Django model instance with fields:
            id, config (JSON), status, logs, current_step, total_steps, etc.
        stop_event: threading.Event to signal cancellation.
    """
    config = experiment.config
    run_id = str(experiment.id)
    base_dir = Path(settings.MEDIA_ROOT) / "experiments"

    def log(msg: str) -> None:
        """Log."""
        experiment.logs = (experiment.logs or "") + msg + "\n"
        experiment.save(update_fields=["logs"])
        WSMessageQueue.send(run_id, {"type": "log", "data": msg})
        broadcast_terminal(msg)

    def update_status(status: str) -> None:
        """Update status."""
        experiment.status = status
        experiment.save(update_fields=["status"])
        WSMessageQueue.send(run_id, {"type": "status", "status": status})

    def progress_callback(current: int, total: int, pct: float) -> None:
        """Progress callback."""
        experiment.current_step = current
        experiment.total_steps = total
        experiment.estimated_remaining_seconds = (total - current) * 0.1  # crude
        experiment.cpu_usage = psutil.cpu_percent()
        experiment.ram_usage_mb = psutil.virtual_memory().used / 1024**2
        experiment.save(
            update_fields=[
                "current_step",
                "total_steps",
                "estimated_remaining_seconds",
                "cpu_usage",
                "ram_usage_mb",
            ]
        )
        WSMessageQueue.send(
            run_id,
            {
                "type": "progress",
                "progress": pct,
                "current_step": current,
                "total_steps": total,
            },
        )

    def log_callback(msg: str):
        # Called by the orchestrator for log messages
        """Log callback."""
        log(msg)

    def final_callback(metrics: dict) -> None:
        """Final callback."""
        experiment.config["final_metrics"] = metrics
        experiment.save(update_fields=["config"])
        WSMessageQueue.send(run_id, {"type": "final", "data": metrics})
        log(f"Training finished. Final metrics: {json.dumps(metrics)}")
        save_artifacts(experiment, metrics)

    try:
        update_status("running")
        orchestrator = MegaOrchestrator(
            config=config,
            run_id=run_id,
            base_dir=base_dir,
            progress_callback=progress_callback,
            log_callback=log_callback,
            final_callback=final_callback,
            stop_event=stop_event,
        )
        orchestrator.run()

        if stop_event.is_set():
            update_status("stopped")
            log("Training cancelled by user.")
        else:
            update_status("completed")

    except Exception as e:
        error_msg = traceback.format_exc()
        log(f"ERROR: {str(e)}\n{error_msg}")
        update_status("failed")


def save_artifacts(experiment, final_metrics: dict) -> None:
    """Save model, plots, history, etc. to the experiment's media directory."""
    output_dir = Path(settings.MEDIA_ROOT) / "experiments" / str(experiment.id)
    output_dir.mkdir(parents=True, exist_ok=True)

    # History is saved by the saver inside orchestrator, we just add a few extra things
    import matplotlib.pyplot as plt

    # Loss plot from training history (if available)
    history = experiment.config.get("history", [])
    if history:
        steps = [h.get("step", i) for i, h in enumerate(history) if "loss" in h]
        losses = [h["loss"] for h in history if "loss" in h]
        if steps:
            plt.figure(figsize=(10, 6))
            plt.plot(steps, losses, label="Train Loss")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig(output_dir / "loss_curve.png", dpi=150)
            plt.close()

    # Final metrics JSON
    with open(output_dir / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    # Copy model if present
    model_src = (
        Path(settings.MEDIA_ROOT)
        / "experiments"
        / str(experiment.id)
        / "artifacts"
        / "model"
    )
    if model_src.exists():
        import shutil

        shutil.copytree(model_src, output_dir / "model", dirs_exist_ok=True)


# -------------------------------------------------------------------------
# AI Assistant (simple example)
# -------------------------------------------------------------------------
class AIAssistant:
    """
    Monitors training metrics and suggests actions (e.g., early stop).
    Uses a callable LLM function for decision making.
    """

    def __init__(self, llm_function) -> None:
        self.llm = llm_function

    def monitor(self, exp_id: str, metrics_history: list) -> None:
        """Monitor."""
        last_losses = [m.get("loss") for m in metrics_history[-5:] if "loss" in m]
        if len(last_losses) >= 5 and all(
            abs(last_losses[i] - last_losses[i - 1]) < 0.01 for i in range(1, 5)
        ):
            suggestion = "early_stop"
            reason = "Loss plateau detected"
            WSMessageQueue.send(
                exp_id,
                {"type": "ai_suggestion", "suggestion": suggestion, "reason": reason},
            )
            # Optionally trigger early stop
            # stop_event.set()
