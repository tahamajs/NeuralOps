# experiment/tasks.py
import os
import json
import traceback
import threading
from pathlib import Path
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer

from django.conf import settings
from .models import Experiment
from framework.experiment import run_training, run_evaluation, run_compression

# -------------------------------------------------------------------------
# WebSocket broadcaster (thread‑safe)
# -------------------------------------------------------------------------
class WSMessageQueue:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loop = None
        return cls._instance

    @staticmethod
    def send(experiment_id, data):
        channel_layer = get_channel_layer()
        # Use async_to_sync from a dedicated event loop
        try:
            async_to_sync(channel_layer.group_send)(
                f"job_{experiment_id}",
                {"type": "job_update", "data": data}
            )
        except Exception as e:
            print(f"WS send error: {e}")

def broadcast_terminal(msg):
    """Send log message to the terminal group."""
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        "terminal",
        {"type": "terminal_message", "data": msg}
    )

# -------------------------------------------------------------------------
# Training runner (background thread)
# -------------------------------------------------------------------------
def run_training_async(experiment_id: int, stop_event: threading.Event):
    exp = Experiment.objects.get(id=experiment_id)
    config = exp.config

    def log(msg):
        exp.logs = (exp.logs or "") + msg + "\n"
        exp.save(update_fields=["logs"])
        WSMessageQueue.send(experiment_id, {"type": "log", "data": msg})
        broadcast_terminal(msg)

    def update_status(status):
        exp.status = status
        exp.save(update_fields=["status"])
        WSMessageQueue.send(experiment_id, {"type": "status", "status": status})

    def progress_callback(current, total, pct):
        exp.current_step = current
        exp.total_steps = total
        exp.estimated_remaining_seconds = (total - current) * 0.1  # crude estimate
        exp.cpu_usage = psutil.cpu_percent()
        exp.ram_usage_mb = psutil.virtual_memory().used / 1024**2
        exp.save(update_fields=["current_step", "total_steps", "estimated_remaining_seconds", "cpu_usage", "ram_usage_mb"])
        WSMessageQueue.send(experiment_id, {
            "type": "progress",
            "progress": pct,
            "current_step": current,
            "total_steps": total,
        })

    def log_callback(step, logs):
        # Send metrics (throttled: only send if step % 5 == 0 or important)
        metrics = {}
        if "loss" in logs:
            metrics["loss"] = logs["loss"]
        if "eval_loss" in logs:
            metrics["eval_loss"] = logs["eval_loss"]
        if "eval_accuracy" in logs:
            metrics["eval_accuracy"] = logs["eval_accuracy"]
        if metrics:
            WSMessageQueue.send(experiment_id, {"type": "metrics", "data": metrics})
        # Always log the raw message for terminal
        log(f"Step {step}: {logs}")

    def final_callback(final_metrics, history):
        exp.config["final_metrics"] = final_metrics
        exp.save(update_fields=["config"])
        WSMessageQueue.send(experiment_id, {"type": "final", "data": final_metrics})
        log(f"Training finished. Final: {json.dumps(final_metrics)}")
        save_experiment_artifacts(exp, history, final_metrics)

    try:
        update_status("running")
        config["output_dir"] = str(Path(settings.MEDIA_ROOT) / str(experiment_id) / "model")
        result = run_training(
            config,
            stop_event=stop_event,
            progress_callback=progress_callback,
            log_callback=log_callback,
            final_callback=final_callback,
        )
        if stop_event and stop_event.is_set():
            update_status("stopped")
            log("Training cancelled.")
        else:
            update_status("completed")
    except Exception as e:
        error_msg = traceback.format_exc()
        log(f"ERROR: {str(e)}\n{error_msg}")
        update_status("failed")
    finally:
        # release any resources
        pass

def save_experiment_artifacts(exp, history, final_metrics):
    # Create timestamped directory and copy model, plots, etc.
    # (similar to original save_experiment_results but using exp.id)
    from datetime import datetime
    import shutil
    import pandas as pd
    import matplotlib.pyplot as plt

    output_dir = Path(settings.MEDIA_ROOT) / str(exp.id)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save history as JSON
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save final metrics
    with open(output_dir / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    # Loss plot
    if history.get("steps"):
        plt.figure(figsize=(10,6))
        plt.plot(history["steps"], history["losses"], label="Train Loss")
        if history.get("eval_steps"):
            plt.plot(history["eval_steps"], history["eval_losses"], "ro-", label="Eval Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / "loss_curve.png", dpi=150)
        plt.close()

    # Copy model if exists
    model_dir = Path(settings.MEDIA_ROOT) / str(exp.id) / "model"
    if model_dir.exists():
        shutil.copytree(model_dir, output_dir / "model", dirs_exist_ok=True)

    # Log completion
    print(f"Artifacts saved to {output_dir}")

# -------------------------------------------------------------------------
# AI Assistant (simple example)
# -------------------------------------------------------------------------
class AIAssistant:
    def __init__(self, llm_function):
        self.llm = llm_function   # expects a prompt and returns a command

    def monitor(self, exp_id, metrics_history):
        # Example: if loss plateau, suggest early stopping
        last_losses = [m.get("loss") for m in metrics_history[-5:] if "loss" in m]
        if len(last_losses) >= 5 and all(abs(last_losses[i] - last_losses[i-1]) < 0.01 for i in range(1,5)):
            # Send command to stop training
            WSMessageQueue.send(exp_id, {"type": "ai_suggestion", "suggestion": "early_stop", "reason": "Loss plateau"})
            # Optionally, auto‑stop
            # stop_event.set()