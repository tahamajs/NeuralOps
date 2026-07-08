"""External monitoring callbacks: Wandb, TensorBoard, Terminal."""

import logging

from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class WandbCallback(TrainerCallback):
    """Logs metrics to Weights & Biases."""

    def __init__(self, project: str = "transformer-toolkit"):
        self.project = project
        self._wandb = None

    def on_train_begin(self, args, state, control, **kwargs):
        try:
            import wandb

            wandb.init(project=self.project)
            self._wandb = wandb
        except ImportError:
            logger.warning("wandb not installed")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self._wandb and logs:
            self._wandb.log(logs)


class TensorBoardCallback(TrainerCallback):
    """Logs to TensorBoard."""

    def __init__(self, log_dir: str = "./runs"):
        self.log_dir = log_dir
        self._writer = None

    def on_train_begin(self, args, state, control, **kwargs):
        try:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=self.log_dir)
        except ImportError:
            logger.warning("tensorboard not installed")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self._writer and logs:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self._writer.add_scalar(k, v, state.global_step)


class TerminalCallback(TrainerCallback):
    """Broadcasts training logs to a terminal group (WebSocket)."""

    def __init__(self, sender_func):
        self.send = sender_func

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.send(f"Step {state.global_step}: {logs}")


class WebSocketCallback(TrainerCallback):
    """Pushes training updates via a generic WebSocket sender."""

    def __init__(self, group_name: str, channel_layer=None):
        self.group_name = group_name
        self.channel_layer = channel_layer

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.channel_layer and logs:
            from asgiref.sync import async_to_sync

            async_to_sync(self.channel_layer.group_send)(
                self.group_name, {"type": "training_update", "data": logs}
            )
