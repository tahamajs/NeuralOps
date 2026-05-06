"""Resource monitoring callbacks: memory, time, GPU."""
import time
import logging
from transformers import TrainerCallback

logger = logging.getLogger(__name__)

class MemoryCallback(TrainerCallback):
    """Logs CPU/GPU memory usage."""
    def __init__(self, log_every_steps: int = 100):
        self.log_every_steps = log_every_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_every_steps == 0:
            import psutil
            mem = psutil.virtual_memory()
            logger.info(f"RAM: {mem.percent}% used ({mem.used // (1024**2)} MB)")


class TimeCallback(TrainerCallback):
    """Tracks elapsed time and estimates remaining time."""
    def __init__(self):
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0:
            elapsed = time.time() - self.start_time
            steps_done = state.global_step
            total = state.max_steps
            if steps_done > 0:
                eta = (elapsed / steps_done) * (total - steps_done)
                logger.info(f"Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")


class GPUStatsCallback(TrainerCallback):
    """Logs GPU utilisation and memory."""
    def on_step_end(self, args, state, control, **kwargs):
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                logger.info(f"GPU memory: {allocated:.0f}MB allocated, {reserved:.0f}MB reserved")
        except ImportError:
            pass