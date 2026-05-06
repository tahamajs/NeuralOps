from transformers import TrainerCallback
from typing import Optional, Callable

class DjangoCallback(TrainerCallback):
    def __init__(self, progress_cb=None, log_cb=None, final_cb=None, stop_event=None):
        self.progress_cb = progress_cb
        self.log_cb = log_cb
        self.final_cb = final_cb
        self.stop_event = stop_event
        self._total_steps = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._total_steps = state.max_steps

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.log_cb:
            self.log_cb(state.global_step, logs or {})

    def on_step_end(self, args, state, control, **kwargs):
        if self.stop_event and self.stop_event.is_set():
            control.should_training_stop = True
        if self.progress_cb and self._total_steps:
            pct = (state.global_step / self._total_steps) * 100
            self.progress_cb(state.global_step, self._total_steps, pct)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.log_cb and metrics:
            self.log_cb(state.global_step, metrics)

    def on_train_end(self, args, state, control, **kwargs):
        if self.final_cb:
            self.final_cb({}, getattr(state, 'log_history', []))