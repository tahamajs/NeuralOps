# framework/callbacks/custom.py
import torch
import copy
from transformers import TrainerCallback
import logging

logger = logging.getLogger(__name__)

class EMACallback(TrainerCallback):
    def __init__(self, decay=0.999, **kwargs):
        self.decay = decay
        self.shadow = {}

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.shadow = {name: param.data.clone() for name, param in model.named_parameters() if param.requires_grad}

    def on_step_end(self, args, state, control, model=None, **kwargs):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def on_train_end(self, args, state, control, model=None, **kwargs):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

class SWACallback(TrainerCallback):
    def __init__(self, start_epoch=5, swa_lr=1e-4, **kwargs):
        self.start_epoch = start_epoch
        self.swa_lr = swa_lr
        self.swa_model = None

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if state.epoch >= self.start_epoch:
            if self.swa_model is None:
                self.swa_model = copy.deepcopy(model)
            else:
                for swa_p, p in zip(self.swa_model.parameters(), model.parameters()):
                    swa_p.data = (swa_p.data * (state.epoch - self.start_epoch) + p.data) / (state.epoch - self.start_epoch + 1)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if self.swa_model is not None:
            for p, swa_p in zip(model.parameters(), self.swa_model.parameters()):
                p.data.copy_(swa_p.data)

class GradientClippingCallback(TrainerCallback):
    def __init__(self, max_norm=1.0, **kwargs):
        self.max_norm = max_norm

    def on_step_end(self, args, state, control, model=None, **kwargs):
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)

class FreezeLayersCallback(TrainerCallback):
    def __init__(self, freeze_until_epoch=3, layers_to_freeze=None, **kwargs):
        self.freeze_until_epoch = freeze_until_epoch
        self.layers_to_freeze = layers_to_freeze

    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        if state.epoch < self.freeze_until_epoch:
            for name, param in model.named_parameters():
                if any(l in name for l in self.layers_to_freeze):
                    param.requires_grad = False

class UnfreezeScheduleCallback(TrainerCallback):
    def __init__(self, unfreeze_epochs=None, **kwargs):
        self.unfreeze_epochs = unfreeze_epochs or [2, 4, 6]

    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        if state.epoch in self.unfreeze_epochs:
            for param in model.parameters():
                param.requires_grad = True

class NanCheckerCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs and (torch.isnan(logs['loss']) or torch.isinf(logs['loss'])):
            logger.error("NaN or Inf loss detected, stopping training")
            control.should_training_stop = True

class ActivationCheckpointingCallback(TrainerCallback):
    def __init__(self, modules_to_checkpoint=None, **kwargs):
        self.modules_to_checkpoint = modules_to_checkpoint

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.modules_to_checkpoint:
            for name, module in model.named_modules():
                if any(m in name for m in self.modules_to_checkpoint):
                    module.gradient_checkpointing_enable()