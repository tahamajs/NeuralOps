import torch.optim.lr_scheduler as lr_scheduler
from framework.core.registry import Registry

SCHEDULER_REGISTRY = Registry("scheduler")

@SCHEDULER_REGISTRY.register("linear")
def create_linear(optimizer, num_warmup_steps=0, num_training_steps=10000, **kwargs):
    return lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_training_steps)

@SCHEDULER_REGISTRY.register("cosine")
def create_cosine(optimizer, T_max=10000, **kwargs):
    return lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

@SCHEDULER_REGISTRY.register("reduce_on_plateau")
def create_rop(optimizer, factor=0.1, patience=10, **kwargs):
    return lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience)

@SCHEDULER_REGISTRY.register("one_cycle")
def create_onecycle(optimizer, max_lr=0.01, steps_per_epoch=1000, epochs=10, **kwargs):
    return lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=epochs)