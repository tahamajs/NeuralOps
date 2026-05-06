import torch.optim as optim
from framework.core.registry import Registry

OPTIMIZER_REGISTRY = Registry("optimizer")

@OPTIMIZER_REGISTRY.register("adamw")
def create_adamw(params, lr=1e-3, weight_decay=0.01, **kwargs):
    return optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)

@OPTIMIZER_REGISTRY.register("sgd")
def create_sgd(params, lr=0.01, momentum=0.9, weight_decay=0.0, **kwargs):
    return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, **kwargs)

@OPTIMIZER_REGISTRY.register("adam")
def create_adam(params, lr=1e-3, weight_decay=0.0, **kwargs):
    return optim.Adam(params, lr=lr, weight_decay=weight_decay, **kwargs)