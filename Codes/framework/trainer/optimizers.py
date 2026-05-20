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

@OPTIMIZER_REGISTRY.register("adabelief")
def create_adabelief(params, lr=1e-3, weight_decay=0.0, **kwargs):
    from adabelief_pytorch import AdaBelief
    return AdaBelief(params, lr=lr, weight_decay=weight_decay, **kwargs)

@OPTIMIZER_REGISTRY.register("ranger")
def create_ranger(params, lr=1e-3, weight_decay=0.0, **kwargs):
    from ranger import Ranger
    return Ranger(params, lr=lr, weight_decay=weight_decay, **kwargs)

@OPTIMIZER_REGISTRY.register("lamb")
def create_lamb(params, lr=1e-3, weight_decay=0.01, **kwargs):
    from torch_optimizer import Lamb
    return Lamb(params, lr=lr, weight_decay=weight_decay, **kwargs)

@OPTIMIZER_REGISTRY.register("lars")
def create_lars(params, lr=1e-3, weight_decay=0.0, **kwargs):
    from torch_optimizer import Lars
    return Lars(params, lr=lr, weight_decay=weight_decay, **kwargs)