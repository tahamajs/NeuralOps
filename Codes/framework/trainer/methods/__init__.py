# framework/trainer/methods/__init__.py
from .registry import register_method, get_method, available_methods
from . import fisher, gelora, llrd, lora, mixout, rdrop_r3f

__all__ = [
    "register_method",
    "get_method",
    "available_methods",
    "fisher",
    "gelora",
    "llrd",
    "lora",
    "mixout",
    "rdrop_r3f",
]