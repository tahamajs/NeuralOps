"""Backward-compatibility shim. Moved to neuralops.core.utils.lora."""
import warnings
warnings.warn(
    "neuralops.utils.lora is deprecated. Use neuralops.core.utils.lora.",
    DeprecationWarning,
    stacklevel=2,
)
from neuralops.core.utils.lora import *  # noqa: F401,F403
from neuralops.core.utils.lora import LoRALinear  # noqa: F401
