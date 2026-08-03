from .awq import AWQQuantizer
from .base import QUANTIZER_REGISTRY, BaseQuantizer
from .bitsandbytes import BitsAndBytesQuantizer
from .fp8 import FP8Quantizer
from .gptq import GPTQQuantizer
from .torchao import TorchAOQuantizer


def list_quantizers() -> Any:
    """List quantizers."""
    return list(QUANTIZER_REGISTRY.keys())
