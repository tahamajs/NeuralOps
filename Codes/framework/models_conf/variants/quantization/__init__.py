from .base import BaseQuantizer, QUANTIZER_REGISTRY
from .bitsandbytes import BitsAndBytesQuantizer
from .gptq import GPTQQuantizer
from .awq import AWQQuantizer
from .fp8 import FP8Quantizer
from .torchao import TorchAOQuantizer

def list_quantizers():
    return list(QUANTIZER_REGISTRY.keys())