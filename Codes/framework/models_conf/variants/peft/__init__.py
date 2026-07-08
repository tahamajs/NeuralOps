from .adapters import AdapterH
from .base import PEFT_REGISTRY, BasePEFT
from .ia3 import IA3PEFT
from .lora import LoRAPEFT
from .prefix_tuning import PrefixTuningPEFT
from .prompt_tuning import PromptTuningPEFT
from .qlora import QLoRAPEFT


def list_peft_methods():
    return list(PEFT_REGISTRY.keys())
