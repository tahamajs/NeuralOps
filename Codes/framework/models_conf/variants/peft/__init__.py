from .base import BasePEFT, PEFT_REGISTRY
from .lora import LoRAPEFT
from .qlora import QLoRAPEFT
from .ia3 import IA3PEFT
from .prefix_tuning import PrefixTuningPEFT
from .prompt_tuning import PromptTuningPEFT
from .adapters import AdapterH

def list_peft_methods():
    return list(PEFT_REGISTRY.keys())