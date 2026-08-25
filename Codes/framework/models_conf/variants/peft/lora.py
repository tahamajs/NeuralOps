from typing import Any
from .base import PEFT_REGISTRY, BasePEFT


@PEFT_REGISTRY.register("lora")
class LoRAPEFT(BasePEFT):
    def apply(self, model, r=8, lora_alpha=16, lora_dropout=0.1, **kwargs) -> Any:
        """Execute the apply operation and return results."""
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError:
            raise ImportError("PEFT library required. Install with: pip install peft")
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        return get_peft_model(model, peft_config)
