from typing import Any
import torch
from .base import PEFT_REGISTRY, BasePEFT


@PEFT_REGISTRY.register("qlora")
class QLoRAPEFT(BasePEFT):
    def apply(self, model, r=8, lora_alpha=16, lora_dropout=0.1, **kwargs) -> Any:
        """Execute the apply operation and return results."""
        try:
            from peft import LoraConfig, TaskType, get_peft_model
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise ImportError("PEFT and bitsandbytes required. Install with: pip install peft bitsandbytes")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
        model.config.quantization_config = bnb_config
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        return get_peft_model(model, peft_config)
