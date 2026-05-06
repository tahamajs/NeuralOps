from peft import get_peft_model, LoraConfig, TaskType
from .base import BasePEFT, PEFT_REGISTRY

@PEFT_REGISTRY.register("lora")
class LoRAPEFT(BasePEFT):
    def apply(self, model, r=8, lora_alpha=16, lora_dropout=0.1, **kwargs):
        peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        return get_peft_model(model, peft_config)