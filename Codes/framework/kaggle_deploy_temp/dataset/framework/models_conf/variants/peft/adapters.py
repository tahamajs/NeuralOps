from typing import Any
from peft import AdaLoraConfig  # using AdaLoRA as a modern adapter proxy
from peft import get_peft_model

from .base import PEFT_REGISTRY, BasePEFT


@PEFT_REGISTRY.register("adapters")
class AdapterH(BasePEFT):
    def apply(self, model, r=8, **kwargs) -> Any:
        """Execute the apply operation and return results."""
        peft_config = AdaLoraConfig(task_type="SEQ_CLS", r=r)
        return get_peft_model(model, peft_config)
