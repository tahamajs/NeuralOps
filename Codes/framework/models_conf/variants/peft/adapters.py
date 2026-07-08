from peft import (AdaLoraConfig,  # using AdaLoRA as a modern adapter proxy
                  get_peft_model)

from .base import PEFT_REGISTRY, BasePEFT


@PEFT_REGISTRY.register("adapters")
class AdapterH(BasePEFT):
    def apply(self, model, r=8, **kwargs):
        peft_config = AdaLoraConfig(task_type="SEQ_CLS", r=r)
        return get_peft_model(model, peft_config)
