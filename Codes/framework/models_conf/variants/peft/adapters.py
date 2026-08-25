from typing import Any
from .base import PEFT_REGISTRY, BasePEFT


@PEFT_REGISTRY.register("adapters")
class AdapterH(BasePEFT):
    def apply(self, model, r=8, **kwargs) -> Any:
        """Execute the apply operation and return results."""
        try:
            from peft import AdaLoraConfig, get_peft_model
        except ImportError:
            raise ImportError("PEFT library required. Install with: pip install peft")
        peft_config = AdaLoraConfig(task_type="SEQ_CLS", r=r)
        return get_peft_model(model, peft_config)
