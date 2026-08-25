from typing import Any
from .base import PEFT_REGISTRY, BasePEFT


@PEFT_REGISTRY.register("prefix_tuning")
class PrefixTuningPEFT(BasePEFT):
    def apply(self, model, num_virtual_tokens=20, **kwargs) -> Any:
        """Execute the apply operation and return results."""
        try:
            from peft import PrefixTuningConfig, TaskType, get_peft_model
        except ImportError:
            raise ImportError("PEFT library required. Install with: pip install peft")
        peft_config = PrefixTuningConfig(
            task_type=TaskType.SEQ_CLS, num_virtual_tokens=num_virtual_tokens
        )
        return get_peft_model(model, peft_config)
