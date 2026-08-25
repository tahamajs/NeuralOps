from typing import Any
from .base import PEFT_REGISTRY, BasePEFT


@PEFT_REGISTRY.register("ia3")
class IA3PEFT(BasePEFT):
    def apply(self, model, **kwargs) -> Any:
        """Execute the apply operation and return results."""
        try:
            from peft import IA3Config, TaskType, get_peft_model
        except ImportError:
            raise ImportError("PEFT library required. Install with: pip install peft")
        peft_config = IA3Config(task_type=TaskType.SEQ_CLS)
        return get_peft_model(model, peft_config)
