from peft import get_peft_model, IA3Config, TaskType
from .base import BasePEFT, PEFT_REGISTRY

@PEFT_REGISTRY.register("ia3")
class IA3PEFT(BasePEFT):
    def apply(self, model, **kwargs):
        peft_config = IA3Config(task_type=TaskType.SEQ_CLS)
        return get_peft_model(model, peft_config)