from peft import IA3Config, TaskType, get_peft_model

from .base import PEFT_REGISTRY, BasePEFT


@PEFT_REGISTRY.register("ia3")
class IA3PEFT(BasePEFT):
    def apply(self, model, **kwargs):
        peft_config = IA3Config(task_type=TaskType.SEQ_CLS)
        return get_peft_model(model, peft_config)
