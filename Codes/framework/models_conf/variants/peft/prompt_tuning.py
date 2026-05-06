from peft import get_peft_model, PromptTuningConfig, TaskType
from .base import BasePEFT, PEFT_REGISTRY

@PEFT_REGISTRY.register("prompt_tuning")
class PromptTuningPEFT(BasePEFT):
    def apply(self, model, num_virtual_tokens=20, **kwargs):
        peft_config = PromptTuningConfig(task_type=TaskType.SEQ_CLS, num_virtual_tokens=num_virtual_tokens)
        return get_peft_model(model, peft_config)