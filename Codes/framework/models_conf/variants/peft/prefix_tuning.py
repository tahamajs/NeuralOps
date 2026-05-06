from peft import get_peft_model, PrefixTuningConfig, TaskType
from .base import BasePEFT, PEFT_REGISTRY

@PEFT_REGISTRY.register("prefix_tuning")
class PrefixTuningPEFT(BasePEFT):
    def apply(self, model, num_virtual_tokens=20, **kwargs):
        peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_CLS, num_virtual_tokens=num_virtual_tokens)
        return get_peft_model(model, peft_config)