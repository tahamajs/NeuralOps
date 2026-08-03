from peft import PromptTuningConfig, TaskType, get_peft_model

from .base import PEFT_REGISTRY, BasePEFT


@PEFT_REGISTRY.register("prompt_tuning")
class PromptTuningPEFT(BasePEFT):
    def apply(self, model, num_virtual_tokens=20, **kwargs) -> Any:
        """Execute the apply operation and return results."""
        peft_config = PromptTuningConfig(
            task_type=TaskType.SEQ_CLS, num_virtual_tokens=num_virtual_tokens
        )
        return get_peft_model(model, peft_config)
