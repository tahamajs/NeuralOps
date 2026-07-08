from .base import COMPRESSOR_REGISTRY, BaseCompressor


@COMPRESSOR_REGISTRY.register("distillation")
class DistillationCompressor(BaseCompressor):
    def apply(self, student_model, teacher_model=None, temperature=2.0, **kwargs):
        if teacher_model is None:
            raise ValueError("Teacher model must be provided for distillation")
        # This is a placeholder; actual KD requires a training loop.
        # The orchestrator can use this info to modify the trainer.
        student_model.config.distillation_config = {
            "teacher": teacher_model,
            "temperature": temperature,
        }
        return student_model
