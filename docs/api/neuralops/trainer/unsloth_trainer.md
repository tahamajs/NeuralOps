Module neuralops.trainer.unsloth_trainer
========================================
UnslothTrainer – Accelerated Fast-LoRA and Memory-Efficient Training Engine.
Integrates Unsloth 2x faster cross-entropy and fast Triton kernels when available,
with a transparent, production-ready fallback to HuggingFaceTrainer on standard systems.

Classes
-------

`UnslothTrainer(model: Any, train_dataset: Any = None, eval_dataset: Any = None, config: dict[str, Any] | None = None, results_manager: Any | None = None, compute_metrics: Callable | None = None, **kwargs: Any)`
:   Accelerated trainer utilizing Unsloth kernels for low-memory, high-throughput fine-tuning.
    Gracefully falls back to optimized HuggingFaceTrainer when Unsloth is not installed.

    ### Ancestors (in MRO)

    * neuralops.trainer.base_trainer.BaseTrainer
    * abc.ABC