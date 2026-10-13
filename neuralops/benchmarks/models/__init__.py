"""Model loading and architecture adaptation subsystem package."""

from neuralops.benchmarks.models.loader import (
    MODEL_ADAPTER_REGISTRY,
    BertFamilyAdapter,
    GptFamilyAdapter,
    LlamaFamilyAdapter,
    ModelAdapter,
    T5FamilyAdapter,
    cleanup_model,
    count_parameters,
    count_trainable_parameters,
    get_model_adapter,
    inspect_model_metadata,
    load_universal_model,
    register_model_adapter,
)

__all__ = [
    "ModelAdapter",
    "register_model_adapter",
    "get_model_adapter",
    "count_parameters",
    "count_trainable_parameters",
    "cleanup_model",
    "inspect_model_metadata",
    "load_universal_model",
    "BertFamilyAdapter",
    "GptFamilyAdapter",
    "LlamaFamilyAdapter",
    "T5FamilyAdapter",
    "MODEL_ADAPTER_REGISTRY",
]

# Auto-wired package members
from neuralops.benchmarks.models import loader
