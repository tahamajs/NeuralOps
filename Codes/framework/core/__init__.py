"""
Efficient Transformer Toolkit – Core module.
Provides model‑agnostic architecture detection, attention injection,
compression pipeline, and base classes.
"""

from .registry import (
    ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY, AUX_REGISTRY,
    MODEL_REGISTRY, DATASET_REGISTRY, LOSS_REGISTRY, TRAINER_REGISTRY,
    VISUALIZER_REGISTRY, list_all_components,
)
from .base import BaseAttention, BaseCompressor
from .architecture import (
    detect_architecture, get_transformer_layers, get_attention_modules,
    get_ffn_modules, get_embedding_module, get_classifier_head,
)
from .model_utils import (
    run_calibration_forward, copy_attention_weights, replace_module,
    get_parent_and_child_name, count_parameters as count_params,
    activation_hooks,
)
from .attention_utils import inject_attention, available_attention_types
from .compression_engine import CompressionPipeline, compress_model
from .utils import set_seed, get_device, format_number, human_readable_time
from .model_wrapper import UnifiedTransformerModel
from .adapters import get_adapter, ADAPTER_MAP
from .model_wrapper import UnifiedTransformerModel, UnifiedTransformerConfig