"""
Efficient Transformer Toolkit – Core module.
Provides model‑agnostic architecture detection, attention injection,
compression pipeline, and base classes.
"""

from .adapters import ADAPTER_MAP, get_adapter
from .architecture import (
    detect_architecture,
    get_attention_modules,
    get_classifier_head,
    get_embedding_module,
    get_ffn_modules,
    get_transformer_layers,
)
from .attention_mask import apply_attention_mask, get_token_valid_mask
from .attention_utils import available_attention_types, inject_attention, softpick
from .base import BaseAttention, BaseCompressor
from .compression_engine import CompressionPipeline, compress_model
from .memory import GPUMemoryGuard, cleanup_gpu_memory, get_gpu_memory_stats
from .model_utils import activation_hooks, copy_attention_weights
from .model_utils import count_parameters as count_params
from .model_utils import (
    get_parent_and_child_name,
    replace_module,
    run_calibration_forward,
)
from .model_wrapper import UnifiedTransformerConfig, UnifiedTransformerModel
from .registry import (
    ATTENTION_REGISTRY,
    AUX_REGISTRY,
    COMPRESSOR_REGISTRY,
    DATASET_REGISTRY,
    LOSS_REGISTRY,
    METHOD_REGISTRY,
    MODEL_REGISTRY,
    TRAINER_REGISTRY,
    VISUALIZER_REGISTRY,
    list_all_components,
)
from .multi_mode_benchmark import MultiModeBenchmarkRunner, SUPPORTED_MODES
from .utils import format_number, get_device, human_readable_time, set_seed
