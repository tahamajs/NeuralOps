# framework/utils/__init__.py
from .seed import set_all_seeds
from .device import get_device
from .helpers import (
    set_seed,
    format_time,
    format_number,
    count_parameters,
    ensure_dir,
    save_json,
    load_json,
    flatten_dict,
    unflatten_dict,
)
from .logger import setup_logger, get_logger, set_log_level
from .metrics import (
    compute_accuracy,
    compute_f1,
    compute_matthews_corrcoef,
    compute_pearson,
    compute_spearman,
    compute_precision_recall,
    compute_metrics_for_task,
)
from .config import load_config, save_config, update_config, merge_configs
from .checkpoint import CheckpointManager
from .profiler import (
    ModelProfiler,
    measure_inference_latency,
    measure_memory_usage,
    profile_context,
)
from .architecture import (
    detect_architecture,
    get_layer_patterns,
    get_compression_targets,
    get_model_info,
    get_layer_count,
    get_hidden_size,
)
from .layer_norm_fix import fix_layernorm_naming
from .save_results import save_experiment_results

# Attention replacement (needs ATTENTION_REGISTRY)
import torch.nn as nn
from framework.attentions.core.registry import ATTENTION_REGISTRY

def replace_attention(model, attention_key, **kwargs):
    attn_class = ATTENTION_REGISTRY.get(attention_key)
    if attn_class is None:
        raise ValueError(f"Unknown attention key: {attention_key}")
    config = getattr(model, 'config', None)

    def _replace_module(module):
        for name, child in module.named_children():
            if 'attention' in name.lower() or 'attn' in name.lower():
                try:
                    new_attn = attn_class(config, **kwargs) if config else attn_class(**kwargs)
                    setattr(module, name, new_attn)
                except Exception as e:
                    print(f"Failed to replace {name}: {e}")
            else:
                _replace_module(child)

    _replace_module(model)
    return model

__all__ = [
    "set_all_seeds",
    "get_device",
    "set_seed",
    "format_time",
    "format_number",
    "count_parameters",
    "ensure_dir",
    "save_json",
    "load_json",
    "flatten_dict",
    "unflatten_dict",
    "setup_logger",
    "get_logger",
    "set_log_level",
    "compute_accuracy",
    "compute_f1",
    "compute_matthews_corrcoef",
    "compute_pearson",
    "compute_spearman",
    "compute_precision_recall",
    "compute_metrics_for_task",
    "load_config",
    "save_config",
    "update_config",
    "merge_configs",
    "CheckpointManager",
    "ModelProfiler",
    "measure_inference_latency",
    "measure_memory_usage",
    "profile_context",
    "detect_architecture",
    "get_layer_patterns",
    "get_compression_targets",
    "get_model_info",
    "get_layer_count",
    "get_hidden_size",
    "fix_layernorm_naming",
    "save_experiment_results",
    "replace_attention",
]