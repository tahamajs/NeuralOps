from . import (
    architecture,
    checkpoint,
    config,
    device,
    helpers,
    layer_norm_fix,
    logger,
    metrics,
    paths,
    profiler,
    replace_attention,
    results_saver,
    seed,
)
from .results_saver import ResultsSaver

__all__ = [
    "architecture",
    "checkpoint",
    "config",
    "device",
    "helpers",
    "layer_norm_fix",
    "logger",
    "metrics",
    "paths",
    "profiler",
    "replace_attention",
    "results_saver",
    "ResultsSaver",
    "seed",
]

# Re-exported utility functions
from .architecture import (  # noqa: E402
    get_compression_targets,
    get_hidden_size,
    get_layer_count,
    get_model_info,
)
from .compat import init_output_scale  # noqa: E402
from .config import merge_configs  # noqa: E402
from .helpers import (  # noqa: E402
    count_parameters,
    format_time,
    load_json,
    save_json,
    set_seed,
    unflatten_dict,
)
from .logger import get_logger, set_log_level  # noqa: E402
from .metrics import compute_metrics_for_task  # noqa: E402
from .paths import resolve_path  # noqa: E402
from .profiler import (  # noqa: E402
    measure_inference_latency,
    measure_memory_usage,
    profile_context,
)
from .qwen_compat import (  # noqa: E402
    is_architecture_error,
    is_qwen_attention,
    qwen_config_proxy,
    qwen_module_policy,
    qwen_projection,
)

__all__ += [
    "format_time",
    "save_json",
    "load_json",
    "unflatten_dict",
    "resolve_path",
    "merge_configs",
    "set_log_level",
    "get_logger",
    "count_parameters",
    "set_seed",
    "measure_inference_latency",
    "measure_memory_usage",
    "profile_context",
    "get_model_info",
    "get_layer_count",
    "get_hidden_size",
    "get_compression_targets",
    "init_output_scale",
    "is_qwen_attention",
    "qwen_config_proxy",
    "qwen_projection",
    "qwen_module_policy",
    "is_architecture_error",
    "compute_metrics_for_task",
    "CheckpointManager",
]

# Checkpoint utilities
from .checkpoint import CheckpointManager  # noqa: E402

# Auto-wired package members
from neuralops.utils import compat
from neuralops.utils import architecture
from neuralops.utils import attention_profiler
from neuralops.utils import attention_utils
from neuralops.utils import checkpoint
from neuralops.utils import compat
from neuralops.utils import config
from neuralops.utils import device
from neuralops.utils import helpers
from neuralops.utils import hf_git_exporter
from neuralops.utils import layer_norm_fix
from neuralops.utils import logger
from neuralops.utils import lora
from neuralops.utils import metrics
from neuralops.utils import model_family
from neuralops.utils import paths
from neuralops.utils import profiler
from neuralops.utils import qwen_compat
from neuralops.utils import replace_attention
from neuralops.utils import results_saver
from neuralops.utils import seed
