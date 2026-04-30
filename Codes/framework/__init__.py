# framework/__init__.py
"""
SVD-Lin Framework - Complete framework for SVD-based transformer optimization.
"""

import os
from pathlib import Path
from .core.registry import (
    ATTENTION_REGISTRY,
    COMPRESSOR_REGISTRY,
    TRAINER_REGISTRY,
    VISUALIZER_REGISTRY,
    DATASET_REGISTRY,
    list_all_components,
)


from . import core
from . import compression
from . import attentions
from . import routing
from . import soa
from .core.model_wrapper import UnifiedTransformerModel
from .quick import quick_svd_compress, quick_finetune, quick_inference
from .soa import SOAConfig, SOABert, migrate_bert_to_soa, compute_global_basis_from_hidden_states
from .core.registry import (
    ATTENTION_REGISTRY,
    COMPRESSOR_REGISTRY,
    list_all_components,
)
from .quick import quick_svd_compress, quick_finetune, quick_inference

# Core wrapper (most flexible)
from .core.model_wrapper import UnifiedTransformerModel

def discover_plugins(plugin_dirs: list = None):
    """
    Auto-discover plugins from directories.
    
    Args:
        plugin_dirs: List of directories to scan for plugins
    """
    if plugin_dirs is None:
        plugin_dirs = [
            os.path.join(os.path.dirname(__file__), "attentions", "custom"),
            os.path.join(os.path.dirname(__file__), "compression", "custom"),
        ]
    
    for plugin_dir in plugin_dirs:
        if os.path.exists(plugin_dir):
            # Import all Python files in the directory
            for file in os.listdir(plugin_dir):
                if file.endswith('.py') and not file.startswith('_'):
                    module_name = file[:-3]
                    try:
                        importlib.import_module(f"framework.attentions.custom.{module_name}")
                    except ImportError:
                        try:
                            importlib.import_module(f"framework.compression.custom.{module_name}")
                        except ImportError:
                            pass

# Auto-discover custom plugins
discover_plugins()


# Export main components
from .core.base import BaseAttention, BaseVisualizer, BaseCompressor
from .core.registry import (
    ATTENTION_REGISTRY,
    COMPRESSOR_REGISTRY,
    TRAINER_REGISTRY,
    VISUALIZER_REGISTRY,
    DATASET_REGISTRY,
)


__version__ = "2.0.0"

__all__ = ['core', 'compression', 'attentions', 'routing', 'soa'    "UnifiedTransformerModel",
    "quick_svd_compress",
    "quick_finetune", 
    "quick_inference",
    
    # Core wrapper
    "UnifiedTransformerModel",
        "ATTENTION_REGISTRY",
    "COMPRESSOR_REGISTRY",
    "TRAINER_REGISTRY",
    "VISUALIZER_REGISTRY",
    "DATASET_REGISTRY",
    "BaseAttention",
    "BaseVisualizer",
    "BaseCompressor",
    "list_all_components",
    "discover_plugins",

    # Registry
    "ATTENTION_REGISTRY",
    "COMPRESSOR_REGISTRY",
    "list_all_components",

    "quick_svd_compress",
    "quick_finetune",
    "quick_inference",
    "SOAConfig",
    "SOABert",
    "migrate_bert_to_soa",
    "compute_global_basis_from_hidden_states",
]
__version__ = "2.0.0"