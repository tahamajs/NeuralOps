"""
Architecture detection and layer pattern management for different transformer models.
"""

import re
from typing import Dict, List, Any


ARCH_PATTERNS: Dict[str, List[str]] = {
    'bert': [
        'attention.self.query',
        'attention.self.key',
        'attention.self.value',
        'attention.output.dense',
        'intermediate.dense',
        'output.dense'
    ],
    'distilbert': [
        'attention.q_lin',
        'attention.k_lin',
        'attention.v_lin',
        'attention.out_lin',
        'ffn.lin1',
        'ffn.lin2'
    ],
    'roberta': [
        'attention.self.query',
        'attention.self.key',
        'attention.self.value',
        'attention.output.dense',
        'intermediate.dense',
        'output.dense'
    ],
    'xlnet': [
        'layer.layer.0.attention.hybrid_attn.q_linear',
        'layer.layer.0.attention.hybrid_attn.k_linear',
        'layer.layer.0.attention.hybrid_attn.v_linear',
        'layer.layer.0.attention.output.dense',
        'layer.layer.0.ff.layer_1',
        'layer.layer.0.ff.layer_2'
    ],
    'albert': [
        'attention_query',
        'attention_key',
        'attention_value',
        'attention_output',
        'ffn',
        'ffn_output'
    ],
    'electra': [
        'attention.self.query',
        'attention.self.key',
        'attention.self.value',
        'attention.output.dense',
        'intermediate.dense',
        'output.dense'
    ],
    'deberta': [
        'attention.self.query_proj',
        'attention.self.key_proj',
        'attention.self.value_proj',
        'attention.output.dense',
        'intermediate.dense',
        'output.dense'
    ],
    'gpt2': [
        'attn.c_attn',
        'attn.c_proj',
        'mlp.c_fc',
        'mlp.c_proj'
    ],
}


def detect_architecture(model_name: str) -> str:
    """
    Detect model architecture from name.

    Args:
        model_name: HuggingFace model name

    Returns:
        Architecture identifier ('bert', 'roberta', 'distilbert', etc.)
    """
    name_lower = model_name.lower()

    if 'distilbert' in name_lower:
        return 'distilbert'
    if 'roberta' in name_lower:
        return 'roberta'
    if 'xlnet' in name_lower:
        return 'xlnet'
    if 'albert' in name_lower:
        return 'albert'
    if 'electra' in name_lower:
        return 'electra'
    if 'deberta' in name_lower:
        return 'deberta'
    if 'gpt2' in name_lower or 'gpt-2' in name_lower:
        return 'gpt2'
    if 'bert' in name_lower:
        return 'bert'

    return 'bert'


def get_layer_patterns(arch: str) -> List[str]:
    """
    Get layer name patterns for a given architecture.

    Args:
        arch: Architecture identifier

    Returns:
        List of pattern strings for matching layer names
    """
    return ARCH_PATTERNS.get(arch, ARCH_PATTERNS['bert'])


def get_compression_targets(model_name: str, include_ffn: bool = True) -> List[str]:
    """
    Get target layer names for compression.

    Args:
        model_name: HuggingFace model name
        include_ffn: Whether to include FFN layers

    Returns:
        List of layer name patterns to compress
    """
    arch = detect_architecture(model_name)
    patterns = get_layer_patterns(arch)

    if not include_ffn:
        ffn_patterns = ['ffn', 'intermediate', 'mlp', 'ff', 'feedforward']
        patterns = [p for p in patterns if not any(fp in p for fp in ffn_patterns)]

    return patterns


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about a model architecture.

    Args:
        model_name: HuggingFace model name

    Returns:
        Dictionary with model info
    """
    arch = detect_architecture(model_name)

    default_configs = {
        'bert': {'hidden_size': 768, 'num_layers': 12, 'num_heads': 12},
        'distilbert': {'hidden_size': 768, 'num_layers': 6, 'num_heads': 12},
        'roberta': {'hidden_size': 768, 'num_layers': 12, 'num_heads': 12},
        'albert': {'hidden_size': 768, 'num_layers': 12, 'num_heads': 12},
        'electra': {'hidden_size': 768, 'num_layers': 12, 'num_heads': 12},
        'deberta': {'hidden_size': 768, 'num_layers': 12, 'num_heads': 12},
        'gpt2': {'hidden_size': 768, 'num_layers': 12, 'num_heads': 12},
        'xlnet': {'hidden_size': 768, 'num_layers': 12, 'num_heads': 12},
    }

    if 'large' in model_name.lower():
        default_configs['bert']['hidden_size'] = 1024
        default_configs['roberta']['hidden_size'] = 1024

    return {
        'architecture': arch,
        'patterns': get_layer_patterns(arch),
        'estimated_config': default_configs.get(arch, default_configs['bert']),
    }


def get_layer_count(model) -> int:
    """
    Get number of transformer layers in a model.

    Args:
        model: HuggingFace model

    Returns:
        Number of layers
    """
    if hasattr(model, 'config') and hasattr(model.config, 'num_hidden_layers'):
        return model.config.num_hidden_layers

    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        return len(model.encoder.layer)

    if hasattr(model, 'transformer') and hasattr(model.transformer, 'layer'):
        return len(model.transformer.layer)

    return 12


def get_hidden_size(model) -> int:
    """
    Get hidden size of a model.

    Args:
        model: HuggingFace model

    Returns:
        Hidden dimension size
    """
    if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
        return model.config.hidden_size

    return 768