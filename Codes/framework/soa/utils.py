# framework/soa/utils.py
"""
Utility functions for SOA models.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def count_parameters(model: nn.Module, trainable_only: bool = True) -> Dict[str, int]:
    """
    Count parameters in SOA model.
    
    Args:
        model: SOA model
        trainable_only: If True, count only trainable parameters
        
    Returns:
        Dictionary with parameter counts by component
    """
    stats = {
        'embeddings': 0,
        'attention_low_rank': 0,
        'attention_full': 0,
        'ffn': 0,
        'pooler_classifier': 0,
        'total': 0,
        'trainable': 0
    }
    
    # Count embeddings
    if hasattr(model, 'word_emb'):
        stats['embeddings'] += model.word_emb.weight.numel()
        stats['embeddings'] += model.pos_emb.weight.numel()
        stats['embeddings'] += model.seg_emb.weight.numel()
    
    # Count attention and FFN layers
    for layer in model.layers:
        # Attention low-rank projections
        stats['attention_low_rank'] += layer.attn.q_proj.weight.numel()
        stats['attention_low_rank'] += layer.attn.k_proj.weight.numel()
        stats['attention_low_rank'] += layer.attn.v_proj.weight.numel()
        
        # Attention full projection
        stats['attention_full'] += layer.attn.o_proj.weight.numel()
        
        # FFN
        stats['ffn'] += layer.ffn[0].weight.numel()
        stats['ffn'] += layer.ffn[2].weight.numel()
    
    # Count pooler and classifier
    if hasattr(model, 'pooler'):
        stats['pooler_classifier'] += model.pooler.weight.numel()
    if hasattr(model, 'cls'):
        stats['pooler_classifier'] += model.cls.weight.numel()
    
    stats['total'] = sum(stats.values())
    
    if trainable_only:
        stats['trainable'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return stats


def print_model_summary(model: nn.Module):
    """Print a summary of the SOA model."""
    params = count_parameters(model)
    
    print("\n" + "=" * 50)
    print("SOA Model Summary")
    print("=" * 50)
    print(f"Embeddings:          {params['embeddings']:,} parameters")
    print(f"Attention (low-rank): {params['attention_low_rank']:,} parameters")
    print(f"Attention (full):    {params['attention_full']:,} parameters")
    print(f"FFN:                 {params['ffn']:,} parameters")
    print(f"Pooler + Classifier: {params['pooler_classifier']:,} parameters")
    print(f"Total:               {params['total']:,} parameters")
    if 'trainable' in params:
        print(f"Trainable:           {params['trainable']:,} parameters")
    print("=" * 50)


def compute_compression_ratio(original_model: nn.Module, soa_model: nn.Module) -> float:
    """
    Compute compression ratio between original and SOA model.
    
    Args:
        original_model: Original BERT model
        soa_model: SOA model
        
    Returns:
        Compression ratio (soa_params / original_params)
    """
    original_params = sum(p.numel() for p in original_model.parameters())
    soa_params = sum(p.numel() for p in soa_model.parameters())
    return soa_params / original_params