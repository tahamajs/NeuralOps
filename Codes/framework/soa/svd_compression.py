# framework/soa/svd_compression.py
"""
SVD compression utilities for BERT models.
"""

import torch
import torch.nn as nn
from ..compression.svd_linear import SVDLinearLayer


def compress_bert_true_svd(model: nn.Module, rank: int) -> nn.Module:
    """
    Compress all linear layers in BERT except embeddings, pooler, classifier.
    
    This function replaces standard linear layers with low-rank SVD approximations.
    
    Args:
        model: BERT model to compress
        rank: Target rank for SVD compression
        
    Returns:
        Compressed model
        
    Example:
        >>> from transformers import BertModel
        >>> model = BertModel.from_pretrained('bert-base-uncased')
        >>> compressed_model = compress_bert_true_svd(model, rank=64)
    """
    print(f"\n🔄 Starting TRUE SVD Compression (Rank={rank})...")
    compressed_count = 0
    total_original = sum(p.numel() for p in model.parameters())
    
    skip_patterns = ['embeddings', 'pooler', 'classifier']
    
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            # Skip certain layers
            if any(skip in name.lower() for skip in skip_patterns):
                continue
            
            try:
                new_layer = SVDLinearLayer(module, rank)
                
                # Replace the layer
                if '.' in name:
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                    child_name = name
                
                setattr(parent, child_name, new_layer)
                compressed_count += 1
                
            except Exception as e:
                print(f"   ⚠️ Failed to compress {name}: {e}")
    
    total_compressed = sum(p.numel() for p in model.parameters())
    reduction = 100 * (1 - total_compressed / total_original)
    print(f"✅ Compressed {compressed_count} layers. Reduction: {reduction:.2f}%")
    
    return model


def svd_compress_input(inputs: torch.Tensor, rank: int) -> torch.Tensor:
    """
    Apply SVD to input embeddings and reconstruct with softmax-weighted rank.
    
    Args:
        inputs: Input tensor of shape (batch, seq_len, hidden_size)
        rank: Target rank for compression
        
    Returns:
        Compressed and reconstructed tensor
    """
    B, S, D = inputs.shape
    X = inputs.view(-1, D)
    
    # Compute SVD
    U, S_val, Vh = torch.linalg.svd(X, full_matrices=False)
    
    # Keep top-k components
    U_r = U[:, :rank]
    S_r = S_val[:rank]
    Vh_r = Vh[:rank, :]
    
    # Softmax weighting of singular values
    softmax_weights = torch.softmax(S_r, dim=0)
    
    # Weighted reconstruction
    compressed = (U_r * softmax_weights.view(1, -1)) @ Vh_r
    
    return compressed.view(B, S, D)


def apply_svd_to_inputs(model: nn.Module, rank: int) -> nn.Module:
    """
    Wrap model to compress inputs before forward pass.
    
    Args:
        model: Model to wrap
        rank: Target rank for input compression
        
    Returns:
        Model with input compression wrapper
    """
    original_forward = model.forward
    embeddings = model.embeddings
    
    def new_forward(input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        # Get embeddings
        if hasattr(embeddings, 'forward'):
            emb = embeddings(input_ids, token_type_ids=token_type_ids)
        else:
            emb = embeddings(input_ids)
        
        # Compress embeddings
        compressed_emb = svd_compress_input(emb, rank)
        
        # Call original forward with compressed inputs
        return original_forward(
            inputs_embeds=compressed_emb,
            attention_mask=attention_mask,
            **kwargs
        )
    
    model.forward = new_forward
    return model