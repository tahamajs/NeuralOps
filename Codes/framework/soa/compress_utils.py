import torch
import torch.nn as nn
from .svd_linear import SVDLinearLayer

def compress_bert_true_svd(model: nn.Module, rank: int) -> nn.Module:
    """Compress all linear layers in BERT except embeddings, pooler, classifier."""
    print(f"\n🔄 Starting TRUE SVD Compression (Rank={rank})...")
    compressed_count = 0
    total_original = sum(p.numel() for p in model.parameters())
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(skip in name for skip in ['embeddings', 'pooler', 'classifier']):
                continue
            new_layer = SVDLinearLayer(module, rank)
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, new_layer)
            compressed_count += 1
    total_compressed = sum(p.numel() for p in model.parameters())
    print(f"✅ Compressed {compressed_count} layers. Reduction: {100*(1 - total_compressed/total_original):.2f}%")
    return model

def svd_compress_input(inputs: torch.Tensor, rank: int) -> torch.Tensor:
    """Apply SVD to input embeddings and reconstruct."""
    B, S, D = inputs.shape
    X = inputs.view(-1, D)
    U, S_val, Vh = torch.linalg.svd(X, full_matrices=False)
    U_r = U[:, :rank]
    S_r = S_val[:rank]
    Vh_r = Vh[:rank, :]
    softmax_weights = torch.softmax(S_r, dim=0)
    compressed = (U_r * softmax_weights.view(1, -1)) @ Vh_r
    return compressed.view(B, S, D)

def apply_svd_to_inputs(model, rank: int):
    """Wrap model to compress inputs before forward pass."""
    original_forward = model.forward
    def new_forward(input_ids=None, attention_mask=None, **kwargs):
        emb = model.embeddings(input_ids)
        compressed_emb = svd_compress_input(emb, rank)
        return original_forward(inputs_embeds=compressed_emb, attention_mask=attention_mask, **kwargs)
    model.forward = new_forward
    return model