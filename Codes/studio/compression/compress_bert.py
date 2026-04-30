# studio/compression/compress_bert.py
import torch
import torch.nn as nn
from .svd_linear import SVDLinearLayer

def compress_bert_true_svd(model: nn.Module, rank: int) -> nn.Module:
    skip = ['embeddings', 'pooler', 'classifier']
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and not any(s in name for s in skip):
            try:
                new_layer = SVDLinearLayer(module, rank)
                parent = model.get_submodule('.'.join(name.split('.')[:-1])) if '.' in name else model
                setattr(parent, name.split('.')[-1], new_layer)
            except Exception:
                pass
    return model