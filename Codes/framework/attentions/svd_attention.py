from ..core.base_attention import BaseAttention
from ..core.registry import ATTENTION_REGISTRY
import torch
import torch.nn.functional as F

@ATTENTION_REGISTRY.register("svd")
class SVDLinearAttention(BaseAttention):
    # ... (your existing SVD attention implementation)
    pass