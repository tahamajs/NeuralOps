# framework/core/base.py (cleaned)
import torch.nn as nn
from typing import Optional, Any, Dict, Tuple
from abc import abstractmethod

class BaseAttention(nn.Module):
    def __init__(self, config, rank: Optional[int] = None):
        super().__init__()
        self.config = config
        self.rank = rank
        self.num_heads = getattr(config, 'num_attention_heads', 12)
        self.hidden_size = getattr(config, 'hidden_size', 768)
        self.head_dim = self.hidden_size // self.num_heads if self.num_heads > 0 else 64

    @abstractmethod
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        pass

    def extra_repr(self) -> str:
        return f"num_heads={self.num_heads}, head_dim={self.head_dim}, rank={self.rank}"


class BaseCompressor:
    """Base class for all compressors. Not a nn.Module to avoid parameter storage."""
    def __init__(self, model: nn.Module, name: str = "base", **kwargs):
        self.model = model
        self.name = name
        self.kwargs = kwargs
        self._original_params = self._count_params(model)

    @abstractmethod
    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        pass

    def _count_params(self, model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

    def get_compression_stats(self) -> Dict[str, Any]:
        compressed = self._count_params(self.model)
        orig = self._original_params
        return {
            "compressor_name": self.name,
            "original_parameters": orig,
            "compressed_parameters": compressed,
            "saved_parameters": orig - compressed,
            "compression_ratio": compressed / orig if orig > 0 else 1.0,
            "reduction_percentage": (orig - compressed) / orig * 100 if orig > 0 else 0.0,
        }

    def summary(self) -> str:
        s = self.get_compression_stats()
        return (
            f"\n========================================\n"
            f"Compression Summary: {self.name}\n"
            f"========================================\n"
            f"Original parameters:  {s['original_parameters']:,}\n"
            f"Compressed parameters: {s['compressed_parameters']:,}\n"
            f"Parameters saved:     {s['saved_parameters']:,}\n"
            f"Compression ratio:    {s['compression_ratio']:.3f}\n"
            f"Reduction:            {s['reduction_percentage']:.2f}%\n"
            f"========================================\n"
        )