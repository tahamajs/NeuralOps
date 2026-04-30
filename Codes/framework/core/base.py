# framework/core/base.py (UPDATED with BaseCompressor)
"""
Base classes for framework components.
"""

from torch import nn
from typing import Optional, Any, Dict, List, Tuple
import torch


class BaseAttention(nn.Module):
    """
    Base class for all attention mechanisms.
    
    All attention implementations should inherit from this class
    and implement the forward method.
    
    Args:
        config: Model configuration (e.g., BertConfig)
        rank: Compression rank (for low-rank attention)
    
    Example:
        >>> class MyAttention(BaseAttention):
        ...     def forward(self, hidden_states, attention_mask=None, **kwargs):
        ...         # custom attention logic
        ...         return output, attention_weights
    """
    
    def __init__(self, config, rank: Optional[int] = None):
        super().__init__()
        self.config = config
        self.rank = rank
        
        # Common attributes for all attention mechanisms
        self.num_heads = getattr(config, 'num_attention_heads', 12)
        self.hidden_size = getattr(config, 'hidden_size', 768)
        self.head_dim = self.hidden_size // self.num_heads if self.num_heads > 0 else 64
        
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        """
        Forward pass for attention.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional mask tensor
            **kwargs: Additional arguments (e.g., output_attentions, head_mask)
            
        Returns:
            Tuple of (output, attention_weights) where attention_weights may be None
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    def extra_repr(self) -> str:
        """String representation of the attention module."""
        return f"num_heads={self.num_heads}, head_dim={self.head_dim}, rank={self.rank}"


class BaseVisualizer:
    """
    Base class for all visualizers.
    
    Visualizers generate plots and visualizations from model outputs.
    
    Example:
        >>> class MyVisualizer(BaseVisualizer):
        ...     def generate(self, model, dataloader, **kwargs):
        ...         # generate visualizations
        ...         return figure_path
    """
    
    def __init__(self, output_dir: str = "visualizations"):
        """
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def generate(self, *args, **kwargs):
        """
        Generate visualizations.
        
        Args:
            *args: Variable arguments
            **kwargs: Keyword arguments
            
        Returns:
            Path or list of paths to generated visualizations
        """
        raise NotImplementedError("Subclasses must implement generate()")
    
    def _save_figure(self, fig, filename: str, dpi: int = 150):
        """Helper to save a matplotlib figure."""
        import os
        import matplotlib.pyplot as plt
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return path


class BaseCompressor:
    """
    Base class for model compressors.
    
    Compressors apply various compression techniques to reduce model size,
    including SVD, pruning, quantization, and decomposition methods.
    
    Attributes:
        model: The model to compress
        name: Name of the compression method
        compression_stats: Dictionary of compression statistics
        
    Example:
        >>> class MyCompressor(BaseCompressor):
        ...     def compress(self):
        ...         # Apply compression logic
        ...         compressed_model = self._apply_compression()
        ...         stats = self.get_compression_stats()
        ...         return compressed_model, stats
    """
    
    def __init__(self, model: nn.Module, name: str = "base", **kwargs):
        """
        Args:
            model: The model to compress
            name: Name of the compression method
            **kwargs: Additional compression parameters
        """
        self.model = model
        self.name = name
        self.kwargs = kwargs
        self._original_params = self._count_params(model)
        self._compressed_params = None
        self._compression_stats = {}
        
    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Apply compression to the model.
        
        Returns:
            Tuple of (compressed_model, compression_statistics)
        """
        raise NotImplementedError("Subclasses must implement compress()")
    
    def _count_params(self, model: nn.Module, trainable_only: bool = False) -> int:
        """
        Count parameters in a model.
        
        Args:
            model: PyTorch model
            trainable_only: If True, count only trainable parameters
            
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        return sum(p.numel() for p in model.parameters())
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the compression.
        
        Returns:
            Dictionary with compression statistics including:
            - original_parameters: Number of parameters before compression
            - compressed_parameters: Number of parameters after compression
            - compression_ratio: Ratio of compressed to original parameters
            - saved_parameters: Number of parameters saved
            - reduction_percentage: Percentage reduction in parameters
        """
        if self._compressed_params is None:
            self._compressed_params = self._count_params(self.model)
        
        original = self._original_params
        compressed = self._compressed_params
        
        stats = {
            "compressor_name": self.name,
            "original_parameters": original,
            "compressed_parameters": compressed,
            "saved_parameters": original - compressed,
            "compression_ratio": compressed / original if original > 0 else 1.0,
            "reduction_percentage": ((original - compressed) / original * 100) if original > 0 else 0.0,
        }
        stats.update(self._compression_stats)
        
        return stats
    
    def _validate_model(self):
        """Validate that the model is compressible."""
        if self.model is None:
            raise ValueError("Model is None. Cannot compress.")
    
    def _log_compression(self, layer_name: str, original_size: int, compressed_size: int):
        """Log compression of a specific layer."""
        reduction = (original_size - compressed_size) / original_size * 100
        print(f"   ✂️ {layer_name}: {original_size:,} → {compressed_size:,} ({reduction:.1f}% reduction)")
    
    def summary(self) -> str:
        """
        Generate a summary string of the compression.
        
        Returns:
            Formatted summary string
        """
        stats = self.get_compression_stats()
        return f"""
========================================
Compression Summary: {self.name}
========================================
Original parameters:  {stats['original_parameters']:,}
Compressed parameters: {stats['compressed_parameters']:,}
Parameters saved:     {stats['saved_parameters']:,}
Compression ratio:    {stats['compression_ratio']:.3f}
Reduction:            {stats['reduction_percentage']:.2f}%
========================================
"""


# ----------------------------------------------------------------------------
# Concrete Compressor Implementations (Examples)
# ----------------------------------------------------------------------------

class SVDCompressor(BaseCompressor):
    """
    SVD-based compression for linear layers.
    
    Replaces linear layers with low-rank approximations using SVD.
    
    Example:
        >>> compressor = SVDCompressor(model, rank=64)
        >>> compressed_model, stats = compressor.compress()
    """
    
    def __init__(self, model: nn.Module, rank: int = 64, target_modules: List[str] = None, **kwargs):
        """
        Args:
            model: Model to compress
            rank: Target rank for SVD approximation
            target_modules: List of module name patterns to compress
        """
        super().__init__(model, name="svd_compression", rank=rank, **kwargs)
        self.rank = rank
        self.target_modules = target_modules or ["query", "key", "value", "dense"]
        self.replaced_layers = []
    
    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply SVD compression to the model."""
        self._validate_model()
        
        # Import SVDLinearLayer (adjust import based on your structure)
        try:
            from ..compression.svd_based import SVDLinearLayer
        except ImportError:
            # Fallback - define a simple SVD layer
            class SVDLinearLayer(nn.Module):
                def __init__(self, linear_layer, rank):
                    super().__init__()
                    import numpy as np
                    self.in_features = linear_layer.in_features
                    self.out_features = linear_layer.out_features
                    self.rank = min(rank, self.in_features, self.out_features)
                    
                    W = linear_layer.weight.data.cpu().numpy()
                    U, S, Vh = np.linalg.svd(W, full_matrices=False)
                    self.U = nn.Parameter(torch.from_numpy(U[:, :self.rank]).float())
                    self.S = nn.Parameter(torch.from_numpy(S[:self.rank]).float())
                    self.Vh = nn.Parameter(torch.from_numpy(Vh[:self.rank, :]).float())
                    self.bias = nn.Parameter(linear_layer.bias.data.clone()) if linear_layer.bias is not None else None
                
                def forward(self, x):
                    x_v = torch.matmul(x, self.Vh.t())
                    sv = self.U.t() * self.S.unsqueeze(0)
                    out = torch.matmul(x_v, sv.t())
                    if self.bias is not None:
                        out += self.bias
                    return out
        
        compressed_count = 0
        
        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear):
                # Check if this layer should be compressed
                should_compress = any(pattern in name.lower() for pattern in self.target_modules)
                if should_compress and "classifier" not in name and "pooler" not in name:
                    try:
                        new_layer = SVDLinearLayer(module, self.rank)
                        
                        # Replace the layer
                        parent_name = '.'.join(name.split('.')[:-1]) if '.' in name else None
                        child_name = name.split('.')[-1]
                        
                        if parent_name:
                            parent = self.model.get_submodule(parent_name)
                        else:
                            parent = self.model
                        
                        setattr(parent, child_name, new_layer)
                        
                        # Track statistics
                        orig_size = module.weight.numel()
                        comp_size = new_layer.U.numel() + new_layer.S.numel() + new_layer.Vh.numel()
                        self._log_compression(name, orig_size, comp_size)
                        
                        self.replaced_layers.append(name)
                        compressed_count += 1
                        
                    except Exception as e:
                        print(f"   ⚠️ Failed to compress {name}: {e}")
        
        self._compressed_params = self._count_params(self.model)
        self._compression_stats = {
            "compressed_layers": compressed_count,
            "replaced_layers": self.replaced_layers,
            "rank_used": self.rank,
        }
        
        print(f"✅ Compressed {compressed_count} layers with rank {self.rank}")
        print(self.summary())
        
        return self.model, self.get_compression_stats()


class PruningCompressor(BaseCompressor):
    """
    Pruning-based compression using magnitude pruning.
    
    Example:
        >>> compressor = PruningCompressor(model, sparsity=0.3)
        >>> compressed_model, stats = compressor.compress()
    """
    
    def __init__(self, model: nn.Module, sparsity: float = 0.3, **kwargs):
        """
        Args:
            model: Model to compress
            sparsity: Fraction of weights to prune (0.0 to 1.0)
        """
        super().__init__(model, name="pruning_compression", sparsity=sparsity, **kwargs)
        self.sparsity = sparsity
    
    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply magnitude pruning to the model."""
        self._validate_model()
        
        pruned_count = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                flat_weight = weight.abs().view(-1)
                
                # Calculate threshold
                k = int(flat_weight.numel() * (1 - self.sparsity))
                if k > 0:
                    threshold = torch.topk(flat_weight, k, largest=True).values[-1]
                else:
                    threshold = float('inf')
                
                # Apply pruning mask
                mask = (weight.abs() >= threshold).float()
                module.weight.data = weight * mask
                pruned_count += 1
        
        self._compressed_params = self._count_params(self.model)
        self._compression_stats = {
            "pruned_layers": pruned_count,
            "sparsity": self.sparsity,
        }
        
        print(f"✅ Pruned {pruned_count} layers with sparsity {self.sparsity}")
        print(self.summary())
        
        return self.model, self.get_compression_stats()


class QuantizationCompressor(BaseCompressor):
    """
    Quantization-based compression.
    
    Example:
        >>> compressor = QuantizationCompressor(model, bits=8)
        >>> compressed_model, stats = compressor.compress()
    """
    
    def __init__(self, model: nn.Module, bits: int = 8, **kwargs):
        """
        Args:
            model: Model to compress
            bits: Number of bits for quantization (4, 8, 16)
        """
        super().__init__(model, name="quantization_compression", bits=bits, **kwargs)
        self.bits = bits
    
    def _quantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize a tensor to the specified number of bits."""
        min_val, max_val = tensor.min(), tensor.max()
        if max_val == min_val:
            return tensor
        
        scale = (max_val - min_val) / (2 ** self.bits - 1)
        quantized = torch.round((tensor - min_val) / scale) * scale + min_val
        return quantized
    
    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply quantization to the model."""
        self._validate_model()
        
        quantized_count = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                module.weight.data = self._quantize_tensor(module.weight.data)
                if module.bias is not None:
                    module.bias.data = self._quantize_tensor(module.bias.data)
                quantized_count += 1
        
        self._compressed_params = self._count_params(self.model)
        self._compression_stats = {
            "quantized_layers": quantized_count,
            "bits": self.bits,
        }
        
        print(f"✅ Quantized {quantized_count} layers to {self.bits}-bit")
        print(self.summary())
        
        return self.model, self.get_compression_stats()


# ----------------------------------------------------------------------------
# Compressor Factory
# ----------------------------------------------------------------------------

class CompressorFactory:
    """
    Factory for creating compressors.
    
    Example:
        >>> compressor = CompressorFactory.create("svd", model, rank=64)
        >>> compressed_model, stats = compressor.compress()
    """
    
    _COMPRESSORS = {
        "svd": SVDCompressor,
        "pruning": PruningCompressor,
        "quantization": QuantizationCompressor,
    }
    
    @classmethod
    def register(cls, name: str, compressor_class):
        """Register a new compressor type."""
        cls._COMPRESSORS[name] = compressor_class
    
    @classmethod
    def create(cls, name: str, model: nn.Module, **kwargs) -> BaseCompressor:
        """
        Create a compressor instance.
        
        Args:
            name: Compressor name ('svd', 'pruning', 'quantization')
            model: Model to compress
            **kwargs: Compressor-specific arguments
            
        Returns:
            Compressor instance
        """
        if name not in cls._COMPRESSORS:
            available = list(cls._COMPRESSORS.keys())
            raise ValueError(f"Unknown compressor: {name}. Available: {available}")
        
        compressor_class = cls._COMPRESSORS[name]
        return compressor_class(model, **kwargs)
    
    @classmethod
    def available(cls) -> List[str]:
        """List available compressor names."""
        return list(cls._COMPRESSORS.keys())


# ----------------------------------------------------------------------------
# Convenience Function
# ----------------------------------------------------------------------------

def compress_model(model: nn.Module, method: str = "svd", **kwargs) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Convenience function to compress a model.
    
    Args:
        model: Model to compress
        method: Compression method ('svd', 'pruning', 'quantization')
        **kwargs: Method-specific arguments
        
    Returns:
        Tuple of (compressed_model, compression_statistics)
        
    Example:
        >>> compressed_model, stats = compress_model(model, "svd", rank=64)
        >>> compressed_model, stats = compress_model(model, "pruning", sparsity=0.3)
    """
    compressor = CompressorFactory.create(method, model, **kwargs)
    return compressor.compress()