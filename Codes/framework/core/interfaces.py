# framework/core/interfaces.py
"""
Core interfaces for all framework components.
This file defines the abstract base classes that all components must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn


class AttentionInterface(ABC):
    """
    Interface for all attention mechanisms.
    
    To add a new attention mechanism:
        1. Inherit from this class
        2. Implement the forward method
        3. Register with @ATTENTION_REGISTRY.register("your_name")
    
    Example:
        @ATTENTION_REGISTRY.register("my_attention")
        class MyAttention(AttentionInterface, nn.Module):
            def __init__(self, config, rank=None):
                super().__init__()
                # your initialization
            
            def forward(self, hidden_states, attention_mask=None, **kwargs):
                # your forward logic
                return output, attention_weights
    """
    
    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of attention mechanism.
        
        Args:
            hidden_states: Input tensor (batch, seq_len, hidden_size)
            attention_mask: Optional attention mask
            **kwargs: Additional arguments (output_attentions, head_mask, etc.)
            
        Returns:
            Tuple of (output, attention_weights)
        """
        pass
    
    @property
    def complexity(self) -> str:
        """Return complexity class: 'quadratic', 'linear', 'sparse'"""
        return "quadratic"
    
    @property
    def supported_ranks(self) -> List[int]:
        """Return supported compression ranks (empty list if not applicable)"""
        return []


class CompressorInterface(ABC):
    """
    Interface for all compression methods.
    
    To add a new compression method:
        1. Inherit from this class
        2. Implement compress() method
        3. Register with @COMPRESSOR_REGISTRY.register("your_name")
    
    Example:
        @COMPRESSOR_REGISTRY.register("my_compressor")
        class MyCompressor(CompressorInterface):
            def __init__(self, model, **kwargs):
                super().__init__(model, **kwargs)
                # your initialization
            
            def compress(self) -> Tuple[nn.Module, Dict]:
                # your compression logic
                return compressed_model, stats
    """
    
    def __init__(self, model: nn.Module, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self._original_params = self._count_params(model)
    
    @abstractmethod
    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply compression and return compressed model with stats."""
        pass
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        compressed_params = self._count_params(self.model)
        return {
            "original_parameters": self._original_params,
            "compressed_parameters": compressed_params,
            "compression_ratio": compressed_params / self._original_params if self._original_params > 0 else 1.0,
            "saved_parameters": self._original_params - compressed_params,
            "reduction_percentage": (1 - compressed_params / self._original_params) * 100 if self._original_params > 0 else 0,
        }
    
    def _count_params(self, model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())
    
    @property
    def name(self) -> str:
        """Return compression method name."""
        return self.__class__.__name__.lower().replace("compressor", "")


class TrainerInterface(ABC):
    """
    Interface for custom trainers.
    
    To add a new trainer:
        1. Inherit from this class
        2. Implement train() and evaluate() methods
        3. Register with @TRAINER_REGISTRY.register("your_name")
    """
    
    @abstractmethod
    def train(self, **kwargs) -> Dict[str, Any]:
        """Train the model and return metrics."""
        pass
    
    @abstractmethod
    def evaluate(self, **kwargs) -> Dict[str, Any]:
        """Evaluate the model and return metrics."""
        pass


class VisualizerInterface(ABC):
    """
    Interface for visualizers.
    
    To add a new visualizer:
        1. Inherit from this class
        2. Implement generate() method
        3. Register with @VISUALIZER_REGISTRY.register("your_name")
    """
    
    @abstractmethod
    def generate(self, model: nn.Module, dataloader, **kwargs) -> List[str]:
        """Generate visualizations and return file paths."""
        pass


class DataLoaderInterface(ABC):
    """
    Interface for custom data loaders.
    
    To add a new dataset:
        1. Inherit from this class
        2. Implement load() method
        3. Register with @DATASET_REGISTRY.register("your_name")
    """
    
    @abstractmethod
    def load(self, split: str = "train") -> torch.utils.data.Dataset:
        """Load dataset for given split."""
        pass