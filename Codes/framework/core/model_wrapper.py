# framework/core/model_wrapper.py
"""
Universal wrapper for any HuggingFace transformer model.
Automatically detects architecture and provides unified interface.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union, Tuple
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
    AutoConfig,
)
from .registry import ATTENTION_REGISTRY
from .architecture import detect_architecture, get_attention_modules, replace_attention_module


class UnifiedTransformerModel(nn.Module):
    """
    Universal wrapper for any HuggingFace transformer model.
    
    Features:
        - Auto-detects model architecture (BERT, RoBERTa, GPT, etc.)
        - Supports any attention replacement
        - Handles different task heads automatically
        - Preserves original functionality
    
    Example:
        >>> from transformers import AutoModelForSequenceClassification
        >>> original_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        >>> model = UnifiedTransformerModel(original_model)
        >>> 
        >>> # Replace attention with SVD attention
        >>> model.replace_attention("svd", rank=64)
        >>> 
        >>> # Or compress linear layers
        >>> model.compress_linear_layers(rank=32)
    """
    
    def __init__(self, model: nn.Module, config: Optional[Any] = None):
        """
        Args:
            model: HuggingFace transformer model (any type)
            config: Model configuration (auto-detected if None)
        """
        super().__init__()
        self.model = model
        self.config = config or getattr(model, 'config', None)
        self.architecture = detect_architecture(model)
        self.original_forward = model.forward
        
        # Store original components for restoration
        self._original_attentions = {}
        self._original_linear_layers = {}
        
    def replace_attention(
        self,
        attention_type: str,
        rank: Optional[int] = None,
        **kwargs
    ) -> 'UnifiedTransformerModel':
        """
        Replace all attention modules with a different attention type.
        
        Args:
            attention_type: Type of attention to use (from registry)
            rank: Compression rank (if applicable)
            **kwargs: Additional arguments for attention module
            
        Returns:
            Self for chaining
        """
        attention_class = ATTENTION_REGISTRY.get(attention_type)
        if attention_class is None:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        # Find and replace all attention modules
        replace_attention_module(self.model, attention_class, rank=rank, **kwargs)
        
        return self
    
    def compress_linear_layers(
        self,
        rank: int,
        target_modules: Optional[List[str]] = None,
        **kwargs
    ) -> 'UnifiedTransformerModel':
        """
        Compress linear layers using SVD.
        
        Args:
            rank: Target rank for compression
            target_modules: List of module name patterns (auto-detected if None)
            **kwargs: Additional arguments
            
        Returns:
            Self for chaining
        """
        from ..compression import compress_linear_layers
        
        compress_linear_layers(self.model, rank, target_modules, **kwargs)
        return self
    
    def apply_petl(
        self,
        method: str = "lora",
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.1,
        target_modules: Optional[List[str]] = None
    ) -> 'UnifiedTransformerModel':
        """
        Apply Parameter-Efficient Fine-Tuning (LoRA, DoRA, etc.).
        
        Args:
            method: PETL method ('lora', 'dora', 'adalora')
            rank: LoRA rank
            alpha: LoRA alpha scaling
            dropout: Dropout probability
            target_modules: Target modules for adaptation
            
        Returns:
            Self for chaining
        """
        from ..peft import apply_lora, apply_dora
        
        if method == "lora":
            apply_lora(self.model, rank, alpha, dropout, target_modules)
        elif method == "dora":
            apply_dora(self.model, rank, alpha, dropout, target_modules)
        else:
            raise ValueError(f"Unknown PETL method: {method}")
        
        return self
    
    def freeze_layers(self, num_layers_to_freeze: int = 6) -> 'UnifiedTransformerModel':
        """
        Freeze early layers for efficient fine-tuning.
        
        Args:
            num_layers_to_freeze: Number of initial layers to freeze
            
        Returns:
            Self for chaining
        """
        layers = self._get_transformer_layers()
        for i, layer in enumerate(layers):
            if i < num_layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
        
        return self
    
    def unfreeze_all(self) -> 'UnifiedTransformerModel':
        """Unfreeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
        return self
    
    def get_attention_maps(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Extract attention maps from all layers.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            **kwargs: Additional arguments
            
        Returns:
            List of attention maps per layer
        """
        self.model.eval()
        
        # Forward with output_attentions=True
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            **kwargs
        )
        
        return outputs.attentions if hasattr(outputs, 'attentions') else []
    
    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Extract hidden states from all layers.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            **kwargs: Additional arguments
            
        Returns:
            List of hidden states per layer
        """
        self.model.eval()
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        
        return outputs.hidden_states if hasattr(outputs, 'hidden_states') else []
    
    def _get_transformer_layers(self) -> List[nn.Module]:
        """Extract transformer layers from model."""
        # Try common patterns
        if hasattr(self.model, 'bert') and hasattr(self.model.bert, 'encoder'):
            return self.model.bert.encoder.layer
        if hasattr(self.model, 'roberta') and hasattr(self.model.roberta, 'encoder'):
            return self.model.roberta.encoder.layer
        if hasattr(self.model, 'distilbert') and hasattr(self.model.distilbert, 'transformer'):
            return self.model.distilbert.transformer.layer
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            return self.model.encoder.layer
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h
        
        return []
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get detailed parameter count."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }
    
    def forward(self, *args, **kwargs):
        """Forward pass through the model."""
        return self.model(*args, **kwargs)
    
    def save_pretrained(self, save_directory: str):
        """Save model to directory."""
        self.model.save_pretrained(save_directory)
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        task: str = "sequence_classification",
        num_labels: Optional[int] = None,
        **kwargs
    ) -> 'UnifiedTransformerModel':
        """
        Load a pretrained model and wrap it.
        
        Args:
            model_name: HuggingFace model name
            task: Task type ('sequence_classification', 'qa', 'token_classification')
            num_labels: Number of labels (auto-detected if None)
            **kwargs: Additional arguments for model loading
            
        Returns:
            Wrapped model
        """
        if task == "sequence_classification":
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_labels or 2, **kwargs
            )
        elif task == "qa":
            model = AutoModelForQuestionAnswering.from_pretrained(model_name, **kwargs)
        elif task == "token_classification":
            model = AutoModelForTokenClassification.from_pretrained(
                model_name, num_labels=num_labels or 2, **kwargs
            )
        else:
            model = AutoModel.from_pretrained(model_name, **kwargs)
        
        config = AutoConfig.from_pretrained(model_name)
        return cls(model, config)