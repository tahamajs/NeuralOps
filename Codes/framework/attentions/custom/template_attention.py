# framework/attentions/custom/template_attention.py
"""
Template for creating new attention mechanisms.

To create a new attention:
    1. Copy this file to your custom directory
    2. Rename the class and file
    3. Implement the forward method
    4. Register with @ATTENTION_REGISTRY.register("your_name")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from ...core.registry import ATTENTION_REGISTRY
from ...core.base import BaseAttention


@ATTENTION_REGISTRY.register("your_attention_name")
class YourCustomAttention(BaseAttention):
    """
    Your custom attention mechanism.
    
    This template provides a starting point for implementing
    new attention mechanisms with the following features:
        - Low-rank approximation support
        - Attention mask handling
        - Optional attention weights output
    """
    
    def __init__(self, config, rank: Optional[int] = None, **kwargs):
        """
        Initialize your custom attention.
        
        Args:
            config: Model configuration (HuggingFace config)
            rank: Optional compression rank
            **kwargs: Additional custom parameters
        """
        super().__init__(config, rank)
        
        # Extract configuration
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.dropout = config.attention_probs_dropout_prob
        
        # Your custom parameters
        self.custom_param = kwargs.get('custom_param', 0.1)
        
        # Projection layers
        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Optional low-rank projections
        if rank is not None and rank > 0:
            self.rank_proj = nn.Linear(self.hidden_size, rank, bias=False)
        else:
            self.rank_proj = None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using standard transformer init."""
        for module in [self.query, self.key, self.value, self.out_proj]:
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _reshape(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Reshape for multi-head attention."""
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of custom attention.
        
        Args:
            hidden_states: Input tensor (batch, seq_len, hidden_size)
            attention_mask: Optional attention mask
            output_attentions: Whether to return attention weights
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        Q = self._reshape(self.query(hidden_states), batch_size)
        K = self._reshape(self.key(hidden_states), batch_size)
        V = self._reshape(self.value(hidden_states), batch_size)
        
        # ============================================================
        # YOUR CUSTOM ATTENTION LOGIC HERE
        # ============================================================
        
        # Example: Apply low-rank projection if rank_proj is set
        if self.rank_proj is not None:
            # Project to low-rank space
            H = hidden_states.mean(dim=1)  # Pool over sequence
            H_proj = self.rank_proj(H)  # (batch, rank)
            
            # Use rank information to modulate attention
            scale = torch.sigmoid(H_proj).mean(dim=1, keepdim=True)  # (batch, 1)
            scale = scale.view(batch_size, 1, 1, 1)
            
            # Scale attention scores
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attention_scores = attention_scores * scale
        else:
            # Standard attention
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply attention mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask.unsqueeze(1)
        
        # Softmax and dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = F.dropout(attention_probs, p=self.dropout, training=self.training)
        
        # Context aggregation
        context = torch.matmul(attention_probs, V)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        output = self.out_proj(context)
        
        if output_attentions:
            return output, attention_probs
        return output, None
    
    @property
    def complexity(self) -> str:
        """Return complexity class."""
        if self.rank_proj is not None:
            return "linear"
        return "quadratic"
    
    def extra_repr(self) -> str:
        return f"num_heads={self.num_heads}, head_dim={self.head_dim}, rank={self.rank}"