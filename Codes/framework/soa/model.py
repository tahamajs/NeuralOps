# framework/soa/model.py
"""
SOA BERT Model - Complete transformer with SVD-optimized architecture.
"""

import torch
import torch.nn as nn
from .config import SOAConfig
from .block import SOABlock


class SOABert(nn.Module):
    """
    SOA BERT Model - BERT architecture with SVD-optimized attention.
    
    This model uses a shared low-rank basis for attention projections,
    significantly reducing the number of parameters while maintaining
    competitive performance.
    
    Example:
        >>> config = SOAConfig(k_rank=64)
        >>> model = SOABert(config)
        >>> logits = model(input_ids, token_type_ids, attention_mask)
    """
    
    def __init__(self, cfg: SOAConfig):
        super().__init__()
        self.cfg = cfg
        
        # Embeddings
        self.word_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_emb = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.seg_emb = nn.Embedding(cfg.type_vocab_size, cfg.hidden_size)
        self.ln = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.dropout_prob)

        # Shared basis (will be filled from migration or training)
        self.register_buffer('basis', torch.empty(cfg.hidden_size, cfg.k_rank))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            SOABlock(cfg, self.basis) for _ in range(cfg.num_hidden_layers)
        ])
        
        # Pooler and classifier
        self.pooler = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.cls = nn.Linear(cfg.hidden_size, cfg.num_labels)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.cfg.hidden_size ** -0.5)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=self.cfg.hidden_size ** -0.5)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through SOA BERT model.
        
        Args:
            input_ids: Input token IDs (batch, seq_len)
            token_type_ids: Token type IDs for segment embeddings
            attention_mask: Attention mask (1 for real tokens, 0 for padding)
            
        Returns:
            Classification logits (batch, num_labels)
        """
        B, S = input_ids.shape
        
        # Position embeddings
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)
        
        # Combine embeddings
        x = self.word_emb(input_ids) + self.pos_emb(pos) + self.seg_emb(token_type_ids)
        x = self.ln(x)
        x = self.dropout(x)
        
        # Create attention mask for causal/self attention
        attn_mask = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * -1e4
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, attn_mask)
        
        # Pool and classify
        cls_token = x[:, 0, :]
        pooled = torch.tanh(self.pooler(cls_token))
        logits = self.cls(pooled)
        
        return logits
    
    def get_attention_maps(self, input_ids, token_type_ids, attention_mask):
        """
        Get attention maps from all layers (for visualization).
        
        Returns:
            List of attention maps per layer
        """
        B, S = input_ids.shape
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)
        x = self.word_emb(input_ids) + self.pos_emb(pos) + self.seg_emb(token_type_ids)
        x = self.ln(x)
        
        attn_mask = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * -1e4
        attn_maps = []
        
        for layer in self.layers:
            h = layer.ln1(x)
            x_latent = h @ self.basis
            # Capture attention map from SOAFusedAttention
            with torch.no_grad():
                q = layer.attn.q_proj(x_latent).view(B, S, layer.attn.num_heads, layer.attn.head_dim).transpose(1, 2)
                k = layer.attn.k_proj(x_latent).view(B, S, layer.attn.num_heads, layer.attn.head_dim).transpose(1, 2)
                scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(layer.attn.head_dim)
                if attn_mask is not None:
                    scores = scores + attn_mask
                attn_map = torch.softmax(scores, dim=-1)
                attn_maps.append(attn_map)
            x = x + layer.attn(x_latent, attn_mask)
            x = x + layer.ffn(layer.ln2(x))
        
        return attn_maps
    
    @classmethod
    def from_pretrained(cls, model_name: str, k_rank: int = 64, **kwargs):
        """
        Create SOABert from a pretrained HuggingFace BERT model.
        
        Args:
            model_name: HuggingFace model name (e.g., 'bert-base-uncased')
            k_rank: Rank for the shared basis
            **kwargs: Additional arguments for SOAConfig
            
        Returns:
            SOABert instance with weights migrated from pretrained model
        """
        from transformers import BertModel
        from .migration import compute_global_basis_from_bert, migrate_bert_to_soa
        
        # Load pretrained BERT
        bert_model = BertModel.from_pretrained(model_name)
        
        # Create SOA config from BERT config
        cfg = SOAConfig.from_pretrained(model_name, k_rank=k_rank, **kwargs)
        
        # Compute shared basis from BERT weights
        basis, _ = compute_global_basis_from_bert(bert_model, k_rank)
        
        # Create SOA model and migrate weights
        soa_model = cls(cfg)
        soa_model = migrate_bert_to_soa(bert_model, soa_model, basis)
        
        return soa_model