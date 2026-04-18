import torch
import torch.nn as nn
from .config import SOAConfig
from .layers import SOABlock

class SOABert(nn.Module):
    def __init__(self, cfg: SOAConfig):
        super().__init__()
        self.cfg = cfg

        # Embeddings
        self.word_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_emb = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.seg_emb = nn.Embedding(cfg.type_vocab_size, cfg.hidden_size)
        self.ln = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

        # Shared basis U, initialized randomly. Will be overwritten by migration.
        self.basis = nn.Parameter(torch.empty(cfg.hidden_size, cfg.k_rank), requires_grad=False)
        
        # Encoder layers
        self.layers = nn.ModuleList([SOABlock(cfg, self.basis) for _ in range(cfg.num_hidden_layers)])
        
        # Pooler and Classifier (for sequence classification)
        self.pooler = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.cls = nn.Linear(cfg.hidden_size, 2)

    def forward(self, input_ids, token_type_ids, attention_mask):
        B, S = input_ids.shape
        device = input_ids.device

        # Position IDs
        pos_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        
        # Sum embeddings
        x = self.word_emb(input_ids) + self.pos_emb(pos_ids) + self.seg_emb(token_type_ids)
        x = self.ln(x)
        x = nn.functional.dropout(x, p=self.cfg.dropout, training=self.training)

        # Create attention mask
        extended_mask = (1.0 - attention_mask[:, None, None, :].float()) * -10000.0

        for layer in self.layers:
            x = layer(x, extended_mask)

        # Pooler
        cls_token = x[:, 0, :]
        pooled = torch.tanh(self.pooler(cls_token))
        logits = self.cls(pooled)
        return logits