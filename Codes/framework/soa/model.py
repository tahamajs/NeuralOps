import torch
import torch.nn as nn
from .config import SOAConfig
from .block import SOABlock

class SOABert(nn.Module):
    def __init__(self, cfg: SOAConfig):
        super().__init__()
        self.cfg = cfg
        self.word_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_emb = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.seg_emb = nn.Embedding(cfg.type_vocab_size, cfg.hidden_size)
        self.ln = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

        # Shared basis (will be filled from migration)
        self.register_buffer('basis', torch.empty(cfg.hidden_size, cfg.k_rank))

        self.layers = nn.ModuleList([SOABlock(cfg, self.basis) for _ in range(cfg.num_hidden_layers)])
        self.pooler = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.cls = nn.Linear(cfg.hidden_size, 2)   # binary classification

    def forward(self, input_ids, token_type_ids, attention_mask):
        B, S = input_ids.shape
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)
        x = self.word_emb(input_ids) + self.pos_emb(pos) + self.seg_emb(token_type_ids)
        x = self.ln(x)

        attn_mask = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * -1e4

        for layer in self.layers:
            x = layer(x, attn_mask)

        cls = x[:, 0, :]
        pooled = torch.tanh(self.pooler(cls))
        logits = self.cls(pooled)
        return logits