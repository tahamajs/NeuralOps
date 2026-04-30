# studio/soa/model.py
import torch
import torch.nn as nn
from .config import SOAConfig
from .attention import SOAFusedAttention

class SOABlock(nn.Module):
    def __init__(self, cfg: SOAConfig, shared_basis):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.ln2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.basis = shared_basis
        self.attn = SOAFusedAttention(cfg.hidden_size, cfg.num_attention_heads, cfg.k_rank)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.intermediate_size),
            nn.GELU(),
            nn.Linear(cfg.intermediate_size, cfg.hidden_size),
        )
    def forward(self, x, attn_mask):
        h = self.ln1(x)
        x_latent = h @ self.basis   # (B,S,k_rank)
        attn_out = self.attn(x_latent, attn_mask)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x

class SOABert(nn.Module):
    def __init__(self, cfg: SOAConfig):
        super().__init__()
        self.cfg = cfg
        self.word_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_emb = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.seg_emb = nn.Embedding(cfg.type_vocab_size, cfg.hidden_size)
        self.ln = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.basis = nn.Parameter(torch.empty(cfg.hidden_size, cfg.k_rank), requires_grad=False)
        self.layers = nn.ModuleList([SOABlock(cfg, self.basis) for _ in range(cfg.num_hidden_layers)])
        self.pooler = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.cls = nn.Linear(cfg.hidden_size, cfg.num_labels)

    def forward(self, input_ids, token_type_ids, attention_mask):
        B, S = input_ids.shape
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)
        x = self.word_emb(input_ids) + self.pos_emb(pos) + self.seg_emb(token_type_ids)
        x = self.ln(x)
        mask = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * -1e4
        for layer in self.layers:
            x = layer(x, mask)
        cls = x[:, 0, :]
        pooled = torch.tanh(self.pooler(cls))
        return self.cls(pooled)