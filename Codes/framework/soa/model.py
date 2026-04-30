# framework/soa/model.py
import torch
import torch.nn as nn
from .config import SOAConfig
from .attention import SOAFusedAttention

class SOABlock(nn.Module):
    def __init__(self, cfg: SOAConfig, shared_basis):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.ln2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.basis = shared_basis                     # (hidden, k_rank)
        self.attn = SOAFusedAttention(cfg.hidden_size, cfg.num_attention_heads, cfg.k_rank)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.intermediate_size),
            nn.GELU(),
            nn.Linear(cfg.intermediate_size, cfg.hidden_size),
        )

    def forward(self, x, attn_mask):
        h = self.ln1(x)
        x_latent = h @ self.basis          # (B, S, k_rank)
        attn_out = self.attn(x_latent, attn_mask)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


class SOABert(nn.Module):
    """Complete BERT model with SVD‑optimised attention."""
    def __init__(self, cfg: SOAConfig):
        super().__init__()
        self.cfg = cfg
        self.word_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_emb = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.seg_emb = nn.Embedding(cfg.type_vocab_size, cfg.hidden_size)
        self.ln = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

        # shared basis (will be filled by migration)
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

        cls_token = x[:, 0, :]
        pooled = torch.tanh(self.pooler(cls_token))
        logits = self.cls(pooled)
        return logits


# -------------------------------------------------------------------
# Optional: Fully SVD‑based block (no basis, dynamic SVD per step)
# -------------------------------------------------------------------
class SVDBasedAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, k_rank):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.k_rank = k_rank

    def forward(self, x, attn_mask):
        B, S, H = x.shape
        x_flat = x.reshape(-1, H)
        U, S_vals, Vh = torch.linalg.svd(x_flat, full_matrices=False)
        U_k = U[:, :self.k_rank]
        S_k = S_vals[:self.k_rank]
        V_approx = U_k * S_k.unsqueeze(0)               # (N, k_rank)
        V = V_approx.reshape(B, S, self.k_rank)

        head_dim = self.k_rank // self.num_heads
        q = V.view(B, S, self.num_heads, head_dim).transpose(1, 2)
        k = q
        v = q

        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        if attn_mask is not None:
            scores = scores + attn_mask
        probs = torch.softmax(scores, dim=-1)
        context = torch.matmul(probs, v)
        context = context.transpose(1, 2).contiguous().view(B, S, self.k_rank)
        return context

class SVDBasedBlock(nn.Module):
    def __init__(self, cfg: SOAConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.ln2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.attn = SVDBasedAttention(cfg.hidden_size, cfg.num_attention_heads, cfg.k_rank)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.intermediate_size),
            nn.GELU(),
            nn.Linear(cfg.intermediate_size, cfg.hidden_size),
        )

    def forward(self, x, attn_mask):
        h = self.ln1(x)
        attn_out = self.attn(h, attn_mask)
        # attn_out is (B,S,k_rank); to keep dimensions, we project back
        if attn_out.shape[-1] != x.shape[-1]:
            attn_out = nn.functional.linear(attn_out, nn.Parameter(torch.eye(x.shape[-1], attn_out.shape[-1])))
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x

class SVDBasedBERT(nn.Module):
    def __init__(self, cfg: SOAConfig):
        super().__init__()
        self.cfg = cfg
        self.word_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.pos_emb = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.seg_emb = nn.Embedding(cfg.type_vocab_size, cfg.hidden_size)
        self.ln = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.layers = nn.ModuleList([SVDBasedBlock(cfg) for _ in range(cfg.num_hidden_layers)])
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
        logits = self.cls(pooled)
        return logits