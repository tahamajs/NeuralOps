"""
MoE-DeltaKV-Paged Hybrid Attention Module.
Combines Mixture-of-Block-Attention (MoBA) with Paged Delta-KV cache compression.

Fixes:
- Straight-Through Estimator (STE) for differentiable 4-bit uniform and SVD quantization.
- Non-causal bidirectional attention for encoder architectures (DistilBERT).
- CLS token preservation in block 0.
- Pretrained weight initialization via copy_weights.
"""

import math
from typing import Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY


def ste_quantize_4bit(x: torch.Tensor) -> torch.Tensor:
    """
    Differentiable 4-bit uniform quantization using Straight-Through Estimator (STE).
    """
    if x.numel() == 0:
        return x
    x_min = x.min(dim=-1, keepdim=True)[0]
    x_max = x.max(dim=-1, keepdim=True)[0]
    scale = torch.clamp((x_max - x_min) / 15.0, min=1e-5)
    x_q_int = torch.clamp(torch.round((x - x_min) / scale), 0, 15)
    x_q = x_q_int * scale + x_min
    # STE: forward uses x_q, backward flows through x
    return x + (x_q - x).detach()


def low_rank_svd_quantize(delta: torch.Tensor, rank: int) -> torch.Tensor:
    """
    Decomposes temporal delta into low-rank SVD components and applies differentiable quantization.
    """
    B, H, N, d = delta.shape
    delta_reshaped = delta.reshape(-1, d)

    if delta_reshaped.shape[0] < rank or delta_reshaped.shape[1] < rank:
        return ste_quantize_4bit(delta)

    delta_f = delta_reshaped.float()
    U, S, Vh = torch.linalg.svd(delta_f, full_matrices=False)
    U = U[:, :rank]
    S = S[:rank]
    Vh = Vh[:rank, :]

    delta_approx = ((U * S.unsqueeze(0)) @ Vh).to(dtype=delta.dtype)
    delta_approx = delta_approx.reshape(B, H, N, d)

    return ste_quantize_4bit(delta_approx)


def simulate_paged_delta_kv(K: torch.Tensor, V: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulates Delta-KV temporal compression with differentiable reconstruction.
    """
    B, H, N, d = K.shape
    if N <= 1:
        return K, V

    K_hat = torch.zeros_like(K)
    V_hat = torch.zeros_like(V)

    K_hat[:, :, 0, :] = K[:, :, 0, :]
    V_hat[:, :, 0, :] = V[:, :, 0, :]

    delta_K = K[:, :, 1:, :] - K[:, :, :-1, :]
    delta_V = V[:, :, 1:, :] - V[:, :, :-1, :]

    delta_K_hat = low_rank_svd_quantize(delta_K, rank)
    delta_V_hat = low_rank_svd_quantize(delta_V, rank)

    K_hat[:, :, 1:, :] = K[:, :, 0:1, :] + torch.cumsum(delta_K_hat, dim=2)
    V_hat[:, :, 1:, :] = V[:, :, 0:1, :] + torch.cumsum(delta_V_hat, dim=2)

    return K_hat, V_hat


@ATTENTION_REGISTRY.register("moe_deltakv_paged")
class MoEDeltaKVPagedAttention(nn.Module):
    r"""
    MoE-DeltaKV-Paged Hybrid Attention Module.
    Combines Mixture-of-Block-Attention (MoBA) with Paged Delta-KV cache compression.
    """

    def __init__(self, config: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__()
        self.hidden_size = getattr(config, "hidden_size", kwargs.get("hidden_size", 768))
        self.num_heads: int = getattr(config, "num_attention_heads", kwargs.get("num_attention_heads", 12))
        self.head_dim: int = self.hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim

        self.page_size = getattr(config, "page_size", kwargs.get("page_size", 64))
        self.k_experts = getattr(config, "k_experts", kwargs.get("k_experts", 4))
        self.svd_rank = getattr(config, "svd_rank", kwargs.get("svd_rank", 16))
        self.is_causal = getattr(config, "is_causal", kwargs.get("is_causal", False))

        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)

        self.W_rg = nn.Linear(self.head_dim, self.head_dim, bias=False)
        nn.init.eye_(self.W_rg.weight)

        self.out_proj = nn.Linear(self.all_head_size, self.hidden_size)

        dropout_p = getattr(
            config,
            "dropout",
            getattr(
                config,
                "hidden_dropout_prob",
                getattr(config, "attention_probs_dropout_prob", 0.1),
            ),
        )
        self.dropout = nn.Dropout(dropout_p)
        self.aux_loss: torch.Tensor = torch.tensor(0.0)

        pretrained_attn = kwargs.get("pretrained_attn", None)
        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, pretrained_attn: nn.Module) -> None:
        with torch.no_grad():
            q_src = getattr(pretrained_attn, "q_lin", getattr(pretrained_attn, "query", getattr(pretrained_attn, "q_proj", None)))
            k_src = getattr(pretrained_attn, "k_lin", getattr(pretrained_attn, "key", getattr(pretrained_attn, "k_proj", None)))
            v_src = getattr(pretrained_attn, "v_lin", getattr(pretrained_attn, "value", getattr(pretrained_attn, "v_proj", None)))
            out_src = getattr(pretrained_attn, "out_lin", getattr(pretrained_attn, "dense", getattr(pretrained_attn, "out_proj", getattr(pretrained_attn, "o_proj", None))))

            if q_src is not None and hasattr(q_src, "weight"):
                self.query.weight.copy_(q_src.weight)
                if q_src.bias is not None and self.query.bias is not None:
                    self.query.bias.copy_(q_src.bias)

            if k_src is not None and hasattr(k_src, "weight"):
                self.key.weight.copy_(k_src.weight)
                if k_src.bias is not None and self.key.bias is not None:
                    self.key.bias.copy_(k_src.bias)

            if v_src is not None and hasattr(v_src, "weight"):
                self.value.weight.copy_(v_src.weight)
                if v_src.bias is not None and self.value.bias is not None:
                    self.value.bias.copy_(v_src.bias)

            if out_src is not None and hasattr(out_src, "weight"):
                self.out_proj.weight.copy_(out_src.weight)
                if out_src.bias is not None and self.out_proj.bias is not None:
                    self.out_proj.bias.copy_(out_src.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim

        q = self.query(hidden_states).view(B, N, H, d).transpose(1, 2)
        k = self.key(hidden_states).view(B, N, H, d).transpose(1, 2)
        v = self.value(hidden_states).view(B, N, H, d).transpose(1, 2)

        # Delta-KV compression with STE
        k_hat, v_hat = simulate_paged_delta_kv(k, v, self.svd_rank)

        num_pages = (N + self.page_size - 1) // self.page_size
        remainder = N % self.page_size
        if remainder != 0:
            pad_len = self.page_size - remainder
            q_padded = F.pad(q, (0, 0, 0, pad_len))
            k_padded = F.pad(k_hat, (0, 0, 0, pad_len))
            v_padded = F.pad(v_hat, (0, 0, 0, pad_len))
            padded_N = N + pad_len
        else:
            q_padded, k_padded, v_padded = q, k_hat, v_hat
            padded_N = N

        if num_pages <= 2:
            # Full attention path when sequence length fits in 1-2 pages
            scale = 1.0 / math.sqrt(d)
            scores = torch.matmul(q, k_hat.transpose(-2, -1)) * scale
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    pad_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
                    scores = scores.masked_fill(pad_mask, -10000.0)
                elif attention_mask.dim() == 4:
                    scores = scores + attention_mask

            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            if head_mask is not None:
                attn = attn * head_mask
            out = torch.matmul(attn, v_hat).transpose(1, 2).contiguous().view(B, N, self.all_head_size)
            return self.out_proj(out), (attn if output_attentions else None)

        # Block Paging
        q_blocks = q_padded.view(B, H, num_pages, self.page_size, d)
        k_blocks = k_padded.view(B, H, num_pages, self.page_size, d)
        v_blocks = v_padded.view(B, H, num_pages, self.page_size, d)

        q_bar = q_blocks.mean(dim=3)  # [B, H, num_pages, d]
        k_bar = k_blocks.mean(dim=3)  # [B, H, num_pages, d]

        router_logits = torch.matmul(self.W_rg(q_bar), k_bar.transpose(-1, -2)) / math.sqrt(d)

        if self.is_causal:
            causal_mask = torch.tril(torch.ones((num_pages, num_pages), device=q.device, dtype=torch.bool))
            router_logits = router_logits.masked_fill(~causal_mask, float("-inf"))

        # Always include page 0 (which contains CLS token at index 0)
        k_experts = min(self.k_experts, num_pages)
        topk_logits, topk_indices = torch.topk(router_logits, k_experts, dim=-1)
        # Ensure page 0 is included
        topk_indices[:, :, :, 0] = 0

        g_j = F.softmax(topk_logits, dim=-1)

        expanded_indices = topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, self.page_size, d)
        k_blocks_exp = k_blocks.unsqueeze(2).expand(-1, -1, num_pages, -1, -1, -1)
        v_blocks_exp = v_blocks.unsqueeze(2).expand(-1, -1, num_pages, -1, -1, -1)

        k_gathered = torch.gather(k_blocks_exp, dim=3, index=expanded_indices)
        v_gathered = torch.gather(v_blocks_exp, dim=3, index=expanded_indices)

        q_expanded = q_blocks.unsqueeze(3)
        attn_scores = torch.matmul(q_expanded, k_gathered.transpose(-1, -2)) / math.sqrt(d)

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context_per_expert = torch.matmul(attn_probs, v_gathered)
        g_j_exp = g_j.unsqueeze(-1).unsqueeze(-1)
        context = (context_per_expert * g_j_exp).sum(dim=3)

        context = context.view(B, H, padded_N, d).transpose(1, 2).contiguous().view(B, padded_N, self.all_head_size)
        out = self.out_proj(context[:, :N, :])

        return out, None
