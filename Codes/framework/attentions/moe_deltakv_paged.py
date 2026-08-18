import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Any, Optional, Tuple

from framework.core.attention_mask import apply_attention_mask
from framework.core.registry import ATTENTION_REGISTRY

def uniform_quantize_4bit(x: torch.Tensor) -> torch.Tensor:
    """
    Simulates 4-bit uniform quantization.
    """
    if x.numel() == 0:
        return x
    x_min = x.min(dim=-1, keepdim=True)[0]
    x_max = x.max(dim=-1, keepdim=True)[0]
    scale = (x_max - x_min) / 15.0
    scale = torch.clamp(scale, min=1e-5)
    x_q = torch.round((x - x_min) / scale)
    x_q = torch.clamp(x_q, 0, 15)
    return x_q * scale + x_min

def low_rank_svd_quantize(delta: torch.Tensor, rank: int) -> torch.Tensor:
    """
    Decomposes the temporal delta into low-rank SVD components and applies 4-bit quantization.
    """
    B, H, N, d = delta.shape
    delta_reshaped = delta.reshape(-1, d)
    
    # If dimensions are smaller than rank, fallback to direct quantization
    if delta_reshaped.shape[0] < rank or delta_reshaped.shape[1] < rank:
        return uniform_quantize_4bit(delta)
        
    # FIX: [Cast to float32 for torch.linalg.svd execution to ensure CPU float16/bfloat16 compatibility]
    delta_f = delta_reshaped.float() if delta_reshaped.dtype in (torch.float16, torch.bfloat16) else delta_reshaped
    U, S, Vh = torch.linalg.svd(delta_f, full_matrices=False)
    U = U[:, :rank]
    S = S[:rank]
    Vh = Vh[:rank, :]
    
    # Reconstruct the low-rank delta
    delta_approx = ((U * S.unsqueeze(0)) @ Vh).to(dtype=delta.dtype)
    delta_approx = delta_approx.reshape(B, H, N, d)
    
    return uniform_quantize_4bit(delta_approx)

def simulate_paged_delta_kv(K: torch.Tensor, V: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulates Delta-KV temporal compression by storing only low-rank quantized frame differences
    between sequential tokens.
    """
    B, H, N, d = K.shape
    if N <= 1:
        return K, V
        
    K_hat = torch.zeros_like(K)
    V_hat = torch.zeros_like(V)
    
    K_hat[:, :, 0, :] = K[:, :, 0, :]
    V_hat[:, :, 0, :] = V[:, :, 0, :]
    
    # Compute temporal frame differences
    delta_K = K[:, :, 1:, :] - K[:, :, :-1, :]
    delta_V = V[:, :, 1:, :] - V[:, :, :-1, :]
    
    # Compress deltas
    delta_K_hat = low_rank_svd_quantize(delta_K, rank)
    delta_V_hat = low_rank_svd_quantize(delta_V, rank)
    
    # Reconstruct full KV states from compressed deltas
    K_hat[:, :, 1:, :] = K[:, :, 0:1, :] + torch.cumsum(delta_K_hat, dim=2)
    V_hat[:, :, 1:, :] = V[:, :, 0:1, :] + torch.cumsum(delta_V_hat, dim=2)
    
    return K_hat, V_hat

@ATTENTION_REGISTRY.register("moe_deltakv_paged")
class MoEDeltaKVPagedAttention(nn.Module):
    """
    MoE-DeltaKV-Paged Hybrid Attention Module.
    Combines Mixture-of-Block-Attention (MoBA) with Paged Delta-KV cache compression.
    """
    def __init__(self, config: Any, **kwargs: Any) -> None:
        super().__init__()
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        
        # Architecture hyperparameters
        self.page_size = getattr(config, "page_size", 64)
        self.k_experts = getattr(config, "k_experts", 4)
        self.svd_rank = getattr(config, "svd_rank", 16)
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        # Router weight matrix W_rg for computing gating scores across block centroids
        self.W_rg = nn.Linear(self.head_dim, self.head_dim, bias=False)
        
        self.out_proj = nn.Linear(self.all_head_size, config.hidden_size)
        
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
        
        q = self.query(hidden_states).view(B, N, H, d).transpose(1, 2) # [B, H, N, d]
        k = self.key(hidden_states).view(B, N, H, d).transpose(1, 2)
        v = self.value(hidden_states).view(B, N, H, d).transpose(1, 2)
        
        # 1. Paged Delta-KV Temporal Compression Engine
        k_hat, v_hat = simulate_paged_delta_kv(k, v, self.svd_rank)
        
        num_pages = N // self.page_size
        remainder = N % self.page_size
        
        # Pad sequence if not divisible by page_size for structured block processing
        if remainder != 0:
            pad_len = self.page_size - remainder
            q = F.pad(q, (0, 0, 0, pad_len))
            k_hat = F.pad(k_hat, (0, 0, 0, pad_len))
            v_hat = F.pad(v_hat, (0, 0, 0, pad_len))
            num_pages += 1
            padded_N = N + pad_len
        else:
            padded_N = N

        if num_pages == 0:
            # Fallback to standard attention if sequence is smaller than page size
            scale = d**-0.5
            scores = torch.matmul(q, k_hat.transpose(-2, -1)) * scale
            scores = apply_attention_mask(scores, attention_mask)
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            if head_mask is not None:
                attn = attn * head_mask
            out = torch.matmul(attn, v_hat)
            out = out.transpose(1, 2).contiguous().view(B, padded_N, self.all_head_size)
            out = self.out_proj(out[:, :N, :])
            return out, attn if output_attentions else None

        # Reshape into discrete physical pages
        q_blocks = q.view(B, H, num_pages, self.page_size, d)
        k_blocks = k_hat.view(B, H, num_pages, self.page_size, d)
        v_blocks = v_hat.view(B, H, num_pages, self.page_size, d)
        
        # 2. Top-k MoBA Block Router
        q_bar = q_blocks.mean(dim=3) # [B, H, num_pages, d]
        k_bar = k_blocks.mean(dim=3) # [B, H, num_pages, d]
        
        # Router correlation scores: [B, H, num_pages, num_pages]
        router_logits = torch.matmul(self.W_rg(q_bar), k_bar.transpose(-1, -2)) / math.sqrt(d)
        
        # Apply causal mask: query blocks can only attend to current and past pages
        causal_mask = torch.ones((num_pages, num_pages), device=q.device, dtype=torch.bool).tril(diagonal=0)
        router_logits = router_logits.masked_fill(~causal_mask, float('-inf'))
        
        # Dispatch to top-k active pages
        k_experts = min(self.k_experts, num_pages)
        topk_logits, topk_indices = torch.topk(router_logits, k_experts, dim=-1) # [B, H, num_pages, k_experts]
        
        g_j = F.softmax(topk_logits, dim=-1) # [B, H, num_pages, k_experts]
        
        # 3. Paged Block-Sparse Attention Computation
        expanded_indices = topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, self.page_size, d)
        
        # Gather active Key and Value pages dynamically
        k_blocks_exp = k_blocks.unsqueeze(2).expand(-1, -1, num_pages, -1, -1, -1)
        v_blocks_exp = v_blocks.unsqueeze(2).expand(-1, -1, num_pages, -1, -1, -1)
        
        k_gathered = torch.gather(k_blocks_exp, dim=3, index=expanded_indices) # [B, H, num_pages, k_experts, page_size, d]
        v_gathered = torch.gather(v_blocks_exp, dim=3, index=expanded_indices) # [B, H, num_pages, k_experts, page_size, d]
        
        q_expanded = q_blocks.unsqueeze(3) # [B, H, num_pages, 1, page_size, d]
        
        # Compute exact dot-product over gathered pages: [B, H, num_pages, k_experts, page_size, page_size]
        attn_scores = torch.matmul(q_expanded, k_gathered.transpose(-1, -2)) / math.sqrt(d)
        
        # Apply intra-block causal masks dynamically to avoid future-token leakage
        query_page_idx = torch.arange(num_pages, device=q.device).view(1, 1, num_pages, 1)
        is_current_block = (topk_indices == query_page_idx) # [B, H, num_pages, k_experts]
        is_current_block_exp = is_current_block.unsqueeze(-1).unsqueeze(-1)
        
        intra_causal_mask = torch.ones((self.page_size, self.page_size), device=q.device, dtype=torch.bool).tril(diagonal=0)
        
        attn_scores = attn_scores.masked_fill(
            is_current_block_exp & ~intra_causal_mask.view(1, 1, 1, 1, self.page_size, self.page_size),
            float('-inf')
        )
        
        attn_probs = F.softmax(attn_scores, dim=-1) # [B, H, num_pages, k_experts, page_size, page_size]
        attn_probs = self.dropout(attn_probs)
        
        # Compute context and weight by dynamic gating factors
        context_per_expert = torch.matmul(attn_probs, v_gathered) # [B, H, num_pages, k_experts, page_size, d]
        g_j_exp = g_j.unsqueeze(-1).unsqueeze(-1)
        context = (context_per_expert * g_j_exp).sum(dim=3) # [B, H, num_pages, page_size, d]
        
        # Reshape to dense output format
        context = context.view(B, H, padded_N, d).transpose(1, 2).contiguous().view(B, padded_N, self.all_head_size)
        out = self.out_proj(context)
        
        # Strip padding for final layer output
        out = out[:, :N, :]
        
        return out, None
