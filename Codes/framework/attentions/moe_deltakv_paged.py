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
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
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



def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    while cos.dim() < q.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    if cos.shape[-2] != q.shape[-2]:
        cos = cos[..., : q.shape[-2], :]
        sin = sin[..., : q.shape[-2], :]

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim <= head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_rot_embed = (q_rot * cos.to(q_rot.dtype)) + (_rotate_half(q_rot) * sin.to(q_rot.dtype))
        k_rot_embed = (k_rot * cos.to(k_rot.dtype)) + (_rotate_half(k_rot) * sin.to(k_rot.dtype))
        q = torch.cat([q_rot_embed, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot_embed
        k = torch.cat([k_rot_embed, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot_embed
    else:
        cos_cut, sin_cut = cos[..., :head_dim], sin[..., :head_dim]
        q = (q * cos_cut.to(q.dtype)) + (_rotate_half(q) * sin_cut.to(q.dtype))
        k = (k * cos_cut.to(k.dtype)) + (_rotate_half(k) * sin_cut.to(k.dtype))

    return q, k

@ATTENTION_REGISTRY.register("moe_deltakv_paged", supports_causal=True, supports_gqa=True)
class MoEDeltaKVPagedAttention(nn.Module):
    r"""
    MoE-DeltaKV-Paged Hybrid Attention Module.
    Combines Mixture-of-Block-Attention (MoBA) with Paged Delta-KV cache compression.
    """

    def __init__(self, config: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__()
        self.hidden_size = getattr(config, "hidden_size", kwargs.get("hidden_size", 768))
        self.num_heads: int = getattr(config, "num_attention_heads", kwargs.get("num_attention_heads", 4))
        self.head_dim: int = self.hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim

        self.page_size = getattr(config, "page_size", kwargs.get("page_size", 64))
        self.k_experts = getattr(config, "k_experts", kwargs.get("k_experts", 4))
        self.svd_rank = getattr(config, "svd_rank", kwargs.get("svd_rank", 16))
        self.is_causal = getattr(config, "is_causal", getattr(config, "is_decoder", kwargs.get("is_causal", kwargs.get("is_decoder", False))))
        self.is_decoder = bool(self.is_causal)
        self.is_nested_encoder = False

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

    def copy_weights(self, pretrained_attn: Any) -> None:
        from framework.core.adapters import UniversalAdapter

        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None), "k": (None, None), "v": (None, None), "o": (None, None)
        }
        for proj in ["q", "k", "v", "o"]:
            try:
                w, b = UniversalAdapter.get_weight_bias(pretrained_attn, proj)
                if w is not None:
                    extracted[proj] = (w, b)
            except Exception:
                pass

        if hasattr(pretrained_attn, "c_attn") and (extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None):
            weight = pretrained_attn.c_attn.weight.data
            bias = getattr(pretrained_attn.c_attn, "bias", None)
            bias_data = bias.data if bias is not None else None
            if weight.dim() == 2 and weight.shape[1] == 3 * self.hidden_size:
                weight = weight.t()
            split_size = weight.shape[0] // 3
            extracted["q"] = (weight[:split_size, :], bias_data[:split_size] if bias_data is not None else None)
            extracted["k"] = (weight[split_size : 2 * split_size, :], bias_data[split_size : 2 * split_size] if bias_data is not None else None)
            extracted["v"] = (weight[2 * split_size :, :], bias_data[2 * split_size :] if bias_data is not None else None)

        if hasattr(pretrained_attn, "c_proj") and extracted["o"][0] is None:
            w_o = pretrained_attn.c_proj.weight.data
            if w_o.dim() == 2 and w_o.shape[0] != self.hidden_size and w_o.shape[1] == self.hidden_size:
                w_o = w_o.t()
            b_o = getattr(pretrained_attn.c_proj, "bias", None)
            extracted["o"] = (w_o, b_o.data if b_o is not None else None)

        if hasattr(pretrained_attn, "self") and (extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None):
            s = getattr(pretrained_attn, "self")
            if hasattr(s, "query"):
                extracted["q"] = (s.query.weight.data, getattr(s.query, "bias", None))
            if hasattr(s, "key"):
                extracted["k"] = (s.key.weight.data, getattr(s.key, "bias", None))
            if hasattr(s, "value"):
                extracted["v"] = (s.value.weight.data, getattr(s.value, "bias", None))

        if hasattr(pretrained_attn, "output") and hasattr(getattr(pretrained_attn, "output"), "dense") and extracted["o"][0] is None:
            dense = pretrained_attn.output.dense
            extracted["o"] = (dense.weight.data, getattr(dense, "bias", None))

        named_map = {
            "q": ["q_proj", "q_lin", "query"],
            "k": ["k_proj", "k_lin", "key"],
            "v": ["v_proj", "v_lin", "value"],
            "o": ["o_proj", "out_proj", "out_lin", "dense"],
        }
        for proj, names in named_map.items():
            if extracted[proj][0] is None:
                for name in names:
                    if hasattr(pretrained_attn, name):
                        layer = getattr(pretrained_attn, name)
                        if hasattr(layer, "weight"):
                            extracted[proj] = (layer.weight.data, getattr(layer, "bias", None))
                            break

        cls_name = pretrained_attn.__class__.__name__
        has_out_proj = any(hasattr(pretrained_attn, attr) for attr in ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "wo", "o", "proj"))
        self.is_nested_encoder = (
            cls_name in ("BertSelfAttention", "RobertaSelfAttention")
            or (cls_name.endswith("SelfAttention") and not has_out_proj)
        ) and not has_out_proj

        missing = [k for k in ["q", "k", "v"] if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            raise RuntimeError(f"MoEDeltaKVPagedAttention missing required weights for projections: {missing}")

        for proj_key in ["q", "k", "v"]:
            w = extracted[proj_key][0]
            if w is not None:
                norm = torch.linalg.norm(w.float())
                if norm < 1e-4:
                    raise RuntimeError(f"MoEDeltaKVPagedAttention extracted weight for {proj_key} has near-zero Frobenius norm ({norm.item():.6f})")

        if extracted["o"][0] is not None:
            norm_o = torch.linalg.norm(extracted["o"][0].float())
            if norm_o < 1e-4:
                raise RuntimeError(f"MoEDeltaKVPagedAttention extracted weight for output projection has near-zero Frobenius norm ({norm_o.item():.6f})")

        def _orient(w: torch.Tensor, in_dim: int) -> torch.Tensor:
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t()
            return w

        with torch.no_grad():
            for proj_key, target in [("q", self.query), ("k", self.key), ("v", self.value)]:
                w, b = extracted[proj_key]
                w = _orient(w, self.hidden_size)
                if w.shape[0] < self.all_head_size and self.all_head_size % w.shape[0] == 0:
                    rep = self.all_head_size // w.shape[0]
                    w = w.repeat(rep, 1)
                    if b is not None: b = b.repeat(rep)
                target.weight.data.copy_(w[:self.all_head_size, :min(self.hidden_size, w.shape[1])])
                if b is not None and target.bias is not None:
                    target.bias.data.copy_(b[:self.all_head_size])

            w_o, b_o = extracted["o"]
            if w_o is not None:
                w_o = _orient(w_o, self.all_head_size)
                self.out_proj.weight.data.copy_(w_o[:self.hidden_size, :min(self.all_head_size, w_o.shape[1])])
                if b_o is not None and self.out_proj.bias is not None:
                    self.out_proj.bias.data.copy_(b_o[:self.hidden_size])
            elif self.is_nested_encoder:
                self.out_proj.weight.data.copy_(torch.eye(self.hidden_size, self.all_head_size))
                if self.out_proj.bias is not None:
                    self.out_proj.bias.data.zero_()

        if hasattr(pretrained_attn, "is_decoder"):
            self.is_decoder = bool(pretrained_attn.is_decoder)
            self.is_causal = self.is_decoder

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim

        q = self.query(hidden_states).view(B, N, H, d).transpose(1, 2)
        k = self.key(hidden_states).view(B, N, H, d).transpose(1, 2)
        v = self.value(hidden_states).view(B, N, H, d).transpose(1, 2)

        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None and encoder_hidden_states is None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

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

        # Convert attention_mask to valid boolean mask [B, padded_N]
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                valid_mask = attention_mask.bool() if attention_mask.dtype == torch.bool else attention_mask > 0
            elif attention_mask.dim() == 4:
                valid_mask = attention_mask[:, 0, 0, :] > -10000.0 if attention_mask.min() < -1000 else attention_mask[:, 0, 0, :] > 0
            else:
                valid_mask = torch.ones(B, N, device=q.device, dtype=torch.bool)
        else:
            valid_mask = torch.ones(B, N, device=q.device, dtype=torch.bool)

        if padded_N > N:
            valid_mask = F.pad(valid_mask, (0, padded_N - N), value=False)

        if num_pages <= 2:
            # Full attention path when sequence length fits in 1-2 pages
            scale = 1.0 / math.sqrt(d)
            scores = torch.matmul(q, k_hat.transpose(-2, -1)) * scale
            if attention_mask is not None:
                pad_mask = (~valid_mask[:, :N]).unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(pad_mask, -10000.0)

            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            if head_mask is not None:
                if head_mask.dim() == 1:
                    head_mask = head_mask.view(1, -1, 1, 1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.unsqueeze(1).unsqueeze(-1)
                elif head_mask.dim() == 3:
                    head_mask = head_mask.unsqueeze(1)
                attn = attn * head_mask
            out = torch.matmul(attn, v_hat).transpose(1, 2).contiguous().view(B, N, self.all_head_size)
            self.aux_loss = torch.tensor(0.0, device=q.device)
            if self.is_nested_encoder:
                return out, (attn if output_attentions else None)
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

        # Boost page 0 logit to ensure it is always selected by top-k (CLS preservation)
        router_logits_biased = router_logits.clone()
        router_logits_biased[:, :, :, 0] = router_logits_biased[:, :, :, 0] + 1e4

        k_experts = min(self.k_experts, num_pages)
        _, topk_indices = torch.topk(router_logits_biased, k_experts, dim=-1)
        topk_logits = torch.gather(router_logits, -1, topk_indices)

        g_j = F.softmax(topk_logits, dim=-1)

        # Auxiliary load balancing loss across pages
        router_probs = F.softmax(router_logits, dim=-1)
        expert_mask = F.one_hot(topk_indices, num_classes=num_pages).float()
        density = expert_mask.sum(dim=-2).mean(dim=(0, 1, 2))
        prob_mean = router_probs.mean(dim=(0, 1, 2))
        self.aux_loss = float(num_pages) * torch.sum(density * prob_mean)

        expanded_indices = topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, self.page_size, d)
        k_blocks_exp = k_blocks.unsqueeze(2).expand(-1, -1, num_pages, -1, -1, -1)
        v_blocks_exp = v_blocks.unsqueeze(2).expand(-1, -1, num_pages, -1, -1, -1)

        k_gathered = torch.gather(k_blocks_exp, dim=3, index=expanded_indices)
        v_gathered = torch.gather(v_blocks_exp, dim=3, index=expanded_indices)

        # Mask padding keys across selected pages
        valid_blocks = valid_mask.view(B, 1, num_pages, self.page_size)
        valid_blocks_exp = valid_blocks.unsqueeze(2).expand(-1, -1, num_pages, -1, -1)
        expanded_idx_mask = topk_indices.unsqueeze(-1).expand(-1, -1, -1, -1, self.page_size)
        k_valid = torch.gather(valid_blocks_exp.expand(-1, H, -1, -1, -1), dim=3, index=expanded_idx_mask)

        q_expanded = q_blocks.unsqueeze(3)
        attn_scores = torch.matmul(q_expanded, k_gathered.transpose(-1, -2)) / math.sqrt(d)
        attn_scores = attn_scores.masked_fill(~k_valid.unsqueeze(-2), -10000.0)

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context_per_expert = torch.matmul(attn_probs, v_gathered)
        g_j_exp = g_j.unsqueeze(-1).unsqueeze(-1)
        context = (context_per_expert * g_j_exp).sum(dim=3)

        context = context.view(B, H, padded_N, d).transpose(1, 2).contiguous().view(B, padded_N, self.all_head_size)
        if self.is_nested_encoder:
            out = context[:, :N, :]
        else:
            out = self.out_proj(context[:, :N, :])

        return out, None
