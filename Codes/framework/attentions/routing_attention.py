"""Routing Attention: Content-Based Clustering for Sparse Attention – Production-Hardened.

Partitions sequence tokens into discrete clusters, confining attention computation
to intra-cluster tokens without materializing full [B, H, L, L] tensors.

"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


# ===========================================================================
# Config getter
# ===========================================================================

def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


# ===========================================================================
# RoPE (dtype-safe)
# ===========================================================================

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    half = d // 2
    x1 = x[..., :half]
    x2 = x[..., half:2 * half]
    rotated = torch.cat((-x2, x1), dim=-1)
    if d % 2 == 1:
        rotated = torch.cat([rotated, x[..., -1:]], dim=-1)
    return rotated


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if cos is None or sin is None:
        return q, k

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
        raise ValueError(
            f"RoPE seq len mismatch: cos {cos.shape[-2]} vs q {q.shape[-2]}"
        )

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim <= head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        cos_q = cos.to(q_rot.dtype)
        sin_q = sin.to(q_rot.dtype)
        cos_k = cos.to(k_rot.dtype)
        sin_k = sin.to(k_rot.dtype)
        q_rot_embed = (q_rot * cos_q) + (_rotate_half(q_rot) * sin_q)
        k_rot_embed = (k_rot * cos_k) + (_rotate_half(k_rot) * sin_k)
        q = torch.cat([q_rot_embed, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot_embed
        k = torch.cat([k_rot_embed, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot_embed
    else:
        cos_cut = cos[..., :head_dim].to(q.dtype)
        sin_cut = sin[..., :head_dim].to(q.dtype)
        q = (q * cos_cut) + (_rotate_half(q) * sin_cut)
        cos_k = cos[..., :head_dim].to(k.dtype)
        sin_k = sin[..., :head_dim].to(k.dtype)
        k = (k * cos_k) + (_rotate_half(k) * sin_k)

    return q, k


# ===========================================================================
# FactorizedLoRALinear (SVD-factorized + LoRA)
# ===========================================================================

class FactorizedLoRALinear(nn.Module):
    """
    SVD-factorized linear + LoRA adapter.
    Forward: y = up(down(x)) + (lora_B @ lora_A)(x) * scaling
    Base (down, up) frozen; LoRA trainable.
    """
    is_custom_attention = False

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = max(1, min(int(rank), self.in_features, self.out_features))
        self.svd_rank = self.rank
        self.lora_rank = int(lora_rank) if lora_rank > 0 else 0
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0

        factory = {"device": device, "dtype": dtype}
        self.down = nn.Linear(self.in_features, self.rank, bias=False, **factory)
        self.up = nn.Linear(self.rank, self.out_features, bias=bias, **factory)

        self.down.weight.requires_grad = False
        self.up.weight.requires_grad = False
        if self.up.bias is not None:
            self.up.bias.requires_grad = False

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(self.lora_rank, self.in_features, **factory))
            self.lora_B = nn.Parameter(torch.empty(self.out_features, self.lora_rank, **factory))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    @property
    def weight(self) -> torch.Tensor:
        w = torch.matmul(self.up.weight, self.down.weight)
        if self.lora_A is not None and self.lora_B is not None:
            w = w + (self.lora_B @ self.lora_A) * self.lora_scaling
        return w

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.up.bias

    def freeze_base_weights(self) -> None:
        self.down.weight.requires_grad = False
        self.up.weight.requires_grad = False
        if self.up.bias is not None:
            self.up.bias.requires_grad = False
        if self.lora_A is not None:
            self.lora_A.requires_grad = True
        if self.lora_B is not None:
            self.lora_B.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.up(self.down(x))
        if self.lora_A is not None and self.lora_B is not None:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        return out


# ===========================================================================
# RoutingAttention
# ===========================================================================

@ATTENTION_REGISTRY.register("routing_class")
@ATTENTION_REGISTRY.register("routing_attention_class")
class RoutingAttention(BaseAttention):
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any = None,
        rank: Optional[int] = None,
        num_clusters: int = 16,
        n_iters: int = 3,
        pretrained_attn: Optional[nn.Module] = None,
        is_decoder: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        try:
            super().__init__(config, rank or 64)
        except TypeError:
            super().__init__(config)

        comp_config = getattr(config, "compression_config", None) if config else None

        self.dim = int(_cfg_get(config, "hidden_size", kwargs.get("hidden_size", 768)))
        self.num_heads = int(_cfg_get(config, "num_attention_heads",
                                      kwargs.get("num_attention_heads", 12)))
        self.num_kv_heads = int(_cfg_get(config, "num_key_value_heads",
                                         kwargs.get("num_key_value_heads", self.num_heads)))

        if self.num_heads <= 0 or self.dim % self.num_heads != 0:
            raise ValueError(f"dim={self.dim} not divisible by num_heads={self.num_heads}")

        self.head_dim = self.dim // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim
        self.num_clusters = int(_cfg_get(comp_config, "num_clusters",
                                         _cfg_get(config, "routing_num_clusters", num_clusters)))
        self.n_iters = max(1, min(int(n_iters), 10))

        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            mt = str(_cfg_get(config, "model_type", "") or "").lower()
            self.is_decoder = (
                bool(_cfg_get(config, "is_decoder", False))
                or bool(kwargs.get("is_causal", False))
                or any(mt.startswith(p) for p in (
                    "gpt", "qwen", "llama", "mistral", "gemma",
                    "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm"))
                or (pretrained_attn is not None and hasattr(pretrained_attn, "c_attn"))
            )
        self.is_causal = self.is_decoder
        self.is_nested_encoder = False

        self.lora_rank = int(_cfg_get(
            config, "lora_rank",
            _cfg_get(comp_config, "lora_rank", kwargs.get("lora_rank", 16)),
        ))
        self.lora_alpha = float(_cfg_get(
            config, "lora_alpha",
            _cfg_get(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)),
        ))
        self.proj_rank = int(kwargs.get(
            "proj_rank",
            _cfg_get(comp_config, "proj_rank",
                     _cfg_get(config, "proj_rank", 64)),
        ))

        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.supports_gqa = True

        # Factorized projections
        FF = FactorizedLoRALinear
        self.q_proj = FF(self.dim, self.all_head_size, rank=self.proj_rank,
                         lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, bias=True)
        self.k_proj = FF(self.dim, self.all_kv_head_size, rank=self.proj_rank,
                         lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, bias=True)
        self.v_proj = FF(self.dim, self.all_kv_head_size, rank=self.proj_rank,
                         lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, bias=True)
        self.out_proj = FF(self.all_head_size, self.dim, rank=self.proj_rank,
                           lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, bias=True)

        dropout_p = _cfg_get(config, "dropout",
                             _cfg_get(config, "attention_probs_dropout_prob", 0.1))
        self.dropout = nn.Dropout(dropout_p)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    # ------------------------------------------------------------------
    @property
    def query(self): return self.q_proj
    @property
    def key(self): return self.k_proj
    @property
    def value(self): return self.v_proj
    @property
    def q_lin(self): return self.q_proj
    @property
    def k_lin(self): return self.k_proj
    @property
    def v_lin(self): return self.v_proj
    @property
    def out_lin(self): return self.out_proj

    # ------------------------------------------------------------------
    @staticmethod
    def _svd_init(w: torch.Tensor, b: Optional[torch.Tensor],
                  proj: FactorizedLoRALinear) -> None:
        with torch.no_grad():
            W = w.detach().float()
            if W.ndim != 2:
                W = W.reshape(W.size(0), -1)

            out_t, in_t = proj.out_features, proj.in_features
            if W.shape[0] != out_t or W.shape[1] != in_t:
                padded = torch.zeros(out_t, in_t, dtype=W.dtype, device=W.device)
                r = min(out_t, W.shape[0])
                c = min(in_t, W.shape[1])
                padded[:r, :c] = W[:r, :c]
                W = padded

            try:
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            except Exception as e:
                logger.warning("RoutingAttention._svd_init SVD failed: %s", e)
                return

            r = min(proj.rank, U.size(1), Vh.size(0))
            proj.down.weight.data.zero_()
            proj.up.weight.data.zero_()
            proj.down.weight.data[:r, :].copy_(Vh[:r, :].to(proj.down.weight.dtype))
            US = U[:, :r] @ torch.diag(S[:r])
            proj.up.weight.data[:, :r].copy_(US.to(proj.up.weight.dtype))

            if b is not None and proj.up.bias is not None:
                n = min(b.numel(), proj.up.bias.numel())
                proj.up.bias.data[:n].copy_(b[:n].to(proj.up.bias.dtype))
            elif proj.up.bias is not None:
                proj.up.bias.data.zero_()

    # ------------------------------------------------------------------
    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = src.__class__.__name__
        has_out_lin = hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj")
        is_nested = False
        if (hasattr(src, "self") and hasattr(getattr(src, "self"), "query")
                and hasattr(src, "output") and hasattr(getattr(src, "output"), "dense")):
            is_nested = True
        elif (cls_name in ["BertSelfAttention", "RobertaSelfAttention",
                           "XLMRobertaSelfAttention", "ElectraSelfAttention",
                           "DebertaSelfAttention"]
              or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())
              ) and not has_out_lin:
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None), "k": (None, None), "v": (None, None), "o": (None, None),
        }

        # GPT-2 fused c_attn
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_attn_mod = src.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                is_conv = (c_attn_mod.__class__.__name__ == "Conv1D"
                           or base_c.__class__.__name__ == "Conv1D"
                           or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1]))
                if is_conv:
                    c_w = c_w.t().contiguous()
                split_size = c_w.size(0) // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_b = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (c_w[:split_size, :], c_b[:split_size] if c_b is not None else None)
                extracted["k"] = (c_w[split_size:2 * split_size, :],
                                  c_b[split_size:2 * split_size] if c_b is not None else None)
                extracted["v"] = (c_w[2 * split_size:, :],
                                  c_b[2 * split_size:] if c_b is not None else None)

            if hasattr(src, "c_proj") and src.c_proj is not None:
                c_proj_mod = src.c_proj
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    is_conv_o = (c_proj_mod.__class__.__name__ == "Conv1D"
                                 or base_o.__class__.__name__ == "Conv1D"
                                 or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1]))
                    if is_conv_o:
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (o_w, bias_o.data.clone() if bias_o is not None else None)

        # Nested / flat
        elif hasattr(src, "self"):
            s = src
            while hasattr(s, "self") and not any(
                hasattr(s, a) for a in ("query", "key", "value", "q_proj", "k_proj", "v_proj")
            ):
                s = s.self
            for proj_key, sub_name in (("q", "query"), ("k", "key"), ("v", "value")):
                mod = getattr(s, sub_name, None)
                if mod is not None and hasattr(mod, "weight"):
                    extracted[proj_key] = (
                        mod.weight.data.clone(),
                        mod.bias.data.clone() if mod.bias is not None else None,
                    )
            if hasattr(src, "output") and hasattr(getattr(src, "output"), "dense"):
                out_mod = src.output.dense
                extracted["o"] = (
                    out_mod.weight.data.clone(),
                    out_mod.bias.data.clone() if out_mod.bias is not None else None,
                )
        else:
            proj_candidates = {
                "q": ["q_lin", "q_proj", "query"],
                "k": ["k_lin", "k_proj", "key"],
                "v": ["v_lin", "v_proj", "value"],
                "o": ["out_lin", "out_proj", "dense", "o_proj"],
            }
            for proj_key, names in proj_candidates.items():
                for name in names:
                    if hasattr(src, name):
                        mod = getattr(src, name)
                        if mod is not None and hasattr(mod, "weight"):
                            extracted[proj_key] = (
                                mod.weight.data.clone(),
                                mod.bias.data.clone() if mod.bias is not None else None,
                            )
                            break

        missing = [k for k, (w, _) in extracted.items()
                   if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing:
            try:
                from framework.core.adapters import UniversalAdapter
                adapter = UniversalAdapter()
                for k in missing:
                    w, b = adapter.get_weight_bias(src, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug("UniversalAdapter failed: %s", e)

        missing_after = [k for k, (w, _) in extracted.items()
                         if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing_after:
            raise RuntimeError(
                f"RoutingAttention.copy_weights: missing {missing_after} from {cls_name}."
            )

        for proj_key, target in (("q", self.q_proj), ("k", self.k_proj),
                                 ("v", self.v_proj), ("o", self.out_proj)):
            w, b = extracted[proj_key]
            if w is None:
                if proj_key == "o" and self.is_nested_encoder:
                    eye = torch.eye(self.all_head_size, self.all_head_size,
                                    device=target.down.weight.device,
                                    dtype=target.down.weight.dtype)
                    self._svd_init(eye, None, target)
                continue

            # GQA resize for k, v
            if proj_key in ("k", "v"):
                out_t = target.out_features
                src_out = w.shape[0]
                if src_out > out_t and src_out % out_t == 0:
                    rep = src_out // out_t
                    w = w.view(out_t, rep, -1).mean(dim=1)
                    if b is not None:
                        b = b.view(out_t, rep).mean(dim=1)
                elif src_out < out_t and out_t % src_out == 0:
                    rep = out_t // src_out
                    w = w.repeat(rep, 1)
                    if b is not None:
                        b = b.repeat(rep)

            self._svd_init(w, b, target)
            target.freeze_base_weights()

        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)

    # ------------------------------------------------------------------
    # Robust valid-mask helper
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_valid_mask(
        attention_mask: Optional[torch.Tensor],
        B: int, L: int, device: torch.device,
    ) -> torch.Tensor:
        if attention_mask is None:
            return torch.ones(B, L, device=device, dtype=torch.bool)
        if attention_mask.dim() == 2:
            if attention_mask.dtype == torch.bool:
                return attention_mask
            if attention_mask.dtype.is_floating_point:
                return attention_mask > -1e4
            return attention_mask > 0
        if attention_mask.dim() == 3:
            if attention_mask.dtype == torch.bool:
                return attention_mask.squeeze(1)
            if attention_mask.dtype.is_floating_point:
                return (attention_mask > -1e4).squeeze(1)
            return (attention_mask > 0).squeeze(1)
        if attention_mask.dim() == 4:
            if attention_mask.dtype == torch.bool:
                v = attention_mask
            elif attention_mask.dtype.is_floating_point:
                v = attention_mask > -1e4
            else:
                v = attention_mask > 0
            return v.any(dim=-2).squeeze(1)
        raise ValueError(f"Unsupported attention_mask dim: {attention_mask.dim()}")

    # ------------------------------------------------------------------
    # Stable k-means
    # ------------------------------------------------------------------
    @staticmethod
    def _kmeans_assign(feat: torch.Tensor, K: int, n_iters: int) -> torch.Tensor:
        """
        Deterministic k-means assignment on (B, H, L, d).
        Uses evenly-spaced initialization + squared-distance expansion.
        """
        B, H, L, d = feat.shape
        K = min(K, L)
        step = max(1, L // K)
        init_idx = torch.arange(0, L, step, device=feat.device)[:K]
        if init_idx.size(0) < K:
            extra = torch.arange(0, K - init_idx.size(0), device=feat.device)
            init_idx = torch.cat([init_idx, extra])

        feat_f = feat.float()
        centroids = feat_f[:, :, init_idx, :].clone()   # (B, H, K, d)
        cluster_ids = torch.zeros(B, H, L, dtype=torch.long, device=feat.device)

        for _ in range(n_iters):
            # (B, H, L, K) via squared-distance expansion
            x_norm = (feat_f ** 2).sum(dim=-1, keepdim=True)          # (B, H, L, 1)
            c_norm = (centroids ** 2).sum(dim=-1).unsqueeze(-2)         # (B, H, 1, K)
            cross = torch.matmul(feat_f, centroids.transpose(-2, -1))   # (B, H, L, K)
            dist = x_norm - 2 * cross + c_norm
            cluster_ids = torch.argmin(dist, dim=-1)

            one_hot = F.one_hot(cluster_ids, num_classes=K).to(feat_f.dtype)
            counts = one_hot.sum(dim=2).clamp(min=1.0).unsqueeze(-1)
            sum_f = torch.einsum("bhlk,bhld->bhkd", one_hot, feat_f)
            centroids = sum_f / counts

        return cluster_ids

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: Optional[bool] = False,
        layer_idx: int = 0,
        *args: Any,
        **kwargs: Any,
    ):
        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        # --- Projections ---
        Q = self.q_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3).contiguous()
        K = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, d).permute(0, 2, 1, 3).contiguous()
        V = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, d).permute(0, 2, 1, 3).contiguous()

        # --- RoPE ---
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            past_len = 0
            if past_key_value is not None:
                if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 1:
                    past_len = past_key_value[0].shape[-2]
                elif hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
                    past_len = past_key_value.key_cache[0].shape[-2]
            cos = cos[..., past_len: past_len + L, :]
            sin = sin[..., past_len: past_len + L, :]
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        # --- GQA ---
        if self.num_heads != self.num_kv_heads:
            repeats = self.num_heads // self.num_kv_heads
            K = repeat_kv(K, repeats)
            V = repeat_kv(V, repeats)

        # --- Valid mask ---
        valid_mask_2d = self._prepare_valid_mask(attention_mask, B, L, device)  # (B, L) bool

        # --- Clustering ---
        K_num = min(max(1, self.num_clusters), L)
        joint_feat = F.normalize(Q + K, p=2, dim=-1)  # (B, H, L, d)
        cluster_ids = self._kmeans_assign(joint_feat, K_num, self.n_iters)  # (B, H, L) long

        scale = d ** -0.5
        is_causal = self.is_decoder or kwargs.get("is_causal", False)

        # ===========================================================
        # Sort + Gather + Chunk attention (real O(N²/K))
        # ===========================================================
        # For each cluster k:
        #   1. Use sort-trick to gather cluster members into a compact block
        #   2. Compute attention on the small (max_q × max_k) matrix
        #   3. Scatter the result back to the original positions
        # ===========================================================

        outputs = torch.zeros_like(Q)  # (B, H, L, d)
        arange_L = torch.arange(L, device=device)

        for k in range(K_num):
            q_mask_k = (cluster_ids == k)                              # (B, H, L) bool
            # Keys: cluster members + token 0 (global anchor) ∩ valid
            k_mask_k = q_mask_k | (arange_L.view(1, 1, -1) == 0)
            k_mask_k = k_mask_k & valid_mask_2d.unsqueeze(1)           # (B, H, L)

            # Determine compact block sizes
            max_q = int(q_mask_k.sum(dim=-1).max().item())
            max_k = int(k_mask_k.sum(dim=-1).max().item())

            if max_q == 0 or max_k == 0:
                continue

            # Sort trick: cluster members get score 2L - pos, others get -pos
            # → argsort descending puts cluster members first (ordered by position desc)
            q_score = q_mask_k.to(torch.int64) * (2 * L) - arange_L.view(1, 1, L)
            k_score = k_mask_k.to(torch.int64) * (2 * L) - arange_L.view(1, 1, L)

            q_idx = q_score.argsort(dim=-1, descending=True)[:, :, :max_q]  # (B, H, max_q)
            k_idx = k_score.argsort(dim=-1, descending=True)[:, :, :max_k]  # (B, H, max_k)

            # Padding mask (which entries are real cluster members)
            valid_q = torch.gather(q_mask_k, 2, q_idx)                 # (B, H, max_q)
            valid_k = torch.gather(k_mask_k, 2, k_idx)                 # (B, H, max_k)

            # Gather Q, K, V for this cluster
            q_idx_exp = q_idx.unsqueeze(-1).expand(-1, -1, -1, d)
            k_idx_exp = k_idx.unsqueeze(-1).expand(-1, -1, -1, d)

            Q_k = torch.gather(Q, 2, q_idx_exp)                        # (B, H, max_q, d)
            K_k = torch.gather(K, 2, k_idx_exp)                        # (B, H, max_k, d)
            V_k = torch.gather(V, 2, k_idx_exp)                        # (B, H, max_k, d)

            # Compact attention: (B, H, max_q, max_k) — this is O((N/K)²), not O(N²)
            scores_k = torch.matmul(Q_k.float(), K_k.float().transpose(-2, -1).contiguous()) * scale

            # Build pair mask
            pair_mask = valid_q.unsqueeze(-1) & valid_k.unsqueeze(-2)

            # Causal mask via ABSOLUTE positions (preserves causality after sort)
            if is_causal:
                abs_pos_q = q_idx.unsqueeze(-1)                        # (B, H, max_q, 1)
                abs_pos_k = k_idx.unsqueeze(-2)                        # (B, H, 1, max_k)
                causal_mask = abs_pos_q >= abs_pos_k
                pair_mask = pair_mask & causal_mask

            scores_k = scores_k.masked_fill(~pair_mask, -1e4)
            scores_k = torch.nan_to_num(scores_k, nan=-1e4, posinf=1e4, neginf=-1e4)

            attn_k = F.softmax(scores_k, dim=-1).to(dtype)
            attn_k = torch.nan_to_num(attn_k, nan=0.0)
            attn_k = self.dropout(attn_k)

            out_k = torch.matmul(attn_k, V_k)                          # (B, H, max_q, d)

            # Zero out padded queries before scattering
            out_k = out_k * valid_q.unsqueeze(-1).to(out_k.dtype)

            # Scatter back (each token belongs to exactly one cluster → safe)
            outputs.scatter_add_(2, q_idx_exp, out_k.to(outputs.dtype))

        # ===========================================================
        # Post-processing
        # ===========================================================

        # head_mask
        if head_mask is not None:
            if head_mask.dim() == 1:
                hm = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                hm = head_mask.unsqueeze(-1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                hm = head_mask[:, :, :1].unsqueeze(-1)
            else:
                hm = head_mask
            outputs = outputs * hm.to(outputs.dtype)

        # Zero out invalid query positions
        outputs = outputs * valid_mask_2d.unsqueeze(1).unsqueeze(-1).to(outputs.dtype)

        # Reshape
        context = outputs.permute(0, 2, 1, 3).contiguous().view(B, L, self.all_head_size)
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            return context, None

        output = self.out_proj(context)
        output = torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)
        return output, None


# ===========================================================================
# Injection with recursion guard
# ===========================================================================

@ATTENTION_REGISTRY.register("routing")
@ATTENTION_REGISTRY.register("routing_attention")
@ATTENTION_REGISTRY.register("routing_inject")
@ATTENTION_REGISTRY.register("inject_routing")
def inject_routing(
    model: Optional[nn.Module] = None,
    layer: Optional[nn.Module] = None,
    model_config: Optional[Any] = None,
    comp_config: Optional[Any] = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    _depth: int = 0,
    **kwargs: Any,
):
    if _depth > 8:
        logger.warning("inject_routing: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer", None))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    # Whole-model traversal
    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    elif hasattr(target, "h") and isinstance(target.h, (torch.nn.ModuleList, list)):
        layer_list = target.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif hasattr(target, "bert") and hasattr(target.bert, "encoder") and hasattr(target.bert.encoder, "layer"):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (torch.nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_routing(
                layer=lyr, model_config=cfg, comp_config=comp_config,
                layer_idx=idx, total_layers=tot,
                _depth=_depth + 1, **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    # Single-layer injection
    try:
        src = old_attn
        if src is None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(target, attr):
                    cand = getattr(target, attr)
                    if isinstance(cand, nn.Module) and cand is not target:
                        src = cand
                        break
        if src is None:
            src = target

        # Double-injection guard
        if isinstance(src, RoutingAttention) or "RoutingAttention" in src.__class__.__name__:
            return None

        # Skip hybrid layers
        c_name = src.__class__.__name__
        if "DeltaNet" in c_name or "LinearAttention" in c_name:
            return None

        # Pass through num_clusters and n_iters
        num_clusters = kwargs.pop(
            "num_clusters",
            _cfg_get(comp_config, "num_clusters",
                     _cfg_get(cfg, "routing_num_clusters", 16)),
        )
        n_iters = kwargs.pop(
            "n_iters",
            _cfg_get(comp_config, "n_iters", 3),
        )

        new_attn = RoutingAttention(
            config=cfg or getattr(src, "config", None),
            rank=kwargs.pop("rank", _cfg_get(comp_config, "rank", 64)),
            num_clusters=num_clusters,
            n_iters=n_iters,
            pretrained_attn=src if src is not target else None,
            **kwargs,
        )

        # Attach to parent layer
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                new_attn.is_nested_encoder = True
                target.attention.self = new_attn
            else:
                new_attn.is_nested_encoder = False
                target.attention = new_attn
        elif hasattr(target, "self_attn"):
            new_attn.is_nested_encoder = False
            target.self_attn = new_attn
        elif hasattr(target, "attn"):
            new_attn.is_nested_encoder = False
            target.attn = new_attn
        elif hasattr(target, "self"):
            new_attn.is_nested_encoder = True
            target.self = new_attn

        return new_attn
    except Exception as e:
        logger.exception("inject_routing failed at layer %d: %s", layer_idx, e)
        return None


# Aliases
inject_routing_attention = inject_routing
inject_routing_into_distilbert = inject_routing


__all__ = [
    "RoutingAttention",
    "FactorizedLoRALinear",
    "inject_routing",
    "inject_routing_attention",
    "inject_routing_into_distilbert",
    "apply_rotary_pos_emb",
]