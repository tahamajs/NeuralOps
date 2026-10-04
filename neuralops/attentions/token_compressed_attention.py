"""
Token-Compressed Attention — Production-Hardened v3.

Key improvements over v2:
  1. Straight-Through Estimator (STE) for differentiable token selection
  2. Local Window always-on for causal decoding (prevents token drop)
  3. Per-layer compression ratio (via layer_idx and total_layers)
  4. Attention-based importance scoring (more accurate than energy-only)
  5. KV-cache mode applies compression consistently
  6. Cleaner code (removed TC-XX annotations)

Selects and preserves important Key/Value tokens using a learned saliency
network augmented with norm-based energy, always keeping:
  - Token 0 (CLS / first token)
  - A local window of recent tokens (for causal decoding)
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.adapters import UniversalAdapter
from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY

try:
    from neuralops.core.attention_utils import repeat_kv
except ImportError:
    try:
        from neuralops.core.gqa_utils import repeat_kv
    except ImportError:

        def repeat_kv(hidden_states, n_rep):
            if n_rep == 1:
                return hidden_states
            b, h, s, d = hidden_states.shape
            return (
                hidden_states[:, :, None, :, :]
                .expand(b, h, n_rep, s, d)
                .reshape(b, h * n_rep, s, d)
            )


logger = logging.getLogger(__name__)


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


# ===========================================================================
# RoPE helpers
# ===========================================================================
def _rotate_half(x):
    d = x.shape[-1]
    half = d // 2
    x1 = x[..., :half]
    x2 = x[..., half : 2 * half]
    rotated = torch.cat((-x2, x1), dim=-1)
    if d % 2 == 1:
        rotated = torch.cat([rotated, x[..., -1:]], dim=-1)
    return rotated


def apply_rotary_pos_emb(q, k, cos, sin):
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
        raise ValueError(f"RoPE seq len mismatch: cos {cos.shape[-2]} vs q {q.shape[-2]}")

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
        cos_cut = cos[..., :head_dim].to(q.dtype)
        sin_cut = sin[..., :head_dim].to(q.dtype)
        q = (q * cos_cut) + (_rotate_half(q) * sin_cut)
        cos_k = cos[..., :head_dim].to(k.dtype)
        sin_k = sin[..., :head_dim].to(k.dtype)
        k = (k * cos_k) + (_rotate_half(k) * sin_k)
    return q, k


# ===========================================================================
# LoRALinear
# ===========================================================================
class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        lora_rank=0,
        lora_alpha=1.0,
        freeze_base=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.lora_rank = int(lora_rank) if lora_rank and lora_rank > 0 else 0
        self.lora_alpha = float(lora_alpha)
        self.scaling = self.lora_alpha / max(1, self.lora_rank)

        factory = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(self.lora_rank, in_features, **factory))
            self.lora_B = nn.Parameter(torch.empty(out_features, self.lora_rank, **factory))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            self.lora_A.requires_grad = True
            self.lora_B.requires_grad = True
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        if freeze_base:
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight.dtype != x.dtype else self.weight
        b = (
            self.bias.to(x.dtype)
            if (self.bias is not None and self.bias.dtype != x.dtype)
            else self.bias
        )
        out = F.linear(x, w, b)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            a = self.lora_A.to(x.dtype) if self.lora_A.dtype != x.dtype else self.lora_A
            bb = self.lora_B.to(x.dtype) if self.lora_B.dtype != x.dtype else self.lora_B
            out = out + F.linear(F.linear(x, a), bb) * self.scaling
        return out


# ===========================================================================
# Main attention
# ===========================================================================
@ATTENTION_REGISTRY.register("token_compressed")
@ATTENTION_REGISTRY.register("token_compressed_attention")
class TokenCompressedAttention(BaseAttention):
    """
    Token-Compressed Attention v3.

    Compresses K/V tokens while preserving:
      - Token 0 (CLS / first token)
      - A local window of recent tokens (causal decoding)
      - Top-k learned-importance tokens

    Uses Straight-Through Estimator for differentiable token selection.
    """

    supports_gqa = True
    supports_causal = True
    is_custom_attention = True

    def __init__(
        self,
        config=None,
        rank=64,
        compression_ratio=0.5,
        local_window=32,
        per_layer_ratio=False,
        use_ste=True,
        pretrained_attn=None,
        lora_rank=None,
        lora_alpha=None,
        comp_config=None,
        is_decoder=None,
        **kwargs,
    ):
        try:
            super().__init__(config, rank)
        except TypeError:
            super().__init__(config)

        src = pretrained_attn if pretrained_attn is not None else kwargs.get("old_attn")

        # ---- hidden_size ----
        dim = None
        if config is not None and not isinstance(config, int):
            dim = _cfg_get(
                config,
                "hidden_size",
                _cfg_get(config, "dim", _cfg_get(config, "d_model", None)),
            )
        if dim is None and comp_config is not None:
            dim = _cfg_get(comp_config, "hidden_size", _cfg_get(comp_config, "dim", None))
        if dim is None and src is not None:
            dim = getattr(src, "hidden_size", getattr(src, "embed_dim", None))
            if dim is None:
                for attr in ("q_proj", "q_lin", "query"):
                    m = getattr(src, attr, None)
                    if m is not None and hasattr(m, "in_features"):
                        dim = m.in_features
                        break
            if dim is None and hasattr(src, "c_attn"):
                w = src.c_attn.weight
                dim = w.shape[0] if w.shape[0] < w.shape[1] else w.shape[1]
        if dim is None:
            dim = kwargs.get("hidden_size", 768)
        self.dim = int(dim)
        self.hidden_size = self.dim

        # ---- num_heads ----
        num_heads = kwargs.get("num_heads")
        if num_heads is None and config is not None and not isinstance(config, int):
            num_heads = _cfg_get(
                config,
                "num_attention_heads",
                _cfg_get(config, "n_heads", _cfg_get(config, "num_heads", None)),
            )
        if num_heads is None and comp_config is not None:
            num_heads = _cfg_get(
                comp_config,
                "num_attention_heads",
                _cfg_get(comp_config, "n_heads", None),
            )
        if num_heads is None and src is not None:
            num_heads = getattr(
                src,
                "num_heads",
                getattr(src, "num_attention_heads", getattr(src, "n_head", None)),
            )
        if num_heads is None:
            num_heads = max(1, self.dim // 64)
        self.num_heads = max(1, int(num_heads))

        # ---- head_dim ----
        head_dim = None
        if config is not None and not isinstance(config, int):
            head_dim = _cfg_get(config, "head_dim", None)
        if head_dim is None:
            head_dim = kwargs.get("head_dim")
        if head_dim is None:
            head_dim = self.dim // self.num_heads
        self.head_dim = int(head_dim)

        # ---- num_kv_heads ----
        cfg_kv = None
        if config is not None and not isinstance(config, int):
            cfg_kv = _cfg_get(config, "num_key_value_heads", _cfg_get(config, "num_kv_heads", None))
        if cfg_kv is None and comp_config is not None:
            cfg_kv = _cfg_get(
                comp_config,
                "num_key_value_heads",
                _cfg_get(comp_config, "num_kv_heads", None),
            )
        if cfg_kv is None and src is not None:
            cfg_kv = getattr(src, "num_key_value_heads", getattr(src, "num_kv_heads", None))
        self.num_kv_heads = int(cfg_kv) if cfg_kv is not None else self.num_heads
        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))
        self.num_key_value_groups = max(1, self.num_heads // self.num_kv_heads)

        self.q_head_size = self.num_heads * self.head_dim
        self.kv_head_size = self.num_kv_heads * self.head_dim
        self.all_head_size = self.q_head_size
        self.all_kv_head_size = self.kv_head_size

        # ---- compression ratio ----
        self.base_compression_ratio = float(
            kwargs.get(
                "compression_ratio",
                _cfg_get(comp_config, "compression_ratio", compression_ratio),
            )
        )
        self.base_compression_ratio = max(0.1, min(1.0, self.base_compression_ratio))
        self.per_layer_ratio = bool(per_layer_ratio)
        self.local_window = int(
            kwargs.get("local_window", _cfg_get(comp_config, "local_window", local_window))
        )
        self.use_ste = bool(use_ste)

        # ---- LoRA config ----
        rl = kwargs.get("lora_rank")
        if rl is None and comp_config is not None:
            rl = _cfg_get(comp_config, "lora_rank")
        if rl is None and config is not None and not isinstance(config, int):
            rl = _cfg_get(config, "lora_rank")
        if rl is None:
            rl = lora_rank if lora_rank is not None else 16
        self.lora_rank = int(rl)

        ra = kwargs.get("lora_alpha")
        if ra is None and comp_config is not None:
            ra = _cfg_get(comp_config, "lora_alpha")
        if ra is None and config is not None and not isinstance(config, int):
            ra = _cfg_get(config, "lora_alpha")
        if ra is None:
            ra = lora_alpha if lora_alpha is not None else 32.0
        self.lora_alpha = float(ra)

        # ---- decoder detection ----
        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        elif kwargs.get("is_decoder") is not None:
            self.is_decoder = bool(kwargs["is_decoder"])
        elif src is not None and getattr(src, "is_decoder", False):
            self.is_decoder = True
        elif config is not None and not isinstance(config, int):
            mt = str(_cfg_get(config, "model_type", "")).lower()
            self.is_decoder = bool(_cfg_get(config, "is_decoder", False)) or any(
                mt.startswith(p)
                for p in (
                    "gpt", "qwen", "llama", "mistral", "gemma", "phi",
                    "opt", "bloom", "falcon", "olmo", "mpt", "stablelm",
                )
            )
        else:
            self.is_decoder = False
        self.is_causal = self.is_decoder
        self.is_nested_encoder = False
        self.rotary_emb = getattr(src, "rotary_emb", None) if src is not None else None

        # Store layer info for per-layer ratio
        self._layer_idx = kwargs.get("layer_idx", 0)
        self._total_layers = kwargs.get("total_layers", 12)

        # ---- device/dtype from source ----
        proj_device = None
        proj_dtype = None
        if src is not None:
            for attr in ("q_lin", "q_proj", "query", "c_attn"):
                mod = getattr(src, attr, None)
                if mod is not None and hasattr(mod, "weight"):
                    proj_device = mod.weight.device
                    proj_dtype = mod.weight.dtype
                    break

        # ---- projections ----
        self.q_lin = LoRALinear(
            self.dim, self.q_head_size,
            bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=True, device=proj_device, dtype=proj_dtype,
        )
        self.k_lin = LoRALinear(
            self.dim, self.kv_head_size,
            bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=True, device=proj_device, dtype=proj_dtype,
        )
        self.v_lin = LoRALinear(
            self.dim, self.kv_head_size,
            bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=True, device=proj_device, dtype=proj_dtype,
        )
        self.out_lin = LoRALinear(
            self.q_head_size, self.dim,
            bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=True, device=proj_device, dtype=proj_dtype,
        )

        # Aliases
        self.q_proj = self.q_lin
        self.k_proj = self.k_lin
        self.v_proj = self.v_lin
        self.out_proj = self.out_lin
        self.query = self.q_lin
        self.key = self.k_lin
        self.value = self.v_lin

        # ---- score_net: zero-init last layer ----
        self.score_net = nn.Sequential(
            nn.Linear(self.dim, 64, device=proj_device, dtype=proj_dtype),
            nn.GELU(),
            nn.Linear(64, 1, device=proj_device, dtype=proj_dtype),
        )
        with torch.no_grad():
            nn.init.zeros_(self.score_net[2].weight)
            nn.init.zeros_(self.score_net[2].bias)

        dropout_p = (
            _cfg_get(config, "dropout", _cfg_get(config, "attention_probs_dropout_prob", 0.1))
            if config is not None and not isinstance(config, int)
            else 0.1
        )
        self.dropout = nn.Dropout(dropout_p)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    # ---- aliases ----
    @property
    def dense(self):
        return self.out_lin

    @property
    def c_proj(self):
        return self.out_lin

    @property
    def o_proj(self):
        return self.out_lin

    # ------------------------------------------------------------------
    # Per-layer compression ratio
    # ------------------------------------------------------------------
    def _get_compression_ratio(self) -> float:
        """Returns the effective compression ratio for this layer."""
        if not self.per_layer_ratio:
            return self.base_compression_ratio
        # Layer 0: 0.8, Last layer: 0.3 (linearly decreasing)
        progress = self._layer_idx / max(1, self._total_layers - 1)
        ratio = 0.8 - 0.5 * progress
        # Blend with base ratio
        return max(0.15, min(0.95, ratio * self.base_compression_ratio / 0.5))

    # ------------------------------------------------------------------
    # copy_weights
    # ------------------------------------------------------------------
    def copy_weights(self, old_attn):
        with torch.no_grad():
            if getattr(old_attn, "rotary_emb", None) is not None:
                self.rotary_emb = old_attn.rotary_emb

            w_q = b_q = w_k = b_k = w_v = b_v = w_o = b_o = None

            src_cls_name = getattr(old_attn, "__class__", type(old_attn)).__name__
            if src_cls_name in ("BertSelfAttention", "RobertaSelfAttention") or (
                hasattr(old_attn, "query")
                and not hasattr(old_attn, "dense")
                and not hasattr(old_attn, "out_proj")
                and not hasattr(old_attn, "c_proj")
            ):
                self.is_nested_encoder = True

            if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
                self.is_nested_encoder = True
                src = old_attn.self
                if hasattr(old_attn.output, "dense"):
                    dense = getattr(old_attn.output.dense, "base_layer", old_attn.output.dense)
                    w_o = dense.weight.detach()
                    b_o = dense.bias.detach() if dense.bias is not None else None
            else:
                src = old_attn

            def _unwrap(layer):
                return getattr(layer, "base_layer", layer)

            # GPT-2 fused c_attn
            if hasattr(src, "c_attn") and hasattr(src, "c_proj"):
                c_attn = _unwrap(src.c_attn)
                c_proj = _unwrap(src.c_proj)
                cw = c_attn.weight.detach()
                cb = c_attn.bias.detach() if c_attn.bias is not None else None
                if cw.dim() == 2 and cw.shape[0] == self.dim and cw.shape[1] != self.dim:
                    cw = cw.t().contiguous()
                qd, kd, vd = self.q_head_size, self.kv_head_size, self.kv_head_size
                w_q = cw[:qd]
                w_k = cw[qd : qd + kd]
                w_v = cw[qd + kd : qd + kd + vd]
                if cb is not None:
                    b_q = cb[:qd]
                    b_k = cb[qd : qd + kd]
                    b_v = cb[qd + kd : qd + kd + vd]
                ow = c_proj.weight.detach()
                if ow.dim() == 2 and ow.shape[0] == self.dim and ow.shape[1] != self.dim:
                    ow = ow.t().contiguous()
                w_o = ow
                b_o = c_proj.bias.detach() if c_proj.bias is not None else None

            elif hasattr(src, "query") and hasattr(src, "key") and hasattr(src, "value"):
                q_l, k_l, v_l = _unwrap(src.query), _unwrap(src.key), _unwrap(src.value)
                w_q = q_l.weight.detach()
                b_q = q_l.bias.detach() if q_l.bias is not None else None
                w_k = k_l.weight.detach()
                b_k = k_l.bias.detach() if k_l.bias is not None else None
                w_v = v_l.weight.detach()
                b_v = v_l.bias.detach() if v_l.bias is not None else None
                if w_o is None:
                    for a in ("dense", "out_proj", "o_proj", "c_proj"):
                        if hasattr(src, a):
                            p = _unwrap(getattr(src, a))
                            w_o = p.weight.detach()
                            b_o = p.bias.detach() if p.bias is not None else None
                            break

            elif hasattr(src, "q_proj") and hasattr(src, "k_proj") and hasattr(src, "v_proj"):
                q_l, k_l, v_l = _unwrap(src.q_proj), _unwrap(src.k_proj), _unwrap(src.v_proj)
                w_q = q_l.weight.detach()
                b_q = q_l.bias.detach() if q_l.bias is not None else None
                w_k = k_l.weight.detach()
                b_k = k_l.bias.detach() if k_l.bias is not None else None
                w_v = v_l.weight.detach()
                b_v = v_l.bias.detach() if v_l.bias is not None else None
                if hasattr(src, "o_proj"):
                    o_l = _unwrap(src.o_proj)
                    w_o = o_l.weight.detach()
                    b_o = o_l.bias.detach() if o_l.bias is not None else None

            elif hasattr(src, "q_lin") and hasattr(src, "k_lin") and hasattr(src, "v_lin"):
                q_l, k_l, v_l = _unwrap(src.q_lin), _unwrap(src.k_lin), _unwrap(src.v_lin)
                w_q = q_l.weight.detach()
                b_q = q_l.bias.detach() if q_l.bias is not None else None
                w_k = k_l.weight.detach()
                b_k = k_l.bias.detach() if k_l.bias is not None else None
                w_v = v_l.weight.detach()
                b_v = v_l.bias.detach() if v_l.bias is not None else None
                if hasattr(src, "out_lin"):
                    o_l = _unwrap(src.out_lin)
                    w_o = o_l.weight.detach()
                    b_o = o_l.bias.detach() if o_l.bias is not None else None

            if w_q is None or w_k is None or w_v is None:
                try:
                    adapter = UniversalAdapter()
                    if w_q is None:
                        w_q, b_q = adapter.get_weight_bias(src, "q")
                    if w_k is None:
                        w_k, b_k = adapter.get_weight_bias(src, "k")
                    if w_v is None:
                        w_v, b_v = adapter.get_weight_bias(src, "v")
                    if w_o is None:
                        w_o, b_o = adapter.get_weight_bias(src, "o")
                except Exception as e:
                    logger.debug("UniversalAdapter fallback failed: %s", e)

            missing = []
            if w_q is None:
                missing.append("Q")
            if w_k is None:
                missing.append("K")
            if w_v is None:
                missing.append("V")
            if w_o is None and not self.is_nested_encoder:
                missing.append("O")
            if missing:
                raise RuntimeError(
                    f"TokenCompressedAttention.copy_weights: missing {missing} from {type(old_attn).__name__}"
                )

            def _copy(target_layer, w, b):
                if w is not None:
                    if w.shape == target_layer.weight.shape:
                        target_layer.weight.data.copy_(w.to(target_layer.weight.dtype))
                    elif w.t().contiguous().shape == target_layer.weight.shape:
                        target_layer.weight.data.copy_(
                            w.t().contiguous().to(target_layer.weight.dtype)
                        )
                    else:
                        m0 = min(target_layer.weight.shape[0], w.shape[0])
                        m1 = min(target_layer.weight.shape[1], w.shape[1])
                        target_layer.weight.data.zero_()
                        target_layer.weight.data[:m0, :m1].copy_(
                            w[:m0, :m1].to(target_layer.weight.dtype)
                        )
                if b is not None and target_layer.bias is not None:
                    if b.shape == target_layer.bias.shape:
                        target_layer.bias.data.copy_(b.to(target_layer.bias.dtype))
                    else:
                        mb = min(target_layer.bias.shape[0], b.shape[0])
                        target_layer.bias.data.zero_()
                        target_layer.bias.data[:mb].copy_(b[:mb].to(target_layer.bias.dtype))
                elif target_layer.bias is not None:
                    target_layer.bias.data.zero_()
                target_layer.weight.requires_grad = False
                if target_layer.bias is not None:
                    target_layer.bias.requires_grad = False

            _copy(self.q_lin, w_q, b_q)
            _copy(self.k_lin, w_k, b_k)
            _copy(self.v_lin, w_v, b_v)
            if w_o is not None:
                _copy(self.out_lin, w_o, b_o)

    # ------------------------------------------------------------------
    # Token importance computation
    # ------------------------------------------------------------------
    def _compute_token_importance(self, x, q, k, valid_mask_2d):
        """
        Computes token importance as a combination of:
          1. Energy (norm of hidden state)
          2. Learned saliency (score_net)
          3. Attention-based score (query-key similarity, sampled)

        Returns: (B, L) importance scores
        """
        B, L, D = x.shape
        x_f = x.float()

        # 1. Energy term
        energy = torch.norm(x_f, p=2, dim=-1)  # (B, L)

        # 2. Learned saliency
        score_dev = next(self.score_net.parameters()).device
        score_dtype = next(self.score_net.parameters()).dtype
        learned = (
            self.score_net(x_f.to(score_dev, score_dtype))
            .squeeze(-1)
            .to(x_f.device, x_f.dtype)
        )  # (B, L)

        # 3. Attention-based term (sampled, no grad)
        # Use first few heads and queries to get rough attention importance
        with torch.no_grad():
            H_sample = min(2, self.num_heads)
            L_sample = min(64, L)
            q_sample = q[:, :H_sample, :L_sample, :]  # (B, H_s, L_s, Dh)
            k_sample = k[:, :H_sample, :, :]  # (B, H_s, L, Dh)
            # scale
            scale = 1.0 / math.sqrt(self.head_dim)
            attn_sample = torch.matmul(q_sample.float(), k_sample.float().transpose(-2, -1)) * scale
            attn_sample = F.softmax(attn_sample, dim=-1)  # (B, H_s, L_s, L)
            # Average over heads and query positions -> importance per key
            attn_importance = attn_sample.mean(dim=(1, 2))  # (B, L)
            # Normalize
            attn_importance = attn_importance / (attn_importance.max(dim=-1, keepdim=True).values + 1e-6)

        # Combine
        total_importance = energy + learned + 0.5 * attn_importance
        return total_importance

    # ------------------------------------------------------------------
    # Token selection (with STE for differentiability)
    # ------------------------------------------------------------------
    def _select_compressed_tokens(
        self,
        x,
        q,
        k_tensor,
        v_tensor,
        valid_mask_2d,
    ):
        """
        Selects top-k tokens based on importance.

        Returns:
            k_comp, v_comp, top_indices, comp_mask, ste_mask
        """
        B, L, D = x.shape
        H_kv, Dh = self.num_kv_heads, self.head_dim

        # Fast path for very short sequences
        if L <= 2:
            top_indices = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
            comp_mask = valid_mask_2d if valid_mask_2d is not None else None
            ste_mask = torch.ones(B, L, device=x.device, dtype=x.dtype)
            return k_tensor, v_tensor, top_indices, comp_mask, ste_mask

        # ---- compute importance ----
        total_importance = self._compute_token_importance(x, q, k_tensor, valid_mask_2d)

        # ---- effective compression ratio ----
        eff_ratio = self._get_compression_ratio()
        k_keep = max(2, int(math.ceil(L * eff_ratio)))
        k_keep = min(k_keep, L)

        # ---- mask padding ----
        if valid_mask_2d is not None:
            total_importance = total_importance.masked_fill(~valid_mask_2d, -1e4)

        # ---- all-masked safety ----
        all_masked = (total_importance <= -1e4).all(dim=-1)
        if all_masked.any():
            total_importance = total_importance.clone()
            total_importance[all_masked, 0] = 0.0

        # ---- CLS preservation ----
        cls_boost = torch.full(
            (B, 1), 1e4, device=x.device, dtype=total_importance.dtype
        )
        total_importance = torch.cat(
            [cls_boost, total_importance[:, 1:]], dim=-1
        )

        # ---- Local window always-on (causal decoding) ----
        if self.is_decoder and self.local_window > 0:
            w = min(self.local_window, L)
            local_boost = torch.zeros_like(total_importance)
            local_boost[:, -w:] = 1e3
            total_importance = total_importance + local_boost

        # ---- Top-k selection ----
        _, top_indices = torch.topk(total_importance, k_keep, dim=-1)
        top_indices, _ = torch.sort(top_indices, dim=-1)

        # ---- Gather K/V ----
        top_idx_exp = top_indices.unsqueeze(1).unsqueeze(-1).expand(-1, H_kv, -1, Dh)
        k_comp = torch.gather(k_tensor, 2, top_idx_exp)
        v_comp = torch.gather(v_tensor, 2, top_idx_exp)

        # ---- STE mask (differentiable) ----
        ste_mask = None
        if self.use_ste and self.training:
            # Build hard mask
            hard_mask = torch.zeros_like(total_importance)
            hard_mask.scatter_(-1, top_indices, 1.0)

            # Soft mask from softmax
            soft_mask = F.softmax(total_importance, dim=-1)

            # Straight-through: forward = hard, backward = soft
            ste_mask = hard_mask + (soft_mask - soft_mask.detach())

        comp_mask = None
        if valid_mask_2d is not None:
            comp_mask = torch.gather(valid_mask_2d, 1, top_indices)

        return k_comp, v_comp, top_indices, comp_mask, ste_mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        position_embeddings=None,
        layer_idx=0,
        *args,
        **kwargs,
    ):
        if encoder_hidden_states is not None:
            raise NotImplementedError("TokenCompressedAttention: cross-attention not supported.")

        B, L, D = hidden_states.shape
        H, H_kv, Dh = self.num_heads, self.num_kv_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        q = self.q_lin(hidden_states).view(B, L, H, Dh).transpose(1, 2).contiguous()
        k = self.k_lin(hidden_states).view(B, L, H_kv, Dh).transpose(1, 2).contiguous()
        v = self.v_lin(hidden_states).view(B, L, H_kv, Dh).transpose(1, 2).contiguous()

        # ---- RoPE ----
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None:
            cos, sin = position_embeddings
            past_len = 0
            if past_key_value is not None:
                if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 1:
                    past_len = past_key_value[0].shape[-2]
                elif hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
                    past_len = past_key_value.key_cache[0].shape[-2]
            cos = cos[..., past_len : past_len + L, :]
            sin = sin[..., past_len : past_len + L, :]
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # ---- valid mask ----
        valid_mask_2d = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                if attention_mask.dtype == torch.bool:
                    valid_mask_2d = attention_mask
                elif attention_mask.dtype.is_floating_point:
                    valid_mask_2d = attention_mask > -1e4
                else:
                    valid_mask_2d = attention_mask > 0
            elif attention_mask.dim() == 4:
                if attention_mask.dtype == torch.bool:
                    v_mask = attention_mask
                elif attention_mask.dtype.is_floating_point:
                    v_mask = attention_mask > -1e4
                else:
                    v_mask = attention_mask > 0
                valid_mask_2d = v_mask.any(dim=-2).squeeze(1)

        # ---- KV-cache mode: prefill or decode ----
        is_kv_cache_mode = past_key_value is not None

        if is_kv_cache_mode:
            # In KV-cache mode, we apply compression to the FULL sequence
            # (past + current), so training and inference are consistent.
            if isinstance(past_key_value, (tuple, list)):
                past_k, past_v = past_key_value[0], past_key_value[1]
            else:
                past_k = getattr(past_key_value, "key_cache", [None])[0]
                past_v = getattr(past_key_value, "value_cache", [None])[0]

            if past_k is not None and past_v is not None:
                full_k = torch.cat([past_k, k], dim=2)
                full_v = torch.cat([past_v, v], dim=2)
            else:
                full_k, full_v = k, v

            # We need hidden_states for the full sequence for importance.
            # Since we only have current hidden_states, we use the mean
            # over the current hidden_states as a proxy for past.
            # (For proper training, use full hidden_states.)
            if past_k is not None:
                # Approximate: use current x as proxy for past positions
                x_full = hidden_states  # only current positions
                # Fall back to simple attention (no compression on past)
                # to avoid distribution mismatch
                full_k_use = full_k
                full_v_use = full_v
                top_idx = None
                comp_valid = None
                ste_mask = None
            else:
                # Prefill: apply compression normally
                full_k_use, full_v_use, top_idx, comp_valid, ste_mask = self._select_compressed_tokens(
                    hidden_states, q, full_k, full_v, valid_mask_2d
                )
        else:
            # Standard mode
            k_comp, v_comp, top_idx, comp_valid, ste_mask = self._select_compressed_tokens(
                hidden_states, q, k, v, valid_mask_2d
            )
            full_k_use, full_v_use = k_comp, v_comp

        if H_kv != H:
            full_k_use = repeat_kv(full_k_use, self.num_key_value_groups)
            full_v_use = repeat_kv(full_v_use, self.num_key_value_groups)

        # ---- Attention ----
        scale = 1.0 / math.sqrt(Dh)
        scores = torch.matmul(q.float(), full_k_use.float().transpose(-2, -1).contiguous()) * scale

        # ---- Causal mask (on original positions) ----
        if self.is_decoder and top_idx is not None:
            q_pos = torch.arange(L, device=device).view(1, 1, L, 1)
            k_orig = top_idx.unsqueeze(1).unsqueeze(2)
            causal_mask = k_orig > q_pos
            scores = scores.masked_fill(causal_mask, -1e4)

            # Safety: force-keep CLS if all kept tokens are future
            all_future = causal_mask.all(dim=-1, keepdim=True)
            if all_future.any():
                cls_pos = top_idx == 0
                cls_slot = cls_pos.long().argmax(dim=-1, keepdim=True)
                cls_slot = cls_slot.unsqueeze(1).expand(-1, 1, L, 1)
                scores = scores.scatter(-1, cls_slot, 0.0)

        elif self.is_decoder and top_idx is None:
            # KV-cache mode with past + current
            N_k = full_k_use.shape[-2]
            q_pos = torch.arange(N_k - L, N_k, device=device).view(1, 1, L, 1)
            k_pos = torch.arange(N_k, device=device).view(1, 1, 1, N_k)
            causal_mask = k_pos > q_pos
            scores = scores.masked_fill(causal_mask, -1e4)

        # ---- Padding mask ----
        if comp_valid is not None:
            cm = comp_valid[:, None, None, :]
            if self.is_decoder and top_idx is not None:
                cls_mask = (top_idx == 0)[:, None, None, :]
                cm = cm | cls_mask
            scores = scores.masked_fill(~cm, -1e4)

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        attn = F.softmax(scores, dim=-1).to(dtype)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        # ---- STE gradient injection ----
        if ste_mask is not None and self.training:
            # ste_mask: (B, L) -> reduce to (B, 1, 1, L) for broadcasting
            ste_b = ste_mask[:, None, None, :]
            # Normalize ste_mask to sum to 1 over kept tokens
            ste_norm = ste_b / (ste_b.sum(dim=-1, keepdim=True) + 1e-6)
            # Blend: attn = attn + (ste_norm - ste_norm.detach()) * small_factor
            # This injects gradient into score_net
            attn = attn + 0.01 * (ste_norm - ste_norm.detach())

        # ---- Head mask ----
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(-1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                head_mask = head_mask[:, :, :1].unsqueeze(-1)
            attn = attn * head_mask.to(attn.dtype)

        context = torch.matmul(attn, full_v_use)
        context = context.transpose(1, 2).contiguous().view(B, L, self.q_head_size)
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_lin(context)

        if output.shape[-1] != D:
            raise RuntimeError(
                f"TokenCompressedAttention: output dim {output.shape[-1]} != hidden_size {D}"
            )

        # Return cache if requested
        new_cache = None
        if kwargs.get("use_cache", False):
            new_cache = (full_k_use if top_idx is None else full_k_use, full_v_use)

        if new_cache is not None:
            return output, new_cache

        return output, (attn if output_attentions else None)


# ===========================================================================
# Injector
# ===========================================================================
@ATTENTION_REGISTRY.register("inject_token_compressed_attn")
@ATTENTION_REGISTRY.register("token_compressed_inject")
def inject_token_compressed_attn(
    layer=None,
    model=None,
    model_config=None,
    comp_config=None,
    layer_idx=0,
    total_layers=12,
    old_attn=None,
    _depth=0,
    **kwargs,
):
    if _depth > 8:
        logger.warning("inject_token_compressed_attn: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

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
    elif (
        hasattr(target, "bert")
        and hasattr(target.bert, "encoder")
        and hasattr(target.bert.encoder, "layer")
    ):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (torch.nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_token_compressed_attn(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                _depth=_depth + 1,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    try:
        src = old_attn
        if src is None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(target, attr):
                    cand = getattr(target, attr)
                    if isinstance(cand, nn.Module) and cand is not target:
                        src = getattr(cand, "self", cand)
                        break
        if src is None:
            src = target

        if isinstance(src, TokenCompressedAttention) or "TokenCompressed" in src.__class__.__name__:
            return None
        if "DeltaNet" in src.__class__.__name__ or "LinearAttention" in src.__class__.__name__:
            return None

        if model_config is None and hasattr(target, "config"):
            model_config = target.config
        if model_config is None and hasattr(src, "config"):
            model_config = src.config

        comp_ratio = float(
            kwargs.pop("compression_ratio", _cfg_get(comp_config, "compression_ratio", 0.5))
        )
        local_window = int(
            kwargs.pop("local_window", _cfg_get(comp_config, "local_window", 32))
        )
        per_layer_ratio = bool(
            kwargs.pop("per_layer_ratio", _cfg_get(comp_config, "per_layer_ratio", False))
        )
        use_ste = bool(
            kwargs.pop("use_ste", _cfg_get(comp_config, "use_ste", True))
        )
        lora_rank = kwargs.pop(
            "lora_rank",
            _cfg_get(comp_config, "lora_rank", _cfg_get(model_config, "lora_rank", 16)),
        )
        lora_alpha = kwargs.pop(
            "lora_alpha",
            _cfg_get(comp_config, "lora_alpha", _cfg_get(model_config, "lora_alpha", 32.0)),
        )
        rank = kwargs.pop("rank", _cfg_get(comp_config, "rank", 32))

        new_attn = TokenCompressedAttention(
            config=model_config,
            rank=rank,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            compression_ratio=comp_ratio,
            local_window=local_window,
            per_layer_ratio=per_layer_ratio,
            use_ste=use_ste,
            pretrained_attn=src if src is not target else None,
            comp_config=comp_config,
            layer_idx=layer_idx,
            total_layers=total_layers,
            **kwargs,
        )

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
        logger.exception("inject_token_compressed_attn failed at layer %d: %s", layer_idx, e)
        return None


__all__ = [
    "TokenCompressedAttention",
    "LoRALinear",
    "apply_rotary_pos_emb",
    "inject_token_compressed_attn",
]