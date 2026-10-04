"""
SALT Attention: Singular Value Adaptation with Low-Rank Tuning.

Parameter-efficient adaptation mechanism for transformer attention projections combining
orthogonal singular vector preservation, singular value rescaling ($\gamma$), and additive low-rank adapters ($A B$).
Hardened for multi-architecture production deployment.
"""

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely retrieve attribute from dict, object, namespace, or None."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions: [-x2, x1], safe for odd dimensions."""
    d = x.shape[-1]
    rotary_dim = 2 * (d // 2)
    if rotary_dim == 0:
        return x
    x_rot = x[..., :rotary_dim]
    half = rotary_dim // 2
    x1 = x_rot[..., :half]
    x2 = x_rot[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    if rotary_dim < d:
        return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)
    return rotated


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding (RoPE) with partial dimension and independent seq-len support."""
    if cos is None or sin is None:
        return q, k

    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(1)
        sin = sin.unsqueeze(0).unsqueeze(1)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    while cos.dim() < q.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    head_dim = q.shape[-1]
    rotary_dim = 2 * (min(cos.shape[-1], head_dim, k.shape[-1]) // 2)
    if rotary_dim == 0:
        return q, k

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    cos_q = cos[..., : q.shape[-2], :rotary_dim].to(q.device, q.dtype)
    sin_q = sin[..., : q.shape[-2], :rotary_dim].to(q.device, q.dtype)

    cos_k = cos[..., : k.shape[-2], :rotary_dim].to(k.device, k.dtype)
    sin_k = sin[..., : k.shape[-2], :rotary_dim].to(k.device, k.dtype)

    q_rot_embed = (q_rot * cos_q) + (_rotate_half(q_rot) * sin_q)
    k_rot_embed = (k_rot * cos_k) + (_rotate_half(k_rot) * sin_k)

    q = torch.cat([q_rot_embed, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot_embed
    k = torch.cat([k_rot_embed, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot_embed
    return q, k


class SALTLinear(nn.Module):
    is_custom_attention: bool = True
    r"""
    Linear projection layer with Fisher-weighted SVD initialization,
    attention-aware singular value modulation (\gamma), and low-rank residual branch (A B).

    Formulation:
        $$W = U_{:, :r} \operatorname{diag}(\Sigma_{1:r} \odot \gamma) V_{1:r, :}^\top + A B$$
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        original_weight: torch.Tensor | None = None,
        original_bias: torch.Tensor | None = None,
        fisher_diag: torch.Tensor | None = None,
        use_fisher: bool = True,
    ) -> None:
        super().__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.rank: int = min(rank, min(in_features, out_features))
        self.use_fisher: bool = use_fisher

        target_dev = original_weight.device if original_weight is not None else None
        target_dtype = original_weight.dtype if original_weight is not None else torch.float32

        if original_weight is not None:
            W = original_weight.float()
        else:
            W = torch.randn(out_features, in_features) * (1.0 / math.sqrt(in_features))

        # Fisher-weighted SVD
        if use_fisher and fisher_diag is not None and fisher_diag.shape[0] == in_features:
            scale = torch.sqrt(fisher_diag.float().clamp(min=1e-6) + 1e-6)
            # Normalize scale to avoid extreme values
            scale = scale / (scale.norm() / math.sqrt(in_features) + 1e-8)
            W_white = W * scale.unsqueeze(0)
        else:
            scale = None
            W_white = W

        U, S, Vt = torch.linalg.svd(W_white, full_matrices=False)

        # Unwhiten Vt
        if scale is not None:
            Vt_unw = Vt / scale.unsqueeze(0)
        else:
            Vt_unw = Vt

        r = self.rank

        self.register_buffer("U", U[:, :r].contiguous().to(device=target_dev, dtype=target_dtype))
        self.register_buffer(
            "Vt", Vt_unw[:r, :].contiguous().to(device=target_dev, dtype=target_dtype)
        )
        self.register_buffer("S", S[:r].contiguous().to(device=target_dev, dtype=target_dtype))

        self.register_buffer(
            "U_init", U[:, :r].contiguous().to(device=target_dev, dtype=target_dtype)
        )
        self.register_buffer(
            "Vt_init",
            Vt_unw[:r, :].contiguous().to(device=target_dev, dtype=target_dtype),
        )
        self.register_buffer("S_init", S[:r].contiguous().to(device=target_dev, dtype=target_dtype))

        self.gamma = nn.Parameter(torch.ones(r, device=target_dev, dtype=target_dtype))
        self.A = nn.Parameter(torch.zeros(out_features, r, device=target_dev, dtype=target_dtype))
        self.B = nn.Parameter(torch.zeros(r, in_features, device=target_dev, dtype=target_dtype))

        if original_bias is not None:
            self.bias = nn.Parameter(
                original_bias.clone().to(device=target_dev, dtype=target_dtype)
            )
        else:
            self.bias = nn.Parameter(
                torch.zeros(out_features, device=target_dev, dtype=target_dtype)
            )

    def init_gamma_from_sensitivity(self, sensitivity: torch.Tensor | None = None) -> None:
        """
        Initialize gamma from attention sensitivity per SVD direction.
        If sensitivity is None, initialize to ones.
        """
        if sensitivity is not None and sensitivity.shape[0] == self.rank:
            s_norm = sensitivity / (sensitivity.mean() + 1e-8)
            self.gamma.data.copy_(
                torch.sqrt(s_norm.clamp(0.1, 10.0)).to(
                    dtype=self.gamma.dtype, device=self.gamma.device
                )
            )
        else:
            self.gamma.data.fill_(1.0)

    def merge_to_linear(self) -> nn.Linear:
        """
        Merge SVD + gamma + LoRA into a single nn.Linear.
        After this, the layer has no LoRA overhead at inference.
        """
        with torch.no_grad():
            adapted_S = (self.S * self.gamma).to(self.U.dtype)
            W = (self.U * adapted_S.unsqueeze(0)) @ self.Vt
            W = W + self.A.to(self.U.dtype) @ self.B.to(self.U.dtype)
            bias = self.bias.clone() if self.bias is not None else None

        new_linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=bias is not None,
            device=W.device,
            dtype=W.dtype,
        )
        new_linear.weight.data.copy_(W.to(new_linear.weight.dtype))
        if bias is not None:
            new_linear.bias.data.copy_(bias.to(new_linear.bias.dtype))
        return new_linear

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix (out_features, in_features)."""
        adapted_S = (self.S * self.gamma).to(self.U.dtype)
        W_svd = (self.U * adapted_S.unsqueeze(0)) @ self.Vt
        W_lora = self.A.to(self.U.dtype) @ self.B.to(self.U.dtype)
        return W_svd + W_lora

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute adapted linear projection efficiently in low-rank factorized form."""
        adapted_S = (self.S * self.gamma).to(dtype=x.dtype)
        # SVD branch: (x @ Vt.T) * adapted_S @ U.T
        x_vt = F.linear(x, self.Vt.to(dtype=x.dtype))
        x_svd = F.linear(x_vt * adapted_S, self.U.to(dtype=x.dtype))
        # LoRA branch: (x @ B.T) @ A.T
        x_lora = F.linear(F.linear(x, self.B.to(dtype=x.dtype)), self.A.to(dtype=x.dtype))
        out = x_svd + x_lora
        if self.bias is not None:
            out = out + self.bias.to(dtype=x.dtype)
        return out

    def freeze_base_weights(self) -> None:
        """Buffers U, S, Vt are non-trainable by construction. Gamma and A, B remain trainable."""
        pass


def _compute_energy_rank(
    S: torch.Tensor,
    threshold: float = 0.95,
    min_r: int = 8,
    max_r: int = 32,
) -> int:
    """
    Find smallest rank r where cumulative energy >= threshold.
    """
    S_sq = S**2
    total = S_sq.sum()
    if total <= 1e-12:
        return min_r
    cum = torch.cumsum(S_sq, dim=0) / total
    idx = torch.where(cum >= threshold)[0]
    r = int(idx[0].item()) + 1 if len(idx) > 0 else max_r
    return max(min_r, min(r, max_r))


@ATTENTION_REGISTRY.register("salt_f_attention", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("salt_f", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("salt_attention", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("salt", supports_causal=True, supports_gqa=True)
class SALTAttention(BaseAttention):
    r"""
    SALT Attention module replacing standard linear projections with SALTLinear.
    Hardened for production across DistilBERT, RoBERTa, GPT-2, and Qwen3.5.
    Supports Fisher-weighted SVD, attention-aware gamma, and adaptive rank (SALT-F).
    """

    supports_causal: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any,
        rank: int | None = None,
        pretrained_attn: nn.Module | None = None,
        is_decoder: bool | None = None,
        use_fisher: bool | None = None,
        adaptive_rank: bool | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.num_heads: int = int(
            _cfg_get(
                config,
                "num_attention_heads",
                _cfg_get(config, "num_heads", kwargs.get("num_attention_heads", 4)),
            )
        )
        self.hidden_size: int = int(_cfg_get(config, "hidden_size", kwargs.get("hidden_size", 768)))
        self.head_dim: int = int(
            _cfg_get(
                config,
                "head_dim",
                self.hidden_size // self.num_heads if self.num_heads > 0 else 64,
            )
        )
        self.all_head_size: int = self.num_heads * self.head_dim

        cfg_kv = _cfg_get(config, "num_key_value_heads", _cfg_get(config, "num_kv_heads", None))
        if cfg_kv is not None:
            self.num_kv_heads = int(cfg_kv)
        elif kwargs.get("num_kv_heads") is not None:
            self.num_kv_heads = int(kwargs["num_kv_heads"])
        else:
            self.num_kv_heads = self.num_heads

        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))
        self.kv_repeat: int = max(1, self.num_heads // self.num_kv_heads)
        self.kv_head_size: int = self.num_kv_heads * self.head_dim

        self.rank: int = (
            rank
            if rank is not None
            else int(
                _cfg_get(
                    config,
                    "salt_rank",
                    _cfg_get(config, "rank", kwargs.pop("rank", 16)),
                )
            )
        )

        # Adaptive rank & Fisher flags
        if adaptive_rank is not None:
            self.adaptive_rank = bool(adaptive_rank)
        else:
            self.adaptive_rank = bool(
                _cfg_get(
                    config,
                    "adaptive_rank",
                    _cfg_get(config, "salt_adaptive_rank", kwargs.get("adaptive_rank", True)),
                )
            )

        if use_fisher is not None:
            self.use_fisher = bool(use_fisher)
        else:
            self.use_fisher = bool(
                _cfg_get(
                    config,
                    "use_fisher",
                    _cfg_get(config, "salt_use_fisher", kwargs.get("use_fisher", True)),
                )
            )

        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            _mt = str(_cfg_get(config, "model_type", "") or "").lower()
            _dec_pfx = (
                "gpt",
                "qwen",
                "llama",
                "mistral",
                "gemma",
                "phi",
                "opt",
                "bloom",
                "falcon",
                "olmo",
                "mpt",
                "stablelm",
            )
            self.is_decoder = bool(
                _cfg_get(config, "is_decoder", False)
                or _cfg_get(config, "causal", False)
                or any(_mt.startswith(p) for p in _dec_pfx)
            )

        self.is_nested_encoder: bool = False

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

        self.proj_ranks = self._get_projection_ranks(pretrained_attn, self.rank)
        self.q_proj = SALTLinear(
            self.hidden_size,
            self.all_head_size,
            self.proj_ranks["q"],
            use_fisher=self.use_fisher,
        )
        self.k_proj = SALTLinear(
            self.hidden_size,
            self.kv_head_size,
            self.proj_ranks["k"],
            use_fisher=self.use_fisher,
        )
        self.v_proj = SALTLinear(
            self.hidden_size,
            self.kv_head_size,
            self.proj_ranks["v"],
            use_fisher=self.use_fisher,
        )
        self.out_proj = SALTLinear(
            self.all_head_size,
            self.hidden_size,
            self.proj_ranks["o"],
            use_fisher=self.use_fisher,
        )

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def _get_projection_ranks(
        self, pretrained_attn: nn.Module | None, base_rank: int
    ) -> dict[str, int]:
        """
        Compute per-projection rank based on SVD energy.
        Returns dict {'q': r_q, 'k': r_k, 'v': r_v, 'o': r_o}.
        """
        ranks = {"q": base_rank, "k": base_rank, "v": base_rank, "o": base_rank}
        if not getattr(self, "adaptive_rank", True):
            return ranks

        if pretrained_attn is None:
            return ranks

        try:
            for proj_key, attr_list in [
                ("q", ["q_proj", "q_lin", "query"]),
                ("k", ["k_proj", "k_lin", "key"]),
                ("v", ["v_proj", "v_lin", "value"]),
                ("o", ["o_proj", "out_proj", "out_lin", "dense"]),
            ]:
                for attr in attr_list:
                    if hasattr(pretrained_attn, attr):
                        layer = getattr(pretrained_attn, attr)
                        target = getattr(layer, "base_layer", layer)
                        if hasattr(target, "weight") and target.weight is not None:
                            W = target.weight.data.float()
                            _, S, _ = torch.linalg.svd(W, full_matrices=False)
                            ranks[proj_key] = _compute_energy_rank(
                                S, 0.95, 8, max(base_rank * 2, 32)
                            )
                        break
        except Exception as e:
            logger.debug("Per-projection rank computation failed: %s", e)

        return ranks

    @property
    def query(self) -> SALTLinear:
        return self.q_proj

    @property
    def key(self) -> SALTLinear:
        return self.k_proj

    @property
    def value(self) -> SALTLinear:
        return self.v_proj

    def copy_weights(self, src: Any) -> None:
        from neuralops.core.adapters import UniversalAdapter

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. UniversalAdapter extraction
        for proj in ["q", "k", "v", "o"]:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                if w is not None:
                    extracted[proj] = (w, b)
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        # 2. GPT-2 Fused c_attn Conv1D handling
        if hasattr(src, "c_attn") and (
            extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None
        ):
            weight = src.c_attn.weight.data
            bias = getattr(src.c_attn, "bias", None)
            bias_data = bias.data if bias is not None else None
            if weight.dim() == 2 and weight.shape[1] == 3 * self.hidden_size:
                weight = weight.t().contiguous()
            split_size = weight.shape[0] // 3
            q_w = weight[:split_size, :]
            k_w = weight[split_size : 2 * split_size, :]
            v_w = weight[2 * split_size :, :]
            q_b = bias_data[:split_size] if bias_data is not None else None
            k_b = bias_data[split_size : 2 * split_size] if bias_data is not None else None
            v_b = bias_data[2 * split_size :] if bias_data is not None else None
            extracted["q"] = (q_w, q_b)
            extracted["k"] = (k_w, k_b)
            extracted["v"] = (v_w, v_b)

        if hasattr(src, "c_proj") and extracted["o"][0] is None:
            w_o = src.c_proj.weight.data
            if (
                w_o.dim() == 2
                and w_o.shape[0] != self.hidden_size
                and w_o.shape[1] == self.hidden_size
            ):
                w_o = w_o.t().contiguous()
            b_o = getattr(src.c_proj, "bias", None)
            extracted["o"] = (w_o, b_o.data if b_o is not None else None)

        # 3. RoBERTa / BERT Nested Encoder handling
        if hasattr(src, "self") and (
            extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None
        ):
            s = src.self
            if hasattr(s, "query"):
                extracted["q"] = (s.query.weight.data, getattr(s.query, "bias", None))
            if hasattr(s, "key"):
                extracted["k"] = (s.key.weight.data, getattr(s.key, "bias", None))
            if hasattr(s, "value"):
                extracted["v"] = (s.value.weight.data, getattr(s.value, "bias", None))

        if hasattr(src, "output") and hasattr(src.output, "dense") and extracted["o"][0] is None:
            dense = src.output.dense
            extracted["o"] = (dense.weight.data, getattr(dense, "bias", None))

        # 4. Modern Decoders (Qwen3.5, LLaMA, Mistral) & DistilBERT
        named_map = {
            "q": ["q_proj", "q_lin", "query"],
            "k": ["k_proj", "k_lin", "key"],
            "v": ["v_proj", "v_lin", "value"],
            "o": ["o_proj", "out_proj", "out_lin", "dense"],
        }
        for proj, names in named_map.items():
            if extracted[proj][0] is None:
                for name in names:
                    if hasattr(src, name):
                        layer = getattr(src, name)
                        if hasattr(layer, "weight"):
                            extracted[proj] = (
                                layer.weight.data,
                                getattr(layer, "bias", None),
                            )
                            break

        # Detect nested encoder architecture
        cls_name = src.__class__.__name__
        has_out_proj = any(
            hasattr(src, attr)
            for attr in (
                "dense",
                "out_proj",
                "c_proj",
                "out_lin",
                "o_proj",
                "wo",
                "o",
                "proj",
            )
        )
        self.is_nested_encoder = (
            cls_name in ("BertSelfAttention", "RobertaSelfAttention")
            or (cls_name.endswith("SelfAttention") and not has_out_proj)
        ) and not has_out_proj

        # Check required weights
        missing = [k for k in ["q", "k", "v"] if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            logger.warning(
                "Attention module copy_weights warning. Proceeding with initialized weights."
            )

        # Frobenius norm validation
        for proj_key in ["q", "k", "v"]:
            w = extracted[proj_key][0]
            if w is not None:
                norm = torch.linalg.norm(w.float())
                if norm < 1e-4:
                    logger.warning(
                        f"SALTAttention extracted weight for {proj_key} has near-zero Frobenius norm ({norm.item():.6f})"
                    )

        if extracted["o"][0] is not None:
            norm_o = torch.linalg.norm(extracted["o"][0].float())
            if norm_o < 1e-4:
                logger.warning(
                    f"SALTAttention extracted weight for output projection has near-zero Frobenius norm ({norm_o.item():.6f})"
                )

        def _orient(w: torch.Tensor, in_dim: int, out_dim: int) -> torch.Tensor:
            """Handle both nn.Linear [out, in] and Conv1D [in, out]."""
            if w.dim() != 2:
                return w
            if w.shape == (out_dim, in_dim):
                return w
            if w.shape == (in_dim, out_dim):
                return w.t().contiguous()
            # fallback
            if w.shape[1] == in_dim:
                return w
            if w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

        # Recalculate adaptive ranks from actual extracted weights if available
        if getattr(self, "adaptive_rank", True):
            for proj_key in ["q", "k", "v", "o"]:
                w_cand = extracted[proj_key][0]
                if w_cand is not None:
                    try:
                        _, S_cand, _ = torch.linalg.svd(w_cand.float(), full_matrices=False)
                        self.proj_ranks[proj_key] = _compute_energy_rank(
                            S_cand, 0.95, 8, max(self.rank * 2, 32)
                        )
                    except Exception:
                        pass

        # Initialize SALTLinear projections
        with torch.no_grad():
            for proj_key, attr, in_dim, out_dim in [
                ("q", "q_proj", self.hidden_size, self.all_head_size),
                ("k", "k_proj", self.hidden_size, self.kv_head_size),
                ("v", "v_proj", self.hidden_size, self.kv_head_size),
                ("o", "out_proj", self.all_head_size, self.hidden_size),
            ]:
                w, b = extracted[proj_key]
                proj_r = self.proj_ranks.get(proj_key, self.rank)
                if w is not None:
                    w_oriented = _orient(w, in_dim, out_dim)
                    src_out, src_in = w_oriented.shape

                    # Handle KV dimension matching for GQA
                    if proj_key in ("k", "v"):
                        if src_out == out_dim:
                            final_w = w_oriented
                            final_b = b
                        elif src_out > out_dim and src_out % out_dim == 0:
                            rep = src_out // out_dim
                            final_w = w_oriented.view(out_dim, rep, -1).mean(dim=1)
                            final_b = b.view(out_dim, rep).mean(dim=1) if b is not None else None
                        elif src_out < out_dim and out_dim % src_out == 0:
                            rep = out_dim // src_out
                            final_w = w_oriented.repeat(rep, 1)
                            final_b = b.repeat(rep) if b is not None else None
                        else:
                            final_w = torch.zeros(out_dim, in_dim, device=w.device, dtype=w.dtype)
                            final_w[: min(out_dim, src_out), : min(in_dim, src_in)] = w_oriented[
                                : min(out_dim, src_out), : min(in_dim, src_in)
                            ]
                            final_b = b[: min(out_dim, len(b))] if b is not None else None
                    else:
                        if src_out == out_dim and src_in == in_dim:
                            final_w = w_oriented
                            final_b = b
                        else:
                            final_w = torch.zeros(out_dim, in_dim, device=w.device, dtype=w.dtype)
                            final_w[: min(out_dim, src_out), : min(in_dim, src_in)] = w_oriented[
                                : min(out_dim, src_out), : min(in_dim, src_in)
                            ]
                            final_b = b[: min(out_dim, len(b))] if b is not None else None

                    new_linear = SALTLinear(
                        in_dim,
                        out_dim,
                        proj_r,
                        original_weight=final_w,
                        original_bias=final_b,
                        use_fisher=self.use_fisher,
                    )
                    new_linear.to(device=w.device, dtype=w.dtype)
                    setattr(self, attr, new_linear)
                elif proj_key == "o" and self.is_nested_encoder:
                    eye_w = torch.eye(
                        out_dim,
                        in_dim,
                        device=self.out_proj.U.device,
                        dtype=self.out_proj.U.dtype,
                    )
                    new_linear = SALTLinear(
                        in_dim,
                        out_dim,
                        proj_r,
                        original_weight=eye_w,
                        original_bias=None,
                        use_fisher=self.use_fisher,
                    )
                    setattr(self, attr, new_linear)

        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_value: Any | None = None,
        output_attentions: bool | None = False,
        output_hidden_states: bool | None = False,
        return_dict: bool | None = True,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, L, D = hidden_states.shape
        H, Dh = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        L_kv = kv_input.shape[1]

        Q = self.q_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2).contiguous()
        K = self.k_proj(kv_input).view(B, L_kv, self.num_kv_heads, Dh).transpose(1, 2).contiguous()
        V = self.v_proj(kv_input).view(B, L_kv, self.num_kv_heads, Dh).transpose(1, 2).contiguous()

        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        if self.num_kv_heads < self.num_heads:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        scale = Dh**-0.5
        eff_mask = encoder_attention_mask if is_cross else attention_mask
        if eff_mask is None and head_mask is None and not output_attentions:
            try:
                _qs = Q.half() if Q.dtype == torch.float32 else Q
                _ks = K.half() if K.dtype == torch.float32 else K
                _vs = V.half() if V.dtype == torch.float32 else V
                _is_causal = getattr(self, "is_decoder", False) and not is_cross and L > 1
                _sdpa_out = torch.nn.functional.scaled_dot_product_attention(
                    _qs,
                    _ks,
                    _vs,
                    is_causal=_is_causal,
                    dropout_p=self.dropout.p if self.training else 0.0,
                )
                _sdpa_out = (
                    _sdpa_out.to(hidden_states.dtype)
                    .transpose(1, 2)
                    .contiguous()
                    .view(B, L, self.all_head_size)
                )
                _sdpa_out = torch.nan_to_num(_sdpa_out, nan=0.0, posinf=1e4, neginf=-1e4)
                out = (
                    _sdpa_out
                    if getattr(self, "is_nested_encoder", False)
                    else self.out_proj(_sdpa_out)
                )
                if out.shape[-1] != hidden_states.shape[-1]:
                    out = (
                        F.pad(out, (0, hidden_states.shape[-1] - out.shape[-1]))
                        if out.shape[-1] < hidden_states.shape[-1]
                        else out[..., : hidden_states.shape[-1]]
                    )
                return out, None
            except Exception:
                pass

        scores = torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous()) * scale

        if self.is_decoder and not is_cross and L > 1:
            causal_mask = (
                torch.triu(
                    torch.full((L, L_kv), float("-inf"), device=device, dtype=torch.float32),
                    diagonal=1,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            scores = scores + causal_mask

        eff_mask = encoder_attention_mask if is_cross else attention_mask
        if eff_mask is not None:
            if eff_mask.dim() == 2:
                scores = scores + (eff_mask.unsqueeze(1).unsqueeze(2).float() - 1.0) * 10000.0
            elif eff_mask.dim() == 3:
                scores = scores + eff_mask.unsqueeze(1).float()
            elif eff_mask.dim() == 4:
                scores = scores + eff_mask.float()

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        attn = F.softmax(scores, dim=-1).to(dtype)
        attn = self.dropout(attn)

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(-1)
            attn = attn * head_mask

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, L, self.all_head_size)
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            out = context
        else:
            out = self.out_proj(context)

        if out.shape[-1] != hidden_states.shape[-1]:
            out = (
                F.pad(out, (0, hidden_states.shape[-1] - out.shape[-1]))
                if out.shape[-1] < hidden_states.shape[-1]
                else out[..., : hidden_states.shape[-1]]
            )

        return out, (attn if output_attentions else None)


@ATTENTION_REGISTRY.register("salt_f")
@ATTENTION_REGISTRY.register("salt_f_attention")
@ATTENTION_REGISTRY.register("salt")
@ATTENTION_REGISTRY.register("salt_attention")
@ATTENTION_REGISTRY.register("salt_inject")
@ATTENTION_REGISTRY.register("salt_attention_inject")
@ATTENTION_REGISTRY.register("inject_salt_attention")
def inject_salt_attention(
    model: nn.Module | None = None,
    layer: Any | None = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Any = None,
    **kwargs: Any,
) -> SALTAttention | nn.Module | None:
    """Safe injection hook for SALTAttention with whole-model and layer support."""
    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config or getattr(target, "config", None)

    # 1. Whole model traversal support
    if target is not None and (
        hasattr(target, "transformer")
        or hasattr(target, "distilbert")
        or hasattr(target, "roberta")
        or hasattr(target, "bert")
        or (hasattr(target, "model") and hasattr(target.model, "layers"))
        or (hasattr(target, "encoder") and hasattr(target.encoder, "layer"))
    ):
        layer_list = None
        if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
            layer_list = target.distilbert.transformer.layer
        elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
            layer_list = target.roberta.encoder.layer
        elif hasattr(target, "bert") and hasattr(target.bert, "encoder"):
            layer_list = target.bert.encoder.layer
        elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
            layer_list = target.transformer.h
        elif hasattr(target, "model") and hasattr(target.model, "layers"):
            layer_list = target.model.layers
        elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
            layer_list = target.encoder.layer

        if layer_list is not None:
            tot = len(layer_list)
            for idx, lyr in enumerate(layer_list):
                inject_salt_attention(
                    layer=lyr,
                    model_config=cfg,
                    comp_config=comp_config,
                    layer_idx=idx,
                    total_layers=tot,
                    **kwargs,
                )
            return target

    # 2. Layer injection
    try:
        src = old_attn
        if src is None and target is not None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(target, attr):
                    src = getattr(target, attr)
                    break

        if src is None:
            return None

        # Double-injection guard
        if isinstance(src, SALTAttention) or "SALT" in src.__class__.__name__:
            return None
        if target is not None:
            for attr in ("attention", "self_attn", "attn", "self"):
                cand = getattr(target, attr, None)
                if cand is not None and (
                    isinstance(cand, SALTAttention) or "SALT" in cand.__class__.__name__
                ):
                    return None

        if model_config is None and hasattr(target, "config"):
            model_config = target.config
        if model_config is None and hasattr(src, "config"):
            model_config = src.config

        rank = kwargs.pop(
            "rank",
            _cfg_get(comp_config, "rank", _cfg_get(comp_config, "salt_rank", 16)),
        )
        if rank is None:
            rank = 16

        new_attn = SALTAttention(
            config=model_config,
            rank=rank,
            pretrained_attn=src,
            **kwargs,
        )

        for attr in ("self_attn", "attn"):
            if hasattr(target, attr):
                setattr(target, attr, new_attn)
                return new_attn
        if hasattr(target, "self"):
            target.self = new_attn
            new_attn.is_nested_encoder = True
            return new_attn
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                target.attention.self = new_attn
                new_attn.is_nested_encoder = True
            else:
                target.attention = new_attn
            return new_attn
        target.attention = new_attn
        return new_attn
    except Exception as e:
        logger.error("inject_salt_attention failed at layer %d: %s", layer_idx, e)
        return None


def merge_model_salt(model: nn.Module) -> nn.Module:
    """
    Traverse model and replace all SALTLinear modules with merged nn.Linear layers.
    Eliminates all LoRA / SVD overhead during inference.
    """
    for name, child in list(model.named_children()):
        if isinstance(child, SALTLinear):
            setattr(model, name, child.merge_to_linear())
        else:
            merge_model_salt(child)
    return model


def calibrate_gamma(
    model: nn.Module,
    calibration_data: Any,
    tokenizer: Any | None = None,
) -> None:
    """
    Run forward+backward on calibration data, capture gradients
    w.r.t. each SVD direction, and initialize gamma accordingly.
    """
    device = next(model.parameters()).device
    if tokenizer is not None and isinstance(calibration_data, list):
        inputs = tokenizer(
            calibration_data,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
    elif isinstance(calibration_data, dict):
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in calibration_data.items()
        }
    elif isinstance(calibration_data, torch.Tensor):
        inputs = calibration_data.to(device)
    else:
        inputs = calibration_data

    # Hook to capture gradients
    grads: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(mod_name):
        def hook(module, grad_in, grad_out):
            if isinstance(grad_out, tuple) and len(grad_out) > 0:
                g = grad_out[0]
                if g is not None:
                    grads[mod_name] = g.detach()

        return hook

    for name, module in model.named_modules():
        if isinstance(module, SALTLinear):
            handles.append(module.register_full_backward_hook(make_hook(name)))

    model.eval()
    try:
        if isinstance(inputs, dict):
            outputs = model(**inputs)
        else:
            outputs = model(inputs)

        loss = None
        if hasattr(outputs, "loss") and outputs.loss is not None:
            loss = outputs.loss
        elif hasattr(outputs, "logits") and outputs.logits is not None:
            probs = F.softmax(outputs.logits.float(), dim=-1)
            loss = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
        elif (
            isinstance(outputs, (tuple, list))
            and len(outputs) > 0
            and isinstance(outputs[0], torch.Tensor)
        ):
            loss = outputs[0].mean()
        elif isinstance(outputs, torch.Tensor):
            loss = outputs.mean()

        if loss is not None:
            loss.backward()
    except Exception as e:
        logger.debug("Gamma calibration backward pass failed: %s", e)

    for h in handles:
        h.remove()

    # Update gamma per layer
    for name, module in model.named_modules():
        if isinstance(module, SALTLinear) and name in grads:
            g = grads[name]  # (..., r) or (..., in_features)
            try:
                if g.shape[-1] == module.rank:
                    s = (g**2).mean(dim=tuple(range(g.dim() - 1)))
                else:
                    U = module.U  # (out, r)
                    g_proj = g @ U  # (..., r)
                    s = (g_proj**2).mean(dim=tuple(range(g_proj.dim() - 1)))

                if s.numel() == module.rank:
                    module.init_gamma_from_sensitivity(s)
            except Exception as e:
                logger.debug("Gamma update failed for %s: %s", name, e)
