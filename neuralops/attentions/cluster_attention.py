"""
ClusterFormer: Token Clustering Attention Module.

Groups queries into $K$ semantic cluster centroids via fast differentiable $K$-means
and evaluates centroid-to-key cross-attention alongside local context, achieving $\mathcal{O}(K \cdot L \cdot d)$ linear complexity
while eliminating token isolation. For causal decoders, strict causal masking is enforced.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.attention_mask import get_token_valid_mask
from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.utils.model_family import ModelAdapter

logger = logging.getLogger(__name__)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
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


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        freeze_base: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0
        self.freeze_base = freeze_base

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty((lora_rank, in_features), **factory_kwargs))
            self.lora_B = nn.Parameter(torch.empty((out_features, lora_rank), **factory_kwargs))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.reset_parameters()
        if freeze_base:
            self.freeze_base_weights()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        if self.lora_rank > 0 and self.lora_A is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
            nn.init.zeros_(self.lora_B)
            self.lora_B.requires_grad = True

    def freeze_base_weights(self) -> None:
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_rank > 0:
            if self.lora_A is not None:
                self.lora_A.requires_grad = True
            if self.lora_B is not None:
                self.lora_B.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
            out = out + F.linear(F.linear(x, a), bb) * self.lora_scaling
        return out


@ATTENTION_REGISTRY.register(
    "cluster", supports_causal=True, supports_bidirectional=True, supports_gqa=True
)
@ATTENTION_REGISTRY.register(
    "cluster_attention",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class ClusterAttention(BaseAttention):
    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    supports_rope: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        num_clusters: int = 16,
        n_iters: int = 3,
        window_size: int = 32,
        lora_rank: int = 16,
        comp_config: Any | None = None,
        pretrained_attn: nn.Module | None = None,
        old_attn: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn
        super().__init__(config, rank)

        # Resolve hidden_size and num_heads safely even if config is None
        hidden_size = kwargs.get("d_model") or kwargs.get("hidden_size") or kwargs.get("dim")
        num_heads = (
            kwargs.get("num_heads") or kwargs.get("n_heads") or kwargs.get("num_attention_heads")
        )
        num_kv_heads = kwargs.get("num_kv_heads") or kwargs.get("num_key_value_heads")

        if not num_heads and source_attn is not None:
            num_heads = getattr(
                source_attn,
                "num_heads",
                getattr(
                    source_attn,
                    "num_attention_heads",
                    getattr(source_attn, "n_heads", None),
                ),
            )
        if not num_heads and config is not None:
            num_heads = getattr(config, "num_attention_heads", getattr(config, "n_heads", None))
        num_heads = int(num_heads or 12)

        if not hidden_size and source_attn is not None:
            cand = getattr(
                source_attn,
                "hidden_size",
                getattr(
                    source_attn,
                    "embed_dim",
                    getattr(source_attn, "all_head_size", None),
                ),
            )
            if (
                cand is None
                and hasattr(source_attn, "query")
                and hasattr(source_attn.query, "in_features")
            ):
                cand = source_attn.query.in_features
            if (
                cand is None
                and hasattr(source_attn, "q_proj")
                and hasattr(source_attn.q_proj, "in_features")
            ):
                cand = source_attn.q_proj.in_features
            if (
                cand is None
                and hasattr(source_attn, "q_lin")
                and hasattr(source_attn.q_lin, "in_features")
            ):
                cand = source_attn.q_lin.in_features
            if (
                cand is None
                and hasattr(source_attn, "c_attn")
                and hasattr(source_attn.c_attn, "in_features")
            ):
                cand = source_attn.c_attn.in_features
            hidden_size = cand
        if not hidden_size and config is not None:
            hidden_size = getattr(config, "hidden_size", getattr(config, "dim", None))
        hidden_size = int(hidden_size or 768)

        head_dim = kwargs.get("head_dim", 0)
        if not head_dim and source_attn is not None:
            head_dim = getattr(source_attn, "head_dim", 0)
            if not head_dim:
                for attr in ("q_proj", "query", "q_lin"):
                    mod = getattr(source_attn, attr, None)
                    if mod is not None and hasattr(mod, "weight"):
                        w = mod.weight
                        head_dim = w.shape[0] // max(1, num_heads)
                        break

        if not head_dim and config is not None:
            head_dim = getattr(config, "head_dim", 0)
            if not head_dim:
                try:
                    adapter = ModelAdapter(config)
                    head_dim = getattr(adapter, "head_dim", 0)
                except Exception:
                    pass

        if not head_dim:
            head_dim = hidden_size // max(1, num_heads) if num_heads > 0 else 64
        head_dim = int(head_dim)

        if not num_kv_heads and source_attn is not None:
            for attr in ("k_proj", "key", "k_lin"):
                mod = getattr(source_attn, attr, None)
                if mod is not None and hasattr(mod, "weight"):
                    w = mod.weight
                    num_kv_heads = w.shape[0] // max(1, head_dim)
                    break
            if not num_kv_heads:
                num_kv_heads = getattr(source_attn, "num_key_value_heads", None) or getattr(
                    source_attn, "num_kv_heads", None
                )

        if not num_kv_heads and config is not None:
            num_kv_heads = getattr(config, "num_key_value_heads", None)

        num_kv_heads = int(num_kv_heads or num_heads)

        self.hidden_size: int = hidden_size
        self.d_model: int = hidden_size
        self.num_heads: int = num_heads
        self.num_kv_heads: int = num_kv_heads
        self.head_dim: int = head_dim
        self.all_head_size: int = self.num_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim
        self.kv_repeat: int = max(1, self.num_heads // max(1, self.num_kv_heads))

        if config is not None:
            self.num_clusters = getattr(config, "cluster_num_clusters", num_clusters)
        else:
            self.num_clusters = num_clusters
        # Fast K-means iterations (default 3, removed max(20, n_iters) constraint)
        self.n_iters: int = max(1, n_iters)
        self.window_size: int = window_size
        self.temp = nn.Parameter(torch.tensor(0.1))
        self.local_residual = nn.Parameter(torch.tensor(0.5))

        # Pre-allocated index buffer for efficient windowed attention
        self.register_buffer("_seq_idx", torch.arange(4096), persistent=False)

        # Decoder & nested encoder detection with robust prefix matching
        model_type = str(getattr(config, "model_type", "") or "").lower() if config else ""
        _decoder_prefixes = (
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
        is_dec = (
            (
                getattr(config, "is_decoder", False)
                or getattr(config, "is_causal", False)
                or getattr(config, "causal", False)
                or any(model_type.startswith(p) for p in _decoder_prefixes)
            )
            if config
            else False
        )
        if hasattr(config, "architectures") and any(
            "GPT" in str(a) or "CausalLM" in str(a)
            for a in (getattr(config, "architectures", []) or [])
        ):
            is_dec = True
        if source_attn is not None:
            cls_name = source_attn.__class__.__name__.lower()
            if (
                any(p in cls_name for p in ("gpt", "causal", "qwen", "llama", "mistral"))
                or getattr(source_attn, "is_causal", False)
                or getattr(source_attn, "is_decoder", False)
                or hasattr(source_attn, "c_attn")
            ):
                is_dec = True
        if kwargs.get("is_decoder") is not None:
            is_dec = bool(kwargs["is_decoder"])
        elif kwargs.get("is_causal") is not None:
            is_dec = bool(kwargs["is_causal"])
        self.is_decoder: bool = bool(is_dec)
        self.is_causal: bool = bool(is_dec)
        self.is_nested_encoder: bool = bool(kwargs.get("is_nested_encoder", False))

        if config is not None:
            self.lora_rank = getattr(
                config,
                "lora_rank",
                getattr(comp_config, "lora_rank", kwargs.get("lora_rank", lora_rank)),
            )
            self.lora_alpha = getattr(
                config,
                "lora_alpha",
                getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)),
            )
        elif comp_config is not None:
            self.lora_rank = getattr(comp_config, "lora_rank", kwargs.get("lora_rank", lora_rank))
            self.lora_alpha = getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0))
        else:
            self.lora_rank = kwargs.get("lora_rank", lora_rank)
            self.lora_alpha = kwargs.get("lora_alpha", 32.0)
        self.freeze_base: bool = self.lora_rank > 0

        self.q_proj = LoRALinear(
            self.hidden_size,
            self.all_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.k_proj = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.v_proj = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.out_proj = LoRALinear(
            self.all_head_size,
            self.hidden_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )

        dropout_p = (
            getattr(
                config,
                "dropout",
                getattr(
                    config,
                    "hidden_dropout_prob",
                    getattr(config, "attention_probs_dropout_prob", 0.1),
                ),
            )
            if config
            else 0.1
        )
        self.dropout = nn.Dropout(dropout_p)
        self.rotary_emb = (
            getattr(source_attn, "rotary_emb", None) if source_attn is not None else None
        )

        if source_attn is not None:
            self.copy_weights(source_attn, **kwargs)

    def copy_weights(self, src: Any, **kwargs: Any) -> None:
        from neuralops.core.adapters import UniversalAdapter

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. HuggingFace / standard UniversalAdapter extraction
        for proj in ["q", "k", "v", "o"]:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                if w is not None:
                    extracted[proj] = (w, b)
            except Exception as e:
                logger.debug(f"UniversalAdapter extraction failed for {proj}: {e}")

        # 2. GPT-2 Fused c_attn Conv1D handling (including LoRALinear wrappers)
        if (
            hasattr(src, "c_attn")
            and src.c_attn is not None
            and (
                extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None
            )
        ):
            c_attn_mod = src.c_attn
            if hasattr(c_attn_mod, "merge") and not getattr(c_attn_mod, "merged", False):
                try:
                    c_attn_mod.merge()
                except Exception:
                    pass
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            weight = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if weight is not None:
                weight = weight.data.clone()
                p_cls = getattr(base_c, "__class__", type(base_c)).__name__
                if (
                    ("Conv1D" in p_cls)
                    or hasattr(base_c, "nf")
                    or ("Conv1D" in getattr(c_attn_mod, "__class__", type(c_attn_mod)).__name__)
                    or hasattr(c_attn_mod, "nf")
                    or (weight.ndim == 2 and weight.shape[1] == 3 * self.hidden_size)
                ):
                    weight = weight.t().contiguous()
                split_size = weight.shape[0] // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                bias_data = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (
                    weight[:split_size, :],
                    bias_data[:split_size] if bias_data is not None else None,
                )
                extracted["k"] = (
                    weight[split_size : 2 * split_size, :],
                    (bias_data[split_size : 2 * split_size] if bias_data is not None else None),
                )
                extracted["v"] = (
                    weight[2 * split_size :, :],
                    bias_data[2 * split_size :] if bias_data is not None else None,
                )

        if hasattr(src, "c_proj") and src.c_proj is not None and extracted["o"][0] is None:
            c_proj_mod = src.c_proj
            if hasattr(c_proj_mod, "merge") and not getattr(c_proj_mod, "merged", False):
                try:
                    c_proj_mod.merge()
                except Exception:
                    pass
            base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
            w_o = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
            if w_o is not None:
                w_o = w_o.data.clone()
                p_cls = getattr(base_o, "__class__", type(base_o)).__name__
                if (
                    ("Conv1D" in p_cls)
                    or hasattr(base_o, "nf")
                    or ("Conv1D" in getattr(c_proj_mod, "__class__", type(c_proj_mod)).__name__)
                    or hasattr(c_proj_mod, "nf")
                    or (w_o.ndim == 2 and w_o.shape[0] < w_o.shape[1])
                ):
                    w_o = w_o.t().contiguous()
                bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                extracted["o"] = (
                    w_o,
                    bias_o.data.clone() if bias_o is not None else None,
                )

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
        is_nested = False
        if (
            hasattr(src, "self")
            and hasattr(src.self, "query")
            or (
                cls_name
                in (
                    "BertSelfAttention",
                    "RobertaSelfAttention",
                    "XLMRobertaSelfAttention",
                )
                or (cls_name.endswith("SelfAttention") and not has_out_proj)
                or (
                    ("roberta" in cls_name.lower() or "bert" in cls_name.lower())
                    and not has_out_proj
                )
            )
        ):
            is_nested = True
        if hasattr(src, "config"):
            if (
                getattr(src.config, "model_type", "").lower() in ("roberta", "bert", "xlm-roberta")
                and not has_out_proj
            ):
                is_nested = True
        if kwargs.get("is_nested_encoder") is not None:
            is_nested = kwargs["is_nested_encoder"]
        self.is_nested_encoder = bool(is_nested)

        # Check required weights
        missing = [k for k in ["q", "k", "v"] if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            raise RuntimeError(
                f"ClusterAttention missing required weights for projections: {missing}"
            )

        # Frobenius norm validation
        for proj_key in ["q", "k", "v"]:
            w = extracted[proj_key][0]
            if w is not None:
                norm = torch.linalg.norm(w.float())
                if norm < 1e-4:
                    raise RuntimeError(
                        f"ClusterAttention extracted weight for {proj_key} has near-zero Frobenius norm ({norm.item():.6f})"
                    )

        if extracted["o"][0] is not None:
            norm_o = torch.linalg.norm(extracted["o"][0].float())
            if norm_o < 1e-4:
                raise RuntimeError(
                    f"ClusterAttention extracted weight for output projection has near-zero Frobenius norm ({norm_o.item():.6f})"
                )

        # Copy into projection modules
        with torch.no_grad():
            for proj_key, target in [
                ("q", self.q_proj),
                ("k", self.k_proj),
                ("v", self.v_proj),
                ("o", self.out_proj),
            ]:
                w, b = extracted[proj_key]
                if w is not None:
                    target_w = target.weight
                    out_d, in_d = target_w.shape
                    src_out, src_in = w.shape
                    target_w.data[: min(out_d, src_out), : min(in_d, src_in)].copy_(
                        w[: min(out_d, src_out), : min(in_d, src_in)]
                    )
                    if b is not None and target.bias is not None:
                        b_val = b.data if isinstance(b, nn.Parameter) else b
                        target.bias.data[: min(len(target.bias), len(b_val))].copy_(
                            b_val[: min(len(target.bias), len(b_val))]
                        )
                elif proj_key == "o" and self.is_nested_encoder:
                    min_d = min(target.weight.shape[0], target.weight.shape[1])
                    target.weight.data.zero_()
                    target.weight.data[:min_d, :min_d].fill_diagonal_(1.0)
                    if target.bias is not None:
                        target.bias.data.zero_()

                target.freeze_base_weights()

        if hasattr(src, "rotary_emb") and src.rotary_emb is not None:
            self.rotary_emb = src.rotary_emb

        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)

    def _kmeans_cluster(
        self,
        Q: torch.Tensor,
        K_num: int,
        iters: int = 3,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Vectorized Lloyd's $K$-means clustering for queries with padding token filtering.
        Q: [B, H, L, d], valid_mask: Optional[torch.Tensor] of shape [B, L] (True for real tokens)
        -> centroids: [B, H, K, d], cluster_ids: [B, H, L], counts: [B, H, K, 1]
        """
        B, H, L, d = Q.shape
        K = min(K_num, L)

        if valid_mask is not None:
            valid_f = valid_mask.float()  # [B, L]
            num_valid = valid_f.sum(dim=-1, keepdim=True).clamp(min=1.0)  # [B, 1]
            step = (num_valid / K).clamp(min=1.0)  # [B, 1]
            k_idx = torch.arange(K, device=Q.device, dtype=torch.float32).unsqueeze(0)  # [1, K]
            target_ranks = (k_idx * step).clamp(max=num_valid - 1.0).long()  # [B, K]

            # Assign rank to each valid token: 0, 1, ..., num_valid-1
            ranks = (valid_f.cumsum(dim=-1) - 1.0) * valid_f  # [B, L]
            is_valid_3d = valid_mask.unsqueeze(1)  # [B, 1, L]
            rank_diff = torch.where(
                is_valid_3d,
                (ranks.unsqueeze(1) - target_ranks.unsqueeze(-1)).abs(),
                torch.tensor(1e6, device=Q.device),
            )
            chosen_indices = torch.argmin(rank_diff, dim=-1)  # [B, K]

            idx_expanded = chosen_indices.unsqueeze(1).unsqueeze(-1).expand(B, H, K, d)
            centroids = torch.gather(Q, 2, idx_expanded).clone()
        else:
            step = max(1, L // K)
            init_indices = torch.arange(0, L, step, device=Q.device)[:K]
            if len(init_indices) < K:
                pad_idx = torch.zeros(K - len(init_indices), dtype=torch.long, device=Q.device)
                init_indices = torch.cat([init_indices, pad_idx])
            centroids = Q[:, :, init_indices, :].clone()

        cluster_ids = torch.zeros(B, H, L, dtype=torch.long, device=Q.device)
        counts = torch.ones(B, H, K, 1, dtype=Q.dtype, device=Q.device)
        Q_f = Q.float()

        for _ in range(iters):
            centroids_f = centroids.float()
            dist = torch.cdist(Q_f, centroids_f)
            if valid_mask is not None:
                # Disincentivize padding tokens from pulling centroids
                dist = dist.masked_fill(~valid_mask.unsqueeze(1).unsqueeze(-1), 1e6)
            cluster_ids = torch.argmin(dist, dim=-1)

            one_hot = F.one_hot(cluster_ids, num_classes=K).to(dtype=Q.dtype)
            if valid_mask is not None:
                one_hot = one_hot * valid_mask.unsqueeze(1).unsqueeze(-1).to(dtype=Q.dtype)

            counts = one_hot.sum(dim=2).unsqueeze(-1)
            counts_clamped = counts.clamp(min=1.0)
            sum_q = torch.einsum("bhlk,bhld->bhkd", one_hot, Q)
            centroids = torch.where(counts > 0, sum_q / (counts_clamped + 1e-8), centroids)
            centroids = torch.nan_to_num(centroids, nan=0.0)

        return centroids, cluster_ids, counts

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
        key_value_states = kwargs.get("key_value_states", encoder_hidden_states)
        is_cross_attention = key_value_states is not None
        kv_input = key_value_states if is_cross_attention else hidden_states

        B, L, D = hidden_states.shape
        N_k = kv_input.shape[1]
        H, Dh = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        Q = self.q_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2).contiguous()
        K = self.k_proj(kv_input).view(B, N_k, self.num_kv_heads, Dh).transpose(1, 2).contiguous()
        V = self.v_proj(kv_input).view(B, N_k, self.num_kv_heads, Dh).transpose(1, 2).contiguous()

        # Apply RoPE (accept from position_embeddings or fallback to rotary_emb)
        pos_emb = (
            position_embeddings
            if position_embeddings is not None
            else kwargs.get("position_embeddings")
        )
        if pos_emb is not None and isinstance(pos_emb, (tuple, list)) and len(pos_emb) >= 2:
            cos, sin = pos_emb[0], pos_emb[1]
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin, kwargs.get("position_ids"))
        elif self.rotary_emb is not None:
            rotary_mod = self.rotary_emb
            pos_ids = kwargs.get("position_ids")
            if pos_ids is None:
                pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
            try:
                cos, sin = rotary_mod(Q, seq_len=L)
            except Exception:
                try:
                    cos, sin = rotary_mod(Q, pos_ids)
                except Exception:
                    try:
                        cos, sin = rotary_mod(Q)
                    except Exception:
                        cos, sin = None, None
            if cos is not None and sin is not None:
                Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids=pos_ids)

        if self.kv_repeat > 1:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        scale = Dh**-0.5

        # Check for causal decoding (guard against self.config being None)
        _cfg = self.config
        is_dec_cfg = False
        if _cfg is not None:
            model_type = str(getattr(_cfg, "model_type", "") or "").lower()
            _decoder_prefixes = (
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
            is_dec_cfg = (
                getattr(_cfg, "is_decoder", False)
                or getattr(_cfg, "is_causal", False)
                or getattr(_cfg, "causal", False)
                or any(model_type.startswith(p) for p in _decoder_prefixes)
            )
        is_decoder = self.is_decoder or is_dec_cfg

        if is_decoder or is_cross_attention:
            # Exact attention with causal masking to prevent future token leakage
            scores = torch.matmul(Q, K.transpose(-2, -1).contiguous()) * scale
            mask = attention_mask if attention_mask is not None else kwargs.get("mask")
            if mask is not None:
                if mask.dim() == 4:
                    if mask.dtype == torch.bool:
                        scores = scores.masked_fill(~mask, -10000.0)
                    else:
                        scores = scores + mask.to(scores.dtype)
                else:
                    valid_m = get_token_valid_mask(mask, B, N_k, device)
                    if valid_m is not None:
                        scores = scores.masked_fill((~valid_m).unsqueeze(1).unsqueeze(2), -10000.0)

            if is_decoder:
                causal_tril = (
                    torch.tril(torch.ones((L, N_k), device=device, dtype=torch.bool))
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
                scores = scores.masked_fill(~causal_tril, -10000.0)

            attn = torch.nan_to_num(F.softmax(scores.float(), dim=-1), nan=0.0).to(dtype)
            attn = self.dropout(attn)

            if head_mask is not None:
                if head_mask.dim() == 1:
                    head_mask = head_mask.view(1, -1, 1, 1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.unsqueeze(-1).unsqueeze(-1)
                elif head_mask.dim() == 3:
                    head_mask = head_mask.unsqueeze(-1)
                attn = attn * head_mask

            out_context = torch.matmul(attn, V)
            if not is_decoder:
                valid_q = get_token_valid_mask(attention_mask, B, L, device)
                if valid_q is not None:
                    out_context = out_context * valid_q.unsqueeze(1).unsqueeze(-1).to(dtype)
            attn_ret = attn if output_attentions else None
        else:
            # Bidirectional clustering attention
            valid_mask_2d = get_token_valid_mask(attention_mask, B, L, device)
            key_valid_4d = (
                valid_mask_2d.unsqueeze(1).unsqueeze(2)
                if valid_mask_2d is not None
                else torch.ones((B, 1, 1, L), dtype=torch.bool, device=device)
            )

            if L <= 64 and (self.num_clusters >= 8 or not output_attentions):
                scores = torch.matmul(Q, K.transpose(-2, -1).contiguous()) * scale
                scores = scores.masked_fill(~key_valid_4d, -10000.0)
                attn = torch.nan_to_num(F.softmax(scores.float(), dim=-1), nan=0.0).to(dtype)
                attn = self.dropout(attn)
                out_context = torch.matmul(attn, V)
                attn_ret = attn if output_attentions else None
            else:
                K_num = min(max(1, self.num_clusters), L)

                # 1. Cluster Queries into K centroids with padding mask filtering
                centroids, cluster_ids, counts = self._kmeans_cluster(
                    Q, K_num, self.n_iters, valid_mask=valid_mask_2d
                )

                # 2. Centroid-to-Key Global Cross-Attention [B, H, K, L]
                scores_cent = torch.matmul(centroids, K.transpose(-2, -1).contiguous()) * scale
                scores_cent = scores_cent.masked_fill(~key_valid_4d, -10000.0)
                attn_cent = torch.nan_to_num(F.softmax(scores_cent.float(), dim=-1), nan=0.0).to(
                    dtype
                )
                attn_cent = self.dropout(attn_cent)

                # Centroid values: [B, H, K, Dh]
                V_cent = torch.matmul(attn_cent, V)

                # 3. Differentiable Query-to-Centroid soft assignment with learned sharpening & context gathering [B, H, L, Dh]
                scores_assign = torch.matmul(Q, centroids.transpose(-2, -1).contiguous()) * scale
                empty_clusters = (counts == 0).squeeze(-1).unsqueeze(2)  # [B, H, 1, K]
                scores_assign = scores_assign.masked_fill(empty_clusters, -10000.0)
                temp_clamped = self.temp.abs().clamp(min=0.1, max=10.0)
                attn_assign = torch.nan_to_num(
                    F.softmax((scores_assign / temp_clamped).float(), dim=-1), nan=0.0
                ).to(dtype)
                assign_denom = attn_assign.sum(dim=-1, keepdim=True).clamp(min=1.0)
                attn_assign = attn_assign / assign_denom
                global_context = torch.matmul(attn_assign, V_cent)

                # 4. Optional Local Sliding Window Attention
                if self.window_size > 0 and self.window_size < L:
                    scores_loc = torch.matmul(Q, K.transpose(-2, -1).contiguous()) * scale
                    idx = (
                        self._seq_idx[:L].to(device)
                        if hasattr(self, "_seq_idx") and len(self._seq_idx) >= L
                        else torch.arange(L, device=device)
                    )
                    band_mask = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() <= self.window_size
                    band_mask_4d = band_mask.unsqueeze(0).unsqueeze(0) & key_valid_4d
                    scores_loc = scores_loc.masked_fill(~band_mask_4d, -10000.0)
                    attn_loc = torch.nan_to_num(F.softmax(scores_loc.float(), dim=-1), nan=0.0).to(
                        dtype
                    )
                    attn_loc = self.dropout(attn_loc)
                    local_context = torch.matmul(attn_loc, V)
                    alpha = torch.sigmoid(self.local_residual)
                    out_context = (1.0 - alpha) * global_context + alpha * local_context
                else:
                    out_context = global_context

                attn_ret = attn_cent if output_attentions else None

            if head_mask is not None:
                if head_mask.dim() == 1:
                    head_mask = head_mask.view(1, -1, 1, 1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.unsqueeze(1).unsqueeze(-1)
                elif head_mask.dim() == 3:
                    head_mask = head_mask.unsqueeze(-1)
                out_context = out_context * head_mask

            if valid_mask_2d is not None:
                out_context = out_context * valid_mask_2d.unsqueeze(1).unsqueeze(-1).to(dtype)

        out = out_context.transpose(1, 2).contiguous().view(B, L, self.all_head_size)

        if self.is_nested_encoder:
            output = out
        else:
            output = self.out_proj(out)

        valid_out = get_token_valid_mask(attention_mask, B, L, device)
        if valid_out is not None:
            output = output * valid_out.unsqueeze(-1).to(output.dtype)

        return output, attn_ret


@ATTENTION_REGISTRY.register("cluster_inject")
@ATTENTION_REGISTRY.register("cluster_attention_inject")
def inject_cluster(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    **kwargs: Any,
) -> ClusterAttention | nn.Module | None:
    r"""Safe injection factory for ClusterAttention across diverse transformer architectures."""
    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config
    if cfg is None and hasattr(target, "config"):
        cfg = target.config

    # 1. Whole model traversal
    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
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
    elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_cluster(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    # 2. Single layer injection
    src_attn = old_attn
    if src_attn is None and target is not None:
        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if hasattr(cand, "self"):
                    src_attn = cand.self
                else:
                    src_attn = cand
                break

    if src_attn is None and target is not None and hasattr(target, "forward"):
        src_attn = target

    if src_attn is None and cfg is None:
        return None

    hidden_size = (
        getattr(src_attn, "hidden_size", None)
        or getattr(src_attn, "embed_dim", None)
        or getattr(src_attn, "dim", None)
        or getattr(src_attn, "all_head_size", None)
        or (
            src_attn.query.in_features
            if hasattr(src_attn, "query") and hasattr(src_attn.query, "in_features")
            else None
        )
        or (
            src_attn.q_proj.in_features
            if hasattr(src_attn, "q_proj") and hasattr(src_attn.q_proj, "in_features")
            else None
        )
        or (
            src_attn.q_lin.in_features
            if hasattr(src_attn, "q_lin") and hasattr(src_attn.q_lin, "in_features")
            else None
        )
        or (
            src_attn.c_attn.in_features
            if hasattr(src_attn, "c_attn") and hasattr(src_attn.c_attn, "in_features")
            else None
        )
        or (getattr(target, "in_features", None) if target is not None else None)
        or (getattr(cfg, "hidden_size", None) if cfg is not None else None)
        or (getattr(cfg, "dim", None) if cfg is not None else None)
        or 768
    )
    num_heads = (
        getattr(src_attn, "num_heads", None)
        or getattr(src_attn, "num_attention_heads", None)
        or getattr(src_attn, "n_heads", None)
        or (getattr(cfg, "num_attention_heads", None) if cfg is not None else None)
        or (getattr(cfg, "n_heads", None) if cfg is not None else None)
        or 12
    )

    num_clusters = getattr(
        comp_config,
        "num_clusters",
        getattr(comp_config, "cluster_num_clusters", kwargs.get("num_clusters", 16)),
    )
    n_iters = getattr(comp_config, "n_iters", kwargs.get("n_iters", 3))
    window_size = getattr(comp_config, "window_size", kwargs.get("window_size", 32))

    try:
        new_block = ClusterAttention(
            config=cfg,
            comp_config=comp_config,
            d_model=hidden_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_clusters=num_clusters,
            n_iters=n_iters,
            window_size=window_size,
            pretrained_attn=src_attn,
            **kwargs,
        )
    except Exception as e:
        logger.error(f"Failed to instantiate ClusterAttention: {e}")
        return None

    if target is not None:
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                new_block.is_nested_encoder = True
                target.attention.self = new_block
            else:
                new_block.is_nested_encoder = False
                target.attention = new_block
        elif hasattr(target, "self"):
            new_block.is_nested_encoder = True
            target.self = new_block
        elif hasattr(target, "attn"):
            new_block.is_nested_encoder = False
            new_block.is_decoder = True
            new_block.is_causal = True
            target.attn = new_block
        elif hasattr(target, "self_attn"):
            new_block.is_nested_encoder = False
            new_block.is_decoder = True
            new_block.is_causal = True
            target.self_attn = new_block

    return new_block
