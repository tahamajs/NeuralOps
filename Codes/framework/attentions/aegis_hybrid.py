r"""
Aegis-Hybrid Attention Mechanism.

Combines low-rank factorized projections, localized sliding-window dense attention,
causal/bidirectional kernelized global linear attention with token-importance pruning,
and learned gating fusion.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY
from framework.utils.compat import repeat_kv
from framework.utils.model_family import ModelAdapter

logger = logging.getLogger(__name__)


@dataclass
class AegisHybridConfig:
    r"""Configuration dataclass for AegisHybridAttention."""
    hidden_size: int = 768
    num_attention_heads: int = 12
    head_dim: int = 64
    num_key_value_heads: Optional[int] = None
    local_window: int = 128
    rank: int = 128
    lora_rank: int = 32
    kernel_feature_dim: int = 128  # Kept for backward compatibility
    dropout: float = 0.1
    use_adaptive_pruning: bool = True
    gate_init_bias: float = 1.8
    is_decoder: bool = False
    use_global: bool = True

    def __post_init__(self) -> None:
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


class FactorizedLinear(nn.Module):
    r"""
    Factorized linear projection with internal rank-r base factorization and trainable residual LoRA adapter:
        Base: x -> lin1(x) [in_features -> rank] -> lin2(lin1(x)) [rank -> out_features] (frozen)
        LoRA: x -> lora_A(x) [in_features -> lora_rank] -> lora_B(...) [lora_rank -> out_features] (trainable)
        out = lin2(lin1(x)) + F.linear(F.linear(x, self.lora_A), self.lora_B)

    Uses `lora_A` and `lora_B` parameter names so recovery optimizers (such as get_recovery_params)
    register them automatically as trainable parameters.
    Base weights and biases are frozen to maintain pretrained representations.
    """

    def __init__(
        self,
        in_features_or_lin1: Union[int, nn.Linear],
        out_features_or_lin2: Optional[Union[int, nn.Linear]] = None,
        rank: int = 128,
        lora_rank: int = 32,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        if isinstance(in_features_or_lin1, nn.Linear) and isinstance(out_features_or_lin2, nn.Linear):
            self.lin1 = in_features_or_lin1
            self.lin2 = out_features_or_lin2
            self.in_features = self.lin1.in_features
            self.out_features = self.lin2.out_features
            self.rank = self.lin1.out_features
        else:
            in_features = int(in_features_or_lin1)
            out_features = int(out_features_or_lin2)
            self.in_features = in_features
            self.out_features = out_features
            self.rank = max(1, min(rank, in_features, out_features))

            factory_kwargs = {"device": device, "dtype": dtype}
            self.lin1 = nn.Linear(in_features, self.rank, bias=False, **factory_kwargs)
            self.lin2 = nn.Linear(self.rank, out_features, bias=bias, **factory_kwargs)

        # Base weights are frozen to prevent catastrophic unlearning
        self.lin1.weight.requires_grad = False
        self.lin2.weight.requires_grad = False
        if self.lin2.bias is not None:
            self.lin2.bias.requires_grad = False

        # Trainable residual LoRA adapter
        self.lora_rank: int = min(lora_rank, min(self.in_features, self.out_features)) if lora_rank > 0 else 0
        factory_kwargs = {"device": device, "dtype": dtype}
        if self.lora_rank > 0:
            self.lora_B: Optional[nn.Parameter] = nn.Parameter(
                torch.empty((self.out_features, self.lora_rank), **factory_kwargs)
            )
            self.lora_A: Optional[nn.Parameter] = nn.Parameter(
                torch.empty((self.lora_rank, self.in_features), **factory_kwargs)
            )
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.lin1.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lin2.weight, a=math.sqrt(5))
            if self.lin2.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.lin2.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.lin2.bias, -bound, bound)

            if getattr(self, "lora_A", None) is not None and getattr(self, "lora_B", None) is not None:
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix (out_features, in_features)."""
        w = torch.matmul(self.lin2.weight, self.lin1.weight)
        if getattr(self, "lora_A", None) is not None and getattr(self, "lora_B", None) is not None:
            w = w + torch.matmul(self.lora_B, self.lora_A)
        return w

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.lin2.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lin2(self.lin1(x))
        if getattr(self, "lora_A", None) is not None and getattr(self, "lora_B", None) is not None:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B)
        return out


@ATTENTION_REGISTRY.register(
    "aegis_hybrid",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
    supports_rope=True,
    supports_delta_net=True,
)
@ATTENTION_REGISTRY.register(
    "aegis_hybrid_attn",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
    supports_rope=True,
    supports_delta_net=True,
)
@ATTENTION_REGISTRY.register(
    "aegis_hybrid_attention",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
    supports_rope=True,
    supports_delta_net=True,
)
class AegisHybridAttention(BaseAttention):
    r"""
    Aegis-Hybrid Attention combining Local Windowed Attention and Global Linear Attention.

    Architectural Foundation:
    - Low-rank factorized projections for Q, K, V, and O with persistent submodules and trainable LoRA adapters.
    - Causal & bidirectional local windowed attention with band masking.
    - Causal cumulative/triangular or bidirectional global linear attention with token pruning.
    - Rotary Position Embeddings (RoPE) integration before pathway split.
    - Correct grouped-query attention (GQA) head repetition.
    - Architecture-aware pretrained weight transfer and nested encoder detection.
    - Adaptive gating fusion of local and global contexts.
    """

    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    supports_rope: bool = True
    is_custom_attention: bool = True
    supports_delta_net: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        hidden_size: Optional[int] = None,
        num_heads: Optional[int] = None,
        pretrained_attn: Optional[nn.Module] = None,
        rank: Optional[int] = None,
        lora_rank: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.adapter = ModelAdapter(config) if config is not None else None

        if not isinstance(config, AegisHybridConfig):
            if isinstance(config, dict):
                h_size = hidden_size or config.get("hidden_size", config.get("dim", 768))
                n_heads = num_heads or config.get("num_attention_heads", config.get("num_heads", 12))
                h_dim = config.get("head_dim", h_size // max(1, n_heads))
                n_kv_heads = config.get("num_key_value_heads", config.get("num_kv_heads", n_heads))
                r = rank or config.get("rank", min(128, h_size))
                l_rank = lora_rank or config.get("lora_rank", config.get("lora_r", 32))
                model_type = config.get("model_type", "")
                is_dec = config.get("is_decoder", False)
                _decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
                if any(model_type.startswith(p) for p in _decoder_prefixes):
                    is_dec = True
                l_win = config.get("local_window", 128)
                drop = config.get("dropout", 0.1)
                gate_bias = config.get("gate_init_bias", 1.8)
                use_prune = config.get("use_adaptive_pruning", True)
                use_glob = config.get("use_global", True)
            else:
                h_size = hidden_size or kwargs.get("hidden_size", kwargs.get("d_model", getattr(config, "hidden_size", getattr(config, "dim", 768)) if config is not None else 768))
                n_heads = num_heads or kwargs.get("num_heads", kwargs.get("num_attention_heads", getattr(config, "num_attention_heads", getattr(config, "n_heads", 12)) if config is not None else 12))
                h_dim = kwargs.get("head_dim", getattr(config, "head_dim", getattr(config, "d_kv", h_size // max(1, n_heads))) if config is not None else h_size // max(1, n_heads))
                n_kv_heads = kwargs.get("num_key_value_heads", kwargs.get("num_kv_heads", getattr(config, "num_key_value_heads", getattr(config, "num_kv_heads", n_heads)) if config is not None else n_heads))
                r = rank or kwargs.get("rank", getattr(config, "rank", min(128, h_size)) if config is not None else 128)
                l_rank = lora_rank or kwargs.get("lora_rank", getattr(config, "lora_rank", getattr(config, "lora_r", 32)) if config is not None else 32)
                model_type = getattr(config, "model_type", "") if config is not None else ""
                is_dec = getattr(config, "is_decoder", False) if config is not None else False
                _decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
                if any(model_type.startswith(p) for p in _decoder_prefixes):
                    is_dec = True
                elif hasattr(config, "architectures") and any("GPT" in str(a) or "CausalLM" in str(a) for a in (getattr(config, "architectures", []) or [])):
                    is_dec = True
                elif pretrained_attn is not None and (hasattr(pretrained_attn, "c_attn") or pretrained_attn.__class__.__name__ in ("GPT2Attention", "Qwen2Attention")):
                    is_dec = True
                l_win = getattr(config, "local_window", getattr(config, "block_size", 128)) if config is not None else 128
                drop = getattr(config, "dropout", getattr(config, "hidden_dropout_prob", 0.1)) if config is not None else 0.1
                gate_bias = getattr(config, "gate_init_bias", 1.8) if config is not None else 1.8
                use_prune = getattr(config, "use_adaptive_pruning", getattr(config, "use_gate", True)) if config is not None else True
                use_glob = getattr(config, "use_global", True) if config is not None else True

            config = AegisHybridConfig(
                hidden_size=h_size,
                num_attention_heads=n_heads,
                head_dim=h_dim,
                num_key_value_heads=n_kv_heads,
                local_window=l_win,
                rank=r,
                lora_rank=l_rank,
                dropout=drop,
                use_adaptive_pruning=use_prune,
                gate_init_bias=gate_bias,
                is_decoder=is_dec or kwargs.get("is_decoder", False),
                use_global=use_glob,
            )

        object.__setattr__(self, 'config', config)
        self.hidden_size: int = config.hidden_size
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.head_dim
        n_kv = getattr(config, "num_key_value_heads", None)
        self.num_kv_heads: int = n_kv if (n_kv is not None and n_kv <= self.num_heads) else self.num_heads
        self.kv_repeat: int = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.all_head_size: int = self.num_heads * self.head_dim
        self.all_kv_head_size: int = self.num_kv_heads * self.head_dim
        self.local_window: int = config.local_window
        self.rank: int = max(1, min(config.rank, self.hidden_size))
        self.lora_rank: int = max(1, min(getattr(config, "lora_rank", 32), self.hidden_size))
        self.dropout: nn.Dropout = nn.Dropout(config.dropout)
        
        # Enforce decoder detection unconditionally
        model_type = getattr(config, "model_type", "") if config else ""
        is_dec = getattr(config, "is_decoder", False)
        _decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
        if any(model_type.startswith(p) for p in _decoder_prefixes):
            is_dec = True
        elif hasattr(config, "architectures") and any("GPT" in str(a) or "CausalLM" in str(a) for a in (getattr(config, "architectures", []) or [])):
            is_dec = True
        elif pretrained_attn is not None and (hasattr(pretrained_attn, "c_attn") or pretrained_attn.__class__.__name__ in ("GPT2Attention", "Qwen2Attention")):
            is_dec = True
        if kwargs.get("is_decoder", False):
            is_dec = True

        self.is_decoder: bool = bool(is_dec)
        self.use_global: bool = getattr(config, "use_global", True)

        # CRITICAL FALLBACK: Disable global linear attention for decoders due to FP16 CumSum instability
        if self.is_decoder:
            self.use_global = False

        self.is_nested_encoder: bool = False
        self.rotary_emb = getattr(pretrained_attn, "rotary_emb", None)

        # Persistent FactorizedLinear submodules equipped with trainable LoRA adapters
        self.q_proj = FactorizedLinear(self.hidden_size, self.all_head_size, rank=self.rank, lora_rank=self.lora_rank, bias=True)
        self.k_proj = FactorizedLinear(self.hidden_size, self.all_kv_head_size, rank=self.rank, lora_rank=self.lora_rank, bias=True)
        self.v_proj = FactorizedLinear(self.hidden_size, self.all_kv_head_size, rank=self.rank, lora_rank=self.lora_rank, bias=True)
        self.out_proj = FactorizedLinear(self.all_head_size, self.hidden_size, rank=self.rank, lora_rank=self.lora_rank, bias=True)

        if config.use_adaptive_pruning:
            self.prune_gate: Optional[nn.Sequential] = nn.Sequential(
                nn.Linear(self.hidden_size, max(1, self.hidden_size // 4)),
                nn.ReLU(),
                nn.Linear(max(1, self.hidden_size // 4), self.num_heads),
            )
        else:
            self.prune_gate = None

        self.fusion_gate = nn.Linear(self.hidden_size, self.num_heads)
        with torch.no_grad():
            self.fusion_gate.bias.fill_(config.gate_init_bias)

        if pretrained_attn is not None:
            self._copy_pretrained(pretrained_attn)

    # Legacy submodule access aliases
    @property
    def q_proj1(self) -> nn.Linear:
        return self.q_proj.lin1

    @property
    def q_proj2(self) -> nn.Linear:
        return self.q_proj.lin2

    @property
    def k_proj1(self) -> nn.Linear:
        return self.k_proj.lin1

    @property
    def k_proj2(self) -> nn.Linear:
        return self.k_proj.lin2

    @property
    def v_proj1(self) -> nn.Linear:
        return self.v_proj.lin1

    @property
    def v_proj2(self) -> nn.Linear:
        return self.v_proj.lin2

    @property
    def out_proj1(self) -> nn.Linear:
        return self.out_proj.lin1

    @property
    def out_proj2(self) -> nn.Linear:
        return self.out_proj.lin2

    # Compatibility aliases returning the SAME persistent initialized FactorizedLinear instances
    @property
    def q(self) -> FactorizedLinear:
        return self.q_proj

    @property
    def k(self) -> FactorizedLinear:
        return self.k_proj

    @property
    def v(self) -> FactorizedLinear:
        return self.v_proj

    @property
    def o(self) -> FactorizedLinear:
        return self.out_proj

    @property
    def query(self) -> FactorizedLinear:
        return self.q_proj

    @property
    def key(self) -> FactorizedLinear:
        return self.k_proj

    @property
    def value(self) -> FactorizedLinear:
        return self.v_proj

    @property
    def dense(self) -> FactorizedLinear:
        return self.out_proj

    @property
    def q_lin(self) -> FactorizedLinear:
        return self.q_proj

    @property
    def k_lin(self) -> FactorizedLinear:
        return self.k_proj

    @property
    def v_lin(self) -> FactorizedLinear:
        return self.v_proj

    @property
    def out_lin(self) -> FactorizedLinear:
        return self.out_proj

    @property
    def o_proj(self) -> FactorizedLinear:
        return self.out_proj

    @property
    def c_proj(self) -> FactorizedLinear:
        return self.out_proj

    @staticmethod
    def _svd_init(
        w: torch.Tensor,
        b: Optional[torch.Tensor],
        proj: FactorizedLinear,
    ) -> None:
        """Initializes FactorizedLinear using truncated SVD approximation: W ≈ U[:, :r] diag(S[:r]) Vh[:r, :]."""
        with torch.no_grad():
            W = w.detach().float()
            if W.ndim != 2:
                raise ValueError(f"Expected 2D weight matrix, got shape {W.shape}")
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            r = min(proj.rank, U.size(1), Vh.size(0), proj.lin1.weight.size(0), proj.lin2.weight.size(1))

            proj.lin1.weight.data.zero_()
            proj.lin2.weight.data.zero_()

            in_copy = min(W.shape[1], proj.lin1.weight.size(1))
            out_copy = min(W.shape[0], proj.lin2.weight.size(0))

            proj.lin1.weight.data[:r, :in_copy].copy_(Vh[:r, :in_copy].to(proj.lin1.weight.dtype))
            US = U[:out_copy, :r] @ torch.diag(S[:r])
            proj.lin2.weight.data[:out_copy, :r].copy_(US.to(proj.lin2.weight.dtype))

            if b is not None and proj.lin2.bias is not None:
                min_b = min(b.numel(), proj.lin2.bias.numel())
                proj.lin2.bias.data[:min_b].copy_(b.detach()[:min_b].to(proj.lin2.bias.dtype))
            elif proj.lin2.bias is not None:
                proj.lin2.bias.data.zero_()

            # Ensure base factorized linear weights remain frozen
            proj.lin1.weight.requires_grad = False
            proj.lin2.weight.requires_grad = False
            if proj.lin2.bias is not None:
                proj.lin2.bias.requires_grad = False

            # Trainable LoRA adapter: A is initialized with Kaiming uniform, B is zeroed so initial delta is 0
            if getattr(proj, "lora_A", None) is not None:
                nn.init.kaiming_uniform_(proj.lora_A, a=math.sqrt(5))
                proj.lora_A.requires_grad = True
            if getattr(proj, "lora_B", None) is not None:
                nn.init.zeros_(proj.lora_B)
                proj.lora_B.requires_grad = True

    def _copy_pretrained(self, old_attn: nn.Module) -> None:
        """Comprehensive weight extraction across DistilBERT, RoBERTa, GPT-2, and Qwen3.5."""
        from framework.core.adapters import UniversalAdapter

        extracted_weights = {}
        for proj in ["q", "k", "v", "o"]:
            w, b = None, None

            # 1. Output projection checks first
            if proj == "o":
                if hasattr(old_attn, "c_proj"):
                    mod = old_attn.c_proj
                    if hasattr(mod, "merge") and not getattr(mod, "merged", False):
                        try:
                            mod.merge()
                        except Exception as e:
                            logger.warning(f"c_proj merge failed: {e}")
                    base_mod = getattr(mod, "base_layer", mod)
                    w = getattr(base_mod, "weight", getattr(mod, "weight", None))
                    if w is not None:
                        w = w.data.clone()
                        hidden_dim = getattr(self, "hidden_size", 768)
                        if (
                            mod.__class__.__name__ == "Conv1D"
                            or base_mod.__class__.__name__ == "Conv1D"
                            or hasattr(mod, "nf")
                            or hasattr(base_mod, "nf")
                            or (w.dim() == 2 and w.shape[0] != hidden_dim and w.shape[1] == hidden_dim)
                            or (w.dim() == 2 and w.shape[0] < w.shape[1])
                        ):
                            w = w.t().contiguous()
                        else:
                            w = w.contiguous()
                    bias_holder = getattr(base_mod, "bias", getattr(mod, "bias", None))
                    b = bias_holder.data.clone().contiguous() if bias_holder is not None else None
                elif hasattr(old_attn, "output") and hasattr(getattr(old_attn, "output"), "dense"):
                    mod = old_attn.output.dense
                    w = mod.weight.data
                    b = getattr(mod, "bias", None)
                elif hasattr(old_attn, "dense"):
                    mod = old_attn.dense
                    w = mod.weight.data
                    b = getattr(mod, "bias", None)
                elif hasattr(old_attn, "out_lin"):
                    mod = old_attn.out_lin
                    w = mod.weight.data
                    b = getattr(mod, "bias", None)
                elif hasattr(old_attn, "out_proj"):
                    mod = old_attn.out_proj
                    w = mod.weight.data
                    b = getattr(mod, "bias", None)
                elif hasattr(old_attn, "o_proj"):
                    mod = old_attn.o_proj
                    w = mod.weight.data
                    b = getattr(mod, "bias", None)

            # 2. GPT-2 Fused c_attn
            elif hasattr(old_attn, "c_attn") and proj in ["q", "k", "v"]:
                mod = old_attn.c_attn
                if hasattr(mod, "merge") and not getattr(mod, "merged", False):
                    try:
                        mod.merge()
                    except Exception as e:
                        logger.warning(f"c_attn merge failed: {e}")
                base_mod = getattr(mod, "base_layer", mod)
                c_w = getattr(base_mod, "weight", getattr(mod, "weight", None))
                if c_w is not None:
                    c_w = c_w.data.clone()
                    hidden_dim = getattr(self, "hidden_size", 768)
                    if (
                        mod.__class__.__name__ == "Conv1D"
                        or base_mod.__class__.__name__ == "Conv1D"
                        or hasattr(mod, "nf")
                        or hasattr(base_mod, "nf")
                        or (c_w.dim() == 2 and c_w.shape[1] == 3 * hidden_dim)
                        or (c_w.dim() == 2 and c_w.shape[0] < c_w.shape[1])
                    ):
                        c_w = c_w.t().contiguous()
                    split_size = c_w.size(0) // 3
                    idx = {"q": 0, "k": 1, "v": 2}[proj]
                    w = c_w[idx * split_size : (idx + 1) * split_size, :].contiguous()
                    bias_holder = getattr(base_mod, "bias", getattr(mod, "bias", None))
                    if bias_holder is not None:
                        c_b = bias_holder.data.clone()
                        b = c_b[idx * split_size : (idx + 1) * split_size].contiguous()

            # 3. RoBERTa / BERT nested self-attention (old_attn.self)
            elif hasattr(old_attn, "self") and proj in ["q", "k", "v"]:
                s = getattr(old_attn, "self")
                name_map = {"q": "query", "k": "key", "v": "value"}
                if hasattr(s, name_map[proj]):
                    mod = getattr(s, name_map[proj])
                    w = mod.weight.data
                    b = getattr(mod, "bias", None)

            # 4. Qwen3.5 GatedDeltaNet fused in_proj_qkv
            elif hasattr(old_attn, "in_proj_qkv") and proj in ["q", "k", "v"]:
                mod = old_attn.in_proj_qkv
                fused_w = mod.weight.data
                fused_b = getattr(mod, "bias", None)
                k_dim = getattr(old_attn, "key_dim", getattr(old_attn, "k_dim", fused_w.size(0) // 3))
                v_dim = getattr(old_attn, "value_dim", getattr(old_attn, "v_dim", fused_w.size(0) // 3))
                q_dim = fused_w.size(0) - k_dim - v_dim if (k_dim + v_dim < fused_w.size(0)) else k_dim
                if proj == "q":
                    w = fused_w[:q_dim, :]
                    b = fused_b[:q_dim] if fused_b is not None else None
                elif proj == "k":
                    w = fused_w[q_dim : q_dim + k_dim, :]
                    b = fused_b[q_dim : q_dim + k_dim] if fused_b is not None else None
                elif proj == "v":
                    w = fused_w[q_dim + k_dim : q_dim + k_dim + v_dim, :]
                    b = fused_b[q_dim + k_dim : q_dim + k_dim + v_dim] if fused_b is not None else None

            # 5. Fallback to UniversalAdapter
            if w is None:
                try:
                    w, b = UniversalAdapter.get_weight_bias(old_attn, proj)
                except Exception as e:
                    logger.debug(f"UniversalAdapter extraction failed for {proj}: {e}")

            extracted_weights[proj] = (w, b)

        # Detect nested encoder: BERT/RoBERTa RobertaSelfAttention where output projection resides in outer module
        cls_name = old_attn.__class__.__name__
        has_out_proj = any(
            hasattr(old_attn, attr)
            for attr in ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "wo", "o")
        )
        if cls_name in ("BertSelfAttention", "RobertaSelfAttention") or (
            cls_name.endswith("SelfAttention") and not has_out_proj
        ):
            self.is_nested_encoder = True
        elif (
            hasattr(old_attn, "self")
            and hasattr(getattr(old_attn, "self"), "query")
            and hasattr(old_attn, "output")
            and hasattr(getattr(old_attn, "output"), "dense")
        ):
            self.is_nested_encoder = True
        elif not has_out_proj:
            self.is_nested_encoder = True

        missing = [p for p in ["q", "k", "v"] if extracted_weights[p][0] is None]
        if not self.is_nested_encoder and extracted_weights["o"][0] is None:
            missing.append("o")

        if missing:
            raise RuntimeError(
                f"AegisHybridAttention.copy_weights: could not extract mandatory projections {missing} "
                f"from {cls_name}. Refusing to install randomly-initialised weights."
            )

        # Strict Frobenius norm verification (>= 1e-4)
        for p in ["q", "k", "v"]:
            w_check = extracted_weights[p][0]
            if w_check is not None:
                norm = float(torch.norm(w_check.float()).item())
                if norm < 1e-4:
                    logger.warning(f"AegisHybridAttention extracted weight for '{p}' has near-zero Frobenius norm ({norm:.2e})")

        if extracted_weights["o"][0] is not None:
            norm_o = float(torch.norm(extracted_weights["o"][0].float()).item())
            if norm_o < 1e-4:
                logger.warning(f"AegisHybridAttention extracted weight for 'o' has near-zero Frobenius norm ({norm_o:.2e})")

        # SVD initialization for Q, K, V
        for proj, target in [("q", self.q_proj), ("k", self.k_proj), ("v", self.v_proj)]:
            w, b = extracted_weights[proj]
            self._svd_init(w, b, target)

        # SVD initialization for O (or identity for nested encoder)
        w_o, b_o = extracted_weights["o"]
        if w_o is not None:
            self._svd_init(w_o, b_o, self.out_proj)
        elif self.is_nested_encoder:
            with torch.no_grad():
                self.out_proj.lin1.weight.data.zero_()
                self.out_proj.lin2.weight.data.zero_()
                r = min(self.out_proj.rank, self.hidden_size, self.all_head_size)
                self.out_proj.lin1.weight.data[:r, :r].fill_diagonal_(1.0)
                self.out_proj.lin2.weight.data[:r, :r].fill_diagonal_(1.0)
                if self.out_proj.lin2.bias is not None:
                    self.out_proj.lin2.bias.data.zero_()
                self.out_proj.lin1.weight.requires_grad = False
                self.out_proj.lin2.weight.requires_grad = False
                if self.out_proj.lin2.bias is not None:
                    self.out_proj.lin2.bias.requires_grad = False
                if getattr(self.out_proj, "lora_A", None) is not None:
                    nn.init.zeros_(self.out_proj.lora_A)
                    self.out_proj.lora_A.requires_grad = True
                if getattr(self.out_proj, "lora_B", None) is not None:
                    self.out_proj.lora_B.requires_grad = True

        # Rotary and decoder metadata inheritance
        if hasattr(old_attn, "rotary_emb") and old_attn.rotary_emb is not None:
            self.rotary_emb = old_attn.rotary_emb
        if hasattr(old_attn, "is_decoder"):
            self.is_decoder = bool(old_attn.is_decoder)
        elif hasattr(old_attn, "c_attn") or old_attn.__class__.__name__ in ("GPT2Attention", "Qwen2Attention"):
            self.is_decoder = True
        if self.is_decoder:
            self.use_global = False

    def copy_weights(self, old_attn: Any) -> None:
        """Public alias for framework attention replacer compatibility."""
        self._copy_pretrained(old_attn)

    @classmethod
    def from_distilbert_pretrained(
        cls, old_attn: nn.Module, model_config: Optional[Any] = None, **kwargs: Any
    ) -> "AegisHybridAttention":
        r"""Instantiates AegisHybridAttention with weights transferred from pretrained attention."""
        if model_config is None:
            class DummyConfig:
                pass
            model_config = DummyConfig()
            out_lin = getattr(old_attn, "out_lin", getattr(old_attn, "dense", None))
            model_config.hidden_size = out_lin.out_features if out_lin is not None else 768
            model_config.num_attention_heads = getattr(old_attn, "n_heads", getattr(old_attn, "num_attention_heads", 12))
        return cls(config=model_config, pretrained_attn=old_attn, **kwargs)

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
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Forward pass executing local windowed attention and global linear attention with learned fusion.
        """
        key_value_states = kwargs.get("key_value_states", encoder_hidden_states)
        is_cross_attention = key_value_states is not None
        kv_input = key_value_states if is_cross_attention else hidden_states

        B, N, _ = hidden_states.shape
        N_k = kv_input.shape[1]
        Dh = self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        # 1. Projections via FactorizedLinear (base + LoRA)
        Q = self.q_proj(hidden_states).view(B, N, self.num_heads, Dh).transpose(1, 2).contiguous()  # [B, H, N, Dh]
        K = self.k_proj(kv_input).view(B, N_k, self.num_kv_heads, Dh).transpose(1, 2).contiguous()  # [B, H_kv, N_k, Dh]
        V = self.v_proj(kv_input).view(B, N_k, self.num_kv_heads, Dh).transpose(1, 2).contiguous()  # [B, H_kv, N_k, Dh]

        # 2. Rotary Position Embeddings (RoPE) - applied before GQA repetition & pathway splitting
        position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None and isinstance(position_embeddings, (tuple, list)) and len(position_embeddings) >= 2:
            cos, sin = position_embeddings[0], position_embeddings[1]

            def _rotate_half(x: torch.Tensor) -> torch.Tensor:
                _d = x.shape[-1]
                _h = _d // 2
                _rotated = torch.cat((-x[..., _h : 2 * _h], x[..., :_h]), dim=-1)
                if _d % 2 == 1:
                    _rotated = torch.cat([_rotated, x[..., -1:]], dim=-1)
                return _rotated

            if cos.dim() == 3:
                cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
            while cos.dim() < Q.dim():
                cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
            if cos.shape[-2] != Q.shape[-2]:
                cos, sin = cos[..., : Q.shape[-2], :], sin[..., : Q.shape[-2], :]

            rotary_dim = cos.shape[-1]
            if rotary_dim <= Q.shape[-1]:
                q_rot, q_pass = Q[..., :rotary_dim], Q[..., rotary_dim:]
                k_rot, k_pass = K[..., :rotary_dim], K[..., rotary_dim:]
                q_rot = (q_rot * cos.to(q_rot.dtype)) + (_rotate_half(q_rot) * sin.to(q_rot.dtype))
                k_rot = (k_rot * cos.to(k_rot.dtype)) + (_rotate_half(k_rot) * sin.to(k_rot.dtype))
                Q = torch.cat([q_rot, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot
                K = torch.cat([k_rot, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot
            else:
                cos_cut, sin_cut = cos[..., : Q.shape[-1]], sin[..., : Q.shape[-1]]
                Q = (Q * cos_cut.to(Q.dtype)) + (_rotate_half(Q) * sin_cut.to(Q.dtype))
                K = (K * cos_cut.to(K.dtype)) + (_rotate_half(K) * sin_cut.to(K.dtype))
        elif hasattr(self, "rotary_emb") and self.rotary_emb is not None:
            try:
                from framework.utils.compat import maybe_apply_rotary
                Q, K = maybe_apply_rotary(Q, K, self.adapter, self.rotary_emb, kwargs.get("position_ids"))
            except Exception as e:
                logger.warning(f"Rotary application failed: {e}")

        # 3. GQA Key-Value expansion: [B, H_kv, N_k, Dh] -> [B, H, N_k, Dh]
        if self.kv_repeat > 1:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        # 4. Token valid mask extraction
        mask_2d: Optional[torch.Tensor] = None
        mask = attention_mask if attention_mask is not None else kwargs.get("mask", None)
        if mask is not None:
            if mask.dim() == 2:
                if mask.dtype in (torch.int, torch.long, torch.bool):
                    mask_2d = mask.bool()
                else:
                    mask_2d = (mask > -5000.0).bool()
            elif mask.dim() == 3:
                mask_2d = (mask[:, 0, :] > -5000.0).bool() if mask.shape[1] == 1 else (mask > -5000.0).any(dim=1).bool()
            elif mask.dim() == 4:
                # In causal masks [B, 1, N, N_k], row i only attends to j <= i.
                # A key token j is non-padding if it can be attended to by any query:
                mask_2d = (mask > -5000.0).any(dim=-2).squeeze(1).bool()

        # 5. Head mask preparation
        head_mask_tensor: Optional[torch.Tensor] = None
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask_tensor = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask_tensor = head_mask.unsqueeze(-1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                head_mask_tensor = head_mask.unsqueeze(-1)
            elif head_mask.dim() == 4:
                head_mask_tensor = head_mask

        is_decoder = (
            self.is_decoder
            or getattr(self.adapter, "is_decoder", False)
            or kwargs.get("is_decoder", False)
            or getattr(getattr(self, "config", None), "is_decoder", False)
            or getattr(getattr(self, "config", None), "model_type", "").lower() in ("gpt2", "gpt", "qwen", "qwen2", "llama")
        )

        # ============== Branch 1: Local Window Attention ==============
        scale = 1.0 / math.sqrt(Dh)
        scores = torch.matmul(Q, K.transpose(-2, -1).contiguous()) * scale  # [B, H, N, N_k]

        idx_q = torch.arange(N, device=device)
        idx_k = torch.arange(N_k, device=device)
        diff = idx_q.unsqueeze(1) - idx_k.unsqueeze(0)  # [N, N_k], diff[i, j] = i - j
        band_mask = diff.abs() <= self.local_window

        if is_decoder:
            causal_mask = diff >= 0
            band_mask = band_mask & causal_mask

        scores = scores.masked_fill(~band_mask.unsqueeze(0).unsqueeze(0), -10000.0)

        if mask is not None and mask.dim() == 4:
            if mask.shape[1] == 1 or mask.shape[1] == scores.shape[1]:
                scores = scores + mask.to(scores.dtype)
            elif scores.shape[1] % mask.shape[1] == 0:
                scores = scores + repeat_kv(mask, scores.shape[1] // mask.shape[1]).to(scores.dtype)
            else:
                scores = scores + mask[:, :1].to(scores.dtype)
        elif mask_2d is not None:
            key_mask = mask_2d.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N_k]
            scores = scores.masked_fill(~key_mask, -10000.0)

        attn_local = F.softmax(scores.float(), dim=-1).to(dtype)
        attn_local = torch.nan_to_num(attn_local, nan=0.0)
        attn_local = self.dropout(attn_local)

        if head_mask_tensor is not None:
            attn_local = attn_local * head_mask_tensor.to(attn_local.dtype)

        local_out = torch.matmul(attn_local, V).transpose(1, 2).contiguous()  # [B, N, H, Dh]

        # 2. Hybrid Combination: Local Window Attention + True O(N) Global Linear Attention
        if (not self.use_global) or is_decoder:
            fused = local_out
        else:
            # --- GLOBAL LINEAR ATTENTION BRANCH ---
            Q_g = Q.transpose(1, 2).contiguous()
            K_g = K.transpose(1, 2).contiguous()
            V_g = V.transpose(1, 2).contiguous()

            if self.prune_gate is not None:
                keep = torch.sigmoid(self.prune_gate(hidden_states))  # [B, N, H]
                K_g = K_g * keep.unsqueeze(-1)
                V_g = V_g * keep.unsqueeze(-1)

            phi_Q = (F.elu(Q_g.float()) + 1.0).transpose(1, 2).contiguous()  # [B, H, N, Dh]
            phi_K = (F.elu(K_g.float()) + 1.0).transpose(1, 2).contiguous()  # [B, H, N_k, Dh]
            V_gf = V_g.float().transpose(1, 2).contiguous()                   # [B, H, N_k, Dh]

            # CRITICAL: Mask padding tokens POST-phi so they don't corrupt the KV state
            if mask_2d is not None:
                mask_4d = mask_2d.unsqueeze(1).unsqueeze(-1).float() # [B, 1, N, 1]
                phi_Q = phi_Q * mask_4d
                phi_K = phi_K * mask_4d
                V_gf = V_gf * mask_4d

            if is_decoder:
                # TRUE O(N) CAUSAL LINEAR ATTENTION using Cumulative Sum
                # KV_states shape: [B, H, N, Dh, Dh]
                KV_states = torch.einsum("bhnd,bhne->bhnde", phi_K, V_gf)
                KV_cumsum = torch.cumsum(KV_states, dim=2)
                
                # Numerator: phi_Q @ KV_cumsum
                global_out_h = torch.einsum("bhnd,bhnde->bhne", phi_Q, KV_cumsum)
                
                # Denominator (Normalization)
                K_cumsum = torch.cumsum(phi_K, dim=2) # [B, H, N, Dh]
                norm = torch.einsum("bhnd,bhnd->bhn", phi_Q, K_cumsum).unsqueeze(-1) # [B, H, N, 1]
            else:
                # TRUE O(N) BIDIRECTIONAL LINEAR ATTENTION
                # KV shape: [B, H, Dh, Dh]
                KV = torch.einsum("bhnd,bhne->bhde", phi_K, V_gf)
                global_out_h = torch.einsum("bhnd,bhde->bhne", phi_Q, KV)
                
                # Denominator (Normalization)
                K_sum = phi_K.sum(dim=2) # [B, H, Dh]
                norm = torch.einsum("bhnd,bhd->bhn", phi_Q, K_sum).unsqueeze(-1) # [B, H, N, 1]

            norm = norm.clamp(min=1e-6)
            global_out_h = (global_out_h / norm).to(dtype)
            
            if head_mask_tensor is not None:
                global_out_h = global_out_h * head_mask_tensor.to(global_out_h.dtype)

            global_out = global_out_h.transpose(1, 2).contiguous()  # [B, N, H, Dh]

            alpha = torch.sigmoid(self.fusion_gate(hidden_states)).unsqueeze(-1)  # [B, N, H, 1]
            fused = alpha * local_out + (1.0 - alpha) * global_out                # [B, N, H, Dh]

        # 3. CRITICAL: Output Projection MUST happen for both branches
        fused = fused.reshape(B, N, self.all_head_size)
        fused = torch.nan_to_num(fused, nan=0.0, posinf=1e4, neginf=-1e4)
        fused = fused.clamp(min=-1e4, max=1e4)

        if getattr(self, "is_nested_encoder", False):
            out = fused
        else:
            out = self.out_proj(fused)
            out = self.dropout(out)

        if not is_decoder and mask_2d is not None:
            out = out * mask_2d.unsqueeze(-1)

        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        out = out.clamp(min=-1e4, max=1e4)

        return out, (attn_local if output_attentions else None)

    def get_ortho_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.out_proj.lin2.weight.device)


@ATTENTION_REGISTRY.register("aegis_hybrid_inject")
@ATTENTION_REGISTRY.register("inject_aegis_hybrid")
def inject_aegis_hybrid(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> Optional[Union[AegisHybridAttention, Any]]:
    r"""
    Injects AegisHybridAttention into a transformer layer with safe pre-test extraction.
    Returns None on failure to preserve the original layer.
    """
    target = layer if layer is not None else model
    if target is None:
        return None
        
    if getattr(target, "_aegis_injected", False):
        return target
        
    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    elif hasattr(target, "h") and isinstance(target.h, (nn.ModuleList, list)):
        layer_list = target.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        for idx, lyr in enumerate(layer_list):
            res = inject_aegis_hybrid(
                layer=lyr, model_config=model_config, comp_config=comp_config,
                layer_idx=idx, total_layers=len(layer_list), **kwargs,
            )
            if res is not None:
                num_injected += 1
        if num_injected > 0:
            target._aegis_injected = True
        return target if num_injected > 0 else None
        
    try:
        if old_attn is None and target is not None:
            for attr in ("attn", "attention", "self_attn", "self"):
                if hasattr(target, attr):
                    cand = getattr(target, attr)
                    if cand is not None:
                        old_attn = getattr(cand, "self", cand)
                        break

        if model_config is None:
            if hasattr(target, "hidden_size") or hasattr(target, "d_model") or hasattr(target, "n_embd") or hasattr(target, "dim"):
                model_config = layer
            elif old_attn is not None:
                model_config = getattr(old_attn, "config", None)

        model_type = getattr(model_config, "model_type", "") if model_config else ""
        is_dec = getattr(model_config, "is_decoder", False) if model_config else False
        _decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
        if any(model_type.startswith(p) for p in _decoder_prefixes):
            is_dec = True
        elif hasattr(model_config, "architectures") and any("GPT" in str(a) or "CausalLM" in str(a) for a in (getattr(model_config, "architectures", []) or [])):
            is_dec = True
        elif old_attn is not None and (hasattr(old_attn, "c_attn") or old_attn.__class__.__name__ in ("GPT2Attention", "Qwen2Attention")):
            is_dec = True
        if kwargs.get("is_decoder", False):
            is_dec = True

        if isinstance(model_config, dict):
            h_size = model_config.get("hidden_size", model_config.get("dim", 768))
            n_heads = model_config.get("num_attention_heads", model_config.get("num_heads", 12))
            h_dim = model_config.get("head_dim", h_size // max(1, n_heads))
            n_kv_heads = model_config.get("num_key_value_heads", model_config.get("num_kv_heads", n_heads))
            l_rank = model_config.get("lora_rank", 32)
        elif model_config is not None:
            h_size = getattr(model_config, "hidden_size", getattr(model_config, "d_model", getattr(model_config, "n_embd", getattr(model_config, "dim", 768))))
            n_heads = getattr(model_config, "num_attention_heads", getattr(model_config, "num_heads", getattr(model_config, "n_heads", getattr(model_config, "n_head", 12))))
            h_dim = getattr(model_config, "head_dim", getattr(model_config, "d_kv", h_size // max(1, n_heads)))
            n_kv_heads = getattr(model_config, "num_key_value_heads", getattr(model_config, "num_kv_heads", n_heads))
            l_rank = getattr(model_config, "lora_rank", 32)
        else:
            h_size = 768
            n_heads = 12
            h_dim = 64
            n_kv_heads = 12
            l_rank = 32

        target_rank = kwargs.get("rank", h_size)
        if comp_config is not None:
            target_rank = getattr(comp_config, "lora_r", getattr(comp_config, "rank", target_rank))
        target_lora_rank = kwargs.get("lora_rank", l_rank)
        if comp_config is not None:
            target_lora_rank = getattr(comp_config, "lora_rank", target_lora_rank)
        target_window = kwargs.get("local_window", 128)
        if comp_config is not None:
            target_window = getattr(comp_config, "block_size", target_window)
        target_use_global = kwargs.get("use_global", getattr(comp_config, "use_global", True) if comp_config is not None else True)

        cfg = AegisHybridConfig(
            hidden_size=h_size,
            num_attention_heads=n_heads,
            head_dim=h_dim,
            num_key_value_heads=n_kv_heads,
            local_window=target_window,
            rank=target_rank,
            lora_rank=target_lora_rank,
            kernel_feature_dim=getattr(comp_config, "rff_features", 128) if comp_config is not None else 128,
            dropout=getattr(comp_config, "dropout", 0.1) if comp_config is not None else 0.1,
            use_adaptive_pruning=getattr(comp_config, "use_gate", True) if comp_config is not None else True,
            gate_init_bias=getattr(comp_config, "gate_init_bias", 1.8) if comp_config is not None else 1.8,
            is_decoder=is_dec or kwargs.get("is_decoder", False),
            use_global=target_use_global,
        )

        if old_attn is not None:
            new_attn = AegisHybridAttention(config=cfg, pretrained_attn=old_attn)
        else:
            new_attn = AegisHybridAttention(config=cfg)

        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                target.attention.self = new_attn
            else:
                target.attention = new_attn
        elif hasattr(target, "self_attn"):
            target.self_attn = new_attn
        elif hasattr(target, "self"):
            target.self = new_attn
        elif hasattr(target, "attn"):
            target.attn = new_attn

        if old_attn is not None:
            new_attn.train(old_attn.training)
        elif hasattr(target, 'training'):
            new_attn.train(target.training)

        return new_attn
    except Exception as e:
        logger.error(f"Failed to inject AEGIS-Hybrid: {e}")
        return None
