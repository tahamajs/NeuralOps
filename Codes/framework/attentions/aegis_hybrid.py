r"""
Aegis-Hybrid Attention (Corrected).

Combines low-rank factorized projections, localized sliding-window dense attention,
causal/bidirectional kernelized global linear attention with token-importance pruning,
and learned gating fusion.

Bug-fix map:
  AH-FIX-01  config assignment via self.config (not object.__setattr__).
  AH-FIX-02  decoder global attention enabled via chunked fp32 cumsum.
  AH-FIX-03  chunked causal cumsum (no 5D [B,H,N,Dh,Dh] tensor).
  AH-FIX-04  KV-cache implemented (tuple + DynamicCache compatible).
  AH-FIX-05  RoPE past_len offset for KV-cache decoding.
  AH-FIX-06  is_nested_encoder detection tightened.
  AH-FIX-07  ModelAdapter construction guarded.
  AH-FIX-08  mask_2d inference from 4D masks corrected.
  AH-FIX-09  injector recursion guard.
  AH-FIX-10  unified _detect_decoder helper.
  AH-FIX-11  from_distilbert_pretrained sets head_dim.
  AH-FIX-12  head_mask application consistent across branches.
  AH-FIX-13  clean-up of duplicate decoder detection in __init__.
  AH-FIX-14  local_window semantics documented.
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
from framework.core.attention_utils import repeat_kv

logger = logging.getLogger(__name__)


# ===========================================================================
# AH-FIX-10: unified decoder detection
# ===========================================================================

def _detect_decoder(
    config: Any,
    is_decoder: Optional[bool],
    old_attn: Optional[nn.Module],
) -> bool:
    if is_decoder is not None:
        return bool(is_decoder)
    if config is not None:
        if getattr(config, "is_decoder", False) or getattr(config, "causal", False):
            return True
        mt = str(getattr(config, "model_type", "")).lower()
        if mt.startswith(("gpt", "qwen", "llama", "mistral", "gemma", "phi",
                          "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")):
            return True
        arch = getattr(config, "architectures", None)
        if arch and any(("GPT" in str(a) or "CausalLM" in str(a)) for a in arch):
            return True
    if old_attn is not None:
        if getattr(old_attn, "is_decoder", False):
            return True
        if hasattr(old_attn, "c_attn"):
            return True
        if "gpt" in old_attn.__class__.__name__.lower():
            return True
    return False


# ===========================================================================
# Config (dataclass with __post_init__ works correctly here)
# ===========================================================================

@dataclass
class AegisHybridConfig:
    hidden_size: int = 768
    num_attention_heads: int = 12
    head_dim: int = 64
    num_key_value_heads: Optional[int] = None
    local_window: int = 128
    rank: int = 128
    lora_rank: int = 32
    kernel_feature_dim: int = 128
    dropout: float = 0.1
    use_adaptive_pruning: bool = True
    gate_init_bias: float = 1.8
    is_decoder: bool = False
    use_global: bool = True
    chunk_size: int = 64   # AH-FIX-03: chunk size for causal cumsum.

    def __post_init__(self) -> None:
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


# ===========================================================================
# FactorizedLinear (adds svd_rank alias for framework FLOPs detection)
# ===========================================================================

class FactorizedLinear(nn.Module):
    is_custom_attention = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 128,
        lora_rank: int = 32,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = max(1, min(int(rank), self.in_features, self.out_features))
        self.svd_rank = self.rank
        self.lora_rank = min(int(lora_rank), min(self.in_features, self.out_features)) \
            if lora_rank > 0 else 0

        factory_kwargs = {"device": device, "dtype": dtype}
        self.lin1 = nn.Linear(self.in_features, self.rank, bias=False, **factory_kwargs)
        self.lin2 = nn.Linear(self.rank, self.out_features, bias=bias, **factory_kwargs)

        self.lin1.weight.requires_grad = False
        self.lin2.weight.requires_grad = False
        if self.lin2.bias is not None:
            self.lin2.bias.requires_grad = False

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(self.lora_rank, self.in_features, **factory_kwargs))
            self.lora_B = nn.Parameter(torch.empty(self.out_features, self.lora_rank, **factory_kwargs))
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
            if self.lora_A is not None:
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)

    @property
    def weight(self) -> torch.Tensor:
        w = torch.matmul(self.lin2.weight, self.lin1.weight)
        if self.lora_A is not None:
            w = w + torch.matmul(self.lora_B, self.lora_A)
        return w

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.lin2.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lin2(self.lin1(x))
        if self.lora_A is not None:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B)
        return out


# ===========================================================================
# AH-FIX-03: chunked causal cumsum in fp32 (numerically stable)
# ===========================================================================

def _causal_global_linear_attention(
    phi_Q: torch.Tensor,   # (B, H, N, Dh)
    phi_K: torch.Tensor,   # (B, H, N_k, Dh)
    V: torch.Tensor,       # (B, H, N_k, Dh)
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    Causal linear attention via chunked cumsum in fp32.

    Memory per chunk: O(B * H * C * Dh * Dh). With C=64, Dh=64, B=8, H=12:
    ~8*12*64*64*64*4 bytes ≈ 100 MB. Safer than the naive (B,H,N,Dh,Dh).
    """
    B, H, N, Dh = phi_Q.shape
    device = phi_Q.device

    state_kv = torch.zeros(B, H, Dh, Dh, device=device, dtype=torch.float32)
    state_k = torch.zeros(B, H, Dh, device=device, dtype=torch.float32)
    outs = []

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        Qc = phi_Q[:, :, start:end].float()
        Kc = phi_K[:, :, start:end].float()
        Vc = V[:, :, start:end].float()

        # Local cumulative outer products: (B, H, C, Dh, Dh)
        KV_local = torch.einsum("bhcd,bhce->bhcde", Kc, Vc).cumsum(dim=2)
        K_local = Kc.cumsum(dim=2)

        KV_total = KV_local + state_kv.unsqueeze(2)
        K_total = K_local + state_k.unsqueeze(2)

        num = torch.einsum("bhcd,bhcde->bhce", Qc, KV_total)
        den = torch.einsum("bhcd,bhcd->bhc", Qc, K_total).unsqueeze(-1).clamp(min=1e-6)
        outs.append(num / den)

        state_kv = KV_total[:, :, -1]
        state_k = K_total[:, :, -1]

    return torch.cat(outs, dim=2)


def _noncausal_global_linear_attention(
    phi_Q: torch.Tensor,   # (B, H, N, Dh)
    phi_K: torch.Tensor,   # (B, H, N_k, Dh)
    V: torch.Tensor,       # (B, H, N_k, Dh)
) -> torch.Tensor:
    """Bidirectional linear attention via KV-first associativity."""
    KV = torch.einsum("bhne,bhnd->bhed", phi_K, V)   # (B, H, Dh, Dh)
    Z = phi_K.sum(dim=2, keepdim=True)               # (B, H, 1, Dh)
    num = torch.einsum("bhnd,bhde->bhne", phi_Q, KV) # (B, H, N, Dh)
    den = torch.einsum("bhnd,bhnd->bhn", phi_Q, Z).unsqueeze(-1).clamp(min=1e-6)
    return num / den


# ===========================================================================
# Main attention
# ===========================================================================

@ATTENTION_REGISTRY.register("aegis_hybrid", supports_causal=True, supports_gqa=True, supports_rope=True)
@ATTENTION_REGISTRY.register("aegis_hybrid_attn", supports_causal=True, supports_gqa=True, supports_rope=True)
@ATTENTION_REGISTRY.register("aegis_hybrid_attention", supports_causal=True, supports_gqa=True, supports_rope=True)
class AegisHybridAttention(BaseAttention):
    supports_causal: bool = True
    supports_gqa: bool = True
    supports_rope: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        hidden_size: Optional[int] = None,
        num_heads: Optional[int] = None,
        pretrained_attn: Optional[nn.Module] = None,
        rank: Optional[int] = None,
        lora_rank: Optional[int] = None,
        is_decoder: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        # super().__init__ with base config; rank passed as keyword-only.
        try:
            super().__init__(config, rank)
        except TypeError:
            super().__init__(config)

        if not isinstance(config, AegisHybridConfig):
            cfg_src = config if config is not None else kwargs
            if isinstance(cfg_src, dict):
                h_size = hidden_size or cfg_src.get("hidden_size", cfg_src.get("dim", 768))
                n_heads = num_heads or cfg_src.get("num_attention_heads", cfg_src.get("num_heads", 12))
                n_kv = cfg_src.get("num_key_value_heads", n_heads)
                h_dim = cfg_src.get("head_dim", h_size // max(1, n_heads))
                r = rank or cfg_src.get("rank", min(128, h_size))
                l_r = lora_rank or cfg_src.get("lora_rank", cfg_src.get("lora_r", 32))
                l_win = cfg_src.get("local_window", 128)
                drop = cfg_src.get("dropout", 0.1)
                gate_b = cfg_src.get("gate_init_bias", 1.8)
                use_prune = cfg_src.get("use_adaptive_pruning", True)
                use_g = cfg_src.get("use_global", True)
            else:
                h_size = hidden_size or kwargs.get("hidden_size",
                         getattr(cfg_src, "hidden_size",
                         getattr(cfg_src, "d_model", 768)))
                n_heads = num_heads or kwargs.get("num_heads",
                          getattr(cfg_src, "num_attention_heads",
                          getattr(cfg_src, "num_heads", 12)))
                n_kv = getattr(cfg_src, "num_key_value_heads",
                       getattr(cfg_src, "num_kv_heads", n_heads))
                h_dim = kwargs.get("head_dim",
                        getattr(cfg_src, "head_dim",
                        getattr(cfg_src, "d_kv", h_size // max(1, n_heads))))
                r = rank or kwargs.get("rank", getattr(cfg_src, "rank", min(128, h_size)))
                l_r = lora_rank or kwargs.get("lora_rank", getattr(cfg_src, "lora_rank",
                                     getattr(cfg_src, "lora_r", 32)))
                l_win = getattr(cfg_src, "local_window", getattr(cfg_src, "block_size", 128))
                drop = getattr(cfg_src, "dropout", 0.1)
                gate_b = getattr(cfg_src, "gate_init_bias", 1.8)
                use_prune = getattr(cfg_src, "use_adaptive_pruning", getattr(cfg_src, "use_gate", True))
                use_g = getattr(cfg_src, "use_global", True)

            dec = _detect_decoder(config, is_decoder, pretrained_attn)
            config = AegisHybridConfig(
                hidden_size=h_size,
                num_attention_heads=n_heads,
                head_dim=h_dim,
                num_key_value_heads=n_kv,
                local_window=l_win,
                rank=r,
                lora_rank=l_r,
                dropout=drop,
                use_adaptive_pruning=use_prune,
                gate_init_bias=gate_b,
                is_decoder=dec,
                use_global=use_g,
            )

        # AH-FIX-01: normal attribute assignment (nn.Module handles this).
        self.config = config

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
        self.dropout = nn.Dropout(config.dropout)

        # AH-FIX-13: single decoder detection at init, no duplicates.
        self.is_decoder: bool = _detect_decoder(config, is_decoder, pretrained_attn)
        self.use_global: bool = bool(getattr(config, "use_global", True))

        # AH-FIX-02: global attention for decoders is now safe (chunked fp32 cumsum),
        # so we no longer force-disable it.
        self.is_nested_encoder: bool = False
        self.rotary_emb = getattr(pretrained_attn, "rotary_emb", None)

        # Projections.
        self.q_proj = FactorizedLinear(self.hidden_size, self.all_head_size,
                                       rank=self.rank, lora_rank=self.lora_rank, bias=True)
        self.k_proj = FactorizedLinear(self.hidden_size, self.all_kv_head_size,
                                       rank=self.rank, lora_rank=self.lora_rank, bias=True)
        self.v_proj = FactorizedLinear(self.hidden_size, self.all_kv_head_size,
                                       rank=self.rank, lora_rank=self.lora_rank, bias=True)
        self.out_proj = FactorizedLinear(self.all_head_size, self.hidden_size,
                                         rank=self.rank, lora_rank=self.lora_rank, bias=True)

        # Pruning gate.
        if config.use_adaptive_pruning:
            self.prune_gate = nn.Sequential(
                nn.Linear(self.hidden_size, max(1, self.hidden_size // 4)),
                nn.ReLU(),
                nn.Linear(max(1, self.hidden_size // 4), self.num_heads),
            )
        else:
            self.register_module("prune_gate", None)

        # Fusion gate.
        self.fusion_gate = nn.Linear(self.hidden_size, self.num_heads)
        with torch.no_grad():
            self.fusion_gate.bias.fill_(config.gate_init_bias)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    # ------------------------------------------------------------------
    # Aliases (legacy + modern)
    # ------------------------------------------------------------------
    @property
    def q_proj1(self): return self.q_proj.lin1
    @property
    def q_proj2(self): return self.q_proj.lin2
    @property
    def k_proj1(self): return self.k_proj.lin1
    @property
    def k_proj2(self): return self.k_proj.lin2
    @property
    def v_proj1(self): return self.v_proj.lin1
    @property
    def v_proj2(self): return self.v_proj.lin2
    @property
    def out_proj1(self): return self.out_proj.lin1
    @property
    def out_proj2(self): return self.out_proj.lin2

    @property
    def q(self): return self.q_proj
    @property
    def k(self): return self.k_proj
    @property
    def v(self): return self.v_proj
    @property
    def o(self): return self.out_proj
    @property
    def query(self): return self.q_proj
    @property
    def key(self): return self.k_proj
    @property
    def value(self): return self.v_proj
    @property
    def dense(self): return self.out_proj
    @property
    def q_lin(self): return self.q_proj
    @property
    def k_lin(self): return self.k_proj
    @property
    def v_lin(self): return self.v_proj
    @property
    def out_lin(self): return self.out_proj
    @property
    def o_proj(self): return self.out_proj
    @property
    def c_proj(self): return self.out_proj

    # ------------------------------------------------------------------
    # SVD init
    # ------------------------------------------------------------------
    @staticmethod
    def _svd_init(w: torch.Tensor, b: Optional[torch.Tensor], proj: FactorizedLinear) -> None:
        with torch.no_grad():
            W = w.detach().float()
            if W.ndim != 2:
                raise ValueError(f"Expected 2D weight, got shape {tuple(W.shape)}")
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            r = min(proj.rank, U.size(1), Vh.size(0))
            proj.lin1.weight.data.zero_()
            proj.lin2.weight.data.zero_()
            in_copy = min(W.shape[1], proj.lin1.weight.size(1))
            out_copy = min(W.shape[0], proj.lin2.weight.size(0))
            proj.lin1.weight.data[:r, :in_copy].copy_(Vh[:r, :in_copy].to(proj.lin1.weight.dtype))
            US = U[:out_copy, :r] @ torch.diag(S[:r])
            proj.lin2.weight.data[:out_copy, :r].copy_(US.to(proj.lin2.weight.dtype))
            if b is not None and proj.lin2.bias is not None:
                n = min(b.numel(), proj.lin2.bias.numel())
                proj.lin2.bias.data[:n].copy_(b.detach()[:n].to(proj.lin2.bias.dtype))
            elif proj.lin2.bias is not None:
                proj.lin2.bias.data.zero_()
            proj.lin1.weight.requires_grad = False
            proj.lin2.weight.requires_grad = False
            if proj.lin2.bias is not None:
                proj.lin2.bias.requires_grad = False
            if proj.lora_A is not None:
                nn.init.kaiming_uniform_(proj.lora_A, a=math.sqrt(5))
                nn.init.zeros_(proj.lora_B)

    # ------------------------------------------------------------------
    # Weight extraction (universal across architectures)
    # ------------------------------------------------------------------
    def _extract_proj_weights(self, old_attn: nn.Module):
        from framework.core.adapters import UniversalAdapter
        extracted = {"q": (None, None), "k": (None, None), "v": (None, None), "o": (None, None)}

        # GPT-2 fused c_attn
        if hasattr(old_attn, "c_attn"):
            mod = old_attn.c_attn
            if hasattr(mod, "merge") and not getattr(mod, "merged", False):
                try: mod.merge()
                except Exception: pass
            base = getattr(mod, "base_layer", mod)
            c_w = getattr(base, "weight", getattr(mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data
                is_conv1d = (mod.__class__.__name__ == "Conv1D" or
                             base.__class__.__name__ == "Conv1D" or
                             hasattr(mod, "nf") or hasattr(base, "nf"))
                if is_conv1d or (c_w.dim() == 2 and c_w.shape[1] == 3 * self.hidden_size):
                    c_w = c_w.t().contiguous()
                split = c_w.size(0) // 3
                bias_holder = getattr(base, "bias", getattr(mod, "bias", None))
                c_b = bias_holder.data if bias_holder is not None else None
                for i, p in enumerate(("q", "k", "v")):
                    extracted[p] = (c_w[i * split:(i + 1) * split, :],
                                    c_b[i * split:(i + 1) * split] if c_b is not None else None)

        # GPT-2 c_proj
        if hasattr(old_attn, "c_proj") and extracted["o"][0] is None:
            mod = old_attn.c_proj
            if hasattr(mod, "merge") and not getattr(mod, "merged", False):
                try: mod.merge()
                except Exception: pass
            base = getattr(mod, "base_layer", mod)
            w = getattr(base, "weight", getattr(mod, "weight", None))
            if w is not None:
                w = w.data
                is_conv1d = (mod.__class__.__name__ == "Conv1D" or
                             base.__class__.__name__ == "Conv1D" or
                             hasattr(mod, "nf") or hasattr(base, "nf"))
                if is_conv1d or (w.dim() == 2 and w.shape[0] != self.hidden_size
                                 and w.shape[1] == self.hidden_size):
                    w = w.t().contiguous()
                b = getattr(base, "bias", getattr(mod, "bias", None))
                extracted["o"] = (w, b.data if b is not None else None)

        # BERT / RoBERTa nested
        if hasattr(old_attn, "self"):
            s = old_attn.self
            if hasattr(s, "query") and extracted["q"][0] is None:
                extracted["q"] = (s.query.weight.data, getattr(s.query, "bias", None))
            if hasattr(s, "key") and extracted["k"][0] is None:
                extracted["k"] = (s.key.weight.data, getattr(s.key, "bias", None))
            if hasattr(s, "value") and extracted["v"][0] is None:
                extracted["v"] = (s.value.weight.data, getattr(s.value, "bias", None))

        # BERT output.dense
        if (hasattr(old_attn, "output")
                and hasattr(getattr(old_attn, "output"), "dense")
                and extracted["o"][0] is None):
            d = old_attn.output.dense
            extracted["o"] = (d.weight.data, getattr(d, "bias", None))

        # Named projections
        named_map = {
            "q": ["q_proj", "q_lin", "query"],
            "k": ["k_proj", "k_lin", "key"],
            "v": ["v_proj", "v_lin", "value"],
            "o": ["o_proj", "out_proj", "out_lin", "dense"],
        }
        for proj, names in named_map.items():
            if extracted[proj][0] is None:
                for n in names:
                    if hasattr(old_attn, n):
                        m = getattr(old_attn, n)
                        if hasattr(m, "weight"):
                            extracted[proj] = (m.weight.data, getattr(m, "bias", None))
                            break

        # UniversalAdapter fallback
        for p in ("q", "k", "v", "o"):
            if extracted[p][0] is None:
                try:
                    w, b = UniversalAdapter.get_weight_bias(old_attn, p)
                    if w is not None:
                        extracted[p] = (w, b)
                except Exception as e:
                    logger.debug("UniversalAdapter failed for %s: %s", p, e)

        return extracted

    def copy_weights(self, old_attn: nn.Module) -> None:
        extracted = self._extract_proj_weights(old_attn)

        # AH-FIX-06: tighter nested encoder detection.
        cls_name = old_attn.__class__.__name__
        has_out = any(hasattr(old_attn, a) for a in
                      ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "wo", "o"))
        if hasattr(old_attn, "out_lin") or hasattr(old_attn, "c_proj") or hasattr(old_attn, "o_proj"):
            self.is_nested_encoder = False
        elif (hasattr(old_attn, "self") and hasattr(old_attn.self, "query")
              and hasattr(old_attn, "output") and hasattr(old_attn.output, "dense")):
            self.is_nested_encoder = True
        elif cls_name in ("BertSelfAttention", "RobertaSelfAttention",
                          "XLMRobertaSelfAttention", "ElectraSelfAttention"):
            self.is_nested_encoder = True
        else:
            self.is_nested_encoder = False

        missing = [p for p in ("q", "k", "v") if extracted[p][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            raise RuntimeError(
                f"AegisHybridAttention.copy_weights: missing {missing} from "
                f"{cls_name}. Refusing random init."
            )

        for proj, target in (("q", self.q_proj), ("k", self.k_proj), ("v", self.v_proj)):
            w, b = extracted[proj]
            if w is not None:
                self._svd_init(w, b, target)

        if self.is_nested_encoder:
            with torch.no_grad():
                self.out_proj.lin1.weight.data.zero_()
                self.out_proj.lin2.weight.data.zero_()
                r = min(self.out_proj.rank, self.hidden_size, self.all_head_size)
                self.out_proj.lin1.weight.data[:r, :r].fill_diagonal_(1.0)
                self.out_proj.lin2.weight.data[:r, :r].fill_diagonal_(1.0)
                if self.out_proj.lin2.bias is not None:
                    self.out_proj.lin2.bias.data.zero_()
        elif extracted["o"][0] is not None:
            self._svd_init(extracted["o"][0], extracted["o"][1], self.out_proj)

        if hasattr(old_attn, "rotary_emb") and old_attn.rotary_emb is not None:
            self.rotary_emb = old_attn.rotary_emb
        if hasattr(old_attn, "is_decoder"):
            self.is_decoder = bool(old_attn.is_decoder)

    # ------------------------------------------------------------------
    # AH-FIX-11: safe classmethod
    # ------------------------------------------------------------------
    @classmethod
    def from_distilbert_pretrained(
        cls, old_attn: nn.Module, model_config: Optional[Any] = None, **kwargs: Any,
    ) -> "AegisHybridAttention":
        if model_config is None:
            out_lin = getattr(old_attn, "out_lin", None) or getattr(old_attn, "dense", None)
            if out_lin is None or not hasattr(out_lin, "out_features"):
                raise ValueError(
                    "from_distilbert_pretrained: could not infer hidden_size from "
                    "out_lin or dense. Pass model_config explicitly."
                )
            class DummyConfig:
                pass
            model_config = DummyConfig()
            model_config.hidden_size = out_lin.out_features
            n_heads = getattr(old_attn, "n_heads",
                              getattr(old_attn, "num_attention_heads",
                                      getattr(old_attn, "num_heads", 12)))
            model_config.num_attention_heads = n_heads
            model_config.head_dim = model_config.hidden_size // max(1, n_heads)
        return cls(config=model_config, pretrained_attn=old_attn, **kwargs)

    # ------------------------------------------------------------------
    # Mask helper (AH-FIX-08)
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_valid_mask(
        attention_mask: Optional[torch.Tensor],
        B: int, N_kv: int, device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Return bool (B, N_kv): True = valid key. None if no mask."""
        if attention_mask is None:
            return None
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
            # (B, 1, N_q, N_kv): a key is valid iff some query can attend to it.
            if attention_mask.dtype == torch.bool:
                v = attention_mask
            elif attention_mask.dtype.is_floating_point:
                v = attention_mask > -1e4
            else:
                v = attention_mask > 0
            return v.any(dim=-2).squeeze(1)
        raise ValueError(f"Unsupported attention_mask dim: {attention_mask.dim()}")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
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
        layer_idx: int = 0,
        *args: Any,
        **kwargs: Any,
    ):
        B, N, _ = hidden_states.shape
        device = hidden_states.device
        Dh = self.head_dim

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        N_kv_new = kv_input.shape[1]

        # ---- Projections ----
        Q = self.q_proj(hidden_states).view(B, N, self.num_heads, Dh).transpose(1, 2).contiguous()
        K_new = self.k_proj(kv_input).view(B, N_kv_new, self.num_kv_heads, Dh).transpose(1, 2).contiguous()
        V_new = self.v_proj(kv_input).view(B, N_kv_new, self.num_kv_heads, Dh).transpose(1, 2).contiguous()

        # ---- RoPE with past offset (AH-FIX-05) ----
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            past_len_rope = 0
            if past_key_value is not None:
                if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 1:
                    past_len_rope = past_key_value[0].shape[-2]
                elif hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
                    past_len_rope = past_key_value.key_cache[0].shape[-2]
            cos = cos[..., past_len_rope: past_len_rope + N, :]
            sin = sin[..., past_len_rope: past_len_rope + N, :]
            Q, K_new = self._apply_rope(Q, K_new, cos, sin)
        elif hasattr(self, "rotary_emb") and self.rotary_emb is not None:
            try:
                from framework.utils.compat import maybe_apply_rotary
                Q, K_new = maybe_apply_rotary(Q, K_new, None, self.rotary_emb,
                                              kwargs.get("position_ids"))
            except Exception as e:
                logger.warning("maybe_apply_rotary failed: %s", e)

        # ---- KV cache (AH-FIX-04) ----
        past_k = past_v = None
        if past_key_value is not None and not is_cross:
            if isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2:
                past_k, past_v = past_key_value[0], past_key_value[1]
            elif hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
                idx = min(layer_idx, len(past_key_value.key_cache) - 1)
                past_k = past_key_value.key_cache[idx]
                past_v = past_key_value.value_cache[idx]

        if past_k is not None and not is_cross:
            K_cat = torch.cat([past_k, K_new], dim=-2)
            V_cat = torch.cat([past_v, V_new], dim=-2)
        else:
            K_cat = K_new
            V_cat = V_new

        if past_key_value is not None and hasattr(past_key_value, "update") and not is_cross:
            past_key_value.update(K_new, V_new, layer_idx)

        N_kv = K_cat.shape[2]
        past_len = N_kv - N if (self.is_decoder and not is_cross) else 0

        # GQA expansion.
        K_full = repeat_kv(K_cat, self.kv_repeat) if self.kv_repeat > 1 else K_cat
        V_full = repeat_kv(V_cat, self.kv_repeat) if self.kv_repeat > 1 else V_cat

        # ---- Mask ----
        eff_mask = encoder_attention_mask if is_cross else attention_mask
        token_valid = self._prepare_valid_mask(eff_mask, B, N_kv, device)

        # Build a full token_valid if cache extends the sequence.
        if past_len > 0 and token_valid is not None:
            past_valid = torch.ones(B, past_len, device=device, dtype=torch.bool)
            token_valid = torch.cat([past_valid, token_valid], dim=1)

        # ---- Local windowed attention ----
        scale = Dh ** -0.5
        scores = torch.matmul(Q.float(), K_full.float().transpose(-2, -1).contiguous()) * scale

        idx_q = (past_len + torch.arange(N, device=device)).unsqueeze(1)
        idx_k = torch.arange(N_kv, device=device).unsqueeze(0)
        band = (idx_q - idx_k).abs() <= self.local_window
        if self.is_decoder and not is_cross:
            band = band & (idx_k <= idx_q)
        band = band.unsqueeze(0).unsqueeze(0)

        mask_val = -1e4
        scores = scores.masked_fill(~band, mask_val)
        if token_valid is not None:
            scores = scores.masked_fill(~token_valid[:, None, None, :].bool(), mask_val)

        scores = torch.nan_to_num(scores, nan=mask_val, posinf=1e4, neginf=mask_val)
        any_valid = band if token_valid is None else (band & token_valid[:, None, None, :])
        any_valid = any_valid.any(dim=-1, keepdim=True).to(scores.dtype)

        attn_local = F.softmax(scores, dim=-1).to(Q.dtype) * any_valid
        attn_local = self.dropout(attn_local)

        local_out = torch.matmul(attn_local, V_full)  # (B, H, N, Dh)

        # ---- Global linear attention ----
        if not self.use_global or is_cross:
            fused = local_out
        else:
            # Pruning gate.
            if self.prune_gate is not None:
                keep = torch.sigmoid(self.prune_gate(hidden_states))  # (B, N, H)
                if self.kv_repeat > 1:
                    keep_kv = keep[:, :, :self.num_kv_heads]
                    keep_kv_full = repeat_kv(
                        keep_kv.transpose(1, 2).unsqueeze(-1), self.kv_repeat
                    ).squeeze(-1)
                else:
                    keep_kv_full = keep.transpose(1, 2)  # (B, H, N)
            else:
                keep_kv_full = None

            phi_Q = F.elu(Q.float()) + 1.0                     # (B, H, N, Dh)
            phi_K = F.elu(K_full.float()) + 1.0                # (B, H, N_k, Dh)
            V_f = V_full.float()                               # (B, H, N_k, Dh)

            if keep_kv_full is not None:
                k_expanded = keep_kv_full.unsqueeze(-1)        # (B, H, N, 1)
                phi_K = phi_K * k_expanded
                V_f = V_f * k_expanded

            # Padding mask applied after phi so state is clean.
            if token_valid is not None:
                mask_kv = token_valid[:, None, :, None].to(phi_K.dtype)
                phi_K = phi_K * mask_kv
                V_f = V_f * mask_kv

            if self.is_decoder and not is_cross:
                global_out = _causal_global_linear_attention(
                    phi_Q, phi_K, V_f, chunk_size=self.config.chunk_size
                )
            else:
                global_out = _noncausal_global_linear_attention(phi_Q, phi_K, V_f)

            global_out = torch.nan_to_num(global_out, nan=0.0).to(Q.dtype)

            # Fusion gate.
            alpha = torch.sigmoid(self.fusion_gate(hidden_states)).unsqueeze(-1)  # (B, N, H, 1)
            alpha = alpha.transpose(1, 2)                                          # (B, H, N, 1)
            fused = alpha * local_out + (1.0 - alpha) * global_out

        # ---- head_mask (AH-FIX-12: single application point) ----
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(-1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                head_mask = head_mask[:, :, :1].unsqueeze(-1)
            fused = fused * head_mask.to(fused.dtype)

        # ---- Reshape and output projection ----
        out = fused.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            final = out
        else:
            final = self.out_proj(out)
            final = self.dropout(final)
        final = torch.nan_to_num(final, nan=0.0, posinf=1e4, neginf=-1e4)

        attn_ret = attn_local if output_attentions else None
        return final, attn_ret

    @staticmethod
    def _apply_rope(q, k, cos, sin):
        def _rotate_half(x):
            d = x.shape[-1]
            h = d // 2
            rot = torch.cat((-x[..., h:2 * h], x[..., :h]), dim=-1)
            if d % 2 == 1:
                rot = torch.cat([rot, x[..., -1:]], dim=-1)
            return rot

        if cos.dim() == 3:
            cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        while cos.dim() < q.dim():
            cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)

        if cos.shape[-2] != q.shape[-2]:
            raise ValueError(
                f"RoPE seq len mismatch: cos {cos.shape[-2]} vs q {q.shape[-2]}"
            )

        rotary_dim = cos.shape[-1]
        head_dim = q.shape[-1]
        if rotary_dim <= head_dim:
            q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
            k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
            q_rot = (q_rot * cos.to(q_rot.dtype)) + (_rotate_half(q_rot) * sin.to(q_rot.dtype))
            k_rot = (k_rot * cos.to(k_rot.dtype)) + (_rotate_half(k_rot) * sin.to(k_rot.dtype))
            q = torch.cat([q_rot, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot
            k = torch.cat([k_rot, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot
        else:
            cos_cut, sin_cut = cos[..., :head_dim], sin[..., :head_dim]
            q = (q * cos_cut.to(q.dtype)) + (_rotate_half(q) * sin_cut.to(q.dtype))
            k = (k * cos_cut.to(k.dtype)) + (_rotate_half(k) * sin_cut.to(k.dtype))
        return q, k

    def get_ortho_loss(self) -> torch.Tensor:
        return torch.zeros((), device=self.out_proj.lin1.weight.device)


# ===========================================================================
# Injection (AH-FIX-09: recursion guard)
# ===========================================================================

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
    _depth: int = 0,
    **kwargs: Any,
):
    if _depth > 8:
        logger.warning("inject_aegis_hybrid: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer", None))
    if target is None:
        return None

    if getattr(target, "_aegis_injected", False):
        return target

    cfg_root = model_config if model_config is not None else getattr(target, "config", None)

    # Whole-model traversal.
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
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif hasattr(target, "bert") and hasattr(target.bert, "encoder") and hasattr(target.bert.encoder, "layer"):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_aegis_hybrid(
                layer=lyr, model_config=cfg_root, comp_config=comp_config,
                layer_idx=idx, total_layers=tot, _depth=_depth + 1, **kwargs,
            )
            if res is not None:
                num_injected += 1
        if num_injected > 0:
            target._aegis_injected = True
        return target if num_injected > 0 else None

    target = layer
    try:
        if old_attn is None:
            for attr in ("attention", "attn", "self_attn", "self"):
                if hasattr(target, attr):
                    cand = getattr(target, attr)
                    if cand is not None:
                        old_attn = getattr(cand, "self", cand)
                        break

        if model_config is None:
            if any(hasattr(target, a) for a in ("hidden_size", "d_model", "n_embd", "dim")):
                model_config = target
            elif old_attn is not None:
                model_config = getattr(old_attn, "config", None)

        h_size = getattr(model_config, "hidden_size",
                 getattr(model_config, "d_model",
                 getattr(model_config, "n_embd",
                 getattr(model_config, "dim", 768)))) if model_config is not None else 768
        n_heads = getattr(model_config, "num_attention_heads",
                  getattr(model_config, "num_heads",
                  getattr(model_config, "n_heads",
                  getattr(model_config, "n_head", 12)))) if model_config is not None else 12
        h_dim = getattr(model_config, "head_dim",
                getattr(model_config, "d_kv",
                h_size // max(1, n_heads))) if model_config is not None else h_size // max(1, n_heads)
        n_kv_heads = getattr(model_config, "num_key_value_heads",
                     getattr(model_config, "num_kv_heads", n_heads)) if model_config is not None else n_heads

        is_dec = _detect_decoder(model_config, kwargs.get("is_decoder", None), old_attn)

        target_rank = kwargs.get("rank", getattr(comp_config, "rank",
                          getattr(comp_config, "lora_r", 64)) if comp_config else 64)
        target_lora = kwargs.get("lora_rank", getattr(comp_config, "lora_rank", 32) if comp_config else 32)
        target_window = kwargs.get("local_window",
                        getattr(comp_config, "block_size", 128) if comp_config else 128)
        target_use_global = kwargs.get("use_global",
                            getattr(comp_config, "use_global", True) if comp_config else True)

        cfg = AegisHybridConfig(
            hidden_size=h_size,
            num_attention_heads=n_heads,
            head_dim=h_dim,
            num_key_value_heads=n_kv_heads,
            local_window=target_window,
            rank=target_rank,
            lora_rank=target_lora,
            dropout=getattr(comp_config, "dropout", 0.1) if comp_config else 0.1,
            use_adaptive_pruning=getattr(comp_config, "use_gate", True) if comp_config else True,
            gate_init_bias=getattr(comp_config, "gate_init_bias", 1.8) if comp_config else 1.8,
            is_decoder=is_dec,
            use_global=target_use_global,
            chunk_size=getattr(comp_config, "chunk_size", 64) if comp_config else 64,
        )

        try:
            new_attn = AegisHybridAttention(
                config=cfg, pretrained_attn=old_attn,
            )
        except Exception as e:
            logger.error("inject_aegis_hybrid: construction failed: %s", e)
            return None

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
        return new_attn
    except Exception as e:
        logger.exception("Failed to inject AEGIS-Hybrid: %s", e)
        return None