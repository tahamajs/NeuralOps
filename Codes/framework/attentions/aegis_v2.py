r"""
Aegis-Attention V2 Module.

Implements hybrid low-rank factorized attention with local sliding window attention,
global linear kernel attention with adaptive soft-pruning gating, and dynamic multi-head fusion.
Hardened for multi-architecture deployment across DistilBERT, RoBERTa, GPT-2, and Qwen3.5.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY
from framework.core.attention_utils import repeat_kv

logger = logging.getLogger(__name__)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    half = d // 2
    x1 = x[..., :half]
    x2 = x[..., half : 2 * half]  # safe for odd d: leaves trailing element out
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


@dataclass
class AegisAttentionConfigV2:
    r"""Configuration dataclass for AegisAttentionV2."""
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_heads: int = 12
    head_dim: int = 64
    num_key_value_heads: int = 12
    local_window: int = 64
    rank: int = 384
    kernel_feature_dim: int = 128
    dropout: float = 0.1
    use_adaptive_pruning: bool = True
    gate_init_bias: float = 1.8
    is_decoder: bool = False


AegisConfigV2 = AegisAttentionConfigV2


class FactorisedLinear(nn.Module):
    is_custom_attention: bool = True
    r"""
    SVD-factorized linear layer with energy-based rank truncation and trainable LoRA adapters.
    Base SVD weights are frozen; LoRA branch provides trainable parameters.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 384,
        bias: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        energy: float = 0.95,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, in_features, out_features)
        self.energy = energy
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0

        self.down = nn.Linear(in_features, self.rank, bias=False)
        self.up = nn.Linear(self.rank, out_features, bias=bias)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.randn(lora_rank, in_features) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.down.weight)
        nn.init.orthogonal_(self.up.weight)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)
        if self.lora_rank > 0 and self.lora_A is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
            nn.init.zeros_(self.lora_B)
            self.lora_B.requires_grad = True

    def decompose_from_weight(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        energy: Optional[float] = None,
    ) -> None:
        with torch.no_grad():
            W = weight.float()
            if W.shape[0] < self.out_features and self.out_features % W.shape[0] == 0:
                rep = self.out_features // W.shape[0]
                W = W.repeat(rep, 1)
                if bias is not None:
                    bias = bias.repeat(rep)
            elif W.shape[0] > self.out_features and W.shape[0] % self.out_features == 0:
                rep = W.shape[0] // self.out_features
                W = W.view(self.out_features, rep, -1).mean(dim=1)
                if bias is not None:
                    bias = bias.view(self.out_features, rep).mean(dim=1)
            elif W.shape[0] != self.out_features or W.shape[1] != self.in_features:
                padded = torch.zeros(self.out_features, self.in_features, device=W.device, dtype=W.dtype)
                padded[:min(self.out_features, W.shape[0]), :min(self.in_features, W.shape[1])] = W[:min(self.out_features, W.shape[0]), :min(self.in_features, W.shape[1])]
                W = padded

            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            S = torch.clamp(S, min=1e-6)
            target_energy = energy if energy is not None else self.energy
            total = (S**2).sum()
            cum = torch.cumsum(S**2, dim=0) / (total + 1e-5)
            calc_rank = int((cum < target_energy).sum().item()) + 1
            r = min(self.rank, calc_rank, U.size(1), Vt.size(0))
            r = max(1, r)

            A = U[:, :r] * S[:r].unsqueeze(0)
            B = Vt[:r, :]

            self.down.weight.data.zero_()
            self.up.weight.data.zero_()
            self.down.weight.data[:r, :] = B.to(self.down.weight.dtype)
            self.up.weight.data[:, :r] = A.to(self.up.weight.dtype)

            if bias is not None and self.up.bias is not None:
                self.up.bias.data[:min(len(self.up.bias), len(bias))].copy_(bias[:min(len(self.up.bias), len(bias))].to(self.up.bias.dtype))
            elif self.up.bias is not None:
                self.up.bias.data.zero_()

            self.freeze_base_weights()

    def freeze_base_weights(self) -> None:
        self.down.weight.requires_grad = False
        self.up.weight.requires_grad = False
        if self.up.bias is not None:
            self.up.bias.requires_grad = False
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            self.lora_A.requires_grad = True
            self.lora_B.requires_grad = True

    @property
    def weight(self) -> torch.Tensor:
        base_w = torch.matmul(self.up.weight, self.down.weight)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            return base_w + (self.lora_B @ self.lora_A) * self.lora_scaling
        return base_w

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.up.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.up(self.down(x))
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        return out


@ATTENTION_REGISTRY.register("aegis_v2", supports_causal=True, supports_gqa=True)
class AegisAttentionV2(BaseAttention):
    r"""AEGIS-Attention V2 Multi-Head Layer."""

    supports_causal: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any = None,
        pretrained_attn: Optional[nn.Module] = None,
        is_decoder: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        if config is None or not isinstance(config, AegisConfigV2):
            h_size = kwargs.get("hidden_size", kwargs.get("d_model", getattr(config, "hidden_size", getattr(config, "d_model", getattr(config, "dim", 768))) if config is not None else 768))
            n_heads = kwargs.get("num_heads", kwargs.get("num_attention_heads", getattr(config, "num_attention_heads", getattr(config, "num_heads", 12)) if config is not None else 12))
            h_dim = kwargs.get("head_dim", getattr(config, "head_dim", getattr(config, "d_kv", h_size // max(1, n_heads))) if config is not None else h_size // max(1, n_heads))
            n_kv_heads = kwargs.get("num_key_value_heads", kwargs.get("num_kv_heads", getattr(config, "num_key_value_heads", getattr(config, "num_kv_heads", n_heads)) if config is not None else n_heads))
            dec = is_decoder if is_decoder is not None else (
                getattr(config, "is_decoder", False)
                or getattr(config, "causal", False)
                or getattr(config, "model_type", "").lower() in ("gpt2", "qwen", "qwen2", "llama") or any(str(getattr(cfg if "cfg" in locals() else (config if "config" in locals() else None), "model_type", "")).lower().startswith(p) for p in ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm"))
                if config else False
            )
            config = AegisConfigV2(
                hidden_size=h_size,
                num_attention_heads=n_heads,
                num_heads=n_heads,
                num_key_value_heads=n_kv_heads,
                head_dim=h_dim,
                local_window=kwargs.get("local_window", getattr(config, "block_size", getattr(config, "local_window", 64)) if config is not None else 64),
                rank=kwargs.get("rank", getattr(config, "lora_r", getattr(config, "rank", 64)) if config is not None else 64),
                kernel_feature_dim=getattr(config, "rff_features", getattr(config, "kernel_feature_dim", 128)) if config is not None else 128,
                dropout=kwargs.get("dropout", getattr(config, "dropout", 0.1) if config is not None else 0.1),
                use_adaptive_pruning=getattr(config, "use_gate", True) if config is not None else True,
                gate_init_bias=getattr(config, "gate_init_bias", 1.8) if config is not None else 1.8,
                is_decoder=dec,
            )

        super().__init__(config)
        self.config = config
        self.hidden_size: int = config.hidden_size
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.head_dim
        self.num_kv_heads: int = getattr(config, "num_key_value_heads", self.num_heads)
        self.kv_repeat: int = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.all_head_size: int = self.num_heads * self.head_dim
        self.all_kv_head_size: int = self.num_kv_heads * self.head_dim
        self.supports_gqa: bool = True
        self.local_window: int = config.local_window
        self.rank: int = config.rank
        self.kernel_dim: int = config.kernel_feature_dim
        self.dropout: nn.Dropout = nn.Dropout(config.dropout)
        self.is_decoder: bool = getattr(config, "is_decoder", False)
        self.is_nested_encoder: bool = False

        self.q_proj = FactorisedLinear(self.hidden_size, self.all_head_size, rank=self.rank)
        self.k_proj = FactorisedLinear(self.hidden_size, self.all_kv_head_size, rank=self.rank)
        self.v_proj = FactorisedLinear(self.hidden_size, self.all_kv_head_size, rank=self.rank)
        self.out_proj = FactorisedLinear(self.all_head_size, self.hidden_size, rank=self.rank)

        if config.use_adaptive_pruning:
            self.prune_gate: Optional[nn.Sequential] = nn.Sequential(
                nn.Linear(self.hidden_size, max(1, self.hidden_size // 4)),
                nn.ReLU(),
                nn.Linear(max(1, self.hidden_size // 4), self.num_heads),
            )
        else:
            self.prune_gate = None

        self.fusion_gate = nn.Linear(self.hidden_size, self.num_heads)
        self.fusion_gate.bias.data.fill_(config.gate_init_bias)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    # Legacy sublayer accessors
    @property
    def q_proj1(self) -> nn.Linear:
        return self.q_proj.down

    @property
    def q_proj2(self) -> nn.Linear:
        return self.q_proj.up

    @property
    def k_proj1(self) -> nn.Linear:
        return self.k_proj.down

    @property
    def k_proj2(self) -> nn.Linear:
        return self.k_proj.up

    @property
    def v_proj1(self) -> nn.Linear:
        return self.v_proj.down

    @property
    def v_proj2(self) -> nn.Linear:
        return self.v_proj.up

    @property
    def out_proj1(self) -> nn.Linear:
        return self.out_proj.down

    @property
    def out_proj2(self) -> nn.Linear:
        return self.out_proj.up

    @property
    def query(self) -> FactorisedLinear:
        return self.q_proj

    @property
    def key(self) -> FactorisedLinear:
        return self.k_proj

    @property
    def value(self) -> FactorisedLinear:
        return self.v_proj

    @property
    def q_lin(self) -> FactorisedLinear:
        return self.q_proj

    @property
    def k_lin(self) -> FactorisedLinear:
        return self.k_proj

    @property
    def v_lin(self) -> FactorisedLinear:
        return self.v_proj

    @property
    def out_lin(self) -> FactorisedLinear:
        return self.out_proj

    @property
    def dense(self) -> FactorisedLinear:
        return self.out_proj

    @property
    def o_proj(self) -> FactorisedLinear:
        return self.out_proj

    @property
    def c_proj(self) -> FactorisedLinear:
        return self.out_proj

    def _copy_pretrained(self, old_attn: nn.Module) -> None:
        self.copy_weights(old_attn)

    def copy_weights(self, old_attn: Any) -> None:
        from framework.core.adapters import UniversalAdapter

        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. UniversalAdapter extraction
        for proj in ["q", "k", "v", "o"]:
            try:
                w, b = UniversalAdapter.get_weight_bias(old_attn, proj)
                if w is not None:
                    extracted[proj] = (w, b)
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        # 2. GPT-2 Fused c_attn Conv1D handling
        if hasattr(old_attn, "c_attn") and (extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None):
            c_attn_mod = old_attn.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            weight = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if weight is not None:
                weight = weight.data.clone()
                is_conv1d = (c_attn_mod.__class__.__name__ == "Conv1D") or (base_c.__class__.__name__ == "Conv1D") or hasattr(c_attn_mod, "nf") or hasattr(base_c, "nf")
                if is_conv1d or (weight.dim() == 2 and weight.shape[0] < weight.shape[1]) or (weight.dim() == 2 and weight.shape[1] == 3 * self.hidden_size):
                    weight = weight.t().contiguous()
                bias = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                bias_data = bias.data if bias is not None else None
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

        if hasattr(old_attn, "c_proj") and extracted["o"][0] is None:
            c_proj_mod = old_attn.c_proj
            base_p = getattr(c_proj_mod, "base_layer", c_proj_mod)
            w_o = getattr(base_p, "weight", getattr(c_proj_mod, "weight", None))
            if w_o is not None:
                w_o = w_o.data.clone()
                is_conv1d = (c_proj_mod.__class__.__name__ == "Conv1D") or (base_p.__class__.__name__ == "Conv1D") or hasattr(c_proj_mod, "nf") or hasattr(base_p, "nf")
                if is_conv1d or (w_o.dim() == 2 and w_o.shape[0] < w_o.shape[1]) or (w_o.dim() == 2 and w_o.shape[0] != self.hidden_size and w_o.shape[1] == self.hidden_size):
                    w_o = w_o.t().contiguous()
                b_o = getattr(base_p, "bias", getattr(c_proj_mod, "bias", None))
                extracted["o"] = (w_o, b_o.data if b_o is not None else None)

        # 3. RoBERTa / BERT Nested Encoder handling
        if hasattr(old_attn, "self") and (extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None):
            s = getattr(old_attn, "self")
            if hasattr(s, "query"):
                extracted["q"] = (s.query.weight.data, getattr(s.query, "bias", None))
            if hasattr(s, "key"):
                extracted["k"] = (s.key.weight.data, getattr(s.key, "bias", None))
            if hasattr(s, "value"):
                extracted["v"] = (s.value.weight.data, getattr(s.value, "bias", None))

        if hasattr(old_attn, "output") and hasattr(getattr(old_attn, "output"), "dense") and extracted["o"][0] is None:
            dense = old_attn.output.dense
            extracted["o"] = (dense.weight.data, getattr(dense, "bias", None))

        # 4. Modern Decoders & DistilBERT
        named_map = {
            "q": ["q_proj", "q_lin", "query"],
            "k": ["k_proj", "k_lin", "key"],
            "v": ["v_proj", "v_lin", "value"],
            "o": ["o_proj", "out_proj", "out_lin", "dense"],
        }
        for proj, names in named_map.items():
            if extracted[proj][0] is None:
                for name in names:
                    if hasattr(old_attn, name):
                        layer = getattr(old_attn, name)
                        if hasattr(layer, "weight"):
                            extracted[proj] = (layer.weight.data, getattr(layer, "bias", None))
                            break

        cls_name = old_attn.__class__.__name__
        has_out_proj = any(hasattr(old_attn, attr) for attr in ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "wo", "o", "proj"))
        if (
            hasattr(old_attn, "self")
            and hasattr(getattr(old_attn, "self"), "query")
            and hasattr(old_attn, "output")
            and hasattr(getattr(old_attn, "output"), "dense")
        ):
            self.is_nested_encoder = True
        elif (
            cls_name in ("BertSelfAttention", "RobertaSelfAttention", "XLMRobertaSelfAttention", "ElectraSelfAttention")
            or "RobertaSelfAttention" in str(type(old_attn))
            or "BertSelfAttention" in str(type(old_attn))
            or "roberta" in str(type(old_attn)).lower()
            or ("bert" in str(type(old_attn)).lower() and "self" in str(type(old_attn)).lower())
            or (cls_name.endswith("SelfAttention") and not has_out_proj)
        ):
            self.is_nested_encoder = True
        else:
            self.is_nested_encoder = False

        missing = [k for k in ["q", "k", "v"] if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            raise RuntimeError(
                f"AegisAttentionV2.copy_weights: could not extract mandatory projections {missing} "
                f"from {old_attn.__class__.__name__}. Refusing to install randomly-initialised weights."
            )

        for proj_key in ["q", "k", "v"]:
            w = extracted[proj_key][0]
            if w is not None:
                norm = torch.linalg.norm(w.float())
                if norm < 1e-4:
                    logger.warning(f"AegisAttentionV2 extracted weight for {proj_key} has near-zero Frobenius norm ({norm.item():.6f})")

        if extracted["o"][0] is not None:
            norm_o = torch.linalg.norm(extracted["o"][0].float())
            if norm_o < 1e-4:
                logger.warning(f"AegisAttentionV2 extracted weight for output projection has near-zero Frobenius norm ({norm_o.item():.6f})")

        def _orient(w: torch.Tensor, in_dim: int) -> torch.Tensor:
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

        with torch.no_grad():
            for proj_key, target in [("q", self.q_proj), ("k", self.k_proj), ("v", self.v_proj), ("o", self.out_proj)]:
                w, b = extracted[proj_key]
                if w is not None:
                    in_dim = self.all_head_size if proj_key == "o" else self.hidden_size
                    w_oriented = _orient(w, in_dim)
                    target.decompose_from_weight(w_oriented, b)
                elif proj_key == "o" and self.is_nested_encoder:
                    # For nested encoders the outer dense handles output projection;
                    # initialise out_proj to identity so it acts as a pass-through if called.
                    eye_w = torch.eye(self.all_head_size, self.all_head_size,
                                      device=self.out_proj.up.weight.device,
                                      dtype=self.out_proj.up.weight.dtype)
                    target.decompose_from_weight(eye_w, None)

        if hasattr(old_attn, "is_decoder"):
            self.is_decoder = bool(old_attn.is_decoder)
        elif hasattr(old_attn, "c_attn") or "gpt" in old_attn.__class__.__name__.lower():
            self.is_decoder = True

    @classmethod
    def from_distilbert_pretrained(
        cls, old_attn: nn.Module, model_config: Optional[Any] = None, **kwargs: Any
    ) -> "AegisAttentionV2":
        if model_config is None:
            class DummyConfig:
                pass
            model_config = DummyConfig()
            model_config.hidden_size = getattr(old_attn, "out_lin", getattr(old_attn, "dense", None)).out_features
            model_config.num_attention_heads = getattr(old_attn, "n_heads", getattr(old_attn, "num_attention_heads", 4))
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
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, _ = hidden_states.shape

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        N_kv = kv_input.shape[1]

        eff_mask = encoder_attention_mask if is_cross else attention_mask
        token_valid: Optional[torch.Tensor] = None
        if eff_mask is not None:
            if eff_mask.dim() == 4:
                if eff_mask.shape[-2] == eff_mask.shape[-1]:
                    token_valid = eff_mask[:, 0, 0, :] > -10000.0
                else:
                    token_valid = eff_mask.squeeze(1).squeeze(1) > -10000.0
            elif eff_mask.dim() == 3:
                token_valid = eff_mask.squeeze(1).bool()
            elif eff_mask.dim() == 2:
                token_valid = eff_mask.bool()

        mask_2d = token_valid

        Q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim)
        K = self.k_proj(kv_input).view(B, N_kv, self.num_kv_heads, self.head_dim)
        V = self.v_proj(kv_input).view(B, N_kv, self.num_kv_heads, self.head_dim)

        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            Q_r, K_r = apply_rotary_pos_emb(Q.transpose(1, 2).contiguous(), K.transpose(1, 2).contiguous(), cos, sin)
            Q, K = Q_r.transpose(1, 2).contiguous(), K_r.transpose(1, 2).contiguous()

        if self.kv_repeat > 1:
            K = repeat_kv(K.transpose(1, 2).contiguous(), self.kv_repeat).transpose(1, 2).contiguous()
            V = repeat_kv(V.transpose(1, 2).contiguous(), self.kv_repeat).transpose(1, 2).contiguous()

        if mask_2d is not None and not self.is_decoder:
            m4 = mask_2d.unsqueeze(-1).unsqueeze(-1)
            Q = Q * m4
            K = K * m4
            V = V * m4

        # ---- Local attention ----
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.einsum("bnhd,bmhd->bhnm", Q.float(), K.float()) * scale
        idx_q = torch.arange(N, device=hidden_states.device).unsqueeze(1)
        idx_k = torch.arange(N_kv, device=hidden_states.device).unsqueeze(0)
        band = (idx_q - idx_k).abs() <= self.local_window
        band = band.unsqueeze(0).unsqueeze(0)

        if self.is_decoder and not is_cross and N > 1:
            causal = (idx_k <= idx_q).unsqueeze(0).unsqueeze(0)
            band = band & causal

        mask_val = -10000.0
        scores = scores.masked_fill(~band, mask_val)

        if eff_mask is not None and eff_mask.dim() == 4:
            if eff_mask.dtype == torch.bool:
                scores = scores.masked_fill(~eff_mask, mask_val)
            else:
                scores = scores + eff_mask.float()
        elif mask_2d is not None:
            key_mask = mask_2d.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~key_mask, mask_val)

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        attn_local = F.softmax(scores, dim=-1).to(Q.dtype)
        attn_local = self.dropout(attn_local)

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(-1)
            attn_local = attn_local * head_mask

        local_out = torch.einsum("bhnm,bmhd->bnhd", attn_local, V)

        if self.is_decoder:
            fused = local_out
        else:
            # ---- Global linear attention ----
            def phi(x: torch.Tensor) -> torch.Tensor:
                return F.elu(x) + 1.0

            if self.prune_gate is not None:
                keep = torch.sigmoid(self.prune_gate(hidden_states))
                Kg = K * keep.unsqueeze(-1)
                Vg = V * keep.unsqueeze(-1)
            else:
                Kg, Vg = K, V

            phi_K = phi(Kg)
            phi_Q = phi(Q)

            if mask_2d is not None:
                m4 = mask_2d.unsqueeze(-1).unsqueeze(-1).to(phi_K.dtype)
                phi_K = phi_K * m4
                Vg = Vg * m4

            phi_K_h = phi_K.transpose(1, 2).contiguous()
            Vg_h = Vg.transpose(1, 2).contiguous()
            phi_Q_h = phi_Q.transpose(1, 2).contiguous()

            KV = torch.matmul(phi_K_h.transpose(-2, -1).contiguous(), Vg_h)
            global_out_h = torch.matmul(phi_Q_h, KV)
            Z = phi_K_h.sum(dim=-2, keepdim=True)
            norm = torch.matmul(phi_Q_h, Z.transpose(-2, -1).contiguous())
            global_out_h = global_out_h / (norm.clamp(min=1e-6))
            global_out = global_out_h.transpose(1, 2).contiguous()

            if mask_2d is not None:
                global_out = global_out * mask_2d.unsqueeze(-1).unsqueeze(-1)

            alpha = torch.sigmoid(self.fusion_gate(hidden_states)).unsqueeze(-1)
            fused = alpha * local_out + (1.0 - alpha) * global_out

        fused = fused.reshape(B, N, self.all_head_size)
        fused = torch.nan_to_num(fused, nan=0.0, posinf=1e4, neginf=-1e4).clamp(min=-1e4, max=1e4)

        if self.is_nested_encoder:
            out = fused
        else:
            out = self.out_proj(fused)
            out = self.dropout(out)

        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4).clamp(min=-1e4, max=1e4)
        return out, (attn_local if output_attentions else None)

    def get_ortho_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.out_proj.up.weight.device)


@ATTENTION_REGISTRY.register("aegis_v2_inject")
@ATTENTION_REGISTRY.register("inject_aegis_v2")
def inject_aegis_v2(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
):
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
            res = inject_aegis_v2(
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

    layer = target

    if model_config is None:
        if hasattr(layer, "hidden_size") or hasattr(layer, "d_model") or hasattr(layer, "n_embd") or hasattr(layer, "dim"):
            model_config = layer
        elif hasattr(layer, "attention"):
            old_attn = layer.attention
    h_size = getattr(model_config, "hidden_size", getattr(model_config, "d_model", getattr(model_config, "n_embd", getattr(model_config, "dim", 768)))) if model_config is not None else 768
    n_heads = getattr(model_config, "num_attention_heads", getattr(model_config, "num_heads", getattr(model_config, "n_heads", getattr(model_config, "n_head", 12)))) if model_config is not None else 12
    h_dim = getattr(model_config, "head_dim", getattr(model_config, "d_kv", h_size // max(1, n_heads))) if model_config is not None else h_size // max(1, n_heads)
    n_kv_heads = getattr(model_config, "num_key_value_heads", getattr(model_config, "num_kv_heads", n_heads)) if model_config is not None else n_heads

    cfg = AegisAttentionConfigV2(
        hidden_size=h_size,
        num_attention_heads=n_heads,
        num_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        head_dim=h_dim,
        local_window=getattr(comp_config, "block_size", 64),
        rank=getattr(comp_config, "rank", getattr(comp_config, "lora_r", 384)),
        kernel_feature_dim=getattr(comp_config, "rff_features", 128),
        dropout=getattr(comp_config, "dropout", 0.1),
        use_adaptive_pruning=getattr(comp_config, "use_gate", True),
        gate_init_bias=getattr(comp_config, "gate_init_bias", 1.8),
    )
    new_attn = AegisAttentionV2(config=cfg)
    src_attn = old_attn if old_attn is not None else getattr(layer, "attention", getattr(layer, "self", getattr(layer, "attn", None)))
    if src_attn is not None:
        try:
            new_attn.copy_weights(src_attn)
        except Exception as e:
            logger.error("inject_aegis_v2 copy_weights failed: %s", e)
            return None
        p = next(src_attn.parameters(), None)
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)

    if hasattr(layer, "attention"):
        if hasattr(layer.attention, "self"):
            layer.attention.self = new_attn
        else:
            layer.attention = new_attn
    elif hasattr(layer, "self"):
        layer.self = new_attn
    elif hasattr(layer, "attn"):
        layer.attn = new_attn
    return new_attn
