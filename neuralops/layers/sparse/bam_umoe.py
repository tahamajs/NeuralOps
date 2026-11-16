#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
BAM-UMoE: Unified Mixture of Experts Attention Block

Unified mixture-of-experts attention block combining low-rank factorization with expert routing.
"""

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.attention_mask import apply_attention_mask, get_token_valid_mask
from neuralops.core.attention_utils import apply_rotary_pos_emb
from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.utils.gqa import repeat_kv

logger = logging.getLogger(__name__)


def _clamp_to_dtype(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Sanitize non-finite values and clamp to dtype safe range before casting."""
    if dtype.is_floating_point:
        max_val = torch.finfo(dtype).max * 0.5
        return torch.nan_to_num(tensor, nan=0.0, posinf=max_val, neginf=-max_val).clamp(-max_val, max_val).to(dtype)
    return torch.nan_to_num(tensor, nan=0.0).to(dtype)


class FactorisedLoRALinear(nn.Module):
    """
    Low-rank factorized linear with optional LoRA:

        y = (x V S U^T) + (x lora_A^T lora_B^T) * scaling + b

    Base factors (U, S, V) are frozen parameters (visible in .parameters()).
    """

    is_custom_attention: bool = False

    def __init__(
        self,
        in_features: int,
        out_features: int,
        svd_rank: int,
        lora_rank: int = 0,
        lora_alpha: float = 32.0,
        bias: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.svd_rank = max(1, min(svd_rank, self.in_features, self.out_features))
        self.lora_rank = max(0, int(lora_rank))
        self.lora_scaling = (lora_alpha / self.lora_rank) if self.lora_rank > 0 else 1.0

        fk = {"dtype": dtype, "device": device}
        self.U = nn.Parameter(
            torch.empty(out_features, self.svd_rank, **fk),
            requires_grad=False,
        )
        self.S = nn.Parameter(
            torch.full(
                (self.svd_rank,), math.sqrt(float(out_features) / max(1, self.svd_rank * self.in_features)), **fk
            ),
            requires_grad=False,
        )
        self.V = nn.Parameter(
            torch.empty(in_features, self.svd_rank, **fk),
            requires_grad=False,
        )
        with torch.no_grad():
            if out_features >= self.svd_rank:
                nn.init.orthogonal_(self.U)
            else:
                nn.init.normal_(self.U, std=1.0 / math.sqrt(max(1, out_features)))
            if in_features >= self.svd_rank:
                nn.init.orthogonal_(self.V)
            else:
                nn.init.normal_(self.V, std=1.0 / math.sqrt(max(1, in_features)))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **fk), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(self.lora_rank, in_features, **fk))
            self.lora_B = nn.Parameter(torch.randn(out_features, self.lora_rank, **fk) * 1e-4)
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.normal_(self.lora_B, mean=0.0, std=1e-4)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self._svd_initialized: bool = False
        # Non-persistent staging buffer so external callers (e.g. QwenAttentionAdapter._inherit_weights)
        # writing to `.weight.data.copy_(...)` are captured and factorized into (U, S, V) on first forward.
        init_w = self._reconstruct_usv()
        self.register_buffer("_pending_weight", init_w, persistent=False)
        self._pending_init_sig: tuple[float, float] = (
            float(init_w.sum().item()),
            float(init_w.abs().sum().item()),
        )

    def _reconstruct_usv(self) -> torch.Tensor:
        return ((self.U.float() * self.S.float().unsqueeze(0)) @ self.V.float().t().contiguous()).to(self.U.dtype)

    def _maybe_sync_pending_weight(self) -> None:
        if self._svd_initialized:
            return
        pw = getattr(self, "_pending_weight", None)
        if pw is not None:
            sig = (float(pw.sum().item()), float(pw.abs().sum().item()))
            if sig != self._pending_init_sig:
                self.set_svd_factors(pw)
                return
        self._svd_initialized = True
        self._pending_weight = None

    def set_svd_factors(self, W: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        with torch.no_grad():
            W_f = W.float()
            if W_f.shape != (self.out_features, self.in_features):
                raise ValueError(
                    f"FactorisedLoRALinear: shape mismatch "
                    f"{tuple(W_f.shape)} vs ({self.out_features}, {self.in_features})"
                )
            try:
                U, S, Vt = torch.linalg.svd(W_f, full_matrices=False)
            except Exception:
                U, S, Vt = torch.linalg.svd(W_f.cpu(), full_matrices=False)
                U, S, Vt = U.to(W_f.device), S.to(W_f.device), Vt.to(W_f.device)
            r = min(self.svd_rank, S.shape[0])
            self.U.data.zero_()
            self.S.data.zero_()
            self.V.data.zero_()
            self.U.data[:, :r].copy_(U[:, :r].to(device=self.U.device, dtype=self.U.dtype))
            self.S.data[:r].copy_(S[:r].to(device=self.S.device, dtype=self.S.dtype))
            self.V.data[:, :r].copy_(Vt[:r, :].t().contiguous().to(device=self.V.device, dtype=self.V.dtype))
            if self.bias is not None:
                if bias is not None:
                    n = min(self.bias.numel(), bias.numel())
                    self.bias.data.zero_()
                    self.bias.data[:n].copy_(bias[:n].to(device=self.bias.device, dtype=self.bias.dtype))
                else:
                    self.bias.data.zero_()
            self._svd_initialized = True
            self._pending_weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._maybe_sync_pending_weight()
        orig_dtype = x.dtype
        x_f = x.float()
        xV = torch.matmul(x_f, self.V.float())
        xV_S = xV * self.S.float().unsqueeze(0).unsqueeze(0)
        out = torch.matmul(xV_S, self.U.float().t().contiguous())
        if self.bias is not None:
            out = out + self.bias.float()
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            xa = torch.matmul(x_f, self.lora_A.float().t().contiguous())
            out = out + torch.matmul(xa, self.lora_B.float().t().contiguous()) * self.lora_scaling
        return _clamp_to_dtype(out, orig_dtype)

    @property
    def weight(self) -> torch.Tensor:
        if not self._svd_initialized and getattr(self, "_pending_weight", None) is not None:
            return self._pending_weight
        return self._reconstruct_usv()

    def freeze_base_weights(self) -> None:
        self.U.requires_grad = False
        self.S.requires_grad = False
        self.V.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_A is not None:
            self.lora_A.requires_grad = True
        if self.lora_B is not None:
            self.lora_B.requires_grad = True


class UnifiedMoE(nn.Module):
    """
    Compact MoE with top-k gating.

    BAM-FIX-02 / FIX-12: expert hidden dim is `dim // (num_experts * 4)`,
    output dim is configurable (default `head_dim`, small).
    """

    is_custom_attention: bool = False

    def __init__(
        self,
        dim: int,
        num_experts: int = 2,
        top_k: int = 1,
        output_dim: int | None = None,
        noise_std: float = 0.1,
        expert_hidden_mult: float = 0.25,
        gate_init: float = -3.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_experts = max(1, int(num_experts))
        self.top_k = min(max(1, int(top_k)), self.num_experts)
        self.out_dim = int(output_dim) if output_dim is not None else max(16, self.dim // 4)
        self.noise_std = float(noise_std)

        self.router_ln = nn.LayerNorm(self.dim)
        self.router = nn.Linear(self.dim, self.num_experts, bias=False)
        nn.init.uniform_(self.router.weight, -0.01, 0.01)

        hidden_dim = max(16, int(self.out_dim * expert_hidden_mult))
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, self.out_dim),
                )
                for _ in range(self.num_experts)
            ]
        )

        for exp in self.experts:
            nn.init.normal_(exp[2].weight, std=0.02)
            if exp[2].bias is not None:
                nn.init.zeros_(exp[2].bias)

        self.gate_bias = nn.Parameter(torch.tensor([float(gate_init)]))

        self._aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        orig_dtype = x.dtype
        x_norm = self.router_ln(x)
        logits = self.router(x_norm).float()

        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std

        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_idx = torch.topk(probs, self.top_k, dim=-1)

        if self.training and self.num_experts > 1:
            router_probs = probs.view(-1, self.num_experts)
            expert_mask = F.one_hot(topk_idx.long(), num_classes=self.num_experts).float()
            density = expert_mask.sum(dim=2).view(-1, self.num_experts).mean(dim=0)
            prob_mean = router_probs.mean(dim=0)
            self._aux_loss = self.num_experts * torch.sum(density * prob_mean)
        else:
            self._aux_loss = None

        exp_outs = [exp(x).float() for exp in self.experts]

        if self.top_k == 1:
            # Switch-style unnormalized top-1 probability weighting keeps router gradients non-zero
            # (normalizing p_i / p_i == 1.0 would zero out all task gradients to router and router_ln).
            exp_stacked = torch.stack(exp_outs, dim=-2)  # [B, L, E, out_dim]
            sel_mask = F.one_hot(topk_idx[..., 0].long(), num_classes=self.num_experts).float().unsqueeze(-1)
            out = (exp_stacked * sel_mask).sum(dim=-2) * topk_probs[..., :1]
        else:
            scores = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            scores = torch.nan_to_num(scores, nan=0.0)
            from neuralops.ops.triton.sparse_attn import fused_bam_unified_qkv_moe_triton

            if x.is_cuda and len(self.experts) == 2 and self.top_k == 2:
                try:
                    from neuralops.ops.cuda.samba_moe import samba_moe

                    out = samba_moe(
                        exp_outs[0].to(orig_dtype), exp_outs[1].to(orig_dtype), scores.to(orig_dtype)
                    ).float()
                except Exception:
                    out, _ = fused_bam_unified_qkv_moe_triton(exp_outs, logits, self.top_k)
            else:
                out, _ = fused_bam_unified_qkv_moe_triton(exp_outs, logits, self.top_k)

        gate = torch.sigmoid(self.gate_bias.float())
        return _clamp_to_dtype(out * gate, orig_dtype)

    def get_aux_loss(self, device: torch.device) -> torch.Tensor:
        """BAM-FIX-04: return aux loss on the correct device."""
        if self._aux_loss is None:
            return torch.zeros((), device=device, dtype=torch.float32)
        return self._aux_loss.to(device=device, dtype=torch.float32)


@ATTENTION_REGISTRY.register("bam_umoe", supports_causal=True, supports_bidirectional=True, supports_gqa=True)
class BAMUMoEBlock(BaseAttention):
    """BAMUMoEBlock neural operator.

    BAM-UMoE: Unified Mixture of Experts Attention Block (Activation-Aware).

    Attributes:
        supports_causal (bool): Whether causal autoregressive masking is supported.
        is_custom_attention (bool): Flag indicating specialized non-standard attention mechanics.
    """

    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int = 64,
        num_experts: int = 2,
        top_k: int = 1,
        layer_id: int | None = None,
        num_layers: int | None = None,
        old_attn: nn.Module | None = None,
        comp_config: Any | None = None,
        moe_enabled: bool = True,
        svd_rank: int | None = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        **kwargs: Any,
    ) -> None:
        if config is None:

            class _Cfg:
                pass

            config = _Cfg()
            config.hidden_size = kwargs.get("hidden_size", 768)
            config.num_attention_heads = kwargs.get("num_heads", kwargs.get("num_attention_heads", 12))
            config.head_dim = kwargs.get("head_dim", None)
            config.model_type = kwargs.get("model_type", "")

        try:
            super().__init__(config, rank)
        except TypeError:
            super().__init__(config)

        self.dim = int(
            kwargs.get("hidden_size")
            or getattr(
                config,
                "hidden_size",
                getattr(config, "d_model", getattr(config, "dim", 768)),
            )
        )
        self.hidden_size = self.dim
        self.num_heads = int(
            kwargs.get("num_heads")
            or kwargs.get("num_attention_heads")
            or getattr(
                config,
                "num_attention_heads",
                getattr(config, "num_heads", getattr(config, "n_heads", 12)),
            )
        )

        cfg_hd = getattr(config, "head_dim", None)
        if cfg_hd is None and old_attn is not None:
            cfg_hd = getattr(old_attn, "head_dim", None)
        self.head_dim = int(cfg_hd or kwargs.get("head_dim") or (self.dim // max(1, self.num_heads)))

        nkv = kwargs.get("num_kv_heads") or kwargs.get("num_key_value_heads")
        if nkv is None:
            nkv = getattr(config, "num_key_value_heads", getattr(config, "num_kv_heads", None))
        if nkv is None and old_attn is not None:
            nkv = getattr(old_attn, "num_key_value_heads", getattr(old_attn, "num_kv_heads", None))
            if nkv is None:
                for attr in ("k_proj", "key", "k_lin"):
                    mod = getattr(old_attn, attr, None)
                    if mod is not None and hasattr(mod, "weight"):
                        base_mod = getattr(mod, "base_layer", mod)
                        w = base_mod.weight
                        is_conv = "Conv1D" in type(base_mod).__name__ or hasattr(base_mod, "nf")
                        out_d = w.shape[1] if (is_conv and w.ndim == 2) else w.shape[0]
                        nkv = out_d // max(1, self.head_dim)
                        break
        self.num_kv_heads = max(1, min(int(nkv if nkv is not None else self.num_heads), self.num_heads))
        self.kv_repeat = max(1, self.num_heads // max(1, self.num_kv_heads))

        self.all_head_size = self.num_heads * self.head_dim
        self.all_kv_head_size = self.num_kv_heads * self.head_dim

        self.lora_rank = int(
            getattr(
                config,
                "lora_rank",
                (getattr(comp_config, "lora_rank", lora_rank) if comp_config is not None else lora_rank),
            )
        )
        self.lora_alpha = float(
            getattr(
                config,
                "lora_alpha",
                (getattr(comp_config, "lora_alpha", lora_alpha) if comp_config is not None else lora_alpha),
            )
        )
        self.freeze_base = self.lora_rank > 0

        default_svd = self.dim
        if svd_rank is not None:
            self.svd_rank = int(svd_rank)
        elif comp_config is not None and hasattr(comp_config, "svd_rank"):
            self.svd_rank = int(comp_config.svd_rank)
        elif config is not None and hasattr(config, "svd_rank"):
            self.svd_rank = int(config.svd_rank)
        else:
            self.svd_rank = int(default_svd)

        mt = str(getattr(config, "model_type", "") or "").lower()
        is_dec = bool(
            getattr(config, "is_decoder", False)
            or getattr(config, "is_causal", False)
            or getattr(config, "causal", False)
        )
        if any(
            mt.startswith(p)
            for p in (
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
        ):
            is_dec = True
        arch_list = getattr(config, "architectures", None) or []
        if any("GPT" in str(a) or "CausalLM" in str(a) for a in arch_list):
            is_dec = True
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
            is_dec = True
        if old_attn is not None:
            cls_lower = old_attn.__class__.__name__.lower()
            if (
                hasattr(old_attn, "c_attn")
                or getattr(old_attn, "is_decoder", False)
                or getattr(old_attn, "is_causal", False)
                or any(p in cls_lower for p in ("qwen", "llama", "mistral", "gpt", "causal"))
            ):
                is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = self.is_decoder
        self.is_nested_encoder = False

        self.num_experts = max(1, int(num_experts))
        self.top_k = min(max(1, int(top_k)), self.num_experts)
        self.moe_enabled = bool(moe_enabled)

        Q_rank = min(self.svd_rank, self.dim, self.all_head_size)
        KV_rank = min(self.svd_rank, self.dim, self.all_kv_head_size)
        O_rank = min(self.svd_rank, self.all_head_size, self.dim)

        self.q_lin = FactorisedLoRALinear(
            self.dim,
            self.all_head_size,
            Q_rank,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            bias=True,
        )
        self.k_lin = FactorisedLoRALinear(
            self.dim,
            self.all_kv_head_size,
            KV_rank,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            bias=True,
        )
        self.v_lin = FactorisedLoRALinear(
            self.dim,
            self.all_kv_head_size,
            KV_rank,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            bias=True,
        )
        self.out_lin = FactorisedLoRALinear(
            self.all_head_size,
            self.dim,
            O_rank,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            bias=True,
        )

        if self.moe_enabled:
            self.qkv_moe: UnifiedMoE | None = UnifiedMoE(
                self.dim,
                num_experts=self.num_experts,
                top_k=self.top_k,
                output_dim=self.all_kv_head_size,
                gate_init=kwargs.get("moe_gate_init", -3.0),
            )
        else:
            self.qkv_moe = None

        dropout_p = float(getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1)))
        self.dropout = nn.Dropout(dropout_p)

        self.layer_id = layer_id
        self.num_layers = num_layers
        self.comp_config = comp_config
        self.rotary_emb = None

        self._orig_storage_bytes = 0
        self._comp_storage_bytes = 0

        if old_attn is not None:
            self.copy_weights(old_attn)

    @property
    def q_proj(self):
        return self.q_lin

    @property
    def k_proj(self):
        return self.k_lin

    @property
    def v_proj(self):
        return self.v_lin

    @property
    def o_proj(self):
        return self.out_lin

    @property
    def query(self):
        return self.q_lin

    @property
    def key(self):
        return self.k_lin

    @property
    def value(self):
        return self.v_lin

    @property
    def dense(self):
        return self.out_lin

    @property
    def c_proj(self):
        return self.out_lin

    @property
    def out_proj(self):
        return self.out_lin

    def copy_weights(self, pretrained_attn: nn.Module) -> None:
        from ._bam_umoe_utils import copy_bam_umoe_weights

        copy_bam_umoe_weights(self, pretrained_attn)

    @property
    def aux_loss(self) -> torch.Tensor | None:
        if self.qkv_moe is not None:
            return self.qkv_moe._aux_loss
        return torch.tensor(0.0)

    def get_aux_loss(self, device: torch.device) -> torch.Tensor:
        """BAM-FIX-04: MoE aux loss for training loop."""
        if self.qkv_moe is None:
            return torch.zeros((), device=device, dtype=torch.float32)
        return self.qkv_moe.get_aux_loss(device)

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
        *args: Any,
        **kwargs: Any,
    ):
        B, L, D = hidden_states.shape
        dtype_in = hidden_states.dtype

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        S = kv_input.shape[1]

        q = self.q_lin(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.k_lin(kv_input).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.v_lin(kv_input).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        cos = kwargs.get("cos")
        sin = kwargs.get("sin")
        pos_emb = kwargs.get("position_embeddings")
        if pos_emb is not None and isinstance(pos_emb, (tuple, list)) and len(pos_emb) == 2:
            cos, sin = pos_emb[0], pos_emb[1]

        if cos is not None and sin is not None and not is_cross:
            q, k = apply_rotary_pos_emb(q, k, cos, sin, kwargs.get("position_ids"))
        elif getattr(self, "rotary_emb", None) is not None and not is_cross:
            rotary_mod = self.rotary_emb
            pos_ids = kwargs.get("position_ids")
            if pos_ids is None:
                pos_ids = torch.arange(L, device=hidden_states.device).unsqueeze(0).expand(B, -1)
            try:
                r_cos, r_sin = rotary_mod(q, seq_len=L)
            except Exception:
                try:
                    r_cos, r_sin = rotary_mod(q, pos_ids)
                except Exception:
                    try:
                        r_cos, r_sin = rotary_mod(q)
                    except Exception:
                        r_cos, r_sin = None, None
            if r_cos is not None and r_sin is not None:
                q, k = apply_rotary_pos_emb(q, k, r_cos, r_sin, pos_ids)

        if self.kv_repeat > 1:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        if self.qkv_moe is not None and not is_cross:
            moe_delta_v = self.qkv_moe(hidden_states)
            moe_delta_v = moe_delta_v.view(B, L, self.num_kv_heads, self.head_dim)
            moe_delta_v = moe_delta_v.transpose(1, 2).contiguous()
            if self.kv_repeat > 1:
                moe_delta_v = repeat_kv(moe_delta_v, self.kv_repeat)
            v = (v + 0.1 * moe_delta_v.to(v.dtype)).to(v.dtype)

        scale = 1.0 / math.sqrt(max(self.head_dim, 1))
        eff_mask = encoder_attention_mask if is_cross else attention_mask
        is_causal = self.is_decoder and not is_cross
        device = q.device

        valid_k = get_token_valid_mask(eff_mask, B, S, device) if eff_mask is not None else None
        if valid_k is not None and valid_k.shape[-1] != S:
            valid_k = None

        if not output_attentions and head_mask is None and (eff_mask is None or valid_k is not None):
            attn_mask = None
            all_masked = None
            if valid_k is not None or (is_causal and S != L):
                bool_mask = torch.ones((B, 1, L, S), device=device, dtype=torch.bool)
                if valid_k is not None:
                    bool_mask = bool_mask & valid_k.unsqueeze(1).unsqueeze(2)
                if is_causal:
                    past_len = max(0, S - L)
                    q_idx = torch.arange(L, device=device).unsqueeze(1)
                    k_idx = torch.arange(S, device=device).unsqueeze(0)
                    bool_mask = bool_mask & (k_idx <= (q_idx + past_len)).unsqueeze(0).unsqueeze(0)
                all_masked = ~bool_mask.any(dim=-1, keepdim=True)
                if all_masked.any():
                    safe_mask = bool_mask.clone()
                    safe_mask[..., :1] = safe_mask[..., :1] | all_masked
                    attn_mask = safe_mask
                else:
                    attn_mask = bool_mask
            try:
                context = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=is_causal if attn_mask is None else False,
                    scale=scale,
                )
                if all_masked is not None and all_masked.any():
                    context = context.masked_fill(all_masked, 0.0)
                context = context.transpose(1, 2).contiguous().view(B, L, self.all_head_size)
                context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)
                if self.is_nested_encoder:
                    out = context
                else:
                    out = self.out_lin(context)
                if out.shape[-1] != D:
                    raise RuntimeError(f"BAMUMoE: output dim {out.shape[-1]} != hidden_size {D}")
                return out, None
            except Exception:
                # allowed-swallow: best-effort-telemetry
                pass

        scores = torch.matmul(q.float(), k.float().transpose(-2, -1).contiguous()) * scale
        invalid = torch.zeros((B, 1, L, S), device=device, dtype=torch.bool)

        if is_causal:
            past_len = max(0, S - L)
            q_idx = torch.arange(L, device=device).unsqueeze(1)
            k_idx = torch.arange(S, device=device).unsqueeze(0)
            invalid = invalid | (k_idx > (q_idx + past_len)).unsqueeze(0).unsqueeze(0)

        if eff_mask is not None:
            if valid_k is not None:
                invalid = invalid | (~valid_k).unsqueeze(1).unsqueeze(2)
            else:
                scores = apply_attention_mask(scores, eff_mask, mask_value=-65500.0)
                invalid = invalid | (scores <= -5e4)

        all_masked = invalid.all(dim=-1, keepdim=True)
        safe_invalid = invalid.clone()
        if all_masked.any():
            safe_invalid[..., :1] = safe_invalid[..., :1] & (~all_masked)

        scores = (
            torch.nan_to_num(scores, nan=0.0, posinf=1e4, neginf=-1e4)
            .masked_fill(safe_invalid, -65500.0)
            .clamp(min=-65500.0, max=65500.0)
        )
        probs = F.softmax(scores, dim=-1).masked_fill(all_masked, 0.0)
        probs = self.dropout(probs)

        if head_mask is not None:
            hm = head_mask.float()
            if hm.dim() == 1:
                hm = hm.view(1, -1, 1, 1)
            elif hm.dim() == 2:
                hm = hm.view(hm.shape[0], hm.shape[1], 1, 1)
            elif hm.dim() == 3:
                hm = hm.unsqueeze(1)
            probs = probs * hm

        context = torch.matmul(probs, v.float()).to(dtype_in)
        context = context.transpose(1, 2).contiguous().view(B, L, self.all_head_size)
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            out = context
        else:
            out = self.out_lin(context)

        if out.shape[-1] != D:
            raise RuntimeError(f"BAMUMoE: output dim {out.shape[-1]} != hidden_size {D}")

        return out, (probs.to(dtype_in) if output_attentions else None)


from ._bam_umoe_utils import inject_bam_umoe

__all__ = [
    "BAMUMoEBlock",
    "UnifiedMoE",
    "FactorisedLoRALinear",
    "apply_rotary_pos_emb",
    "inject_bam_umoe",
]
