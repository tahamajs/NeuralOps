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
            torch.full((self.svd_rank,), math.sqrt(float(out_features) / max(1, self.svd_rank * self.in_features)), **fk),
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

                    out = samba_moe(exp_outs[0].to(orig_dtype), exp_outs[1].to(orig_dtype), scores.to(orig_dtype)).float()
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
        self.kv_repeat = self.num_heads // self.num_kv_heads

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
        is_dec = bool(getattr(config, "is_decoder", False) or getattr(config, "is_causal", False) or getattr(config, "causal", False))
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
        if pretrained_attn is None:
            return

        try:
            for p in pretrained_attn.parameters():
                self._orig_storage_bytes += p.numel() * p.element_size()
        except Exception:
            # allowed-swallow: best-effort-telemetry
            pass

        cls_name = pretrained_attn.__class__.__name__
        has_out_lin = any(hasattr(pretrained_attn, a) for a in ("out_lin", "c_proj", "o_proj", "out_proj", "dense"))
        is_nested = False
        if (
            (
                hasattr(pretrained_attn, "self")
                and hasattr(pretrained_attn.self, "query")
                and hasattr(pretrained_attn, "output")
                and hasattr(pretrained_attn.output, "dense")
            )
            or (
                cls_name
                in (
                    "BertSelfAttention",
                    "RobertaSelfAttention",
                    "XLMRobertaSelfAttention",
                    "ElectraSelfAttention",
                    "DebertaSelfAttention",
                )
                or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())
            )
            and not has_out_lin
        ):
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        if hasattr(pretrained_attn, "rotary_emb") and pretrained_attn.rotary_emb is not None:
            self.rotary_emb = pretrained_attn.rotary_emb

        extracted = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        def _extract_mod_wb(mod: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
            if mod is None:
                return None, None
            if hasattr(mod, "merge") and callable(mod.merge) and not getattr(mod, "merged", False):
                try:
                    mod.merge()
                except Exception:
                    # allowed-swallow: best-effort-telemetry
                    pass
            base_m = getattr(mod, "base_layer", mod)
            w_m = getattr(base_m, "weight", getattr(mod, "weight", None))
            if w_m is None or not isinstance(w_m, torch.Tensor) or w_m.ndim != 2:
                return None, None
            w_m = w_m.data.clone()
            cls_m = type(mod).__name__
            base_cls_m = type(base_m).__name__
            if ("Conv1D" in cls_m) or ("Conv1D" in base_cls_m) or hasattr(base_m, "nf") or hasattr(mod, "nf"):
                w_m = w_m.t().contiguous()
            b_mod = getattr(base_m, "bias", getattr(mod, "bias", None))
            b_m = b_mod.data.clone() if isinstance(b_mod, torch.Tensor) else None
            return w_m, b_m

        if hasattr(pretrained_attn, "c_attn") and pretrained_attn.c_attn is not None:
            c = pretrained_attn.c_attn
            base_c = getattr(c, "base_layer", c)
            cw = getattr(base_c, "weight", getattr(c, "weight", None))
            if cw is not None:
                cw = cw.data.clone()
                is_conv = (
                    c.__class__.__name__ == "Conv1D"
                    or base_c.__class__.__name__ == "Conv1D"
                    or (cw.ndim == 2 and cw.shape[0] < cw.shape[1])
                )
                if is_conv:
                    cw = cw.t().contiguous()
                split = cw.size(0) // 3
                bmod = getattr(base_c, "bias", getattr(c, "bias", None))
                cb = bmod.data.clone() if bmod is not None else None
                extracted["q"] = (cw[:split], cb[:split] if cb is not None else None)
                extracted["k"] = (
                    cw[split : 2 * split],
                    cb[split : 2 * split] if cb is not None else None,
                )
                extracted["v"] = (
                    cw[2 * split :],
                    cb[2 * split :] if cb is not None else None,
                )

            if hasattr(pretrained_attn, "c_proj") and pretrained_attn.c_proj is not None:
                o = pretrained_attn.c_proj
                base_o = getattr(o, "base_layer", o)
                ow = getattr(base_o, "weight", getattr(o, "weight", None))
                if ow is not None:
                    ow = ow.data.clone()
                    is_conv_o = (
                        o.__class__.__name__ == "Conv1D"
                        or base_o.__class__.__name__ == "Conv1D"
                        or (ow.ndim == 2 and ow.shape[0] < ow.shape[1])
                    )
                    if is_conv_o:
                        ow = ow.t().contiguous()
                    bo = getattr(base_o, "bias", getattr(o, "bias", None))
                    extracted["o"] = (ow, bo.data.clone() if bo is not None else None)

        elif hasattr(pretrained_attn, "self") and hasattr(pretrained_attn.self, "query"):
            s = pretrained_attn.self
            for key, sub in (("q", "query"), ("k", "key"), ("v", "value")):
                w_m, b_m = _extract_mod_wb(getattr(s, sub, None))
                if w_m is not None:
                    extracted[key] = (w_m, b_m)
            if hasattr(pretrained_attn, "output") and hasattr(pretrained_attn.output, "dense"):
                w_m, b_m = _extract_mod_wb(pretrained_attn.output.dense)
                if w_m is not None:
                    extracted["o"] = (w_m, b_m)

        else:
            candidates = {
                "q": ("q_lin", "q_proj", "query"),
                "k": ("k_lin", "k_proj", "key"),
                "v": ("v_lin", "v_proj", "value"),
                "o": ("out_lin", "out_proj", "dense", "o_proj"),
            }
            for key, names in candidates.items():
                for n in names:
                    if hasattr(pretrained_attn, n):
                        w_m, b_m = _extract_mod_wb(getattr(pretrained_attn, n))
                        if w_m is not None:
                            extracted[key] = (w_m, b_m)
                            break

        missing = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing:
            try:
                from neuralops.core.adapters import UniversalAdapter

                adapter = UniversalAdapter()
                for k in missing:
                    w, b = adapter.get_weight_bias(pretrained_attn, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        missing_after = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing_after:
            raise RuntimeError(
                f"BAMUMoE.copy_weights: missing {missing_after} from " f"{pretrained_attn.__class__.__name__}"
            )

        def _orient(w, in_dim):
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

        target_map = {
            "q": self.q_lin,
            "k": self.k_lin,
            "v": self.v_lin,
            "o": self.out_lin,
        }

        with torch.no_grad():
            for proj_key, target in target_map.items():
                w, b = extracted[proj_key]
                if w is None:
                    if proj_key == "o" and self.is_nested_encoder:
                        eye = torch.eye(
                            target.out_features,
                            target.in_features,
                            device=target.U.device,
                            dtype=target.U.dtype,
                        )
                        target.set_svd_factors(eye, None)
                    continue

                if w.ndim != 2:
                    w = w.reshape(w.size(0), -1)

                in_dim = self.all_head_size if proj_key == "o" else self.dim
                w = _orient(w, in_dim)

                if proj_key in ("k", "v"):
                    src_out = w.shape[0]
                    tgt_out = target.out_features
                    hd = self.head_dim
                    if src_out > tgt_out and src_out % tgt_out == 0 and tgt_out % hd == 0:
                        rep = src_out // tgt_out
                        w = w.view(self.num_kv_heads, rep, hd, -1).mean(dim=1).reshape(tgt_out, -1)
                        if b is not None and b.numel() == src_out:
                            b = b.view(self.num_kv_heads, rep, hd).mean(dim=1).reshape(tgt_out)
                    elif src_out < tgt_out and tgt_out % src_out == 0 and src_out % hd == 0:
                        rep = tgt_out // src_out
                        src_kv = src_out // hd
                        w = w.view(src_kv, hd, -1).repeat_interleave(rep, dim=0).reshape(tgt_out, -1)
                        if b is not None and b.numel() == src_out:
                            b = b.view(src_kv, hd).repeat_interleave(rep, dim=0).reshape(tgt_out)

                out_d = min(target.out_features, w.size(0))
                in_d = min(target.in_features, w.size(1))
                if (w.size(0), w.size(1)) != (target.out_features, target.in_features):
                    padded = torch.zeros(
                        target.out_features,
                        target.in_features,
                        device=w.device,
                        dtype=w.dtype,
                    )
                    padded[:out_d, :in_d] = w[:out_d, :in_d]
                    w = padded

                target.set_svd_factors(w, b)
                target.freeze_base_weights()

        if bool(getattr(pretrained_attn, "is_decoder", False)) or bool(getattr(pretrained_attn, "is_causal", False)):
            self.is_decoder = True
            self.is_causal = True

        try:
            for m in (self.q_lin, self.k_lin, self.v_lin, self.out_lin):
                for p in m.parameters():
                    self._comp_storage_bytes += p.numel() * p.element_size()
        except Exception:
            # allowed-swallow: best-effort-telemetry
            pass

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
                scores = apply_attention_mask(scores, eff_mask, mask_value=-1e9)
                invalid = invalid | (scores <= -5e8)

        all_masked = invalid.all(dim=-1, keepdim=True)
        safe_invalid = invalid.clone()
        if all_masked.any():
            safe_invalid[..., :1] = safe_invalid[..., :1] & (~all_masked)

        scores = torch.nan_to_num(scores, nan=0.0, posinf=1e4, neginf=-1e4).masked_fill(safe_invalid, -1e9)
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


def inject_bam_umoe(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    _depth: int = 0,
    **kwargs: Any,
):
    if _depth > 8:
        logger.warning("inject_bam_umoe: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    layer_list = None
    for path in (
        ("distilbert", "transformer", "layer"),
        ("roberta", "encoder", "layer"),
        ("bert", "encoder", "layer"),
        ("transformer", "h"),
        ("model", "layers"),
        ("encoder", "layer"),
        ("layers",),
        ("h",),
    ):
        obj = target
        ok = True
        for p in path:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                ok = False
                break
        if ok and isinstance(obj, (nn.ModuleList, list)):
            layer_list = obj
            break

    if layer_list is not None:
        n_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_bam_umoe(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                _depth=_depth + 1,
                **kwargs,
            )
            if res is not None:
                n_injected += 1
        return target if n_injected > 0 else None

    layer = target

    src_attn = old_attn
    if src_attn is None:
        for attr in ("attention", "self", "attn", "self_attn"):
            if hasattr(layer, attr):
                cand = getattr(layer, attr)
                if isinstance(cand, nn.Module) and cand is not layer:
                    src_attn = getattr(cand, "self", cand)
                    break

    if src_attn is not None and (isinstance(src_attn, BAMUMoEBlock) or "BAMUMoE" in src_attn.__class__.__name__):
        return None

    if cfg is None and src_attn is not None:
        cfg = getattr(src_attn, "config", None)

    hidden_size = kwargs.get("hidden_size")
    if hidden_size is None and src_attn is not None:
        hidden_size = getattr(src_attn, "hidden_size", None)
        if hidden_size is None and hasattr(src_attn, "q_proj"):
            q = getattr(src_attn, "q_proj", None)
            if q is not None and hasattr(q, "in_features"):
                hidden_size = q.in_features
    if hidden_size is None:
        hidden_size = getattr(cfg, "hidden_size", 768) if cfg else 768

    num_heads = kwargs.get("num_heads", kwargs.get("num_attention_heads"))
    if num_heads is None and src_attn is not None:
        num_heads = getattr(src_attn, "num_heads", getattr(src_attn, "num_attention_heads", None))
    if num_heads is None:
        num_heads = getattr(cfg, "num_attention_heads", 12) if cfg else 12

    moe_enabled = bool(kwargs.pop("moe_enabled", True))
    svd_rank = kwargs.pop("svd_rank", None)

    try:
        kwargs.pop("num_heads", None)
        kwargs.pop("hidden_size", None)
        kwargs.pop("d_model", None)
        new_attn = BAMUMoEBlock(
            config=cfg,
            old_attn=src_attn if src_attn is not layer else None,
            comp_config=comp_config,
            hidden_size=hidden_size,
            num_heads=num_heads,
            layer_id=layer_idx,
            moe_enabled=moe_enabled,
            svd_rank=svd_rank,
            **kwargs,
        )
        if src_attn is not None and src_attn is not layer:
            p = next(src_attn.parameters(), None)
            if p is not None:
                new_attn.to(device=p.device, dtype=p.dtype)

        spliced = False
        if hasattr(layer, "attention"):
            if hasattr(layer.attention, "self"):
                new_attn.is_nested_encoder = True
                layer.attention.self = new_attn
            else:
                new_attn.is_nested_encoder = False
                layer.attention = new_attn
            spliced = True
        elif hasattr(layer, "self"):
            new_attn.is_nested_encoder = True
            layer.self = new_attn
            spliced = True
        elif hasattr(layer, "attn"):
            new_attn.is_nested_encoder = False
            layer.attn = new_attn
            spliced = True
        elif hasattr(layer, "self_attn"):
            new_attn.is_nested_encoder = False
            layer.self_attn = new_attn
            spliced = True

        if not spliced:
            logger.error(
                "inject_bam_umoe: no writable attention attribute on %s",
                layer.__class__.__name__,
            )
            return None
        return new_attn

    except Exception as e:
        logger.exception("inject_bam_umoe failed at layer %d: %s", layer_idx, e)
        return None


__all__ = [
    "BAMUMoEBlock",
    "UnifiedMoE",
    "FactorisedLoRALinear",
    "apply_rotary_pos_emb",
    "inject_bam_umoe",
]
