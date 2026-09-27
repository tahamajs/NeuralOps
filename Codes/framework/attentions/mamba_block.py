"""
Faithful Mamba Block Implementation with SVD-based Weight Distillation.

Based on Gu & Dao, "Mamba: Linear-Time Sequence Modeling" (arXiv:2312.00752).
Features:
- Full state dimension (B, d_inner, N)
- Input-dependent selectivity (Δ, B, C) via x_proj
- Identity-initialized depthwise conv1d to prevent initial context blurring
- SVD distillation mapping W_q -> C and W_k -> B
- Safe injection and architecture detection (GPT-2, LLaMA, Qwen, etc.)
"""

from __future__ import annotations
import logging
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY
from framework.core.selective_scan import selective_scan

logger = logging.getLogger(__name__)


@ATTENTION_REGISTRY.register("mamba")
@ATTENTION_REGISTRY.register("mamba_attention")
@ATTENTION_REGISTRY.register("mamba_block")
class MambaBlock(nn.Module):
    is_custom_attention: bool = True
    supports_causal: bool = True
    supports_bidirectional: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        d_model: Optional[int] = None,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        conv_bias: bool = True,
        bias: bool = False,
        bidirectional: bool = False,
        is_decoder_override: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(config, (int, float)):
            d_model = int(config)
            config = None

        super().__init__()
        self.config = config

        extracted_dim = (
            d_model
            or kwargs.get("hidden_size")
            or kwargs.get("dim")
            or (getattr(config, "hidden_size", None) if config else None)
            or (getattr(config, "d_model", None) if config else None)
            or 768
        )
        self.d_model: int = int(extracted_dim)
        self.d_state: int = int(d_state)
        self.d_conv: int = int(d_conv)
        self.expand: int = int(expand)
        self.d_inner: int = self.expand * self.d_model
        self.dt_rank: int = int(dt_rank or math.ceil(self.d_model / 16))
        self.bidirectional: bool = bool(bidirectional)

        if is_decoder_override is not None:
            self.is_decoder = bool(is_decoder_override)
        else:
            model_type = str(getattr(config, "model_type", "")).lower() if config else ""
            _dec_prefixes = (
                "gpt", "qwen", "llama", "mistral", "gemma",
                "phi", "opt", "bloom", "falcon", "olmo"
            )
            self.is_decoder = (
                bool(getattr(config, "is_decoder", False)) if config else False
            ) or any(model_type.startswith(p) for p in _dec_prefixes)

        # ---- Projections & Parameters ----
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=bias)
        
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            bias=conv_bias,
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

        # ---- Default Initializations ----
        dt_min, dt_max = 0.001, 0.1
        dt_init = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        dt_init_inv = dt_init + torch.log(-torch.expm1(-dt_init))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt_init_inv)
            bound = 1.0 / math.sqrt(self.dt_rank)
            nn.init.uniform_(self.dt_proj.weight, -bound, bound)
            nn.init.kaiming_uniform_(self.conv1d.weight, a=math.sqrt(5))

        # ---- Bidirectional Mode ----
        if self.bidirectional:
            self.A_log_bwd = nn.Parameter(torch.log(A.clone() + 0.01 * torch.randn_like(A).abs()))
            self.dt_proj_bwd = nn.Linear(self.dt_rank, self.d_inner, bias=True)
            with torch.no_grad():
                self.dt_proj_bwd.bias.copy_(dt_init_inv)
                nn.init.uniform_(self.dt_proj_bwd.weight, -bound, bound)
            self.x_proj_bwd = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
            self.D_bwd = nn.Parameter(torch.ones(self.d_inner))

    def copy_weights(self, src: nn.Module) -> None:
        """
        استخراج وزن‌های Attention و تقریب آن‌ها با SVD جهت نگاشت به پارامترهای S6 در Mamba.
        """
        extracted: Dict[str, Optional[torch.Tensor]] = {"q": None, "k": None, "v": None, "o": None}

        # ۱. تلاش برای استخراج وزن‌ها (پشتیبانی از مدل‌های فلت، تو در تو و GPT-2)
        try:
            from framework.core.adapters import UniversalAdapter
            for proj in ["q", "k", "v", "o"]:
                w, _ = UniversalAdapter.get_weight_bias(src, proj)
                extracted[proj] = w
        except Exception:
            pass

        if hasattr(src, "c_attn") and src.c_attn is not None and extracted["q"] is None:
            w = getattr(getattr(src.c_attn, "base_layer", src.c_attn), "weight").data
            if w.dim() == 2 and w.shape[1] == 3 * self.d_model:
                w = w.t().contiguous()
            split = w.size(0) // 3
            extracted["q"] = w[:split, :]
            extracted["k"] = w[split:2 * split, :]
            extracted["v"] = w[2 * split:, :]
        if hasattr(src, "c_proj") and src.c_proj is not None and extracted["o"] is None:
            w_o = getattr(getattr(src.c_proj, "base_layer", src.c_proj), "weight").data
            if w_o.dim() == 2 and w_o.shape[0] < w_o.shape[1]:
                w_o = w_o.t().contiguous()
            extracted["o"] = w_o

        named_map = {
            "q": ["q_proj", "q_lin", "query"],
            "k": ["k_proj", "k_lin", "key"],
            "v": ["v_proj", "v_lin", "value"],
            "o": ["o_proj", "out_proj", "out_lin", "dense"],
        }
        for proj, names in named_map.items():
            if extracted[proj] is None:
                for name in names:
                    if hasattr(src, name):
                        mod = getattr(src, name)
                        if mod is not None and hasattr(mod, "weight"):
                            extracted[proj] = mod.weight.data
                            break

        # ۲. نگاشت و تزریق وزن‌ها به بلوک مامبا
        target_dtype = self.in_proj.weight.dtype
        with torch.no_grad():
            w_v = extracted["v"]
            if w_v is not None:
                w_v_float = w_v.float()
                d_v_out = min(w_v_float.shape[0], self.d_inner)
                d_v_in = min(w_v_float.shape[1], self.d_model)
                self.in_proj.weight.data.zero_()
                # مسیر اصلی (x_ssm) حاوی اطلاعات W_v
                self.in_proj.weight.data[:d_v_out, :d_v_in] = w_v_float[:d_v_out, :d_v_in].to(target_dtype)
                # مسیر گیت (z) با ضریب نرمالیزه باز می‌ماند
                self.in_proj.weight.data[self.d_inner:self.d_inner+d_v_out, :d_v_in] = (w_v_float[:d_v_out, :d_v_in] * 0.5).to(target_dtype)

            # تقریب SVD روی W_k و W_q برای جاسازی در بردار وضعیت B و C
            w_k = extracted["k"]
            if w_k is not None:
                _, _, Vh_k = torch.linalg.svd(w_k.float(), full_matrices=False)
                b_weight = Vh_k[:self.d_state, :min(Vh_k.shape[1], self.d_inner)]
                b_start = self.dt_rank
                b_end = self.dt_rank + self.d_state
                self.x_proj.weight.data.zero_()
                self.x_proj.weight.data[b_start:b_end, :b_weight.shape[1]] = b_weight.to(target_dtype)

            w_q = extracted["q"]
            if w_q is not None:
                _, _, Vh_q = torch.linalg.svd(w_q.float(), full_matrices=False)
                c_weight = Vh_q[:self.d_state, :min(Vh_q.shape[1], self.d_inner)]
                c_start = self.dt_rank + self.d_state
                c_end = c_start + self.d_state
                self.x_proj.weight.data[c_start:c_end, :c_weight.shape[1]] = c_weight.to(target_dtype)

            # مقداردهی خنثی (Identity) به لایه کانوولوشن برای جلوگیری از تخریب کانتکست اولیه
            self.conv1d.weight.data.zero_()
            self.conv1d.weight.data[:, 0, -1] = 1.0  
            if self.conv1d.bias is not None:
                self.conv1d.bias.data.zero_()

            # کپی مستقیم لایه خروجی
            w_o = extracted["o"]
            if w_o is not None:
                w_o_float = w_o.float()
                d_o_out = min(w_o_float.shape[0], self.d_model)
                d_o_in = min(w_o_float.shape[1], self.d_inner)
                self.out_proj.weight.data.zero_()
                self.out_proj.weight.data[:d_o_out, :d_o_in] = w_o_float[:d_o_out, :d_o_in].to(target_dtype)
                if self.out_proj.bias is not None:
                    self.out_proj.bias.data.zero_()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if kwargs.get("encoder_hidden_states") is not None:
            raise NotImplementedError("Mamba does not support cross-attention")

        B, L, _ = hidden_states.shape

        xz = self.in_proj(hidden_states)
        x_ssm, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(
            x_ssm.transpose(1, 2)
        )[..., :L].transpose(1, 2)

        x_ssm = F.silu(x_conv)

        x_dbl = self.x_proj(x_ssm)
        dt, B_ssm, C_ssm = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)

        A = -torch.exp(self.A_log)

        u_t = x_ssm.transpose(1, 2)
        dt_t = dt.transpose(1, 2)
        B_t = B_ssm.transpose(1, 2)
        C_t = C_ssm.transpose(1, 2)

        y = selective_scan(
            u=u_t,
            delta=dt_t,
            A=A,
            B=B_t,
            C=C_t,
            D=self.D,
            delta_softplus=True,
        )
        y = y.transpose(1, 2)

        if self.bidirectional and not self.is_decoder:
            u_rev = torch.flip(x_ssm, dims=[1]).transpose(1, 2)
            x_rev = torch.flip(x_ssm, dims=[1])
            x_dbl_rev = self.x_proj_bwd(x_rev)
            dt_b, B_b, C_b = x_dbl_rev.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt_b = self.dt_proj_bwd(dt_b).transpose(1, 2)
            B_b = B_b.transpose(1, 2)
            C_b = C_b.transpose(1, 2)
            A_b = -torch.exp(self.A_log_bwd)

            y_b = selective_scan(
                u=u_rev,
                delta=dt_b,
                A=A_b,
                B=B_b,
                C=C_b,
                D=self.D_bwd,
                delta_softplus=True,
            )
            y_b = y_b.transpose(1, 2)
            y_b = torch.flip(y_b, dims=[1])
            y = (0.5 * (y + y_b)).to(hidden_states.dtype)

        target_dtype = hidden_states.dtype
        y = (y.to(target_dtype) * F.silu(z.to(target_dtype))).to(target_dtype)
        out = self.out_proj(y)

        if attention_mask is not None and attention_mask.dim() == 2:
            out = out * attention_mask.unsqueeze(-1).to(out.dtype)

        return out.to(target_dtype), None


Mamba = MambaBlock


@ATTENTION_REGISTRY.register("mamba_module_inject")
@ATTENTION_REGISTRY.register("mamba_inject")
@ATTENTION_REGISTRY.register("inject_mamba")
def inject_mamba(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> Optional[Union[nn.Module, Any]]:
    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config
    if cfg is None and hasattr(target, "config"):
        cfg = target.config

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
    elif hasattr(target, "bert") and hasattr(target.bert, "encoder") and hasattr(target.bert.encoder, "layer"):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_mamba(
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

    src_attn = old_attn
    if src_attn is None and target is not None:
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                src_attn = target.attention.self
            else:
                src_attn = target.attention
        elif hasattr(target, "self_attn"):
            src_attn = target.self_attn
        elif hasattr(target, "self"):
            src_attn = target.self
        elif hasattr(target, "attn"):
            src_attn = target.attn

    rank = kwargs.pop("rank", 32)
    d_model = 128
    if cfg is not None:
        d_model = getattr(cfg, "hidden_size", getattr(cfg, "dim", getattr(cfg, "d_model", 128)))
    elif src_attn is not None:
        d_model = (
            getattr(src_attn, "hidden_size", None)
            or getattr(src_attn, "embed_dim", None)
            or getattr(src_attn, "dim", None)
            or (src_attn.q_proj.in_features if hasattr(src_attn, "q_proj") and hasattr(src_attn.q_proj, "in_features") else None)
            or (src_attn.q_lin.in_features if hasattr(src_attn, "q_lin") and hasattr(src_attn.q_lin, "in_features") else None)
            or 128
        )
    elif target is not None and hasattr(target, "config"):
        d_model = getattr(target.config, "hidden_size", getattr(target.config, "dim", 128))
    elif target is not None and hasattr(target, "hidden_size"):
        d_model = getattr(target, "hidden_size", 128)

    try:
        new_module = MambaBlock(d_model=d_model, config=cfg, comp_config=comp_config, rank=rank, **kwargs)
    except Exception as e:
        logger.error("Failed to instantiate MambaBlock: %s", e)
        return None

    if src_attn is not None and hasattr(new_module, "copy_weights"):
        try:
            new_module.copy_weights(src_attn)
        except Exception as e:
            logger.error("copy_weights failed: %s", e)
            return None

    if target is not None:
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                target.attention.self = new_module
            else:
                target.attention = new_module
        elif hasattr(target, "self_attn"):
            target.self_attn = new_module
        elif hasattr(target, "self"):
            target.self = new_module
        elif hasattr(target, "attn"):
            target.attn = new_module
        else:
            target.attention = new_module

    return new_module

__all__ = ["MambaBlock", "Mamba", "inject_mamba"]