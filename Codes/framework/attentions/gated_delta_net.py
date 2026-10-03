"""Gated DeltaNet Attention Module — Production-Hardened (v2).

"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY
try:
    from framework.core.triton_kernels.gdn_scan import gdn_scan_triton, MAX_REGISTER_DIM
    HAS_TRITON_GDN = True
    logger.info("GDN: Triton scan available")
except Exception as _e:
    logger.warning("GDN: Triton scan unavailable: %s", _e)
    gdn_scan_triton = None
    HAS_TRITON_GDN = False
    MAX_REGISTER_DIM = 64


# ===========================================================================
# Helpers
# ===========================================================================
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    b, h, s, d = hidden_states.shape
    return hidden_states[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _delta_chunk_forward(S, q_i, k_i, v_i, beta_i, alpha_i):
    """
    Inner loop over a chunk of size C.

    Recurrence (GDN-FIX-02):
        inter_v = k_t S_{t-1}
        err = v_t - inter_v
        denom = 1 + ||k_t||²
        step = alpha_t / denom
        S_t = beta_t * S_{t-1} + step * k_tᵀ err
        y_t = q_t S_t
    """
    C = q_i.shape[2]
    outs = []
    curr = S
    for t in range(C):
        kt = k_i[:, :, t:t+1, :]
        vt = v_i[:, :, t:t+1, :]
        qt = q_i[:, :, t:t+1, :]
        bt = beta_i[:, :, t:t+1].unsqueeze(-1)
        at = alpha_i[:, :, t:t+1].unsqueeze(-1)

        inter_v = torch.matmul(kt, curr)                       # (B,H,1,d)
        err = vt - inter_v

        norm_k_sq = (kt * kt).sum(dim=-1, keepdim=True)         # (B,H,1,1)
        step = at / (1.0 + norm_k_sq).clamp_min(1.0)

        delta = step * torch.matmul(kt.transpose(-1, -2).contiguous(), err)
        curr = bt * curr + delta
        curr = torch.nan_to_num(curr, nan=0.0, posinf=1e4, neginf=-1e4)

        yt = torch.matmul(qt, curr)
        outs.append(yt)

    return torch.cat(outs, dim=2), curr


def chunked_delta_rule_scan(q, k, v, beta, alpha, chunk_size=32):
    """
    Chunked Delta-Rule scan with O(L·chunk) intermediate memory.

    GD-FIX-09: no inner Python overhead beyond the minimal recurrence.
    """
    B, H, L, d = q.shape
    device = q.device

    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    beta_f = beta.float().clamp(min=1e-4, max=1.0 - 1e-4)
    alpha_f = alpha.float().clamp(min=1e-4, max=1.0)

    k_f = F.normalize(k_f, p=2, dim=-1, eps=1e-6)
    q_f = F.normalize(q_f, p=2, dim=-1, eps=1e-6)

    # --- Triton fast path (head_dim <= 64, on CUDA) ---
    if (HAS_TRITON_GDN and gdn_scan_triton is not None and q.is_cuda and d <= MAX_REGISTER_DIM
            and q.dtype in (torch.float16, torch.bfloat16, torch.float32)):
        try:
            out = gdn_scan_triton(q_f, k_f, v_f, beta_f, alpha_f).to(q.dtype)
            logger.debug("GDN: Triton scan used")
            return out
        except Exception as e:
            logger.warning("Triton gdn_scan failed, falling back to PyTorch: %s", e)
            logger.debug("GDN: Triton FAILED: %s", e)

    # --- PyTorch fallback ---
    logger.debug("GDN: PyTorch fallback")
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len > 0:
        q_f = F.pad(q_f, (0, 0, 0, pad_len))
        k_f = F.pad(k_f, (0, 0, 0, pad_len))
        v_f = F.pad(v_f, (0, 0, 0, pad_len))
        beta_f = F.pad(beta_f, (0, pad_len), value=1.0)
        alpha_f = F.pad(alpha_f, (0, pad_len), value=1.0)

    L_pad = L + pad_len
    n_chunks = L_pad // chunk_size

    q_c = q_f.view(B, H, n_chunks, chunk_size, d)
    k_c = k_f.view(B, H, n_chunks, chunk_size, d)
    v_c = v_f.view(B, H, n_chunks, chunk_size, d)
    beta_c = beta_f.view(B, H, n_chunks, chunk_size)
    alpha_c = alpha_f.view(B, H, n_chunks, chunk_size)

    S = torch.zeros((B, H, d, d), device=device, dtype=torch.float32)
    outputs = []
    for c in range(n_chunks):
        y_i, S = _delta_chunk_forward(
            S,
            q_c[:, :, c], k_c[:, :, c], v_c[:, :, c],
            beta_c[:, :, c], alpha_c[:, :, c],
        )
        outputs.append(y_i)

    y = torch.cat(outputs, dim=2)
    if pad_len > 0:
        y = y[:, :, :L, :]
    return torch.nan_to_num(y, nan=0.0).to(q.dtype)


def compute_relational_kd_loss(
    student_hidden_states: tuple[torch.Tensor, ...],
    teacher_hidden_states: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """
    Vectorized Relational Knowledge Distillation (RKD)
    بهینه‌شده برای سرعت بالا (بدون حلقه For) و پایدار در برابر NaN
    """
    # ۱. تجمیع تمام لایه‌ها در یک تنسور واحد: (Num_Layers, Batch, Seq_Len, Dim)
    s_stacked = torch.stack(student_hidden_states[1:], dim=0).float()
    t_stacked = torch.stack(teacher_hidden_states[1:], dim=0).float()

    # ۲. نرمال‌سازی با eps=1e-8 (محافظت قطعی در برابر خطای NaN)
    s_norm = F.normalize(s_stacked, p=2, dim=-1, eps=1e-8)
    t_norm = F.normalize(t_stacked, p=2, dim=-1, eps=1e-8)

    # ۳. ضرب ماتریسی همزمان برای تمام لایه‌ها و بچ‌ها -> خروجی: (Num_Layers, Batch, Seq_Len, Seq_Len)
    s_rel = torch.matmul(s_norm, s_norm.transpose(-1, -2))
    t_rel = torch.matmul(t_norm, t_norm.transpose(-1, -2))

    # ۴. محاسبه میانگین خطا به صورت یکجا
    return F.smooth_l1_loss(s_rel, t_rel)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    if cos is None or sin is None:
        return q, k
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim < head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q = torch.cat([(q_rot * cos) + (_rotate_half(q_rot) * sin), q_pass], dim=-1)
        k = torch.cat([(k_rot * cos) + (_rotate_half(k_rot) * sin), k_pass], dim=-1)
    else:
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


# ===========================================================================
# LoRALinear
# ===========================================================================
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, lora_rank=16,
                 lora_alpha=32.0, freeze_base=False, device=None, dtype=None):
        super().__init__()
        fk = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0
        self.freeze_base = freeze_base

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **fk))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **fk))
        else:
            self.register_parameter("bias", None)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty((lora_rank, in_features), **fk))
            self.lora_B = nn.Parameter(torch.empty((out_features, lora_rank), **fk))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.reset_parameters()
        if freeze_base:
            self.freeze_base_weights()

    def reset_parameters(self):
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

    def freeze_base_weights(self):
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_rank > 0:
            if self.lora_A is not None:
                self.lora_A.requires_grad = True
            if self.lora_B is not None:
                self.lora_B.requires_grad = True

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight.dtype != x.dtype else self.weight
        b = self.bias.to(x.dtype) if (self.bias is not None and self.bias.dtype != x.dtype) else self.bias
        out = F.linear(x, w, b)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            a = self.lora_A.to(x.dtype) if self.lora_A.dtype != x.dtype else self.lora_A
            bb = self.lora_B.to(x.dtype) if self.lora_B.dtype != x.dtype else self.lora_B
            out = out + F.linear(F.linear(x, a), bb) * self.lora_scaling
        return out


# ===========================================================================
# Main attention
# ===========================================================================
@ATTENTION_REGISTRY.register("gated_delta_net", supports_causal=True,
                              supports_bidirectional=True, supports_gqa=True)
class GatedDeltaNet(BaseAttention):
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        d_model: int | Any | None = None,
        num_heads: int | None = None,
        comp_config: Any | None = None,
        layer_id: int | None = None,
        layer_idx: int | None = None,
        old_attn: nn.Module | None = None,
        pretrained_attn: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn

        # ---- d_model ----
        if d_model is None:
            d_model = kwargs.get("d_model") or kwargs.get("hidden_size") or kwargs.get("dim")
            if d_model is None:
                if config is not None:
                    d_model = getattr(config, "hidden_size", getattr(config, "dim", 768))
                elif source_attn is not None:
                    out_layer = None
                    for attr in ("out_lin", "dense", "c_proj", "o_proj", "out_proj"):
                        if hasattr(source_attn, attr):
                            out_layer = getattr(source_attn, attr)
                            break
                    if out_layer is not None:
                        d_model = getattr(out_layer, "out_features",
                                           getattr(out_layer, "in_features", 768))
                    else:
                        for attr in ("q_proj", "query", "q_lin", "W_q"):
                            q_l = getattr(source_attn, attr, None)
                            if q_l is not None:
                                d_model = getattr(q_l, "in_features",
                                                   getattr(q_l, "out_features", 768))
                                break
                if d_model is None:
                    d_model = 768
        if hasattr(d_model, "hidden_size"):
            d_model = d_model.hidden_size

        # ---- num_heads ----
        if num_heads is None:
            num_heads = kwargs.get("num_heads") or kwargs.get("num_attention_heads")
            if num_heads is None:
                if config is not None:
                    num_heads = getattr(config, "num_attention_heads",
                                         getattr(config, "n_heads",
                                                 getattr(config, "num_heads", 12)))
                elif source_attn is not None:
                    num_heads = getattr(source_attn, "n_heads",
                                         getattr(source_attn, "num_attention_heads",
                                                 getattr(source_attn, "num_heads", 12)))
                else:
                    num_heads = 12

        # GDN-FIX-01: None-safe
        try:
            super().__init__(config, rank if rank is not None else 64)
        except TypeError:
            super().__init__(config)

        self.d_model = int(d_model)
        self.num_heads = max(1, int(num_heads))

        # ---- num_kv_heads ----
        cfg_kv = None
        if config is not None:
            cfg_kv = getattr(config, "num_key_value_heads",
                              getattr(config, "num_kv_heads", None))
        if cfg_kv is None:
            cfg_kv = kwargs.get("num_kv_heads", self.num_heads)
        self.num_kv_heads = max(1, min(int(cfg_kv), self.num_heads))

        # ---- head_dim ----
        head_dim = getattr(config, "head_dim", None) if config else None
        if head_dim is None and source_attn is not None:
            for q_cand in ("q_proj", "query", "q_lin", "W_q"):
                q_mod = getattr(source_attn, q_cand, None)
                if q_mod is not None and hasattr(q_mod, "out_features"):
                    head_dim = q_mod.out_features // max(1, self.num_heads)
                    break
        if head_dim is None:
            head_dim = self.d_model // self.num_heads
        self.head_dim = int(head_dim)
        self.all_head_size = self.num_heads * self.head_dim
        self.kv_head_size = self.num_kv_heads * self.head_dim
        self.kv_repeat = max(1, self.num_heads // self.num_kv_heads)

        resolved_layer_id = (layer_id if layer_id is not None
                              else (layer_idx if layer_idx is not None
                                    else kwargs.get("layer_id", kwargs.get("layer_idx", 0))))
        self.layer_idx = int(resolved_layer_id)

        # ---- LoRA ----
        if config is not None:
            self.lora_rank = int(getattr(config, "lora_rank",
                                  getattr(comp_config, "lora_rank",
                                          kwargs.get("lora_rank", 16))))
            self.lora_alpha = float(getattr(config, "lora_alpha",
                                     getattr(comp_config, "lora_alpha",
                                             kwargs.get("lora_alpha", 32.0))))
        elif comp_config is not None:
            self.lora_rank = int(getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16)))
            self.lora_alpha = float(getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)))
        else:
            self.lora_rank = int(kwargs.get("lora_rank", 16))
            self.lora_alpha = float(kwargs.get("lora_alpha", 32.0))
        self.freeze_base = (self.lora_rank > 0)

        # ---- decoder detection ----
        model_type = str(getattr(config, "model_type", "") if config else "").lower()
        is_dec = False
        if config is not None:
            is_dec = bool(getattr(config, "is_decoder", False)
                          or getattr(config, "is_causal", False))
        decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt",
                             "bloom", "falcon", "olmo", "mpt", "stablelm", "codegen")
        if any(model_type.startswith(p) for p in decoder_prefixes):
            is_dec = True
        if kwargs.get("is_decoder", False) or kwargs.get("is_causal", False):
            is_dec = True
        if source_attn is not None:
            if getattr(source_attn, "is_decoder", False) or hasattr(source_attn, "c_attn"):
                is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = self.is_decoder

        # ---- nested encoder detection ----
        is_nested = False
        if source_attn is not None:
            src_cls = source_attn.__class__.__name__
            if (src_cls in ("BertSelfAttention", "RobertaSelfAttention",
                             "XLMRobertaSelfAttention", "ElectraSelfAttention",
                             "DebertaSelfAttention")
                or (src_cls.endswith("SelfAttention") and "distil" not in src_cls.lower())) or hasattr(source_attn, "self") and hasattr(source_attn, "output"):
                is_nested = True
        if not is_nested and model_type in ("roberta", "bert", "xlm-roberta") and not self.is_decoder:
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        # ---- projections ----
        self.q_proj = LoRALinear(self.d_model, self.all_head_size, bias=True,
                                  lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                  freeze_base=self.freeze_base)
        self.k_proj = LoRALinear(self.d_model, self.kv_head_size, bias=True,
                                  lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                  freeze_base=self.freeze_base)
        self.v_proj = LoRALinear(self.d_model, self.kv_head_size, bias=True,
                                  lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                  freeze_base=self.freeze_base)
        self.z_proj = LoRALinear(self.d_model, self.all_head_size, bias=True,
                                  lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                  freeze_base=self.freeze_base)
        self.out_proj = LoRALinear(self.all_head_size, self.d_model, bias=True,
                                    lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
                                    freeze_base=self.freeze_base)

        # GDN-FIX-13: single output scale
        self.output_scale = nn.Parameter(torch.ones(1))

        # ---- gating ----
        self.beta_proj = nn.Linear(self.d_model, self.num_heads, bias=True)
        self.alpha_proj = nn.Linear(self.d_model, self.num_heads, bias=True)
        with torch.no_grad():
            nn.init.constant_(self.beta_proj.bias, 2.0)   # ~0.88 initial decay
            nn.init.constant_(self.alpha_proj.bias, 0.0)  # ~0.5 initial step

        # ---- chunk size (GDN-FIX-14) ----
        cs = int(_cfg_get(comp_config, "chunk_size", kwargs.get("chunk_size", 32)))
        self.chunk_size = max(8, min(cs, 128))

        self.source_attn = source_attn
        self.rotary_emb = getattr(source_attn, "rotary_emb", None) if source_attn is not None else None
        self.comp_config = comp_config

        # ---- storage accounting ----
        self._orig_storage_bytes = 0
        self._comp_storage_bytes = 0

        if source_attn is not None:
            self.copy_weights(source_attn)

    def freeze_base_weights(self):
        for m in (self.q_proj, self.k_proj, self.v_proj, self.z_proj, self.out_proj):
            if hasattr(m, "freeze_base_weights"):
                m.freeze_base_weights()

    # ---- aliases ----
    @property
    def W_q(self): return self.q_proj
    @property
    def W_k(self): return self.k_proj
    @property
    def W_v(self): return self.v_proj
    @property
    def o_proj(self): return self.out_proj
    @property
    def q_lin(self): return self.q_proj
    @property
    def k_lin(self): return self.k_proj
    @property
    def v_lin(self): return self.v_proj
    @property
    def out_lin(self): return self.out_proj
    @property
    def query(self): return self.q_proj
    @property
    def key(self): return self.k_proj
    @property
    def value(self): return self.v_proj
    @property
    def dense(self): return self.out_proj
    @property
    def c_proj(self): return self.out_proj

    # ------------------------------------------------------------------
    def copy_weights(self, old_attn: Any) -> None:
        if old_attn is None:
            return

        try:
            for p in old_attn.parameters():
                self._orig_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

        cls_name = old_attn.__class__.__name__
        is_nested = False
        if (hasattr(old_attn, "self") and hasattr(old_attn.self, "query")
                and hasattr(old_attn, "output") and hasattr(old_attn.output, "dense")) or (cls_name in ("BertSelfAttention", "RobertaSelfAttention",
                            "XLMRobertaSelfAttention", "ElectraSelfAttention",
                            "DebertaSelfAttention")
              or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())):
            is_nested = True
        if not is_nested and getattr(self, "is_nested_encoder", False):
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        if hasattr(old_attn, "rotary_emb"):
            self.rotary_emb = old_attn.rotary_emb

        extracted = {"q": (None, None), "k": (None, None),
                      "v": (None, None), "o": (None, None)}

        # GPT-2 fused c_attn
        if hasattr(old_attn, "c_attn") and old_attn.c_attn is not None:
            c = old_attn.c_attn
            # GDN-FIX-12: guarded merge
            if hasattr(c, "merge") and callable(c.merge) and not getattr(c, "merged", False):
                try:
                    c.merge()
                except Exception as e:
                    logger.debug("c_attn merge skipped: %s", e)
            base_c = getattr(c, "base_layer", c)
            cw = getattr(base_c, "weight", getattr(c, "weight", None))
            if cw is not None:
                cw = cw.data.clone()
                cls_c = c.__class__.__name__
                base_cls_c = base_c.__class__.__name__
                is_conv = ("Conv1D" in cls_c or "Conv1D" in base_cls_c
                            or hasattr(c, "nf") or hasattr(base_c, "nf")
                            or (cw.ndim == 2 and cw.shape[0] < cw.shape[1]))
                if is_conv:
                    cw = cw.t().contiguous()
                split = cw.size(0) // 3
                bmod = getattr(base_c, "bias", getattr(c, "bias", None))
                cb = bmod.data.clone() if bmod is not None else None
                extracted["q"] = (cw[:split], cb[:split] if cb is not None else None)
                extracted["k"] = (cw[split:2*split], cb[split:2*split] if cb is not None else None)
                extracted["v"] = (cw[2*split:], cb[2*split:] if cb is not None else None)

            if hasattr(old_attn, "c_proj") and old_attn.c_proj is not None:
                o = old_attn.c_proj
                base_o = getattr(o, "base_layer", o)
                ow = getattr(base_o, "weight", getattr(o, "weight", None))
                if ow is not None:
                    ow = ow.data.clone()
                    is_conv_o = ("Conv1D" in o.__class__.__name__
                                  or "Conv1D" in base_o.__class__.__name__
                                  or hasattr(o, "nf") or hasattr(base_o, "nf")
                                  or (ow.ndim == 2 and ow.shape[0] < ow.shape[1]))
                    if is_conv_o:
                        ow = ow.t().contiguous()
                    bo = getattr(base_o, "bias", getattr(o, "bias", None))
                    extracted["o"] = (ow, bo.data.clone() if bo is not None else None)

        # Nested (BERT / RoBERTa)
        elif hasattr(old_attn, "self") and hasattr(old_attn.self, "query"):
            s = old_attn.self
            for key, sub in (("q", "query"), ("k", "key"), ("v", "value")):
                m = getattr(s, sub, None)
                if m is not None and hasattr(m, "weight"):
                    b = getattr(m, "bias", None)
                    extracted[key] = (m.weight.data.clone(),
                                       b.data.clone() if b is not None else None)
            if hasattr(old_attn, "output") and hasattr(old_attn.output, "dense"):
                d = old_attn.output.dense
                b = getattr(d, "bias", None)
                extracted["o"] = (d.weight.data.clone(),
                                   b.data.clone() if b is not None else None)

        # Flat
        else:
            candidates = {
                "q": ("q_lin", "q_proj", "query", "W_q"),
                "k": ("k_lin", "k_proj", "key", "W_k"),
                "v": ("v_lin", "v_proj", "value", "W_v"),
                "o": ("out_lin", "out_proj", "dense", "o_proj", "output", "proj"),
            }
            for key, names in candidates.items():
                for n in names:
                    if hasattr(old_attn, n):
                        m = getattr(old_attn, n)
                        if m is not None and hasattr(m, "weight"):
                            b = getattr(m, "bias", None)
                            extracted[key] = (m.weight.data.clone(),
                                               b.data.clone() if b is not None else None)
                            break

        # UniversalAdapter fallback
        missing = [k for k, (w, _) in extracted.items()
                    if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing:
            try:
                from framework.core.adapters import UniversalAdapter
                adapter = UniversalAdapter()
                for k in missing:
                    w, b = adapter.get_weight_bias(old_attn, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        missing_after = [k for k, (w, _) in extracted.items()
                          if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing_after:
            logger.warning("GatedDeltaNet copy_weights: missing %s", missing_after)
            return

        with torch.no_grad():
            for key, target in (("q", self.q_proj), ("k", self.k_proj),
                                  ("v", self.v_proj), ("o", self.out_proj)):
                w, b = extracted[key]
                if w is None:
                    if key == "o" and self.is_nested_encoder:
                        nn.init.eye_(target.weight)
                        if target.bias is not None:
                            nn.init.zeros_(target.bias)
                    continue

                if w.ndim != 2:
                    w = w.reshape(w.size(0), -1)
                if w.shape != target.weight.shape \
                        and w.t().contiguous().shape == target.weight.shape:
                    w = w.t().contiguous()

                out_d = min(target.weight.size(0), w.size(0))
                in_d = min(target.weight.size(1), w.size(1))
                target.weight.data[:out_d, :in_d].copy_(
                    w[:out_d, :in_d].to(target.weight.device, target.weight.dtype)
                )

                if b is not None and target.bias is not None:
                    mb = min(target.bias.numel(), b.numel())
                    target.bias.data[:mb].copy_(
                        b[:mb].to(target.bias.device, target.bias.dtype)
                    )
                elif target.bias is not None:
                    nn.init.zeros_(target.bias)

                n = float(torch.linalg.norm(target.weight.data.float()).item())
                if n < 1e-4:
                    logger.warning("GatedDeltaNet: '%s' near-zero norm (%.2e)", key, n)

                if self.freeze_base:
                    target.freeze_base_weights()

            # z_proj from q
            w_q = extracted["q"][0]
            if w_q is not None:
                out_z = min(self.z_proj.weight.size(0), w_q.size(0))
                in_z = min(self.z_proj.weight.size(1), w_q.size(1))
                self.z_proj.weight.data[:out_z, :in_z].copy_(
                    w_q[:out_z, :in_z].to(self.z_proj.weight.device, self.z_proj.weight.dtype)
                )
                if self.freeze_base:
                    self.z_proj.freeze_base_weights()

        try:
            for m in (self.q_proj, self.k_proj, self.v_proj, self.z_proj, self.out_proj):
                for p in m.parameters():
                    self._comp_storage_bytes += p.numel() * p.element_size()
        except Exception:
            pass

        if self.freeze_base:
            self.freeze_base_weights()

    # ------------------------------------------------------------------
    # Mask helper
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_valid_mask(mask, B, L, device):
        """GDN-FIX-04: returns (B, L) bool or None."""
        if mask is None:
            return None

        def _to_bool(t):
            if t.dtype == torch.bool:
                return t
            if t.dtype.is_floating_point:
                return t > -1e4
            return t > 0

        if mask.dim() == 2:
            m = _to_bool(mask)
            if m.shape == (B, L):
                return m
        elif mask.dim() == 3:
            m = _to_bool(mask)
            if m.shape[0] == B and m.shape[-1] == L:
                return m[:, -1, :]
        elif mask.dim() == 4:
            m = _to_bool(mask)
            m = m.any(dim=-2)
            if m.dim() == 3:
                m = m.squeeze(1)
            if m.shape == (B, L):
                return m
        return None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
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
        position_ids: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()

        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        dtype_in = hidden_states.dtype

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        kv_len = kv_input.shape[1]

        q = self.q_proj(hidden_states).view(B, L, H, d).transpose(1, 2).contiguous()
        k = self.k_proj(kv_input).view(B, kv_len, self.num_kv_heads, d).transpose(1, 2).contiguous()
        v = self.v_proj(kv_input).view(B, kv_len, self.num_kv_heads, d).transpose(1, 2).contiguous()
        z = self.z_proj(hidden_states)

        # ---- RoPE ----
        if position_embeddings is not None and not is_cross:
            if isinstance(position_embeddings, tuple):
                cos, sin = position_embeddings
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        elif self.rotary_emb is not None and not is_cross:
            if position_ids is None:
                position_ids = torch.arange(kv_len, device=hidden_states.device).unsqueeze(0).expand(B, -1)
            try:
                cos, sin = self.rotary_emb(k, seq_len=kv_len)
            except TypeError:
                try:
                    cos, sin = self.rotary_emb(k, position_ids)
                except Exception:
                    cos, sin = None, None
            if cos is not None and sin is not None:
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        # ---- GQA ----
        if self.num_kv_heads != self.num_heads:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        # ---- Gating ----
        hs_beta = hidden_states.to(self.beta_proj.weight.dtype)
        beta = torch.sigmoid(self.beta_proj(hs_beta)).transpose(1, 2).contiguous()  # (B,H,L)
        hs_alpha = hidden_states.to(self.alpha_proj.weight.dtype)
        alpha = torch.sigmoid(self.alpha_proj(hs_alpha)).transpose(1, 2).contiguous()

        # ---- Mask ----
        eff_mask = encoder_attention_mask if is_cross else attention_mask
        mask_bool = self._normalize_valid_mask(eff_mask, B, kv_len, hidden_states.device)

        # ---- is_causal (GDN-FIX-08) ----
        is_causal = bool(self.is_decoder and not is_cross and L > 1 and past_key_value is None)

        # ---- Delta scan ----
        y_fwd = chunked_delta_rule_scan(
            q, k, v, beta, alpha, chunk_size=self.chunk_size,
        )
        if not self.is_decoder:
            q_b = torch.flip(q, dims=[2])
            k_b = torch.flip(k, dims=[2])
            v_b = torch.flip(v, dims=[2])
            beta_b = torch.flip(beta, dims=[2])
            alpha_b = torch.flip(alpha, dims=[2])
            y_bwd_f = chunked_delta_rule_scan(q_b, k_b, v_b, beta_b, alpha_b,
                                                chunk_size=self.chunk_size)
            y_bwd = torch.flip(y_bwd_f, dims=[2])
            y = 0.5 * (y_fwd + y_bwd)
        else:
            y = y_fwd

        # ---- head_mask (GDN-FIX-11) ----
        if head_mask is not None:
            hm = head_mask.to(y.dtype)
            if hm.dim() == 1:
                hm = hm.view(1, -1, 1, 1)
            elif hm.dim() == 2:
                hm = hm.view(hm.shape[0], hm.shape[1], 1, 1)
            elif hm.dim() == 3:
                hm = hm.unsqueeze(1)
            y = y * hm

        # ---- Recombine + SiLU gating ----
        y = torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)
        y = y.transpose(1, 2).contiguous().view(B, L, self.all_head_size)
        gated_y = y.to(z.dtype) * F.silu(z)

        # ---- Output projection ----
        if self.is_nested_encoder:
            out = gated_y
        else:
            out = self.out_proj(gated_y)

        out = out * self.output_scale

        # ---- mask output (only once; GDN-FIX-05) ----
        if mask_bool is not None:
            out = out * mask_bool.unsqueeze(-1).to(out.dtype)

        return out, None

    # ------------------------------------------------------------------
    def storage_bytes(self) -> dict[str, int]:
        def _sz(p):
            if isinstance(p, (nn.Parameter, torch.Tensor)):
                return p.numel() * p.element_size()
            return 0

        out = {}
        for name, m in (("q_proj", self.q_proj), ("k_proj", self.k_proj),
                         ("v_proj", self.v_proj), ("z_proj", self.z_proj),
                         ("out_proj", self.out_proj),
                         ("beta_proj", self.beta_proj),
                         ("alpha_proj", self.alpha_proj)):
            out[name] = sum(_sz(p) for p in m.parameters())
        out["output_scale"] = _sz(self.output_scale)
        out["total"] = sum(out.values())
        return out


# ===========================================================================
# Injection
# ===========================================================================
@ATTENTION_REGISTRY.register("gated_delta_net_inject")
@ATTENTION_REGISTRY.register("inject_gated_delta_net")
def inject_gated_delta_net(
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
        logger.warning("inject_gated_delta_net: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config
    if cfg is None and hasattr(target, "config"):
        cfg = target.config

    # ---- Whole-model traversal ----
    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "bert") and hasattr(target.bert, "encoder"):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    elif hasattr(target, "transformer") and hasattr(target.transformer, "layer"):
        layer_list = target.transformer.layer
    elif hasattr(target, "h") and isinstance(target.h, (nn.ModuleList, list)):
        layer_list = target.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        n_inj = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_gated_delta_net(
                layer=lyr, model_config=cfg, comp_config=comp_config,
                layer_idx=idx, total_layers=tot, _depth=_depth + 1, **kwargs,
            )
            if res is not None:
                n_inj += 1
        return target if n_inj > 0 else None

    # ---- Single layer ----
    src_attn = old_attn
    if src_attn is None and target is not None:
        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if isinstance(cand, nn.Module) and cand is not target:
                    src_attn = getattr(cand, "self", cand)
                    break
    if src_attn is None and target is not None and hasattr(target, "forward"):
        src_attn = target

    # GDN-FIX-06: idempotency guard
    if src_attn is not None and (isinstance(src_attn, GatedDeltaNet)
                                   or "GatedDeltaNet" in src_attn.__class__.__name__):
        return None

    if src_attn is None and cfg is None:
        return None

    hidden_size = (
        getattr(src_attn, "hidden_size", None)
        or getattr(src_attn, "embed_dim", None)
        or getattr(src_attn, "dim", None)
        or (getattr(src_attn.q_proj, "in_features", None) if hasattr(src_attn, "q_proj") else None)
        or (getattr(target, "in_features", None) if target is not None else None)
        or getattr(cfg, "hidden_size", getattr(cfg, "dim", 64))
    )
    num_heads = (
        getattr(src_attn, "num_heads", None)
        or getattr(src_attn, "num_attention_heads", None)
        or getattr(src_attn, "n_heads", None)
        or getattr(cfg, "num_attention_heads", getattr(cfg, "n_heads", 4))
    )

    try:
        new_block = GatedDeltaNet(
            config=cfg,
            comp_config=comp_config,
            d_model=hidden_size,
            num_heads=num_heads,
            layer_id=layer_idx,
            layer_idx=layer_idx,
            old_attn=src_attn if src_attn is not target else None,
            **kwargs,
        )
    except Exception as e:
        logger.exception("GatedDeltaNet injection failed at layer %d: %s", layer_idx, e)
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
            target.attn = new_block
        elif hasattr(target, "self_attn"):
            new_block.is_nested_encoder = False
            target.self_attn = new_block

    return new_block


__all__ = [
    "GatedDeltaNet",
    "inject_gated_delta_net",
    "chunked_delta_rule_scan",
    "apply_rotary_pos_emb",
    "LoRALinear",
    "repeat_kv",
]
