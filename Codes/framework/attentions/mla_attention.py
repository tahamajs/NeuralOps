"""Multi-Head Latent Attention (MLA) – DeepSeek-V2.

Paper: https://arxiv.org/abs/2405.04434
Hardened for multi-architecture production deployment across DistilBERT, RoBERTa, GPT-2, and Qwen3.5.
"""

import logging
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
from framework.core.registry import ATTENTION_REGISTRY

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


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        freeze_base: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
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
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        return out


@ATTENTION_REGISTRY.register("mla", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("mla_attention", supports_causal=True, supports_gqa=True)
class MLAAttention(BaseAttention):
    r"""Multi-Head Latent Attention (MLA) Mechanism.
    Hardened for multi-architecture deployment across DistilBERT, RoBERTa, GPT-2, and Qwen3.5.
    """

    supports_causal: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = 128,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.hidden_size: int = getattr(config, "hidden_size", getattr(config, "d_model", kwargs.get("hidden_size", 768)))
        self.num_heads: int = getattr(
            config, "num_attention_heads", getattr(config, "n_heads", getattr(config, "n_head", getattr(config, "num_heads", 12)))
        )
        self.head_dim: int = getattr(config, "head_dim", self.hidden_size // max(1, self.num_heads))
        self.q_dim: int = getattr(config, "q_proj_out", self.num_heads * self.head_dim)
        self.all_head_size: int = self.q_dim

        cfg_kv = getattr(config, "num_key_value_heads", getattr(config, "num_kv_heads", None))
        if cfg_kv is not None:
            self.num_kv_heads = int(cfg_kv)
        elif kwargs.get("num_kv_heads") is not None:
            self.num_kv_heads = int(kwargs["num_kv_heads"])
        else:
            self.num_kv_heads = self.num_heads

        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))
        self.kv_dim: int = self.num_kv_heads * self.head_dim
        self.kv_head_size: int = self.kv_dim
        self.kv_repeat: int = max(1, self.num_heads // max(1, self.num_kv_heads))

        self.rank: int = getattr(config, "mla_rank", rank if rank is not None else 128)
        self.lora_rank: int = lora_rank
        self.lora_alpha: float = lora_alpha

        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            _mt = str(getattr(config, "model_type", "") or "").lower()
            _dec_pfx = ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
            self.is_decoder = (
                getattr(config, "is_decoder", False)
                or getattr(config, "causal", False)
                or any(_mt.startswith(p) for p in _dec_pfx)
            )

        self.is_nested_encoder: bool = False

        self.q_proj = LoRALinear(self.hidden_size, self.q_dim, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True)
        self.kv_down = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.k_up = nn.Linear(self.rank, self.kv_dim, bias=False)
        self.v_up = nn.Linear(self.rank, self.kv_dim, bias=False)
        self.out_proj = LoRALinear(self.q_dim, self.hidden_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=True)

        dropout_p = getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1))
        self.dropout: nn.Dropout = nn.Dropout(dropout_p)

        pretrained_attn = kwargs.get("pretrained_attn", None)
        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    @property
    def query(self) -> LoRALinear:
        return self.q_proj

    @property
    def key(self) -> nn.Module:
        return self.k_up

    @property
    def value(self) -> nn.Module:
        return self.v_up

    def copy_weights(self, src: Any) -> None:
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
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                if w is not None:
                    extracted[proj] = (w, b)
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        # 2. GPT-2 Fused c_attn Conv1D handling
        if hasattr(src, "c_attn") and (extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None):
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
            if w_o.dim() == 2 and w_o.shape[0] != self.hidden_size and w_o.shape[1] == self.hidden_size:
                w_o = w_o.t().contiguous()
            b_o = getattr(src.c_proj, "bias", None)
            extracted["o"] = (w_o, b_o.data if b_o is not None else None)

        # 3. RoBERTa / BERT Nested Encoder handling
        if hasattr(src, "self") and (extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None):
            s = getattr(src, "self")
            if hasattr(s, "query"):
                extracted["q"] = (s.query.weight.data, getattr(s.query, "bias", None))
            if hasattr(s, "key"):
                extracted["k"] = (s.key.weight.data, getattr(s.key, "bias", None))
            if hasattr(s, "value"):
                extracted["v"] = (s.value.weight.data, getattr(s.value, "bias", None))

        if hasattr(src, "output") and hasattr(getattr(src, "output"), "dense") and extracted["o"][0] is None:
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
                            extracted[proj] = (layer.weight.data, getattr(layer, "bias", None))
                            break

        # Detect nested encoder architecture
        cls_name = src.__class__.__name__
        has_out_proj = any(hasattr(src, attr) for attr in ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "wo", "o", "proj"))
        is_nested = False
        if hasattr(src, "self") and hasattr(getattr(src, "self"), "query"):
            is_nested = True
        elif cls_name in ("BertSelfAttention", "RobertaSelfAttention", "XLMRobertaSelfAttention") or cls_name.endswith("SelfAttention"):
            is_nested = True
        elif cls_name.endswith("Attention") and not has_out_proj:
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        # Check required weights
        missing = [k for k in ["q", "k", "v"] if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            logger.warning(f"MLAAttention missing required weights for projections: {missing}")
            return

        # Frobenius norm validation
        for proj_key in ["q", "k", "v"]:
            w = extracted[proj_key][0]
            if w is not None:
                norm = torch.linalg.norm(w.float())
                if norm < 1e-4:
                    logger.warning(f"MLAAttention extracted weight for {proj_key} has near-zero Frobenius norm ({norm.item():.6f})")

        if extracted["o"][0] is not None:
            norm_o = torch.linalg.norm(extracted["o"][0].float())
            if norm_o < 1e-4:
                logger.warning(f"MLAAttention extracted weight for output projection has near-zero Frobenius norm ({norm_o.item():.6f})")

        def _orient(w: torch.Tensor, in_dim: int) -> torch.Tensor:
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

        with torch.no_grad():
            # 1. Q projection
            w_q, b_q = extracted["q"]
            w_q_oriented = _orient(w_q, self.hidden_size)
            self.q_proj.weight.data[:min(self.q_dim, w_q_oriented.shape[0]), :min(self.hidden_size, w_q_oriented.shape[1])].copy_(
                w_q_oriented[:min(self.q_dim, w_q_oriented.shape[0]), :min(self.hidden_size, w_q_oriented.shape[1])]
            )
            if b_q is not None and self.q_proj.bias is not None:
                self.q_proj.bias.data[:min(len(self.q_proj.bias), len(b_q))].copy_(b_q[:min(len(self.q_proj.bias), len(b_q))])
            self.q_proj.freeze_base_weights()

            # 2. Output projection
            w_o, b_o = extracted["o"]
            if w_o is not None:
                w_o_oriented = _orient(w_o, self.q_dim)
                self.out_proj.weight.data[:min(self.hidden_size, w_o_oriented.shape[0]), :min(self.q_dim, w_o_oriented.shape[1])].copy_(
                    w_o_oriented[:min(self.hidden_size, w_o_oriented.shape[0]), :min(self.q_dim, w_o_oriented.shape[1])]
                )
                if b_o is not None and self.out_proj.bias is not None:
                    self.out_proj.bias.data[:min(len(self.out_proj.bias), len(b_o))].copy_(b_o[:min(len(self.out_proj.bias), len(b_o))])
            elif self.is_nested_encoder:
                min_d = min(self.out_proj.weight.shape[0], self.out_proj.weight.shape[1])
                self.out_proj.weight.data.zero_()
                self.out_proj.weight.data[:min_d, :min_d].fill_diagonal_(1.0)
                if self.out_proj.bias is not None:
                    self.out_proj.bias.data.zero_()
            self.out_proj.freeze_base_weights()

            # 3. Low-rank K/V SVD decomposition
            w_k, _ = extracted["k"]
            w_v, _ = extracted["v"]
            w_k_f = _orient(w_k, self.hidden_size).float()
            w_v_f = _orient(w_v, self.hidden_size).float()

            # Handle GQA dimension matching
            if w_k_f.shape[0] < self.kv_dim and self.kv_dim % w_k_f.shape[0] == 0:
                rep = self.kv_dim // w_k_f.shape[0]
                w_k_f = w_k_f.repeat(rep, 1)
            elif w_k_f.shape[0] > self.kv_dim and w_k_f.shape[0] % self.kv_dim == 0:
                rep = w_k_f.shape[0] // self.kv_dim
                w_k_f = w_k_f.view(self.kv_dim, rep, -1).mean(dim=1)

            if w_v_f.shape[0] < self.kv_dim and self.kv_dim % w_v_f.shape[0] == 0:
                rep = self.kv_dim // w_v_f.shape[0]
                w_v_f = w_v_f.repeat(rep, 1)
            elif w_v_f.shape[0] > self.kv_dim and w_v_f.shape[0] % self.kv_dim == 0:
                rep = w_v_f.shape[0] // self.kv_dim
                w_v_f = w_v_f.view(self.kv_dim, rep, -1).mean(dim=1)

            w_kv = torch.cat([w_k_f[:self.kv_dim, :self.hidden_size], w_v_f[:self.kv_dim, :self.hidden_size]], dim=0)
            u_kv, s_kv, vh_kv = torch.linalg.svd(w_kv, full_matrices=False)
            r = min(self.rank, s_kv.shape[0], self.kv_down.weight.shape[0], self.k_up.weight.shape[1], self.v_up.weight.shape[1])

            orig_norm = torch.linalg.norm(w_kv)
            approx_norm = torch.linalg.norm(s_kv[:r])
            norm_scale = (orig_norm / torch.clamp(approx_norm, min=1e-12)).to(self.k_up.weight.device, self.k_up.weight.dtype)

            self.kv_down.weight.data.zero_()
            self.k_up.weight.data.zero_()
            self.v_up.weight.data.zero_()

            self.kv_down.weight.data[:r, :].copy_(vh_kv[:r, :].to(self.kv_down.weight.device, self.kv_down.weight.dtype))
            up = (u_kv[:, :r] * s_kv[:r].unsqueeze(0)).to(self.k_up.weight.device, self.k_up.weight.dtype) * norm_scale
            d_k = min(self.kv_dim, self.k_up.weight.shape[0])
            d_v = min(self.kv_dim, self.v_up.weight.shape[0])
            self.k_up.weight.data[:d_k, :r].copy_(up[:d_k, :r])
            self.v_up.weight.data[:d_v, :r].copy_(up[self.kv_dim : self.kv_dim + d_v, :r])

            self.kv_down.weight.requires_grad = False
            self.k_up.weight.requires_grad = False
            self.v_up.weight.requires_grad = False

        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)
        elif cls_name in ("GPT2Attention", "GPT2Block") or hasattr(src, "c_attn") or "gpt" in cls_name.lower():
            self.is_decoder = True

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
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        kv_len = kv_input.shape[1]

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        latent = self.kv_down(kv_input)
        k = self.k_up(latent).view(batch_size, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.v_up(latent).view(batch_size, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.num_kv_heads < self.num_heads:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        scale = 1.0 / math.sqrt(self.head_dim)
        eff_mask = encoder_attention_mask if is_cross else attention_mask
        if eff_mask is None and head_mask is None and not output_attentions:
            try:
                _qs = q.half() if q.dtype == torch.float32 else q
                _ks = k.half() if k.dtype == torch.float32 else k
                _vs = v.half() if v.dtype == torch.float32 else v
                _is_causal = getattr(self, 'is_decoder', False) and not is_cross and seq_len > 1
                _sdpa_out = torch.nn.functional.scaled_dot_product_attention(
                    _qs, _ks, _vs, is_causal=_is_causal, dropout_p=self.dropout.p if self.training else 0.0
                )
                _sdpa_out = _sdpa_out.to(hidden_states.dtype).transpose(1, 2).contiguous().view(batch_size, seq_len, self.all_head_size)
                _sdpa_out = torch.nan_to_num(_sdpa_out, nan=0.0, posinf=1e4, neginf=-1e4)
                return (self.out_proj(_sdpa_out) if not getattr(self, 'is_nested_encoder', False) else _sdpa_out), None
            except Exception:
                pass

        scores = torch.matmul(q.float(), k.float().transpose(-2, -1).contiguous()) * scale

        if self.is_decoder and not is_cross and seq_len > 1:
            causal_mask = torch.triu(
                torch.full((seq_len, kv_len), float("-inf"), device=device, dtype=torch.float32),
                diagonal=1,
            ).unsqueeze(0).unsqueeze(0)
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
        attn_probs = F.softmax(scores, dim=-1).to(dtype)
        attn_probs = self.dropout(attn_probs)

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(-1)
            attn_probs = attn_probs * head_mask

        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.all_head_size)
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            out = context
        else:
            out = self.out_proj(context)

        return out, (attn_probs if output_attentions else None)


@ATTENTION_REGISTRY.register("mla_inject")
def inject_mla(
    model: Optional[nn.Module] = None,
    layer: Optional[nn.Module] = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 1,
    old_attn: Any = None,
    **kwargs: Any,
) -> Optional[Union[MLAAttention, nn.Module]]:
    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config or getattr(target, "config", None)
    if cfg is None:
        cfg = type('Cfg', (), {'hidden_size': 64, 'num_heads': 4, 'rank': 32})()

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
                inject_mla(
                    layer=lyr,
                    model_config=cfg,
                    comp_config=comp_config,
                    layer_idx=idx,
                    total_layers=tot,
                    **kwargs,
                )
            return target

    # 2. Layer injection
    src_attn = old_attn if old_attn is not None else getattr(target, "attention", getattr(target, "self", None))
    if src_attn is not None:
        if isinstance(src_attn, MLAAttention) or "MLA" in src_attn.__class__.__name__:
            return None

    rank = getattr(comp_config, "mla_rank", 128) if comp_config is not None else 128
    new_attn = MLAAttention(cfg, rank=rank)
    if src_attn is not None:
        try:
            new_attn.copy_weights(src_attn)
        except Exception as e:
            logger.warning(f"copy_weights failed: {e}")
        p = next(src_attn.parameters(), None)
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)
    if hasattr(target, "attention"):
        if hasattr(target.attention, "self"):
            new_attn.is_nested_encoder = True
            target.attention.self = new_attn
        else:
            target.attention = new_attn
    elif hasattr(target, "self"):
        new_attn.is_nested_encoder = True
        target.self = new_attn
    return target if target is not None else new_attn


# Aliases
inject_mla_attention = inject_mla
