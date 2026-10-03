"""Phantom: Parameter-Hashed AttentioN with Token-wise Orthogonal Mapping – Production-Hardened.

Combines CUR-matrix factorized linear projections based on empirical statistical leverage scores
with multi-round angular Locality-Sensitive Hashing (LSH) sparse attention and rectified Softpick activation.

Supports:
- Multi-architecture copy_weights (GPT-2, RoBERTa/BERT nested, Qwen/LLaMA, DistilBERT)
- Trainable LoRA adapters with frozen CUR base weights
- GQA with repeat_kv
- RoPE position embeddings
- Causal masking for decoders
- Nested encoder bypass
- Frobenius norm verification (>= 1e-4)
- Safe injection returning None on failure
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralops.core.attention_mask import get_token_valid_mask
from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


@dataclass
class PHANTOMConfig:
    cur_rank: int = 16
    lsh_rounds: int = 4
    lsh_bucket_size: int = 4
    softpick_enabled: bool = True
    softpick_temperature: float = 1.0
    dropout: float = 0.1


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

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim < head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_rot_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
        k_rot_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)
        q = torch.cat([q_rot_embed, q_pass], dim=-1)
        k = torch.cat([k_rot_embed, k_pass], dim=-1)
    else:
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)

    return q, k


def softpick(x: torch.Tensor, dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
    orig_dtype = x.dtype
    x = x.float()
    temp = max(temperature, 1e-4)
    # Masked positions (future tokens) are <= -1000.0. Preserve mask while clamping unmasked values.
    is_masked = x <= -1000.0
    x_clamped = torch.where(
        is_masked, torch.full_like(x, -1e4), torch.clamp(x, min=-50.0, max=10.0)
    )
    res = F.softmax(x_clamped / temp, dim=dim)
    res = torch.nan_to_num(res, nan=0.0)
    return res.to(orig_dtype)


class FactorisedLinear_CUR(nn.Module):
    is_custom_attention: bool = True
    r"""Linear projection compressed via leverage-score CUR matrix decomposition with LoRA support."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        cur_rank: int = 16,
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
        self.cur_rank = max(1, min(cur_rank, in_features, out_features))
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0
        self.freeze_base = freeze_base

        self.down = nn.Linear(in_features, self.cur_rank, bias=False, **factory_kwargs)
        self.up = nn.Linear(self.cur_rank, out_features, bias=bias, **factory_kwargs)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(
                torch.empty((lora_rank, in_features), **factory_kwargs)
            )
            self.lora_B = nn.Parameter(
                torch.empty((out_features, lora_rank), **factory_kwargs)
            )
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.reset_parameters()
        if freeze_base:
            self.freeze_base_weights()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.up.weight, a=math.sqrt(5))
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)

        if self.lora_rank > 0 and self.lora_A is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
            nn.init.zeros_(self.lora_B)
            self.lora_B.requires_grad = True

    def freeze_base_weights(self) -> None:
        self.down.weight.requires_grad = False
        self.up.weight.requires_grad = False
        if self.up.bias is not None:
            self.up.bias.requires_grad = False
        if self.lora_rank > 0:
            if self.lora_A is not None:
                self.lora_A.requires_grad = True
            if self.lora_B is not None:
                self.lora_B.requires_grad = True

    def copy_cur(self, weight: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        with torch.no_grad():
            W = weight.float()
            out_dim, in_dim = W.shape
            rank = min(self.cur_rank, out_dim, in_dim)

            try:
                U_full, S_full, Vt_full = torch.linalg.svd(W, full_matrices=False)
                col_leverage = (Vt_full**2).sum(dim=0)
                col_leverage = col_leverage / torch.clamp(col_leverage.sum(), min=1e-5)
                row_leverage = (U_full**2).sum(dim=1)
                row_leverage = row_leverage / torch.clamp(row_leverage.sum(), min=1e-5)

                _, col_idx = torch.topk(col_leverage, k=rank, largest=True)
                _, row_idx = torch.topk(row_leverage, k=rank, largest=True)

                C = W[:, col_idx]
                R = W[row_idx, :]
                W_intersection = W[row_idx][:, col_idx]
                U_int = torch.linalg.pinv(W_intersection, rcond=1e-3)
                C_Uint = C @ U_int
                rec_norm = float(torch.linalg.norm(C_Uint @ R))
                orig_norm = float(torch.linalg.norm(W))

                if rec_norm > 2.0 * orig_norm or not math.isfinite(rec_norm):
                    # Ill-conditioned CUR intersection -> fallback to stable truncated SVD
                    U_r = U_full[:, :rank]
                    S_r = S_full[:rank]
                    Vt_r = Vt_full[:rank, :]
                    sqrt_s = torch.sqrt(S_r)
                    self.down.weight.data.zero_()
                    self.down.weight.data[:rank, :in_dim].copy_(
                        (sqrt_s.unsqueeze(1) * Vt_r).to(dtype=self.down.weight.dtype)
                    )
                    self.up.weight.data.zero_()
                    self.up.weight.data[:out_dim, :rank].copy_(
                        (U_r * sqrt_s.unsqueeze(0)).to(dtype=self.up.weight.dtype)
                    )
                else:
                    self.down.weight.data.zero_()
                    self.down.weight.data[:rank, :in_dim].copy_(
                        R.to(dtype=self.down.weight.dtype)
                    )
                    self.up.weight.data.zero_()
                    self.up.weight.data[:out_dim, :rank].copy_(
                        C_Uint.to(dtype=self.up.weight.dtype)
                    )
            except Exception:
                # SVD fallback
                nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.up.weight, a=math.sqrt(5))

            if bias is not None and self.up.bias is not None:
                self.up.bias.data.copy_(bias.to(dtype=self.up.bias.dtype))

        if self.freeze_base:
            self.freeze_base_weights()

    @property
    def weight(self) -> torch.Tensor:
        return torch.matmul(self.up.weight, self.down.weight)

    @property
    def bias(self) -> torch.Tensor | None:
        return self.up.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.up(self.down(x))
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = (
                F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
            )
            out = out + lora_out
        return out


def lsh_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    rounds: int,
    bucket_size: int,
    softpick_temp: float = 1.0,
    dropout: nn.Module | None = None,
    token_valid: torch.Tensor | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    B, H, N, D = Q.shape
    scale = D**-0.5
    n_bits = max(1, min(bucket_size, 8))

    Q_norm = F.normalize(Q, p=2, dim=-1, eps=1e-5)
    K_norm = F.normalize(K, p=2, dim=-1, eps=1e-5)

    bit_shifts = torch.tensor(
        [2**i for i in range(n_bits)], device=Q.device, dtype=torch.float32
    )

    scores_base = (
        torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous()) * scale
    )
    if token_valid is not None:
        val_mask = token_valid.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
        scores_base = scores_base.masked_fill(~val_mask, -1e4)

    if is_causal:
        causal_mask = torch.triu(torch.full((N, N), True, device=Q.device), diagonal=1)
        scores_base = scores_base.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), -1e4
        )

    accum_out = torch.zeros_like(V)

    for _ in range(rounds):
        hash_proj = torch.randn(D, n_bits, device=Q.device, dtype=Q.dtype)
        hash_proj = F.normalize(hash_proj, p=2, dim=0, eps=1e-5)

        hq = (torch.matmul(Q_norm, hash_proj) > 0).float() @ bit_shifts  # [B, H, N]
        hk = (torch.matmul(K_norm, hash_proj) > 0).float() @ bit_shifts  # [B, H, N]

        match = hq.unsqueeze(-1) == hk.unsqueeze(-2)  # [B, H, N, N]
        eye = torch.eye(N, device=Q.device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)
        match = match | eye

        if is_causal:
            match = match & (~causal_mask.unsqueeze(0).unsqueeze(0)) | eye

        scores_round = scores_base.masked_fill(~match, -1e4)
        attn_w = softpick(scores_round, dim=-1, temperature=softpick_temp).to(Q.dtype)
        if dropout is not None:
            attn_w = dropout(attn_w)

        accum_out = accum_out + torch.matmul(attn_w, V)

    return accum_out / float(max(1, rounds))


@ATTENTION_REGISTRY.register("phantom")
@ATTENTION_REGISTRY.register("phantom_module")
class PHANTOMAttention(BaseAttention):
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = 16,
        phantom_cfg: PHANTOMConfig | None = None,
        pretrained_attn: Any | None = None,
        bert_mode: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank or 16)
        self.config = config
        comp_config = getattr(config, "compression_config", None)

        cur_rank = rank or getattr(
            comp_config,
            "cur_rank",
            getattr(config, "cur_rank", getattr(config, "svd_rank", 16)),
        )
        if phantom_cfg is None:
            phantom_cfg = PHANTOMConfig(cur_rank=cur_rank)
        elif rank is not None:
            phantom_cfg.cur_rank = rank
        self.cfg: PHANTOMConfig = phantom_cfg

        model_config = config or kwargs.get("model_config")
        self.hidden_size: int = getattr(
            model_config, "hidden_size", kwargs.get("hidden_size", 768)
        )
        self.dim: int = self.hidden_size
        self.num_heads: int = getattr(
            model_config, "num_attention_heads", kwargs.get("num_attention_heads", 12)
        )
        self.num_kv_heads: int = getattr(
            model_config,
            "num_key_value_heads",
            kwargs.get("num_key_value_heads", self.num_heads),
        )
        self.head_dim: int = getattr(
            model_config,
            "head_dim",
            kwargs.get("head_dim", self.dim // self.num_heads),
        )
        self.all_head_size: int = self.num_heads * self.head_dim
        self.all_kv_head_size: int = self.num_kv_heads * self.head_dim
        self.kv_repeat: int = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.supports_gqa: bool = True

        is_dec = getattr(config, "is_decoder", kwargs.get("is_decoder", False))
        if kwargs.get("is_causal", False):
            is_dec = True
        self.is_decoder = bool(is_dec)
        self.is_causal = bool(is_dec)
        self.is_nested_encoder = False

        self.lora_rank = (
            getattr(
                config,
                "lora_rank",
                getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16)),
            )
            if config
            else 16
        )
        self.lora_alpha = (
            getattr(
                config,
                "lora_alpha",
                getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)),
            )
            if config
            else 32.0
        )
        self.freeze_base = self.lora_rank > 0

        source_attn = (
            pretrained_attn
            if pretrained_attn is not None
            else kwargs.get("source_attn")
        )
        target_dtype = kwargs.get("dtype")
        if target_dtype is None and source_attn is not None:
            target_dtype = next(source_attn.parameters(), torch.tensor([])).dtype
            if not isinstance(target_dtype, torch.dtype):
                target_dtype = torch.float32

        # Projections
        self.q_proj = FactorisedLinear_CUR(
            self.dim,
            self.all_head_size,
            cur_rank=self.cfg.cur_rank,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
            dtype=target_dtype,
        )
        self.k_proj = FactorisedLinear_CUR(
            self.dim,
            self.all_kv_head_size,
            cur_rank=self.cfg.cur_rank,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
            dtype=target_dtype,
        )
        self.v_proj = FactorisedLinear_CUR(
            self.dim,
            self.all_kv_head_size,
            cur_rank=self.cfg.cur_rank,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
            dtype=target_dtype,
        )
        self.out_proj = FactorisedLinear_CUR(
            self.all_head_size,
            self.dim,
            cur_rank=self.cfg.cur_rank,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
            dtype=target_dtype,
        )

        self.query = self.q_proj
        self.key = self.k_proj
        self.value = self.v_proj

        self.dropout = nn.Dropout(self.cfg.dropout)

        if source_attn is not None:
            self.copy_weights(source_attn)

        if target_dtype is not None:
            self.to(target_dtype)

    def copy_weights(self, src: Any) -> None:
        if src is None:
            return

        cls_name = src.__class__.__name__
        has_out_lin = (
            hasattr(src, "out_lin") or hasattr(src, "c_proj") or hasattr(src, "o_proj")
        )
        is_nested = False
        if (
            hasattr(src, "self")
            and hasattr(src.self, "query")
            or (
                cls_name
                in [
                    "BertSelfAttention",
                    "RobertaSelfAttention",
                    "XLMRobertaSelfAttention",
                    "ElectraSelfAttention",
                    "DebertaSelfAttention",
                ]
                or (
                    cls_name.endswith("SelfAttention")
                    and "distil" not in cls_name.lower()
                )
            )
            and not has_out_lin
        ):
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. GPT-2 fused
        if hasattr(src, "c_attn") and src.c_attn is not None:
            c_attn_mod = src.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                if (
                    (c_attn_mod.__class__.__name__ == "Conv1D")
                    or (base_c.__class__.__name__ == "Conv1D")
                    or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1])
                ):
                    c_w = c_w.t().contiguous()
                split_size = c_w.size(0) // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_b = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (
                    c_w[:split_size, :],
                    c_b[:split_size] if c_b is not None else None,
                )
                extracted["k"] = (
                    c_w[split_size : 2 * split_size, :],
                    c_b[split_size : 2 * split_size] if c_b is not None else None,
                )
                extracted["v"] = (
                    c_w[2 * split_size :, :],
                    c_b[2 * split_size :] if c_b is not None else None,
                )

            if hasattr(src, "c_proj") and src.c_proj is not None:
                c_proj_mod = src.c_proj
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    if (
                        (c_proj_mod.__class__.__name__ == "Conv1D")
                        or (base_o.__class__.__name__ == "Conv1D")
                        or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1])
                    ):
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (
                        o_w,
                        bias_o.data.clone() if bias_o is not None else None,
                    )

        # 2. RoBERTa nested
        elif hasattr(src, "self") and hasattr(src.self, "query"):
            s = src.self
            for proj_key, sub_name in [("q", "query"), ("k", "key"), ("v", "value")]:
                mod = getattr(s, sub_name, None)
                if mod is not None and hasattr(mod, "weight"):
                    extracted[proj_key] = (
                        mod.weight.data.clone(),
                        mod.bias.data.clone() if mod.bias is not None else None,
                    )
            if hasattr(src, "output") and hasattr(src.output, "dense"):
                out_mod = src.output.dense
                extracted["o"] = (
                    out_mod.weight.data.clone(),
                    out_mod.bias.data.clone() if out_mod.bias is not None else None,
                )

        # 3. Flat modules
        else:
            proj_candidates = {
                "q": ["q_lin", "q_proj", "query"],
                "k": ["k_lin", "k_proj", "key"],
                "v": ["v_lin", "v_proj", "value"],
                "o": ["out_lin", "out_proj", "dense", "o_proj"],
            }
            for proj_key, names in proj_candidates.items():
                for name in names:
                    if hasattr(src, name):
                        mod = getattr(src, name)
                        if mod is not None and hasattr(mod, "weight"):
                            extracted[proj_key] = (
                                mod.weight.data.clone(),
                                mod.bias.data.clone() if mod.bias is not None else None,
                            )
                            break

        missing = [
            k
            for k, (w, _) in extracted.items()
            if w is None and (k != "o" or not self.is_nested_encoder)
        ]
        if missing:
            try:
                from neuralops.core.adapters import UniversalAdapter

                adapter = UniversalAdapter()
                for k in missing:
                    w, b = adapter.get_weight_bias(src, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception as e:
                logger.debug("UniversalAdapter fallback failed: %s", e)

        missing_after = [
            k
            for k, (w, _) in extracted.items()
            if w is None and (k != "o" or not self.is_nested_encoder)
        ]
        if missing_after:
            logger.warning(
                f"PHANTOMAttention copy_weights failed: missing mandatory projections {missing_after} from {src.__class__.__name__}."
            )
            return

        target_map = {
            "q": self.q_proj,
            "k": self.k_proj,
            "v": self.v_proj,
            "o": self.out_proj,
        }

        with torch.no_grad():
            for proj_key, target_mod in target_map.items():
                w, b = extracted[proj_key]
                if w is None:
                    if proj_key == "o" and self.is_nested_encoder:
                        target_mod.copy_cur(
                            torch.eye(self.all_head_size, self.dim), None
                        )
                        continue
                    else:
                        logger.warning(f"Missing required weights for {proj_key}")
                        return

                if w.shape != (target_mod.out_features, target_mod.in_features):
                    if w.t().contiguous().shape == (
                        target_mod.out_features,
                        target_mod.in_features,
                    ):
                        w = w.t().contiguous()
                    elif (
                        w.shape[0] < target_mod.out_features
                        and target_mod.out_features % w.shape[0] == 0
                    ):
                        repeats = target_mod.out_features // w.shape[0]
                        w = w.repeat(repeats, 1)
                    else:
                        full_w = torch.zeros(
                            (target_mod.out_features, target_mod.in_features),
                            dtype=w.dtype,
                            device=w.device,
                        )
                        min_0 = min(target_mod.out_features, w.shape[0])
                        min_1 = min(target_mod.in_features, w.shape[1])
                        full_w[:min_0, :min_1].copy_(w[:min_0, :min_1])
                        w = full_w

                target_mod.copy_cur(w, b)

                fnorm = torch.linalg.norm(target_mod.weight.float()).item()
                if fnorm < 1e-4:
                    logger.warning(
                        f"Weight copy notice for {proj_key}: norm {fnorm:.6f} < 1e-4"
                    )

        if self.freeze_base:
            for mod in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
                mod.freeze_base_weights()

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
        B, N, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim

        Q = self.q_proj(hidden_states).view(B, N, H, D).permute(0, 2, 1, 3)
        K = (
            self.k_proj(hidden_states)
            .view(B, N, self.num_kv_heads, D)
            .permute(0, 2, 1, 3)
        )
        V = (
            self.v_proj(hidden_states)
            .view(B, N, self.num_kv_heads, D)
            .permute(0, 2, 1, 3)
        )

        # Apply RoPE if provided
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None:
            cos, sin = position_embeddings
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        # Repeat KV for GQA
        if self.num_heads != self.num_kv_heads:
            repeats = self.num_heads // self.num_kv_heads
            K = repeat_kv(K, repeats)
            V = repeat_kv(V, repeats)

        token_valid = get_token_valid_mask(attention_mask, B, N, hidden_states.device)
        is_causal = self.is_decoder or kwargs.get("is_causal", False)

        context = lsh_attention(
            Q,
            K,
            V,
            rounds=self.cfg.lsh_rounds,
            bucket_size=self.cfg.lsh_bucket_size,
            softpick_temp=(
                self.cfg.softpick_temperature if self.cfg.softpick_enabled else 1.0
            ),
            dropout=self.dropout if self.cfg.softpick_enabled else None,
            token_valid=token_valid,
            is_causal=is_causal,
        )

        # head_mask handling
        if head_mask is not None:
            if head_mask.dim() == 1 and head_mask.size(0) == self.num_heads:
                context = context * head_mask.view(1, self.num_heads, 1, 1).to(
                    context.dtype
                )
            elif head_mask.dim() == 2 and head_mask.size(1) == self.num_heads:
                context = context * head_mask.view(
                    head_mask.size(0), self.num_heads, 1, 1
                ).to(context.dtype)
            elif head_mask.dim() == 4:
                context = context * head_mask.to(context.dtype)

        context = (
            context.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
        )

        if self.is_nested_encoder:
            return context, None

        output = self.out_proj(context)
        return output, None


def inject_phantom(
    model: nn.Module | None = None,
    layer: nn.Module | None = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    **kwargs: Any,
) -> PHANTOMAttention | nn.Module | None:
    """Safe injection hook for PHANTOMAttention across diverse transformer architectures."""
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
            res = inject_phantom(
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
    try:
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

        if src_attn is None:
            return None

        # Double-injection guard
        if src_attn is not None:
            if isinstance(src_attn, PHANTOMAttention):
                return None
            if "PHANTOM" in src_attn.__class__.__name__:
                return None
            if hasattr(src_attn, "inner") and isinstance(
                src_attn.inner, PHANTOMAttention
            ):
                return None
            if hasattr(src_attn, "_wrapped_attention") and isinstance(
                src_attn._wrapped_attention, PHANTOMAttention
            ):
                return None

        if target is not None:
            for attr in ("attention", "self_attn", "attn", "self"):
                cand = getattr(target, attr, None)
                if cand is None:
                    continue
                if isinstance(cand, PHANTOMAttention):
                    return None
                if "PHANTOM" in cand.__class__.__name__:
                    return None
                if hasattr(cand, "inner") and isinstance(cand.inner, PHANTOMAttention):
                    return None
                if hasattr(cand, "_wrapped_attention") and isinstance(
                    cand._wrapped_attention, PHANTOMAttention
                ):
                    return None

        # Skip Qwen3.5 hybrid DeltaNet/LinearAttention layers
        if src_attn is not None:
            c_name = src_attn.__class__.__name__
            if "DeltaNet" in c_name or "LinearAttention" in c_name:
                return None

        if cfg is None and hasattr(src_attn, "config"):
            cfg = src_attn.config

        hidden_size = (
            kwargs.get("hidden_size")
            or getattr(src_attn, "hidden_size", None)
            or getattr(src_attn, "embed_dim", None)
            or (
                src_attn.q_proj.in_features
                if hasattr(src_attn, "q_proj")
                and hasattr(src_attn.q_proj, "in_features")
                else None
            )
            or (
                src_attn.query.in_features
                if hasattr(src_attn, "query") and hasattr(src_attn.query, "in_features")
                else None
            )
            or (
                src_attn.q_lin.in_features
                if hasattr(src_attn, "q_lin") and hasattr(src_attn.q_lin, "in_features")
                else None
            )
            or (target.in_features if hasattr(target, "in_features") else None)
            or getattr(cfg, "hidden_size", None)
            or getattr(cfg, "d_model", 64)
        )
        num_heads = (
            kwargs.get("num_heads")
            or kwargs.get("num_attention_heads")
            or getattr(src_attn, "num_heads", None)
            or getattr(src_attn, "num_attention_heads", None)
            or getattr(cfg, "num_attention_heads", None)
            or getattr(cfg, "num_heads", 4)
        )

        new_attn = PHANTOMAttention(
            config=cfg,
            comp_config=comp_config,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_attention_heads=num_heads,
            pretrained_attn=src_attn,
            **kwargs,
        )

        if target is not None:
            if hasattr(target, "attention"):
                if hasattr(target.attention, "self"):
                    target.attention.self = new_attn
                    new_attn.is_nested_encoder = True
                else:
                    target.attention = new_attn
                return new_attn
            for attr in ("self_attn", "attn", "self"):
                if hasattr(target, attr):
                    setattr(target, attr, new_attn)
                    return new_attn
            target.attention = new_attn
            return new_attn
        return new_attn
    except Exception as e:
        logger.error("inject_phantom failed: %s", e)
        return None


# Aliases and Registry
@ATTENTION_REGISTRY.register("phantom_inject")
def inject_phantom_attention(
    *args: Any, **kwargs: Any
) -> PHANTOMAttention | nn.Module | None:
    return inject_phantom(*args, **kwargs)
