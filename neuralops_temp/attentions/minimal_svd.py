import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.gqa_utils import repeat_kv
from neuralops.core.registry import ATTENTION_REGISTRY

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
) -> tuple[torch.Tensor, torch.Tensor]:
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


class FactorisedLinear(nn.Module):
    is_custom_attention: bool = True
    r"""SVD-Factorized Linear Transformation for NOVA Architecture."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        bias: bool = True,
        energy: float = 0.95,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, in_features, out_features)
        self.energy = energy
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0

        self.down: nn.Linear = nn.Linear(in_features, self.rank, bias=False)
        self.up: nn.Linear = nn.Linear(self.rank, out_features, bias=bias)

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
        bias: torch.Tensor | None = None,
        energy: float | None = None,
    ) -> None:
        with torch.no_grad():
            W = weight.float()
            # Handle KV dimension matching for GQA
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
                padded_W = torch.zeros(
                    self.out_features, self.in_features, device=W.device, dtype=W.dtype
                )
                padded_W[
                    : min(self.out_features, W.shape[0]),
                    : min(self.in_features, W.shape[1]),
                ] = W[
                    : min(self.out_features, W.shape[0]),
                    : min(self.in_features, W.shape[1]),
                ]
                W = padded_W

            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            target_energy = energy if energy is not None else self.energy
            total_energy = (S**2).sum()
            cum_energy = torch.cumsum(S**2, dim=0) / (total_energy + 1e-12)
            calc_rank = int((cum_energy < target_energy).sum().item()) + 1
            r = min(self.rank, calc_rank, U.size(1), Vt.size(0))
            r = max(1, r)

            sqrt_s = torch.sqrt(S[:r])
            A = (U[:, :r] * sqrt_s).to(self.up.weight.dtype)
            B = (sqrt_s.unsqueeze(1) * Vt[:r, :]).to(self.down.weight.dtype)

            self.up.weight.data.zero_()
            self.down.weight.data.zero_()
            self.up.weight.data[:, :r] = A
            self.down.weight.data[:r, :] = B

            if bias is not None and self.up.bias is not None:
                self.up.bias.data[: min(len(self.up.bias), len(bias))].copy_(
                    bias[: min(len(self.up.bias), len(bias))]
                )

            self.freeze_base_weights()

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

    @property
    def weight(self) -> torch.Tensor:
        base_w = torch.matmul(self.up.weight, self.down.weight)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            return base_w + (self.lora_B @ self.lora_A) * self.lora_scaling
        return base_w

    @property
    def bias(self) -> torch.Tensor | None:
        return self.up.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.up(self.down(x))
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        return out


@ATTENTION_REGISTRY.register("minimal_svd", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("minimal_aurora", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("minimal_prism", supports_causal=True, supports_gqa=True)
@ATTENTION_REGISTRY.register("minimal_svd_v2", supports_causal=True, supports_gqa=True)
class MinimalSVDAttention(BaseAttention):
    r"""
    MinimalSVD: SVD-Factorized Low-Rank Projection Attention.
    Hardened for multi-architecture deployment across DistilBERT, RoBERTa, GPT-2, and Qwen3.5.
    """

    supports_causal: bool = True
    supports_gqa: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        model_config: Any | None = None,
        energy: float = 0.95,
        dropout: float = 0.1,
        pretrained_attn: nn.Module | None = None,
        rank: int | None = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: bool | None = None,
        **kwargs: Any,
    ) -> None:
        cfg = model_config if model_config is not None else config
        actual_rank = getattr(
            cfg,
            "svd_rank",
            rank if rank is not None else getattr(cfg, "rank", 64),
        )
        super().__init__(cfg, actual_rank)
        self.num_heads: int = getattr(
            cfg,
            "num_attention_heads",
            getattr(cfg, "num_heads", kwargs.get("num_attention_heads", 4)),
        )
        self.hidden_size: int = getattr(cfg, "hidden_size", kwargs.get("hidden_size", 768))
        cfg_head_dim = (
            getattr(cfg, "head_dim", None)
            if cfg
            else (getattr(model_config, "head_dim", None) if model_config else None)
        )
        self.head_dim: int = int(
            cfg_head_dim
            or kwargs.get("head_dim")
            or (
                self.hidden_size // self.num_heads
                if self.num_heads > 0
                else (self.hidden_size or 64)
            )
        )
        self.all_head_size: int = self.num_heads * self.head_dim

        cfg_kv = (
            getattr(cfg, "num_key_value_heads", getattr(cfg, "num_kv_heads", None)) if cfg else None
        )
        if cfg_kv is not None:
            self.num_kv_heads = int(cfg_kv)
        elif kwargs.get("num_kv_heads") is not None:
            self.num_kv_heads = int(kwargs["num_kv_heads"])
        else:
            self.num_kv_heads = self.num_heads

        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))
        self.kv_repeat: int = self.num_heads // self.num_kv_heads
        self.kv_head_size: int = self.num_kv_heads * self.head_dim

        self.rank: int = actual_rank
        self.energy: float = energy
        self.lora_rank: int = lora_rank
        self.lora_alpha: float = lora_alpha

        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            self.is_decoder = (
                getattr(cfg, "is_decoder", False)
                or getattr(cfg, "causal", False)
                or getattr(cfg, "model_type", "").lower() in ("gpt2", "qwen", "qwen2", "llama")
                or any(
                    str(
                        getattr(
                            (
                                cfg
                                if "cfg" in locals()
                                else (config if "config" in locals() else None)
                            ),
                            "model_type",
                            "",
                        )
                    )
                    .lower()
                    .startswith(p)
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
                )
                if cfg
                else False
            )

        model_type = getattr(cfg, "model_type", "").lower() if cfg else ""
        self.is_nested_encoder: bool = getattr(cfg, "is_nested_encoder", False) or model_type in (
            "roberta",
            "bert",
            "xlm-roberta",
        )

        self.q_proj = FactorisedLinear(
            self.hidden_size,
            self.all_head_size,
            rank=self.rank,
            bias=True,
            energy=energy,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
        )
        self.k_proj = FactorisedLinear(
            self.hidden_size,
            self.kv_head_size,
            rank=self.rank,
            bias=True,
            energy=energy,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
        )
        self.v_proj = FactorisedLinear(
            self.hidden_size,
            self.kv_head_size,
            rank=self.rank,
            bias=True,
            energy=energy,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
        )
        self.out_proj = FactorisedLinear(
            self.all_head_size,
            self.hidden_size,
            rank=self.rank,
            bias=True,
            energy=energy,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
        )

        dropout_p = (
            getattr(cfg, "dropout", getattr(cfg, "attention_probs_dropout_prob", dropout))
            if cfg
            else dropout
        )
        self.dropout: nn.Dropout = nn.Dropout(dropout_p)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    @property
    def query(self) -> FactorisedLinear:
        return self.q_proj

    @property
    def key(self) -> FactorisedLinear:
        return self.k_proj

    @property
    def value(self) -> FactorisedLinear:
        return self.v_proj

    def copy_weights(self, src: Any) -> None:
        from neuralops.core.adapters import UniversalAdapter

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
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
                logger.debug(f"UniversalAdapter extraction failed for {proj}: {e}")

        # 2. GPT-2 Fused c_attn / c_proj Conv1D handling
        if hasattr(src, "c_attn") and (
            extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None
        ):
            c_attn_mod = src.c_attn
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            weight = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if weight is not None:
                weight = weight.data.clone()
                bias = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                bias_data = bias.data.clone() if bias is not None else None
                p_cls = getattr(base_c, "__class__", type(base_c)).__name__
                if (
                    ("Conv1D" in p_cls)
                    or hasattr(base_c, "nf")
                    or ("Conv1D" in getattr(c_attn_mod, "__class__", type(c_attn_mod)).__name__)
                    or hasattr(c_attn_mod, "nf")
                    or (weight.dim() == 2 and weight.shape[1] == 3 * self.hidden_size)
                ):
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
            c_proj_mod = src.c_proj
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
                    or (w_o.dim() == 2 and w_o.shape[0] < w_o.shape[1])
                ):
                    w_o = w_o.t().contiguous()
                b_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                extracted["o"] = (w_o, b_o.data.clone() if b_o is not None else None)

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
        self.is_nested_encoder = bool(is_nested)

        # Check required weights
        missing = [k for k in ["q", "k", "v"] if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            logger.warning(
                f"MinimalSVDAttention missing required weights for projections: {missing}. Proceeding with initialized weights."
            )

        # Frobenius norm validation
        for proj_key in ["q", "k", "v"]:
            w = extracted[proj_key][0]
            if w is not None:
                norm = torch.linalg.norm(w.float())
                if norm < 1e-4:
                    logger.warning(
                        f"MinimalSVDAttention extracted weight for {proj_key} has near-zero Frobenius norm ({norm.item():.6f})"
                    )

        if extracted["o"][0] is not None:
            norm_o = torch.linalg.norm(extracted["o"][0].float())
            if norm_o < 1e-4:
                logger.warning(
                    f"MinimalSVDAttention extracted weight for output projection has near-zero Frobenius norm ({norm_o.item():.6f})"
                )

        def _orient(w: torch.Tensor, in_dim: int) -> torch.Tensor:
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t().contiguous()
            return w

        # Copy into factorized projection modules
        with torch.no_grad():
            for proj_key, target in [
                ("q", self.q_proj),
                ("k", self.k_proj),
                ("v", self.v_proj),
                ("o", self.out_proj),
            ]:
                w, b = extracted[proj_key]
                if w is not None:
                    in_dim = self.all_head_size if proj_key == "o" else self.hidden_size
                    w_oriented = _orient(w, in_dim)
                    target.decompose_from_weight(w_oriented, b, self.energy)
                elif proj_key == "o" and self.is_nested_encoder:
                    eye_w = torch.eye(
                        self.all_head_size,
                        self.all_head_size,
                        device=self.out_proj.up.weight.device,
                        dtype=self.out_proj.up.weight.dtype,
                    )
                    target.decompose_from_weight(eye_w, None, self.energy)

                target.freeze_base_weights()

        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)

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
        device = hidden_states.device
        dtype = hidden_states.dtype

        if self.q_proj.down.weight.device != device or self.q_proj.down.weight.dtype != dtype:
            self.to(device=device, dtype=dtype)

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        N_kv = kv_input.shape[1]

        Q = self.q_proj(hidden_states).view(B, N, H, D).transpose(1, 2).contiguous()
        K = self.k_proj(kv_input).view(B, N_kv, self.num_kv_heads, D).transpose(1, 2).contiguous()
        V = self.v_proj(kv_input).view(B, N_kv, self.num_kv_heads, D).transpose(1, 2).contiguous()

        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        if self.num_kv_heads < self.num_heads:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        scale = 1.0 / math.sqrt(D)
        # Memory-efficient: use SDPA to avoid OOM on large seqs; fallback to explicit if needed

        try:
            _qs = Q.half() if Q.dtype == torch.float32 else Q

            _ks = K.half() if K.dtype == torch.float32 else K

            _vs = V.half() if V.dtype == torch.float32 else V

            _is_causal = (
                getattr(self, "is_decoder", False) and not is_cross and N > 1
                if "is_cross" in dir()
                else getattr(self, "is_decoder", False) and N > 1
            )

            _sdpa_out = torch.nn.functional.scaled_dot_product_attention(
                _qs,
                _ks,
                _vs,
                is_causal=_is_causal,
                dropout_p=self.dropout.p if self.training else 0.0,
            )

            _sdpa_out = (
                _sdpa_out.to(hidden_states.dtype)
                .transpose(1, 2)
                .contiguous()
                .view(B, N, self.all_head_size)
            )

            _sdpa_out = torch.nan_to_num(_sdpa_out, nan=0.0, posinf=1e4, neginf=-1e4)

            return (
                self.out_proj(_sdpa_out)
                if not getattr(self, "is_nested_encoder", False)
                else _sdpa_out
            ), None

        except Exception:
            pass

        scores = torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous()) * scale

        if self.is_decoder and not is_cross and N > 1:
            causal_mask = (
                torch.triu(
                    torch.full((N, N_kv), float("-inf"), device=device, dtype=torch.float32),
                    diagonal=1,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            scores = scores + causal_mask

        eff_mask = encoder_attention_mask if is_cross else attention_mask
        if eff_mask is not None:
            if eff_mask.dim() == 2:
                if eff_mask.dtype == torch.bool:
                    scores = scores.masked_fill(~eff_mask.unsqueeze(1).unsqueeze(2), -10000.0)
                elif torch.is_floating_point(eff_mask) and (eff_mask < 0).any():
                    scores = scores + eff_mask.unsqueeze(1).unsqueeze(2).float()
                else:
                    scores = scores + (eff_mask.unsqueeze(1).unsqueeze(2).float() - 1.0) * 10000.0
            elif eff_mask.dim() == 3:
                scores = scores + eff_mask.unsqueeze(1).float()
            elif eff_mask.dim() == 4:
                if eff_mask.dtype == torch.bool:
                    scores = scores.masked_fill(~eff_mask, -10000.0)
                else:
                    scores = scores + eff_mask.float()

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        attn_weights = F.softmax(scores, dim=-1).to(dtype)
        attn_weights = self.dropout(attn_weights)

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(-1)
            attn_weights = attn_weights * head_mask

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, N, H * D)
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_proj(context)

        return output, (attn_weights if output_attentions else None)

    def get_ortho_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.out_proj.up.weight.device)


@ATTENTION_REGISTRY.register("minimal_svd_inject")
@ATTENTION_REGISTRY.register("inject_minimal_svd")
@ATTENTION_REGISTRY.register("inject_minimal_aurora")
@ATTENTION_REGISTRY.register("inject_minimal_prism")
@ATTENTION_REGISTRY.register("inject_minimal_svd_v2")
def inject_minimal_svd(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    **kwargs: Any,
):
    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
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
    elif (
        hasattr(target, "bert")
        and hasattr(target.bert, "encoder")
        and hasattr(target.bert.encoder, "layer")
    ):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (torch.nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_minimal_svd(
                layer=lyr,
                model_config=cfg,
                comp_config=(
                    comp_config if "comp_config" in locals() or "comp_config" in kwargs else None
                ),
                layer_idx=idx,
                total_layers=tot,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    layer = target

    if model_config is None:
        model_config = type("Cfg", (), {"hidden_size": 64, "num_heads": 4, "rank": 32})()

    r"""Injection hook for MinimalSVDAttention into transformer layer."""
    energy = getattr(comp_config, "svd_energy", 0.95) if comp_config else 0.95
    dropout = getattr(comp_config, "dropout", 0.1) if comp_config else 0.1
    new_attn = MinimalSVDAttention(
        model_config=model_config,
        energy=energy,
        dropout=dropout,
    )
    src_attn = (
        old_attn
        if old_attn is not None
        else getattr(layer, "attention", getattr(layer, "self", None))
    )
    if src_attn is not None:
        try:
            new_attn.copy_weights(src_attn)
        except Exception as e:
            logger.error(f"copy_weights failed for minimal_svd: {e}")
            return None
        p = next(src_attn.parameters(), None)
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)

    if hasattr(layer, "attention"):
        if hasattr(layer.attention, "self"):
            layer.attention.self = new_attn
            new_attn.is_nested_encoder = True
        else:
            layer.attention = new_attn
            new_attn.is_nested_encoder = False
    elif hasattr(layer, "self"):
        layer.self = new_attn
        new_attn.is_nested_encoder = True
    else:
        new_attn.is_nested_encoder = False
    return new_attn


MinimalAuroraAttention = MinimalSVDAttention
MinimalPRISMAttention = MinimalSVDAttention
MinimalSVDAttentionV2 = MinimalSVDAttention
inject_minimal_aurora = inject_minimal_svd
inject_minimal_prism = inject_minimal_svd
inject_minimal_svd_v2 = inject_minimal_svd
