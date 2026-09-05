"""
ASVD (Activation-Aware Singular Value Decomposition) Attention.
Paper: https://arxiv.org/abs/2307.07382

Hardened for production use across diverse transformer architectures:
- DistilBERT (Flat MHA)
- RoBERTa / BERT (Nested Encoder Self-Attention)
- GPT-2 (Causal Decoder with Conv1D)
- Qwen / LLaMA (GQA with Rotary Position Embeddings)
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.adapters import UniversalAdapter
from framework.core.attention_mask import apply_attention_mask
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
from framework.core.registry import ATTENTION_REGISTRY

logger = logging.getLogger(__name__)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding (RoPE) to query and key tensors."""
    if cos.dim() == 2 and position_ids is not None:
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    elif cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(1)
        sin = sin.unsqueeze(0).unsqueeze(1)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    while cos.dim() < q.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    if cos.shape[-2] != q.shape[-2]:
        seq_len = q.shape[-2]
        cos = cos[..., :seq_len, :]
        sin = sin[..., :seq_len, :]

    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


class ASVDLinear(nn.Module):
    r"""Low-rank factorized linear layer with explicit singular value scaling and trainable LoRA adapters:
    
    $$y = ((x V^\top) \odot \Sigma) U^\top + (x A^\top) B^\top + b$$
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        lora_rank: int = 0,
        freeze_base: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_rank = lora_rank
        self.freeze_base = freeze_base

        factory_kwargs = {"dtype": dtype, "device": device}
        self.U = nn.Parameter(torch.empty(out_features, rank, **factory_kwargs))
        self.S = nn.Parameter(torch.empty(rank, **factory_kwargs))
        self.V = nn.Parameter(torch.empty(rank, in_features, **factory_kwargs))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        # Trainable LoRA adapter parameters
        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features, **factory_kwargs))
            self.lora_B = nn.Parameter(torch.empty(out_features, lora_rank, **factory_kwargs))
            self.lora_scaling = 1.0
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)
            self.lora_scaling = 1.0

        self.reset_parameters()
        if self.freeze_base:
            self.freeze_base_weights()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.U.dtype in (torch.float16, torch.bfloat16):
                u_fp32 = torch.empty_like(self.U, dtype=torch.float32, device="cpu")
                nn.init.orthogonal_(u_fp32)
                self.U.copy_(u_fp32.to(dtype=self.U.dtype, device=self.U.device))
            else:
                nn.init.orthogonal_(self.U)

            nn.init.ones_(self.S)

            if self.V.dtype in (torch.float16, torch.bfloat16):
                v_fp32 = torch.empty_like(self.V, dtype=torch.float32, device="cpu")
                nn.init.orthogonal_(v_fp32)
                self.V.copy_(v_fp32.to(dtype=self.V.dtype, device=self.V.device))
            else:
                nn.init.orthogonal_(self.V)

            if self.bias is not None:
                nn.init.zeros_(self.bias)

            if getattr(self, "lora_A", None) is not None and self.lora_A is not None:
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                self.lora_A.requires_grad = True
            if getattr(self, "lora_B", None) is not None and self.lora_B is not None:
                nn.init.zeros_(self.lora_B)
                self.lora_B.requires_grad = True

    def freeze_base_weights(self) -> None:
        """Freeze U, S, V and bias parameters so only LoRA adapters adapt during recovery fine-tuning."""
        self.U.requires_grad = False
        self.S.requires_grad = False
        self.V.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if getattr(self, "lora_A", None) is not None and self.lora_A is not None:
            self.lora_A.requires_grad = True
        if getattr(self, "lora_B", None) is not None and self.lora_B is not None:
            self.lora_B.requires_grad = True

    def unfreeze_base_weights(self) -> None:
        """Unfreeze base weights for standard end-to-end training."""
        self.U.requires_grad = True
        self.S.requires_grad = True
        self.V.requires_grad = True
        if self.bias is not None:
            self.bias.requires_grad = True

    def set_svd_factors(
        self,
        U: torch.Tensor,
        V: torch.Tensor,
        S: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        with torch.no_grad():
            r = min(U.size(1), V.size(0))
            if r != self.rank:
                device, dtype = self.U.device, self.U.dtype
                req_grad = self.U.requires_grad
                self.rank = r
                self.U = nn.Parameter(torch.zeros(self.out_features, r, device=device, dtype=dtype), requires_grad=req_grad)
                self.S = nn.Parameter(torch.ones(r, device=device, dtype=dtype), requires_grad=req_grad)
                self.V = nn.Parameter(torch.zeros(r, self.in_features, device=device, dtype=dtype), requires_grad=req_grad)

            self.U.data.zero_()
            self.V.data.zero_()
            self.S.data.zero_()

            out_copy = min(self.out_features, U.size(0))
            in_copy = min(self.in_features, V.size(1))

            self.U.data[:out_copy, :r].copy_(U[:out_copy, :r].to(device=self.U.device, dtype=self.U.dtype))
            self.V.data[:r, :in_copy].copy_(V[:r, :in_copy].to(device=self.V.device, dtype=self.V.dtype))
            if S is not None:
                self.S.data[:r].copy_(S[:r].to(device=self.S.device, dtype=self.S.dtype))
            else:
                self.S.data[:r].fill_(1.0)

            if bias is not None and self.bias is not None:
                min_b = min(self.out_features, bias.numel())
                self.bias.data[:min_b].copy_(bias[:min_b].to(device=self.bias.device, dtype=self.bias.dtype))
            elif self.bias is not None:
                self.bias.data.zero_()

    def set_identity(self) -> None:
        """Initializes factorized linear transformation to act as an identity pass-through."""
        with torch.no_grad():
            self.U.data.zero_()
            self.V.data.zero_()
            self.S.data.fill_(1.0)
            r = min(self.rank, self.in_features, self.out_features)
            self.U.data[:r, :r].fill_diagonal_(1.0)
            self.V.data[:r, :r].fill_diagonal_(1.0)
            if self.bias is not None:
                self.bias.data.zero_()
            self.freeze_base_weights()

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix for inspection & compatibility."""
        W = torch.matmul(self.U * self.S.unsqueeze(0), self.V)
        if getattr(self, "lora_A", None) is not None and self.lora_A is not None:
            W = W + torch.matmul(self.lora_B, self.lora_A) * self.lora_scaling
        return W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (*, in_features)
        # 1. x @ V.T -> (*, rank)
        Vx = F.linear(x, self.V)
        # 2. scale by singular values S
        SVx = Vx * self.S
        # 3. SVx @ U.T -> (*, out_features)
        out = F.linear(SVx, self.U)
        if self.bias is not None:
            out = out + self.bias

        # 4. Residual LoRA adapter path: (x @ lora_A.T) @ lora_B.T
        if getattr(self, "lora_A", None) is not None and self.lora_A is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
            out = out + lora_out * self.lora_scaling

        return out


@ATTENTION_REGISTRY.register("asvd")
@ATTENTION_REGISTRY.register("asvd_attention")
class ASVDAttention(nn.Module):
    r"""Hardened ASVD Attention with Factored Low-Rank Matrix Decomposition, GQA, RoPE, and LoRA.

    Computational Complexity:
        - Factorized Projections: $\mathcal{O}(B \cdot N \cdot r \cdot (d + d_{\text{all}}))$
        - Attention Scoring: $\mathcal{O}(B \cdot H \cdot N^2 \cdot d_k)$

    Tensor Dimensions:
        - hidden_states: $[B, N, D]$ where $B$ is batch size, $N$ is sequence length, and $D$ is hidden dimension.
        - Output: Tuple of $([B, N, D], \text{None})$.
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = None,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size: int = config.hidden_size
        self.num_heads: int = config.num_attention_heads
        self.num_kv_heads: int = getattr(config, "num_key_value_heads", self.num_heads)
        self.head_dim: int = getattr(config, "head_dim", None) or (self.hidden_size // self.num_heads)
        self.all_head_size: int = self.num_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim

        # Rank configuration
        self.rank: int = getattr(
            config,
            "asvd_rank",
            rank if rank is not None else getattr(config, "rank", 32),
        )
        self.lora_rank: int = getattr(config, "lora_rank", kwargs.get("lora_rank", 0))
        self.freeze_base: bool = getattr(config, "freeze_base", kwargs.get("freeze_base", self.lora_rank > 0))
        self.energy_threshold: float = getattr(config, "asvd_energy_threshold", kwargs.get("energy_threshold", 0.95))
        self.error_threshold: Optional[float] = getattr(config, "asvd_error_threshold", kwargs.get("error_threshold", 0.20))
        self.dynamic_rank: bool = getattr(config, "asvd_dynamic_rank", kwargs.get("dynamic_rank", True))

        # Flags & metadata
        self.supports_gqa: bool = True
        self.supports_rope: bool = True
        self.is_nested_encoder: bool = False
        self.rotary_emb: Optional[Any] = None

        # Decoder detection
        model_type = getattr(config, "model_type", "").lower()
        is_dec = getattr(config, "is_decoder", False)
        if model_type in ["gpt2", "gpt", "qwen", "qwen2", "qwen2.5", "qwen3.5", "llama", "mistral"]:
            is_dec = True
        elif hasattr(config, "architectures") and any(
            "GPT" in str(a) or "CausalLM" in str(a) for a in (getattr(config, "architectures", []) or [])
        ):
            is_dec = True
        self.is_decoder: bool = bool(is_dec)

        # Projections
        self.query = ASVDLinear(self.hidden_size, self.all_head_size, self.rank, bias=True, lora_rank=self.lora_rank, freeze_base=self.freeze_base)
        self.key = ASVDLinear(self.hidden_size, self.kv_head_size, self.rank, bias=True, lora_rank=self.lora_rank, freeze_base=self.freeze_base)
        self.value = ASVDLinear(self.hidden_size, self.kv_head_size, self.rank, bias=True, lora_rank=self.lora_rank, freeze_base=self.freeze_base)
        self.out_proj = ASVDLinear(self.all_head_size, self.hidden_size, self.rank, bias=True, lora_rank=self.lora_rank, freeze_base=self.freeze_base)

        dropout_p = getattr(
            config,
            "dropout",
            getattr(
                config,
                "hidden_dropout_prob",
                getattr(config, "attention_probs_dropout_prob", 0.1),
            ),
        )
        self.dropout = nn.Dropout(dropout_p)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    # --------------------------------------------------------------------------
    # Compatibility properties for parameter inspection & framework integration
    # --------------------------------------------------------------------------
    @property
    def q_proj(self) -> ASVDLinear:
        return self.query

    @property
    def k_proj(self) -> ASVDLinear:
        return self.key

    @property
    def v_proj(self) -> ASVDLinear:
        return self.value

    @property
    def o_proj(self) -> ASVDLinear:
        return self.out_proj

    def _decompose_and_set(
        self,
        target_linear: ASVDLinear,
        src_weight: torch.Tensor,
        src_bias: Optional[torch.Tensor],
        proj_name: str = "",
    ) -> float:
        """Decomposes src_weight into low-rank SVD factors and verifies relative Frobenius error."""
        with torch.no_grad():
            W = src_weight.detach().float()
            # Ensure correct orientation: [out_features, in_features]
            if W.ndim != 2:
                raise ValueError(f"Expected 2D weight matrix for projection {proj_name}, got shape {W.shape}")
            if W.shape[0] != target_linear.out_features and W.shape[1] == target_linear.out_features:
                W = W.t()

            # Perform SVD strictly in FP32
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)

            # Energy-based dynamic rank selection
            total_energy = torch.sum(S ** 2)
            cum_energy = torch.cumsum(S ** 2, dim=0) / (total_energy + 1e-12)

            target_energy = max(self.energy_threshold, 1.0 - (self.error_threshold ** 2) if self.error_threshold is not None else 0.95)
            indices = (cum_energy >= target_energy).nonzero()
            if len(indices) > 0:
                r_energy = indices[0].item() + 1
            else:
                r_energy = min(U.size(1), Vt.size(0))

            if self.dynamic_rank:
                r = max(16, min(r_energy, U.size(1), Vt.size(0)))
            else:
                r = min(self.rank, U.size(1), Vt.size(0))

            w_rec = (U[:, :r] * S[:r].unsqueeze(0)) @ Vt[:r, :]
            diff = torch.norm(W - w_rec, p="fro")
            denom = torch.norm(W, p="fro") + 1e-9
            rel_err = float((diff / denom).item())

            # If error exceeds threshold under dynamic rank, expand rank to meet error constraint
            if self.error_threshold is not None and rel_err > self.error_threshold:
                if self.dynamic_rank and r < min(U.size(1), Vt.size(0)):
                    indices_strict = (cum_energy >= (1.0 - (self.error_threshold ** 2) + 1e-5)).nonzero()
                    if len(indices_strict) > 0:
                        r = indices_strict[0].item() + 1
                        w_rec = (U[:, :r] * S[:r].unsqueeze(0)) @ Vt[:r, :]
                        rel_err = float((torch.norm(W - w_rec) / denom).item())

                if rel_err > self.error_threshold:
                    raise RuntimeError(
                        f"Relative Frobenius reconstruction error ({rel_err:.2%}) for projection '{proj_name}' "
                        f"exceeds {self.error_threshold:.0%} threshold. Aborting injection."
                    )

            target_linear.set_svd_factors(U[:, :r], Vt[:r, :], S[:r], bias=src_bias)
            return rel_err

    def copy_weights(self, src: Any) -> None:
        """Extracts and factorizes pretrained weights across DistilBERT, RoBERTa, GPT-2, Qwen/LLaMA."""
        extracted_weights: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. GPT-2: Fused Conv1D c_attn + Conv1D c_proj
        if hasattr(src, "c_attn") and src.c_attn is not None and hasattr(src.c_attn, "weight"):
            c_w = src.c_attn.weight.data
            c_b = getattr(src.c_attn, "bias", None)
            c_b = c_b.data if c_b is not None else None
            # GPT-2 Conv1D orientation: [in_features, 3 * out_features]
            if src.c_attn.__class__.__name__ == "Conv1D" or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1]):
                c_w = c_w.t()
            split_size = c_w.size(0) // 3
            extracted_weights["q"] = (c_w[:split_size, :], c_b[:split_size] if c_b is not None else None)
            extracted_weights["k"] = (c_w[split_size : 2 * split_size, :], c_b[split_size : 2 * split_size] if c_b is not None else None)
            extracted_weights["v"] = (c_w[2 * split_size :, :], c_b[2 * split_size :] if c_b is not None else None)

            if hasattr(src, "c_proj") and src.c_proj is not None and hasattr(src.c_proj, "weight"):
                o_w = src.c_proj.weight.data
                o_b = getattr(src.c_proj, "bias", None)
                o_b = o_b.data if o_b is not None else None
                if src.c_proj.__class__.__name__ == "Conv1D" or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1]):
                    o_w = o_w.t()
                extracted_weights["o"] = (o_w, o_b)

        # 2. RoBERTa / BERT Nested Encoder: self.query, self.key, self.value
        s = getattr(src, "self", src)
        name_map = {"q": "query", "k": "key", "v": "value"}
        for proj, attr in name_map.items():
            if extracted_weights[proj][0] is None and hasattr(s, attr) and hasattr(getattr(s, attr), "weight"):
                mod = getattr(s, attr)
                b = getattr(mod, "bias", None)
                extracted_weights[proj] = (mod.weight.data, b.data if b is not None else None)

        if extracted_weights["o"][0] is None and hasattr(src, "output") and hasattr(src.output, "dense"):
            mod = src.output.dense
            b = getattr(mod, "bias", None)
            extracted_weights["o"] = (mod.weight.data, b.data if b is not None else None)

        # 3. Qwen / LLaMA: q_proj, k_proj, v_proj, o_proj
        for proj in ["q", "k", "v", "o"]:
            if extracted_weights[proj][0] is None:
                for candidate in [f"{proj}_proj", f"{proj}_lin", proj]:
                    if hasattr(src, candidate) and hasattr(getattr(src, candidate), "weight"):
                        mod = getattr(src, candidate)
                        b = getattr(mod, "bias", None)
                        extracted_weights[proj] = (mod.weight.data, b.data if b is not None else None)
                        break

        # 4. Fallback to UniversalAdapter
        for proj in ["q", "k", "v", "o"]:
            if extracted_weights[proj][0] is None:
                try:
                    w, b = UniversalAdapter.get_weight_bias(src, proj)
                    if w is not None:
                        extracted_weights[proj] = (w, b)
                except Exception:
                    pass

        # 5. Nested Encoder Detection
        # Strict nested encoder detection: RoBERTa and BERT use two cascaded modules:
        # (RobertaAttention -> self: RobertaSelfAttention, output: RobertaSelfOutput).
        # In contrast, DistilBERT is flat and owns 'out_lin'.
        cls_name = src.__class__.__name__
        has_out_lin = hasattr(src, "out_lin")
        is_known_nested = (
            cls_name in ["BertSelfAttention", "RobertaSelfAttention", "XLMRobertaSelfAttention", "ElectraSelfAttention", "DebertaSelfAttention"]
            or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())
        ) and not has_out_lin

        self.is_nested_encoder = bool(is_known_nested)

        # 6. Check for missing essential projections
        missing = [p for p in ["q", "k", "v"] if extracted_weights[p][0] is None]
        if not self.is_nested_encoder and extracted_weights["o"][0] is None:
            missing.append("o")
        if missing:
            raise RuntimeError(f"Missing essential pretrained weights for ASVDAttention: {missing}. Aborting injection.")

        # 7. SVD Factorization & Reconstruction Verification
        for proj, target in [("q", self.query), ("k", self.key), ("v", self.value), ("o", self.out_proj)]:
            if proj == "o" and self.is_nested_encoder:
                self.out_proj.set_identity()
                continue
            w, b = extracted_weights[proj]
            if w is not None:
                self._decompose_and_set(target, w, b, proj_name=proj)

        # 8. Inherit Decoder & Rotary Metadata
        if hasattr(src, "rotary_emb"):
            self.rotary_emb = src.rotary_emb
        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)
        elif hasattr(src, "c_attn") or cls_name in ("GPT2Attention", "Qwen2Attention", "Qwen2_5_VLAttention", "QwenAttention"):
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
        position_ids: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for factorized ASVD attention."""
        B, N, D = hidden_states.shape
        H = self.num_heads
        d = self.head_dim
        kv_H = self.num_kv_heads

        # Extract rotary kwargs if supplied
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings", None)
        if position_ids is None:
            position_ids = kwargs.get("position_ids", None)

        # 1. Project Q, K, V using low-rank factorized linear layers
        q: torch.Tensor = self.query(hidden_states).view(B, N, H, d).transpose(1, 2)
        k: torch.Tensor = self.key(hidden_states).view(B, N, kv_H, d).transpose(1, 2)
        v: torch.Tensor = self.value(hidden_states).view(B, N, kv_H, d).transpose(1, 2)

        # 2. Apply Rotary Position Embedding (RoPE) before GQA expansion and masking
        if position_embeddings is not None and isinstance(position_embeddings, (tuple, list)) and len(position_embeddings) >= 2:
            cos, sin = position_embeddings[0], position_embeddings[1]
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        elif self.rotary_emb is not None:
            try:
                rotary_out = self.rotary_emb(v, position_ids) if position_ids is not None else self.rotary_emb(v)
                if isinstance(rotary_out, (tuple, list)) and len(rotary_out) >= 2:
                    cos, sin = rotary_out[0], rotary_out[1]
                    q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
            except Exception:
                pass

        # 3. GQA Expansion: Expand key/value heads to match query heads if kv_H < H
        if kv_H < H:
            k = repeat_kv(k, H // kv_H)
            v = repeat_kv(v, H // kv_H)

        # 4. Attention Scoring strictly in FP32 to prevent FP16 overflow
        scale: float = 1.0 / math.sqrt(d)
        scores: torch.Tensor = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale

        # 5. Causal Masking for Decoder Models
        if getattr(self, "is_decoder", False):
            causal_mask = torch.triu(
                torch.full((N, N), -10000.0, device=scores.device, dtype=scores.dtype),
                diagonal=1,
            )
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        # 6. Padding Attention Mask
        if attention_mask is not None:
            scores = apply_attention_mask(scores, attention_mask)

        # 7. Softmax & NaN Protection
        attn: torch.Tensor = F.softmax(scores, dim=-1).to(hidden_states.dtype)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        # 8. Safe Head Mask Handling
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask_tensor = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask_tensor = head_mask.view(head_mask.size(0), -1, 1, 1)
            elif head_mask.dim() == 3:
                head_mask_tensor = head_mask.unsqueeze(1)
            else:
                head_mask_tensor = head_mask
            attn = attn * head_mask_tensor

        # 9. Context Aggregation
        context: torch.Tensor = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)

        # 10. Output Projection (Bypassed for nested encoders to prevent double projection)
        if getattr(self, "is_nested_encoder", False):
            out = context
        else:
            out = self.out_proj(context)

        return out, attn if output_attentions else None


@ATTENTION_REGISTRY.register("asvd_fn")
@ATTENTION_REGISTRY.register("asvd_inject")
def inject_asvd(
    layer: Any = None,
    config: Any = None,
    rank: Optional[int] = None,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> Optional[nn.Module]:
    """Universal injection factory for ASVDAttention with pre-flight extraction validation."""
    if old_attn is None and layer is not None:
        old_attn = layer

    target_config = config
    if target_config is None and old_attn is not None:
        target_config = getattr(old_attn, "config", None)

    if target_config is None:
        logger.warning("inject_asvd: Neither config nor old_attn.config provided. Aborting injection.")
        return None

    # Pre-flight extraction test: ensure source layer has extractable projections
    if old_attn is not None:
        try:
            # Instantiate replacement and copy pretrained weights
            new_attn = ASVDAttention(target_config, rank=rank, pretrained_attn=old_attn, **kwargs)
            return new_attn
        except RuntimeError as err:
            logger.warning("inject_asvd: Pre-flight weight extraction failed on %s: %s. Aborting replacement.", type(old_attn).__name__, err)
            return None
        except Exception as err:
            logger.warning("inject_asvd: Unexpected error initializing ASVDAttention: %s. Aborting replacement.", err)
            return None

    # Fallback to random initialization with a warning
    logger.warning("inject_asvd: old_attn is None. Initializing ASVDAttention with random weights.")
    return ASVDAttention(target_config, rank=rank, **kwargs)
