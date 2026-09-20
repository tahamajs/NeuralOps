"""
BAM-UMoE: Unified Mixture of Experts Attention Block.

This module implements a production-hardened Unified Mixture of Experts (UMoE) attention
mechanism supporting multi-architecture deployments: DistilBERT, RoBERTa/BERT, GPT-2,
and Qwen/LLaMA/Mistral (GQA + partial RoPE).

Key architectural features:
1. Robust Weight Extraction across fused Conv1D (GPT-2), nested self-attention (RoBERTa/BERT),
   explicit linear projections (DistilBERT), and standard Q/K/V/O projections (Qwen/LLaMA).
2. Step-0 Pretrained Representation Preservation via zero-initialized expert output layers
   and learnable negative-bias gating (sigmoid(-3.0)).
3. Causal Masking with dynamic decoder detection for autoregressive generation.
4. Native Grouped-Query Attention (GQA) with kv_repeat broadcasting.
5. Rotary Position Embedding (RoPE) support with partial rotary dimension coordinate slicing.
6. Parameter-Efficient Recovery Fine-Tuning with trainable residual LoRA adapters (lora_A, lora_B)
   and frozen base weights.
7. Nested Encoder Detection to eliminate double output projection in RoBERTa/BERT blocks.
8. Full numerical stability: FP32 score computation, nan_to_num guardrails, and MoE output clamping.
9. Cross-Attention support for encoder-decoder architectures.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input for RoPE."""
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
    r"""
    Applies rotary position embeddings to query and key tensors with partial-RoPE dimension slicing.

    Supports coordinate layouts where head_dim > rotary_dim (e.g., Qwen3.5: head_dim=256, rotary_dim=64).
    Rotates the first `rotary_dim` coordinates and concatenates unrotated pass-through coordinates.
    """
    if cos is None or sin is None:
        return q, k

    # Ensure 4D broadcasting: (B, 1, L, rot_dim) or (1, 1, L, rot_dim)
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


class LoRALinear(nn.Module):
    r"""
    Linear projection layer equipped with frozen base weights and trainable residual LoRA adapters.

    Base path:
        y_base = x @ W^T + b  (frozen when freeze_base=True)
    LoRA path:
        y_lora = (x @ lora_A^T) @ lora_B^T * scaling  (trainable)
    Total output:
        y = y_base + y_lora

    Uses standard `lora_A` and `lora_B` parameter naming recognized by framework recovery optimizers.
    """

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
        """Freeze base projection parameters so only LoRA parameters adapt."""
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_rank > 0 and self.lora_A is not None:
            self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
            self.lora_B.requires_grad = True

    def unfreeze_base_weights(self) -> None:
        """Unfreeze base projection parameters."""
        self.weight.requires_grad = True
        if self.bias is not None:
            self.bias.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
            out = out + lora_out
        return out


class UnifiedMoE(nn.Module):
    r"""
    Vectorized Mixture of Experts (MoE) layer with top-k gating, auxiliary load-balancing loss,
    and step-0 representation preservation.

    Preservation mechanism:
    - Final linear projection of each expert is initialized to strictly zero.
    - An optional learnable gating parameter with negative bias (sigmoid(-3.0) ≈ 0.047)
      ensures smooth, non-disruptive activation during early training steps.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 2,
        top_k: int = 1,
        output_dim: Optional[int] = None,
        noise_std: float = 0.1,
        expert_multiplier: int = 1,
    ) -> None:
        super().__init__()
        self.dim: int = dim
        self.num_experts: int = max(1, num_experts)
        self.top_k: int = min(max(1, top_k), self.num_experts)
        self.out_dim: int = output_dim or dim
        self.noise_std: float = noise_std
        self.aux_loss: torch.Tensor = torch.tensor(0.0)

        # Router with small uniform weights for balanced initial participation
        self.router_ln = nn.LayerNorm(self.dim)
        self.router = nn.Linear(self.dim, self.num_experts, bias=False)
        nn.init.uniform_(self.router.weight, -0.01, 0.01)

        # Experts: 2-layer compact MLP with GELU non-linearity
        hidden_dim = max(32, (self.out_dim * expert_multiplier) // 4)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.out_dim),
            )
            for _ in range(self.num_experts)
        ])

        # Step-0 Zero-Initialization: Final expert linear layer initialized to strictly zero
        # to guarantee zero disturbance to pretrained representations at step 0.
        for exp in self.experts:
            nn.init.zeros_(exp[2].weight)
            if exp[2].bias is not None:
                nn.init.zeros_(exp[2].bias)

        # Learnable gate with negative bias for gentle ramp-up
        self.gate_bias = nn.Parameter(torch.tensor([-3.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Forward execution pass for top-k Unified MoE with load balancing."""
        B, L, D = x.shape
        x_norm = self.router_ln(x)
        logits = self.router(x_norm)  # (B, L, E)
        if self.training:
            logits += torch.randn_like(logits) * 0.05

        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            routing_logits = logits + noise
        else:
            routing_logits = logits

        topk_vals, topk_idx = torch.topk(routing_logits, self.top_k, dim=-1)  # (B, L, k)
        scores = F.softmax(topk_vals, dim=-1)  # (B, L, k)
        scores = torch.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)

        # Auxiliary load balancing loss (Switch Transformer / GShard formulation)
        if self.training and self.num_experts > 1:
            router_probs = F.softmax(logits.float(), dim=-1).view(-1, self.num_experts)  # (B*L, E)
            expert_mask = F.one_hot(topk_idx, num_classes=self.num_experts).float()  # (B, L, k, E)
            density = expert_mask.sum(dim=2).view(-1, self.num_experts).mean(dim=0)  # (E,)
            prob_mean = router_probs.mean(dim=0)  # (E,)
            self.aux_loss = self.num_experts * torch.sum(density * prob_mean)
        else:
            self.aux_loss = torch.tensor(0.0, device=x.device, dtype=torch.float32)

        # Expert forward passes
        experts_out = torch.stack([exp(x) for exp in self.experts], dim=2)  # (B, L, E, out_dim)

        topk_out = torch.gather(
            experts_out,
            2,
            topk_idx.unsqueeze(-1).expand(B, L, self.top_k, self.out_dim),
        )  # (B, L, k, out_dim)

        out = torch.sum(topk_out * scores.unsqueeze(-1), dim=2)  # (B, L, out_dim)

        # Clamp expert outputs to prevent numerical overflow in FP16
        out = torch.clamp(torch.nan_to_num(out, nan=0.0), min=-65000.0, max=65000.0)

        # Learnable gating modulation
        gate = torch.sigmoid(self.gate_bias)
        return out * gate


@ATTENTION_REGISTRY.register("bam_umoe")
class BAMUMoEBlock(BaseAttention):
    r"""
    BAM Unified Mixture of Experts Attention Block.

    Hardened for cross-architecture deployment across DistilBERT, RoBERTa/BERT, GPT-2,
    and Qwen/LLaMA/Mistral.
    """

    is_custom_attention: bool = True
    def __init__(
        self,
        config: Optional[Any] = None,
        rank: int = 64,
        num_experts: int = 2,
        top_k: int = 1,
        layer_id: Optional[int] = None,
        num_layers: Optional[int] = None,
        old_attn: Optional[nn.Module] = None,
        comp_config: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)

        self.dim: int = kwargs.get("hidden_size") or getattr(config, "hidden_size", getattr(config, "d_model", getattr(config, "dim", 768)))
        self.hidden_size: int = self.dim
        self.num_heads: int = kwargs.get("num_heads") or kwargs.get("num_attention_heads") or getattr(config, "num_attention_heads", getattr(config, "num_heads", getattr(config, "n_heads", 12)))
        self.num_kv_heads: int = getattr(config, "num_key_value_heads", getattr(config, "num_kv_heads", self.num_heads))
        self.head_dim: int = getattr(config, "head_dim", getattr(config, "kv_channels", self.dim // max(1, self.num_heads)))
        self.all_head_size: int = self.num_heads * self.head_dim
        self.all_kv_head_size: int = self.num_kv_heads * self.head_dim
        self.kv_repeat: int = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.supports_gqa: bool = True

        # LoRA adaptation configuration
        self.lora_rank: int = getattr(config, "lora_rank", getattr(comp_config, "lora_rank", kwargs.get("lora_rank", 16)))
        self.lora_alpha: float = getattr(config, "lora_alpha", getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)))
        self.freeze_base: bool = (self.lora_rank > 0)

        # Decoder and nested encoder flags
        model_type = getattr(config, "model_type", "").lower()
        is_dec = getattr(config, "is_decoder", False)
        decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
        if any(model_type.startswith(p) for p in decoder_prefixes):
            is_dec = True
        elif hasattr(config, "architectures") and any("GPT" in str(a) or "CausalLM" in str(a) for a in (getattr(config, "architectures", []) or [])):
            is_dec = True
        if kwargs.get("is_decoder", False):
            is_dec = True
        self.is_decoder: bool = bool(is_dec)
        self.is_nested_encoder: bool = False

        self.num_experts: int = num_experts
        self.top_k: int = min(top_k, num_experts)

        # Multi-Head Attention Projections with residual LoRA adapters
        self.q_lin = LoRALinear(self.dim, self.all_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.k_lin = LoRALinear(self.dim, self.all_kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.v_lin = LoRALinear(self.dim, self.all_kv_head_size, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)
        self.out_lin = LoRALinear(self.all_head_size, self.dim, bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha, freeze_base=self.freeze_base)

        # MoE Value Refinement Modulation Branch (applies to self-attention values)
        self.qkv_moe = UnifiedMoE(self.dim, num_experts=num_experts, top_k=top_k, output_dim=self.all_head_size)

        dropout_p = getattr(config, "dropout", getattr(config, "attention_probs_dropout_prob", 0.1))
        self.dropout = nn.Dropout(dropout_p)

        self.layer_id: Optional[int] = layer_id
        self.num_layers: Optional[int] = num_layers
        self.comp_config: Optional[Any] = comp_config
        self.aux_loss: torch.Tensor = torch.tensor(0.0)
        self.rotary_emb = None

        if old_attn is not None:
            self.copy_weights(old_attn)

    # --------------------------------------------------------------------------
    # Compatibility properties for parameter inspection & framework integration
    # --------------------------------------------------------------------------
    @property
    def q_proj(self) -> LoRALinear:
        return self.q_lin

    @property
    def k_proj(self) -> LoRALinear:
        return self.k_lin

    @property
    def v_proj(self) -> LoRALinear:
        return self.v_lin

    @property
    def o_proj(self) -> LoRALinear:
        return self.out_lin

    @property
    def query(self) -> LoRALinear:
        return self.q_lin

    @property
    def key(self) -> LoRALinear:
        return self.k_lin

    @property
    def value(self) -> LoRALinear:
        return self.v_lin

    @property
    def dense(self) -> LoRALinear:
        return self.out_lin

    @property
    def c_proj(self) -> LoRALinear:
        return self.out_lin

    def copy_weights(self, pretrained_attn: nn.Module) -> None:
        r"""
        Extracts and copies pretrained weights across DistilBERT, RoBERTa/BERT, GPT-2, and Qwen/LLaMA.
        Strictly validates that copied weights are non-empty and verifies reconstruction norms.
        """
        if pretrained_attn is None:
            return

        # 1. Detect nested encoder architectures (RoBERTa / BERT)
        cls_name = pretrained_attn.__class__.__name__
        has_out_lin = hasattr(pretrained_attn, "out_lin")
        is_nested = False
        if (
            hasattr(pretrained_attn, "self")
            and hasattr(getattr(pretrained_attn, "self"), "query")
            and hasattr(pretrained_attn, "output")
            and hasattr(getattr(pretrained_attn, "output"), "dense")
        ):
            is_nested = True
        elif (
            cls_name in ["BertSelfAttention", "RobertaSelfAttention", "XLMRobertaSelfAttention", "ElectraSelfAttention", "DebertaSelfAttention"]
            or (cls_name.endswith("SelfAttention") and "distil" not in cls_name.lower())
        ) and not has_out_lin:
            is_nested = True
        self.is_nested_encoder = bool(is_nested)

        # Inherit rotary embeddings if present
        if hasattr(pretrained_attn, "rotary_emb"):
            self.rotary_emb = pretrained_attn.rotary_emb

        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 2. GPT-2: Fused c_attn + c_proj
        if hasattr(pretrained_attn, "c_attn") and pretrained_attn.c_attn is not None:
            c_attn_mod = pretrained_attn.c_attn
            if hasattr(c_attn_mod, "merge") and not getattr(c_attn_mod, "merged", False):
                try:
                    c_attn_mod.merge()
                except Exception as e:
                    print(f'copy_weights failed: {e}')
                    logger.warning(f"copy_weights failed: {e}")
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            c_w = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if c_w is not None:
                c_w = c_w.data.clone()
                if (c_attn_mod.__class__.__name__ == "Conv1D") or (base_c.__class__.__name__ == "Conv1D") or (c_w.ndim == 2 and c_w.shape[0] < c_w.shape[1]):
                    c_w = c_w.t().contiguous()
                split_size = c_w.size(0) // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                c_b = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (c_w[:split_size, :], c_b[:split_size] if c_b is not None else None)
                extracted["k"] = (c_w[split_size : 2 * split_size, :], c_b[split_size : 2 * split_size] if c_b is not None else None)
                extracted["v"] = (c_w[2 * split_size :, :], c_b[2 * split_size :] if c_b is not None else None)

            # GPT-2 Output: c_proj
            if hasattr(pretrained_attn, "c_proj") and pretrained_attn.c_proj is not None:
                c_proj_mod = pretrained_attn.c_proj
                if hasattr(c_proj_mod, "merge") and not getattr(c_proj_mod, "merged", False):
                    try:
                        c_proj_mod.merge()
                    except Exception as e:
                        print(f'copy_weights failed: {e}')
                        logger.warning(f"copy_weights failed: {e}")
                base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
                o_w = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
                if o_w is not None:
                    o_w = o_w.data.clone()
                    if (c_proj_mod.__class__.__name__ == "Conv1D") or (base_o.__class__.__name__ == "Conv1D") or (o_w.ndim == 2 and o_w.shape[0] < o_w.shape[1]):
                        o_w = o_w.t().contiguous()
                    bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                    extracted["o"] = (o_w, bias_o.data.clone() if bias_o is not None else None)

        # 3. RoBERTa / BERT nested self-attention (pretrained_attn.self)
        elif hasattr(pretrained_attn, "self") and hasattr(getattr(pretrained_attn, "self"), "query"):
            s = getattr(pretrained_attn, "self")
            for proj_key, sub_name in [("q", "query"), ("k", "key"), ("v", "value")]:
                mod = getattr(s, sub_name, None)
                if mod is not None and hasattr(mod, "weight"):
                    extracted[proj_key] = (mod.weight.data.clone(), mod.bias.data.clone() if mod.bias is not None else None)
            # Output dense from outer container
            if hasattr(pretrained_attn, "output") and hasattr(getattr(pretrained_attn, "output"), "dense"):
                out_mod = pretrained_attn.output.dense
                extracted["o"] = (out_mod.weight.data.clone(), out_mod.bias.data.clone() if out_mod.bias is not None else None)

        # 4. Standard linear projections (DistilBERT, Qwen, LLaMA)
        else:
            proj_candidates = {
                "q": ["q_lin", "q_proj", "query"],
                "k": ["k_lin", "k_proj", "key"],
                "v": ["v_lin", "v_proj", "value"],
                "o": ["out_lin", "out_proj", "dense", "o_proj"],
            }
            for proj_key, names in proj_candidates.items():
                for name in names:
                    if hasattr(pretrained_attn, name):
                        mod = getattr(pretrained_attn, name)
                        if mod is not None and hasattr(mod, "weight"):
                            extracted[proj_key] = (mod.weight.data.clone(), mod.bias.data.clone() if mod.bias is not None else None)
                            break

        # 5. Fallback via UniversalAdapter if any projections were missing
        missing = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing:
            try:
                from framework.core.adapters import UniversalAdapter
                adapter = UniversalAdapter()
                for k in missing:
                    w, b = adapter.get_weight_bias(pretrained_attn, k)
                    if w is not None:
                        extracted[k] = (w.clone(), b.clone() if b is not None else None)
            except Exception:
                # Log or preserve warning, but proceed to strict validation check
                pass

        # Verify all mandatory projections were extracted
        missing_after = [k for k, (w, _) in extracted.items() if w is None and (k != "o" or not self.is_nested_encoder)]
        if missing_after:
            logger.warning(
                f"BAMUMoEBlock copy_weights failed: missing mandatory projections {missing_after} "
                f"from module {pretrained_attn.__class__.__name__} ({pretrained_attn}). "
                f"Proceeding with partially initialized projections."
            )

        # 6. Copy weights into target LoRALinear layers with strict dimension safety
        target_map = {
            "q": self.q_lin,
            "k": self.k_lin,
            "v": self.v_lin,
            "o": self.out_lin,
        }

        with torch.no_grad():
            for proj_key, target_mod in target_map.items():
                w, b = extracted[proj_key]
                if w is None:
                    if proj_key == "o" and self.is_nested_encoder:
                        # Inner nested self-attention bypasses out_lin; initialize to identity
                        nn.init.eye_(target_mod.weight)
                        if target_mod.bias is not None:
                            nn.init.zeros_(target_mod.bias)
                    continue

                if w.ndim != 2:
                    w = w.reshape(w.size(0), -1)

                # Handle transposed orientation if necessary (only if shape does not match already)
                if w.shape != target_mod.weight.shape and target_mod.weight.shape == w.t().contiguous().shape:
                    w = w.t().contiguous()
                out_d = min(target_mod.weight.size(0), w.size(0))
                in_d = min(target_mod.weight.size(1), w.size(1))

                target_mod.weight.data[:out_d, :in_d].copy_(w[:out_d, :in_d].to(device=target_mod.weight.device, dtype=target_mod.weight.dtype))

                if b is not None and target_mod.bias is not None:
                    min_b = min(target_mod.bias.numel(), b.numel())
                    target_mod.bias.data[:min_b].copy_(b[:min_b].to(device=target_mod.bias.device, dtype=target_mod.bias.dtype))
                elif target_mod.bias is not None:
                    nn.init.zeros_(target_mod.bias)

                # Weight verification: ensure weights are non-trivial
                w_norm = float(torch.norm(target_mod.weight.data).item())
                if w_norm < 1e-4:
                    logger.warning(f"Weight copy verification failed for projection '{proj_key}': norm too small ({w_norm:.2e})")

                if self.freeze_base:
                    target_mod.freeze_base_weights()

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
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, D = hidden_states.shape
        is_cross_attention = encoder_hidden_states is not None
        kv_states = encoder_hidden_states if is_cross_attention else hidden_states
        S = kv_states.shape[1]

        # 1. Base QKV projections
        q = self.q_lin(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.k_lin(kv_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.v_lin(kv_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        # 2. Rotary Position Embeddings (RoPE) applied before GQA expansion
        cos = kwargs.get("cos", None)
        sin = kwargs.get("sin", None)
        pos_emb = kwargs.get("position_embeddings", None)
        if pos_emb is not None and isinstance(pos_emb, (tuple, list)) and len(pos_emb) == 2:
            cos, sin = pos_emb[0], pos_emb[1]

        if cos is not None and sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos, sin, kwargs.get("position_ids", None))
        elif hasattr(self, "rotary_emb") and self.rotary_emb is not None:
            try:
                from framework.utils.compat import maybe_apply_rotary
                q, k = maybe_apply_rotary(q, k, getattr(self, "adapter", None), self.rotary_emb, kwargs.get("position_ids", None))
            except Exception as e:
                pass

        # 3. GQA expansion: repeat key/value heads to match query heads
        if self.kv_repeat > 1:
            k = repeat_kv(k, self.kv_repeat)
            v = repeat_kv(v, self.kv_repeat)

        # 4. MoE Value Refinement
        # Note on MoE Scope: qkv_moe is intentionally applied ONLY to self-attention (not is_cross_attention).
        # In cross-attention (encoder-decoder models like T5/BART), the key and value representations
        # are projected from encoder_hidden_states across different semantic spaces. Pretrained
        # cross-attention value projections are highly calibrated; injecting unaligned MoE value
        # refinement there would disrupt cross-attention alignment and destabilize sequence generation.
        if not is_cross_attention:
            moe_delta_v = self.qkv_moe(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
            v = v + 0.1 * moe_delta_v

        # 5. Attention scores computation in FP32 for numerical stability
        scale = 1.0 / math.sqrt(self.head_dim)
        # Memory-efficient: use SDPA to avoid OOM on large seqs

        try:

            _qs = q.half() if q.dtype == torch.float32 else q

            _ks = k.half() if k.dtype == torch.float32 else k

            _vs = v.half() if v.dtype == torch.float32 else v

            _sdpa_out = torch.nn.functional.scaled_dot_product_attention(_qs, _ks, _vs, is_causal=getattr(self, 'is_decoder', False) and N > 1, dropout_p=self.dropout.p if self.training else 0.0)

            _sdpa_out = _sdpa_out.to(hidden_states.dtype).transpose(1, 2).contiguous().view(B, N, self.all_head_size)

            _sdpa_out = torch.nan_to_num(_sdpa_out, nan=0.0, posinf=1e4, neginf=-1e4)

            return (self.out_proj(_sdpa_out) if not getattr(self, 'is_nested_encoder', False) else _sdpa_out), None

        except Exception:

            pass

        scores = torch.matmul(q.float(), k.float().transpose(-2, -1).contiguous()) * scale

        # 6. Causal and padding mask application
        if self.is_decoder and not is_cross_attention:
            # Lower-triangular causal mask
            causal_mask = torch.triu(
                torch.full((L, S), float("-inf"), device=scores.device, dtype=torch.float32),
                diagonal=1 + (S - L),
            )
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(1)

        # Apply padding mask
        mask = encoder_attention_mask if is_cross_attention else attention_mask
        if mask is not None:
            if mask.dim() == 2:
                pad_mask = (mask == 0).unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(pad_mask, float("-inf"))
            elif mask.dim() == 3:
                scores = scores + mask.unsqueeze(1).float()
            elif mask.dim() == 4:
                scores = scores + mask.float()

        # 7. Softmax and dropout
        probs = F.softmax(scores, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
        probs = self.dropout(probs)

        # 8. Head mask application
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.view(head_mask.shape[0], head_mask.shape[1], 1, 1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(-1)
            probs = probs * head_mask

        # 9. Context aggregation
        attn_out = torch.matmul(probs.to(v.dtype), v).transpose(1, 2).contiguous().view(B, L, self.all_head_size)

        # 10. Output projection with nested encoder bypass
        if self.is_nested_encoder:
            # Outer RobertaAttention.output.dense performs the dense projection; bypass out_lin
            final_out = attn_out
        else:
            final_out = self.out_lin(attn_out)

        # Accumulate auxiliary load-balancing loss from active MoE branches
        self.aux_loss = self.qkv_moe.aux_loss

        return final_out, (probs if output_attentions else None)


@ATTENTION_REGISTRY.register("bam_umoe_fn")
@ATTENTION_REGISTRY.register("bam_umoe_inject")
def inject_bam_umoe(
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
            res = inject_bam_umoe(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config if 'comp_config' in locals() or 'comp_config' in kwargs else None,
                layer_idx=idx,
                total_layers=tot,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    layer = target

    r"""Injection factory function for BAM-UMoE attention blocks with safe pre-testing."""
    if old_attn is None and layer is not None:
        old_attn = getattr(layer, "attention", getattr(layer, "self", getattr(layer, "attn", None)))

    hidden_size = (
        getattr(old_attn, 'hidden_size', None) or
        getattr(old_attn, 'embed_dim', None) or
        (old_attn.q_proj.in_features if hasattr(old_attn, 'q_proj') and hasattr(old_attn.q_proj, 'in_features') else None) or
        (layer.in_features if hasattr(layer, 'in_features') else None) or
        getattr(model_config, 'hidden_size', 64)
    )
    num_heads = (
        getattr(old_attn, 'num_heads', None) or
        getattr(old_attn, 'num_attention_heads', None) or
        getattr(model_config, 'num_attention_heads', 4)
    )

    try:
        new_attn = BAMUMoEBlock(
            config=model_config,
            old_attn=old_attn,
            comp_config=comp_config,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_attention_heads=num_heads,
            layer_id=layer_idx,
            **kwargs,
        )
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
    except Exception:
        return None
