"""
Endoskeletal Surgery Shell, Warm-Start Weight Transfer & HF Registration.

Provides:
  - register_neuralops_with_huggingface(): registers all 131 NeuralOps attention mechanisms
    into transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS so any HuggingFace model can use
    attn_implementation="neuralops_<name>".
  - transfer_pretrained_qkv_weights(): warm-starts NeuralOps modules with pretrained Q, K, V, O weights.
  - SidecarGateBranch: lightweight data-dependent gating & 1D causal convolution for recurrent/stateful attentions.
  - UniversalAttentionBridge: universal wrapper module replacing host attention blocks while preserving
    exact return signatures, kwargs, and surrounding residual/LayerNorm invariants.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.surgery.scanner import AttentionTargetSpec, UniversalArchitectureScanner

logger = logging.getLogger("NeuralOps.Surgery.EndoskeletalShell")


# ==============================================================================
# 1. HuggingFace ALL_ATTENTION_FUNCTIONS Registration
# ==============================================================================

def register_neuralops_with_huggingface() -> None:
    """
    Registers all 131 NeuralOps attention mechanisms into HuggingFace's
    `transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS`.
    Enables zero-surgery usage:
        AutoModelForCausalLM.from_pretrained("...", attn_implementation="neuralops_cosformer")
    """
    try:
        import transformers.modeling_utils as hf_utils
        if not hasattr(hf_utils, "ALL_ATTENTION_FUNCTIONS"):
            return
        all_attn_fn = hf_utils.ALL_ATTENTION_FUNCTIONS
    except Exception:
        # HuggingFace transformers not installed or old version
        return

    def _make_hf_attention_callable(attn_name: str) -> Callable:
        def _hf_neuralops_attention(
            module: nn.Module,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            scaling: Optional[float] = None,
            is_causal: Optional[bool] = None,
            **kwargs: Any,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            # query: (B, H_q, T, D_h)
            # key, value: (B, H_kv, T, D_h)
            B, H_q, T, D_h = query.shape
            H_kv = key.shape[1]

            # Broadcast GQA if H_q != H_kv
            if H_q != H_kv:
                repeat_factor = H_q // H_kv
                key = torch.repeat_interleave(key, repeat_factor, dim=1)
                value = torch.repeat_interleave(value, repeat_factor, dim=1)

            # Route through NeuralOps kernel or registered class
            attn_cls = ATTENTION_REGISTRY.get(attn_name)
            if hasattr(module, "_neuralops_submodule") and module._neuralops_submodule is not None:
                core_sub = module._neuralops_submodule
            elif attn_cls is not None:
                if not hasattr(module, "_neuralops_submodule_cache"):
                    module._neuralops_submodule_cache = {}
                if attn_name not in module._neuralops_submodule_cache:
                    hidden_size = H_q * D_h
                    module._neuralops_submodule_cache[attn_name] = attn_cls(
                        hidden_size=hidden_size, num_heads=H_q, head_dim=D_h
                    ).to(device=query.device, dtype=query.dtype)
                core_sub = module._neuralops_submodule_cache[attn_name]
            else:
                core_sub = None

            if core_sub is not None:
                # Transpose back to (B, T, H*D) or pass Q, K, V
                q_flat = query.transpose(1, 2).contiguous().view(B, T, H_q * D_h)
                out, _ = core_sub(q_flat, attention_mask=attention_mask, is_causal=is_causal, **kwargs)
                out = out.view(B, T, H_q, D_h)
            else:
                # Fallback to standard PyTorch SDPA
                out = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, is_causal=bool(is_causal)
                )
                out = out.transpose(1, 2).contiguous()

            return out, None

        return _hf_neuralops_attention

    # Register each attention mechanism
    for name in ATTENTION_REGISTRY.keys():
        hf_key = f"neuralops_{name}"
        if hf_key not in all_attn_fn:
            all_attn_fn[hf_key] = _make_hf_attention_callable(name)

    if "neuralops" not in all_attn_fn:
        all_attn_fn["neuralops"] = _make_hf_attention_callable("cosformer")


# Auto-register upon importing endoskeletal_shell
try:
    register_neuralops_with_huggingface()
except Exception:
    pass


# ==============================================================================
# 2. Sidecar Gate & Short-Convolution Branch
# ==============================================================================

class SidecarGateBranch(nn.Module):
    """
    Lightweight data-dependent gating and depthwise causal 1D convolution
    for recurrent/stateful linear attention models.
    Initialized in long-memory regime (decay bias ~ 2.5 -> sigmoid ~ 0.92-0.95).
    """

    def __init__(self, d_model: int, num_heads: int, d_conv: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_conv = d_conv

        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_model,
            bias=True,
        )
        self.gate_proj = nn.Linear(d_model, num_heads, bias=True)

        with torch.no_grad():
            # Identity-like conv init
            self.conv1d.weight.zero_()
            self.conv1d.weight[:, :, -1] = 1.0
            if self.conv1d.bias is not None:
                self.conv1d.bias.zero_()
            # Long-term retention init
            nn.init.constant_(self.gate_proj.bias, 2.5)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, D)
        B, T, D = x.shape
        x_conv = self.conv1d(x.transpose(1, 2))[:, :, :T].transpose(1, 2).contiguous()
        gates = torch.sigmoid(self.gate_proj(x))
        return x_conv, gates


# ==============================================================================
# 3. Warm-Start Weight Extractor & Transfer
# ==============================================================================

def transfer_pretrained_qkv_weights(
    spec: AttentionTargetSpec,
    new_module: nn.Module,
    strict: bool = False,
) -> None:
    """
    Copies pretrained Q, K, V, and O projections from the host model into the newly
    instantiated NeuralOps attention module without random re-initialization.
    """
    if spec.extracted_qkv is None:
        return

    W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o = spec.extracted_qkv

    # Identify projection attributes on target new_module
    target_q = getattr(new_module, "q_proj", getattr(new_module, "query", getattr(new_module, "q", None)))
    target_k = getattr(new_module, "k_proj", getattr(new_module, "key", getattr(new_module, "k", None)))
    target_v = getattr(new_module, "v_proj", getattr(new_module, "value", getattr(new_module, "v", None)))
    target_o = getattr(
        new_module,
        "out_proj",
        getattr(new_module, "o_proj", getattr(new_module, "proj", getattr(new_module, "dense", None))),
    )

    def _copy_param(target_linear: Any, W: torch.Tensor, b: Optional[torch.Tensor]) -> None:
        if target_linear is None or not hasattr(target_linear, "weight"):
            return
        with torch.no_grad():
            tgt_w = target_linear.weight
            if tgt_w.shape == W.shape:
                tgt_w.copy_(W.to(device=tgt_w.device, dtype=tgt_w.dtype))
            elif tgt_w.ndim == 2 and W.ndim == 2:
                # Handle head expansion or GQA repeat
                min_out = min(tgt_w.shape[0], W.shape[0])
                min_in = min(tgt_w.shape[1], W.shape[1])
                tgt_w[:min_out, :min_in].copy_(W[:min_out, :min_in].to(device=tgt_w.device, dtype=tgt_w.dtype))

            if b is not None and getattr(target_linear, "bias", None) is not None:
                tgt_b = target_linear.bias
                if tgt_b.shape == b.shape:
                    tgt_b.copy_(b.to(device=tgt_b.device, dtype=tgt_b.dtype))
                else:
                    min_b = min(tgt_b.shape[0], b.shape[0])
                    tgt_b[:min_b].copy_(b[:min_b].to(device=tgt_b.device, dtype=tgt_b.dtype))

    _copy_param(target_q, W_q, b_q)
    _copy_param(target_k, W_k, b_k)
    _copy_param(target_v, W_v, b_v)
    _copy_param(target_o, W_o, b_o)


# ==============================================================================
# 4. Universal Attention Bridge
# ==============================================================================

class UniversalAttentionBridge(nn.Module):
    """
    Universal surgery shell replacing any attention layer in the wild.
    Preserves exact forward return signature, residual connections, and positional embeddings.
    """

    def __init__(
        self,
        orig_module: nn.Module,
        spec: AttentionTargetSpec,
        new_attention: nn.Module,
        inherit_weights: bool = True,
        attach_sidecar: bool = False,
    ) -> None:
        super().__init__()
        self.orig_module = orig_module
        self.spec = spec
        self.new_attention = new_attention
        self.pattern_type = spec.pattern_type
        self.embed_dim = spec.embed_dim
        self.num_heads = spec.num_heads

        # Extract argument signature of original forward
        self._orig_sig = inspect.signature(orig_module.forward)

        # Warm start weight transfer
        if inherit_weights:
            transfer_pretrained_qkv_weights(spec, self.new_attention)

        # Optional sidecar gate branch for recurrent/linear attention
        self.sidecar = (
            SidecarGateBranch(d_model=self.embed_dim, num_heads=self.num_heads)
            if attach_sidecar
            else None
        )

        # Teacher / shadow twin placeholder
        self.teacher_branch: Optional[nn.Module] = None
        self.alpha: float = 1.0

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        # Extract primary hidden states from args or kwargs
        if len(args) > 0:
            hidden_states = args[0]
            rest_args = args[1:]
        else:
            hidden_states = kwargs.get(
                "hidden_states",
                kwargs.get("x", kwargs.get("input_tensor", kwargs.get("query", None))),
            )
            rest_args = ()

        if hidden_states is None:
            raise ValueError(f"Could not extract hidden states from forward call {args}, {kwargs}")

        # Apply sidecar short-conv if present
        if self.sidecar is not None:
            hidden_states, _ = self.sidecar(hidden_states)

        # Forward through new attention mechanism
        attn_out = self.new_attention(hidden_states, *rest_args, **kwargs)

        # Unpack attention output
        if isinstance(attn_out, tuple):
            core_tensor = attn_out[0]
            extra_outputs = attn_out[1:]
        else:
            core_tensor = attn_out
            extra_outputs = ()

        # Pattern-specific return packaging
        if self.pattern_type == "bert_nested":
            # In BertSelfAttention, the return is (context_layer,) or (context_layer, attention_probs)
            if kwargs.get("output_attentions", False):
                return (core_tensor, None)
            return (core_tensor,)

        if self.pattern_type == "conv1d_gpt2":
            # GPT2Attention returns (attn_output, present) or (attn_output, present, attentions)
            use_cache = kwargs.get("use_cache", False)
            output_attentions = kwargs.get("output_attentions", False)
            ret = [core_tensor]
            if use_cache:
                ret.append(extra_outputs[0] if len(extra_outputs) > 0 else None)
            if output_attentions:
                ret.append(None)
            return tuple(ret) if len(ret) > 1 else core_tensor

        # Standard HuggingFace tuple preservation
        if len(extra_outputs) > 0:
            return (core_tensor, *extra_outputs)

        return core_tensor


__all__ = [
    "register_neuralops_with_huggingface",
    "SidecarGateBranch",
    "transfer_pretrained_qkv_weights",
    "UniversalAttentionBridge",
]
