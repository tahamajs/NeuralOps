"""Attention Replace Operation Contract and Implementation.

Specification Section 6.2: attention.replace@1.0.0
Replaces attention mechanisms in target models with linear, flash, or hybrid attention blocks.
Integrates with UniversalSurgeryDirector and fx_surgery bridges.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from neuralops.operations.errors import OperationError
from neuralops.operations.base import Operation, OperationContract
from neuralops.operations.registry import register_operation

logger = logging.getLogger("NeuralOps.Operations.Attention.Replace")

CONTRACT = OperationContract(
    name="attention.replace",
    version="1.0.0",
    input_state="loaded",
    output_state="attention_replaced",
    params_schema={
        "type": "object",
        "properties": {
            "target_attention": {
                "type": "string",
            },
            "target": {
                "type": "string",
            },
            "layer_indices": {
                "type": "array",
                "items": {"type": "integer"},
            },
            "chunk_size": {"type": "integer", "minimum": 16},
            "preserve_weights": {"type": "boolean"},
        },
        "required": [],
        "additionalProperties": False,
    },
    preconditions=("model.has_attention",),
    postconditions=("model.output_shape_preserved",),
    composable_with_self=False,
)


class ReplacementAttentionBlock(nn.Module):
    """Drop-in attention replacement supporting linear, SDPA, and hybrid modes."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mode: str = "sdpa",
        chunk_size: int = 64,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = max(1, embed_dim // max(1, num_heads))
        self.mode = mode
        self.chunk_size = chunk_size

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        b, s, d = hidden_states.shape
        q = self.q_proj(hidden_states).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

        if self.mode == "linear":
            # Linear causal / kernel attention: (Q K^T) V -> Q (K^T V)
            q_norm = torch.nn.functional.elu(q) + 1.0
            k_norm = torch.nn.functional.elu(k) + 1.0
            # Context computation
            kv = torch.einsum("bhsk,bhsd->bhkd", k_norm, v)
            out = torch.einsum("bhsk,bhkd->bhsd", q_norm, kv)
            # Normalization
            z = 1.0 / (torch.einsum("bhsk,bhk->bhs", q_norm, k_norm.sum(dim=2)).unsqueeze(-1) + 1e-6)
            out = out * z
            out = out.transpose(1, 2).contiguous().view(b, s, d)
        else:
            # Native PyTorch SDPA (supports Flash, Mem-Efficient, Math)
            is_causal = attention_mask is None and s > 1
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask, is_causal=is_causal
            )
            out = out.transpose(1, 2).contiguous().view(b, s, d)

        out = self.out_proj(out)
        if getattr(self, "return_tuple", False) or len(args) >= 2 or "output_attentions" in kwargs:
            return (out, None)
        return out


def _find_target_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    attn_modules = []
    attn_keywords = ("self_attn", "attn", "attention", "cross_attn")
    for name, module in model.named_modules():
        if module is model:
            continue
        lower_name = name.lower()
        if any(kw in lower_name for kw in attn_keywords):
            # check if it's a leaf attention block
            has_sub_attn = any(
                any(kw in sub_name.lower() for kw in attn_keywords)
                for sub_name, _ in module.named_children()
            )
            if not has_sub_attn or hasattr(module, "q_proj") or hasattr(module, "q_lin"):
                attn_modules.append((name, module))
    return attn_modules


@register_operation(override=True)
class AttentionReplaceOperation(Operation):
    """Operation to replace standard attention with optimized/linear alternatives."""

    def __init__(self) -> None:
        super().__init__(contract=CONTRACT)

    def apply(self, model: nn.Module, config: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        target_mode: str = config.get("target_attention", config.get("target", "sdpa"))
        layer_indices: Optional[List[int]] = config.get("layer_indices")
        chunk_size: int = config.get("chunk_size", 64)
        preserve_weights: bool = config.get("preserve_weights", True)

        attn_modules = _find_target_modules(model)
        if not attn_modules:
            raise OperationError("Precondition failed: model.has_attention — no attention modules found.")

        replaced_count = 0
        replaced_names = []

        for idx, (name, mod) in enumerate(attn_modules):
            if layer_indices is not None and idx not in layer_indices:
                continue

            # Infer embed_dim
            embed_dim = None
            if hasattr(mod, "q_proj") and hasattr(mod.q_proj, "in_features"):
                embed_dim = mod.q_proj.in_features
            elif hasattr(mod, "q_lin") and hasattr(mod.q_lin, "in_features"):
                embed_dim = mod.q_lin.in_features
            elif hasattr(mod, "query") and hasattr(mod.query, "in_features"):
                embed_dim = mod.query.in_features
            elif hasattr(mod, "dim"):
                embed_dim = mod.dim
            elif hasattr(mod, "embed_dim"):
                embed_dim = mod.embed_dim
            elif hasattr(mod, "hidden_size"):
                embed_dim = mod.hidden_size
            elif hasattr(mod, "in_proj_weight"):
                embed_dim = mod.in_proj_weight.shape[1]
            else:
                embed_dim = 64

            num_heads = getattr(mod, "num_heads", getattr(mod, "n_heads", getattr(mod, "num_attention_heads", 8)))
            new_block = ReplacementAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mode=target_mode,
                chunk_size=chunk_size,
            )
            if hasattr(mod, "q_lin") or hasattr(mod, "n_heads") or "distilbert" in type(mod).__name__.lower():
                new_block.return_tuple = True

            # Move new block to device and dtype of existing module
            try:
                first_param = next(mod.parameters(), None)
                if first_param is not None:
                    new_block = new_block.to(device=first_param.device, dtype=first_param.dtype)
            except Exception:
                pass

            if preserve_weights:
                try:
                    if hasattr(mod, "q_proj") and hasattr(new_block, "q_proj"):
                        new_block.q_proj.weight.data.copy_(mod.q_proj.weight.data)
                    if hasattr(mod, "k_proj") and hasattr(new_block, "k_proj"):
                        new_block.k_proj.weight.data.copy_(mod.k_proj.weight.data)
                    if hasattr(mod, "v_proj") and hasattr(new_block, "v_proj"):
                        new_block.v_proj.weight.data.copy_(mod.v_proj.weight.data)
                    if hasattr(mod, "out_proj") and hasattr(new_block, "out_proj"):
                        new_block.out_proj.weight.data.copy_(mod.out_proj.weight.data)
                except Exception:
                    pass

            # Swap in model
            parent_name, _, child_name = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, new_block)
            replaced_names.append(name)
            replaced_count += 1

        evidence: Dict[str, Any] = {
            "params_used": config,
            "operation_version": self.version,
            "output_state": self.output_state,
            "replaced_count": replaced_count,
            "replaced_modules": replaced_names,
            "target_mode": target_mode,
        }
        logger.info("AttentionReplaceOperation: Replaced %d modules with %s", replaced_count, target_mode)
        return model, evidence
