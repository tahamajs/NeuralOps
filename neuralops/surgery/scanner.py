"""
Topological & Subgraph Attention Scanner for Universal Model Surgery.

Inspects any PyTorch or HuggingFace neural network architecture and discovers
all attention blocks and linear projections across 5 universal patterns:
  - Pattern A: Split 4-Projection Topology (q_proj, k_proj, v_proj, o_proj)
  - Pattern B: Fused QKV Linear Topology (qkv, c_attn, query_key_value)
  - Pattern C: Transposed HuggingFace Conv1D Topology (GPT-2)
  - Pattern D: Native torch.nn.MultiheadAttention Parameter Topology
  - Pattern E: Custom/Nested Self-Attention Cores (BERT/RoBERTa self vs output)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

logger = logging.getLogger("NeuralOps.Surgery.Scanner")


@dataclass
class AttentionTargetSpec:
    """Structured metadata specifying an identified attention block and its projections."""

    module_path: str
    parent_module: nn.Module
    attr_name: str
    target_module: nn.Module
    pattern_type: str  # "split_4proj", "fused_qkv", "conv1d_gpt2", "native_mha", "bert_nested"

    # Dimensional characteristics
    embed_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    is_causal: bool = False
    has_bias: bool = True

    # Weight and bias accessors/extractors
    # Returns (W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o) with canonical shapes:
    # W_q: (num_heads * head_dim, embed_dim)
    # W_k: (num_kv_heads * head_dim, embed_dim)
    # W_v: (num_kv_heads * head_dim, embed_dim)
    # W_o: (embed_dim, num_heads * head_dim)
    extracted_qkv: Optional[
        Tuple[
            torch.Tensor,
            Optional[torch.Tensor],
            torch.Tensor,
            Optional[torch.Tensor],
            torch.Tensor,
            Optional[torch.Tensor],
            Optional[torch.Tensor],
            Optional[torch.Tensor],
        ]
    ] = None

    # Auxiliary module hooks (e.g. for BERT where BertSelfOutput houses dense + LayerNorm + residual)
    output_dense_module: Optional[nn.Module] = None
    output_layernorm_module: Optional[nn.Module] = None
    extra_metadata: Optional[Dict[str, Any]] = None


class Conv1DLinearAdapter(nn.Module):
    """
    Adapter converting HuggingFace transformers Conv1D to standard nn.Linear semantics.
    
    In HuggingFace GPT-2:
        weight shape is (in_features, out_features)
        bias shape is (out_features,)
        y = x @ weight + bias
    Standard nn.Linear:
        weight shape is (out_features, in_features)
        y = x @ weight.t() + bias
    """

    def __init__(self, conv1d_module: Any) -> None:
        super().__init__()
        self.conv1d_module = conv1d_module
        raw_w = getattr(conv1d_module, "weight", None)
        raw_b = getattr(conv1d_module, "bias", None)

        if raw_w is not None and isinstance(raw_w, torch.Tensor):
            self.in_features = raw_w.shape[0]
            self.out_features = raw_w.shape[1]
        else:
            self.in_features = getattr(conv1d_module, "nf", 768)
            self.out_features = getattr(conv1d_module, "nx", 768)

    @property
    def weight(self) -> torch.Tensor:
        """Transposed weight tensor with standard (out_features, in_features) layout."""
        w = self.conv1d_module.weight
        return w.t().contiguous() if w.ndim == 2 else w

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return getattr(self.conv1d_module, "bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1d_module(x)


class UniversalArchitectureScanner:
    """Intelligent structural scanner discovering attention blocks across all 5 topological patterns."""

    @staticmethod
    def is_conv1d(module: nn.Module) -> bool:
        """Check if module is a HuggingFace Conv1D projection."""
        cls_name = module.__class__.__name__
        if "Conv1D" in cls_name:
            return True
        w = getattr(module, "weight", None)
        # Check if weight exists and has non-standard shape
        if isinstance(w, (torch.Tensor, nn.Parameter)) and hasattr(module, "nf"):
            return True
        return False

    @staticmethod
    def _extract_weight_bias(
        linear_or_conv: Union[nn.Module, nn.Parameter, torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extract canonical weight (out_features, in_features) and bias (out_features,) from any projection."""
        if isinstance(linear_or_conv, (nn.Linear, nn.modules.linear.NonDynamicallyQuantizableLinear)):
            return linear_or_conv.weight, linear_or_conv.bias

        if UniversalArchitectureScanner.is_conv1d(linear_or_conv):
            w = linear_or_conv.weight
            b = getattr(linear_or_conv, "bias", None)
            if w.ndim == 2:
                # Transpose from (in_features, out_features) to (out_features, in_features)
                return w.t().contiguous(), b
            return w, b

        if isinstance(linear_or_conv, (torch.Tensor, nn.Parameter)):
            return linear_or_conv, None

        if hasattr(linear_or_conv, "weight"):
            w = linear_or_conv.weight
            b = getattr(linear_or_conv, "bias", None)
            return w, b

        raise ValueError(f"Cannot extract weight and bias from module {linear_or_conv}")

    @classmethod
    def scan(cls, model: nn.Module) -> List[AttentionTargetSpec]:
        """
        Recursively scan any nn.Module and return all identified attention blocks.
        Guarantees that replacing the target module preserves surrounding LayerNorm and residuals.
        """
        targets: List[AttentionTargetSpec] = []
        visited_ids = set()

        for name, module in model.named_modules():
            if id(module) in visited_ids:
                continue

            spec = cls._inspect_module(model, name, module)
            if spec is not None:
                targets.append(spec)
                visited_ids.add(id(spec.target_module))

        return targets

    @classmethod
    def _inspect_module(
        cls, root_model: nn.Module, path: str, module: nn.Module
    ) -> Optional[AttentionTargetSpec]:
        """Inspect a single module and match against universal patterns."""
        cls_name = module.__class__.__name__.lower()

        # Find parent and attribute name
        parts = path.split(".")
        if len(parts) == 1 and parts[0] == "":
            parent = root_model
            attr_name = ""
        elif len(parts) == 1:
            parent = root_model
            attr_name = parts[0]
        else:
            parent_path = ".".join(parts[:-1])
            attr_name = parts[-1]
            parent = root_model.get_submodule(parent_path)

        # -------------------------------------------------------------
        # 1. Pattern D: Native torch.nn.MultiheadAttention
        # -------------------------------------------------------------
        if isinstance(module, nn.MultiheadAttention):
            embed_dim = module.embed_dim
            num_heads = module.num_heads
            head_dim = embed_dim // num_heads
            is_causal = getattr(module, "batch_first", False)  # generic proxy

            in_w = module.in_proj_weight
            in_b = module.in_proj_bias
            out_proj = module.out_proj

            W_q, W_k, W_v = in_w.chunk(3, dim=0)
            b_q, b_k, b_v = in_b.chunk(3, dim=0) if in_b is not None else (None, None, None)
            W_o, b_o = out_proj.weight, out_proj.bias

            return AttentionTargetSpec(
                module_path=path,
                parent_module=parent,
                attr_name=attr_name,
                target_module=module,
                pattern_type="native_mha",
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_heads,
                head_dim=head_dim,
                is_causal=is_causal,
                has_bias=in_b is not None,
                extracted_qkv=(W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o),
            )

        # -------------------------------------------------------------
        # 2. Pattern E: BERT / RoBERTa Nested Attention (BertAttention vs BertSelfAttention)
        # CRITICAL: If module is BertAttention (contains self + output), target module.self directly
        # so output (dense + LayerNorm + residual) is never destroyed!
        # -------------------------------------------------------------
        if hasattr(module, "self") and hasattr(module, "output") and ("bert" in cls_name or "roberta" in cls_name):
            self_attn = getattr(module, "self")
            output_block = getattr(module, "output")
            output_dense = getattr(output_block, "dense", None)
            output_ln = getattr(output_block, "LayerNorm", getattr(output_block, "layer_norm", None))

            q_mod = getattr(self_attn, "query", None)
            k_mod = getattr(self_attn, "key", None)
            v_mod = getattr(self_attn, "value", None)

            if q_mod is not None and k_mod is not None and v_mod is not None and output_dense is not None:
                W_q, b_q = cls._extract_weight_bias(q_mod)
                W_k, b_k = cls._extract_weight_bias(k_mod)
                W_v, b_v = cls._extract_weight_bias(v_mod)
                W_o, b_o = cls._extract_weight_bias(output_dense)

                embed_dim = W_q.shape[1]
                num_heads = getattr(self_attn, "num_attention_heads", 12)
                head_dim = W_q.shape[0] // num_heads

                return AttentionTargetSpec(
                    module_path=f"{path}.self",
                    parent_module=module,
                    attr_name="self",
                    target_module=self_attn,
                    pattern_type="bert_nested",
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_heads,
                    head_dim=head_dim,
                    is_causal=False,
                    has_bias=b_q is not None,
                    extracted_qkv=(W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o),
                    output_dense_module=output_dense,
                    output_layernorm_module=output_ln,
                )

        # -------------------------------------------------------------
        # 3. Pattern C: Transposed HuggingFace Conv1D Topology (GPT-2)
        # -------------------------------------------------------------
        if hasattr(module, "c_attn") and hasattr(module, "c_proj"):
            c_attn = getattr(module, "c_attn")
            c_proj = getattr(module, "c_proj")

            W_attn, b_attn = cls._extract_weight_bias(c_attn)
            W_o, b_o = cls._extract_weight_bias(c_proj)

            # W_attn shape is (3 * D, D) after transpose
            embed_dim = W_attn.shape[1]
            total_qkv_out = W_attn.shape[0]
            # Standard GPT-2 has 3 equal chunks (Q, K, V)
            split_dim = total_qkv_out // 3
            W_q = W_attn[0:split_dim, :]
            W_k = W_attn[split_dim : 2 * split_dim, :]
            W_v = W_attn[2 * split_dim : 3 * split_dim, :]

            if b_attn is not None:
                b_q = b_attn[0:split_dim]
                b_k = b_attn[split_dim : 2 * split_dim]
                b_v = b_attn[2 * split_dim : 3 * split_dim]
            else:
                b_q, b_k, b_v = None, None, None

            num_heads = getattr(module, "num_heads", getattr(module, "n_head", 12))
            head_dim = split_dim // num_heads

            return AttentionTargetSpec(
                module_path=path,
                parent_module=parent,
                attr_name=attr_name,
                target_module=module,
                pattern_type="conv1d_gpt2",
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_heads,
                head_dim=head_dim,
                is_causal=True,
                has_bias=b_attn is not None,
                extracted_qkv=(W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o),
            )

        # -------------------------------------------------------------
        # 4. Pattern B: Fused QKV Linear Topology (ViT, Swin, DiT)
        # -------------------------------------------------------------
        qkv_mod = getattr(module, "qkv", getattr(module, "query_key_value", None))
        proj_mod = getattr(module, "proj", getattr(module, "out_proj", getattr(module, "o_proj", None)))

        if qkv_mod is not None and proj_mod is not None:
            W_qkv, b_qkv = cls._extract_weight_bias(qkv_mod)
            W_o, b_o = cls._extract_weight_bias(proj_mod)

            embed_dim = W_qkv.shape[1]
            num_heads = getattr(module, "num_heads", 12)
            total_out = W_qkv.shape[0]

            # Determine head_dim and check if GQA or symmetric 3*D
            # If total_out == 3 * embed_dim:
            if total_out % 3 == 0:
                split_dim = total_out // 3
                num_kv_heads = num_heads
                head_dim = split_dim // num_heads
                W_q = W_qkv[0:split_dim, :]
                W_k = W_qkv[split_dim : 2 * split_dim, :]
                W_v = W_qkv[2 * split_dim : 3 * split_dim, :]
                if b_qkv is not None:
                    b_q = b_qkv[0:split_dim]
                    b_k = b_qkv[split_dim : 2 * split_dim]
                    b_v = b_qkv[2 * split_dim : 3 * split_dim]
                else:
                    b_q, b_k, b_v = None, None, None
            else:
                # GQA fused layout: q_dim = num_heads * head_dim, kv_dim = 2 * num_kv_heads * head_dim
                head_dim = embed_dim // num_heads
                q_dim = num_heads * head_dim
                rem_kv = total_out - q_dim
                kv_dim = rem_kv // 2
                num_kv_heads = kv_dim // head_dim
                W_q = W_qkv[0:q_dim, :]
                W_k = W_qkv[q_dim : q_dim + kv_dim, :]
                W_v = W_qkv[q_dim + kv_dim :, :]
                if b_qkv is not None:
                    b_q = b_qkv[0:q_dim]
                    b_k = b_qkv[q_dim : q_dim + kv_dim]
                    b_v = b_qkv[q_dim + kv_dim :]
                else:
                    b_q, b_k, b_v = None, None, None

            return AttentionTargetSpec(
                module_path=path,
                parent_module=parent,
                attr_name=attr_name,
                target_module=module,
                pattern_type="fused_qkv",
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                is_causal=getattr(module, "is_causal", False),
                has_bias=b_qkv is not None,
                extracted_qkv=(W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o),
            )

        # -------------------------------------------------------------
        # 5. Pattern A: Split 4-Projection Topology (LLaMA, Mistral, Qwen, DistilBERT, T5)
        # -------------------------------------------------------------
        q_mod, k_mod, v_mod, o_mod = None, None, None, None

        # Check LLaMA / Qwen / Mistral
        if hasattr(module, "q_proj") and hasattr(module, "k_proj") and hasattr(module, "v_proj") and hasattr(module, "o_proj"):
            q_mod = getattr(module, "q_proj")
            k_mod = getattr(module, "k_proj")
            v_mod = getattr(module, "v_proj")
            o_mod = getattr(module, "o_proj")
        # Check DistilBERT (q_lin, k_lin, v_lin, out_lin)
        elif hasattr(module, "q_lin") and hasattr(module, "k_lin") and hasattr(module, "v_lin") and hasattr(module, "out_lin"):
            q_mod = getattr(module, "q_lin")
            k_mod = getattr(module, "k_lin")
            v_mod = getattr(module, "v_lin")
            o_mod = getattr(module, "out_lin")
        # Check T5 (q, k, v, o)
        elif hasattr(module, "q") and hasattr(module, "k") and hasattr(module, "v") and hasattr(module, "o"):
            q_mod = getattr(module, "q")
            k_mod = getattr(module, "k")
            v_mod = getattr(module, "v")
            o_mod = getattr(module, "o")
        # Check generic query, key, value, out_proj
        elif hasattr(module, "query") and hasattr(module, "key") and hasattr(module, "value") and hasattr(module, "out_proj"):
            q_mod = getattr(module, "query")
            k_mod = getattr(module, "key")
            v_mod = getattr(module, "value")
            o_mod = getattr(module, "out_proj")

        if q_mod is not None and k_mod is not None and v_mod is not None and o_mod is not None:
            W_q, b_q = cls._extract_weight_bias(q_mod)
            W_k, b_k = cls._extract_weight_bias(k_mod)
            W_v, b_v = cls._extract_weight_bias(v_mod)
            W_o, b_o = cls._extract_weight_bias(o_mod)

            embed_dim = W_q.shape[1]
            num_heads = getattr(module, "num_heads", getattr(module, "n_heads", getattr(module, "num_attention_heads", 12)))
            head_dim = getattr(module, "head_dim", W_q.shape[0] // num_heads)
            num_kv_heads = getattr(module, "num_key_value_heads", getattr(module, "num_kv_heads", W_k.shape[0] // head_dim))
            is_causal = getattr(module, "is_causal", True if "llama" in cls_name or "qwen" in cls_name else False)

            return AttentionTargetSpec(
                module_path=path,
                parent_module=parent,
                attr_name=attr_name,
                target_module=module,
                pattern_type="split_4proj",
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                is_causal=is_causal,
                has_bias=b_q is not None,
                extracted_qkv=(W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o),
            )

        return None


__all__ = [
    "AttentionTargetSpec",
    "Conv1DLinearAdapter",
    "UniversalArchitectureScanner",
]
