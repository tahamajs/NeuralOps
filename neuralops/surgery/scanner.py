#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Universal Architecture Attention Scanner

Discovers attention blocks and projection topologies across neural network architectures.
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
    pattern_type: str

    embed_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    is_causal: bool = False
    has_bias: bool = True

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
        if isinstance(w, (torch.Tensor, nn.Parameter)) and hasattr(module, "nf"):
            return True
        return False

    @staticmethod
    def _is_lora_module(module: Any) -> bool:
        """Determines if a module or projection is wrapped with LoRA / PEFT adapters."""
        if not isinstance(module, nn.Module):
            return False
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            return True
        if hasattr(module, "base_layer") and (
            hasattr(module, "scaling") or hasattr(module, "lora_A") or "lora" in module.__class__.__name__.lower()
        ):
            return True
        if hasattr(module, "active_adapters") or hasattr(module, "peft_type"):
            return True
        return False

    @staticmethod
    def _extract_lora_adapters(module: nn.Module) -> Dict[str, Any]:
        """Extracts adapter references and configurations from a LoRA module."""
        adapters: Dict[str, Any] = {}
        for attr in ("lora_A", "lora_B", "scaling", "base_layer", "r", "lora_alpha", "adapter_name"):
            if hasattr(module, attr):
                adapters[attr] = getattr(module, attr)
        return adapters

    @classmethod
    def _extract_weight_bias_with_lora(
        cls, linear_or_conv: Union[nn.Module, nn.Parameter, torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, Any]]]:
        """
        Extract canonical weight and bias. If linear_or_conv is a LoRA/PEFT wrapped module,
        unwraps the base layer while preserving adapter references and merges the adapter weights.
        """
        if not isinstance(linear_or_conv, nn.Module) or not cls._is_lora_module(linear_or_conv):
            w, b = cls._extract_weight_bias(linear_or_conv)
            return w, b, None

        adapter_info = cls._extract_lora_adapters(linear_or_conv)
        base = getattr(linear_or_conv, "base_layer", linear_or_conv)

        if base is not linear_or_conv:
            base_w, base_b = cls._extract_weight_bias(base)
        else:
            raw_w = getattr(linear_or_conv, "weight", None)
            raw_b = getattr(linear_or_conv, "bias", None)
            if raw_w is None:
                raise ValueError(f"Cannot extract base weight from LoRA module {linear_or_conv}")
            base_w = raw_w
            base_b = raw_b

        is_merged = getattr(linear_or_conv, "merged", False)
        if isinstance(is_merged, list):
            is_merged = len(is_merged) > 0 and all(is_merged)

        if is_merged:
            return base_w, base_b, adapter_info

        merged_w = base_w.clone()
        lora_A = getattr(linear_or_conv, "lora_A", None)
        lora_B = getattr(linear_or_conv, "lora_B", None)
        scaling = getattr(linear_or_conv, "scaling", {})

        if lora_A is not None and lora_B is not None:
            if isinstance(lora_A, (nn.ModuleDict, dict)) and isinstance(lora_B, (nn.ModuleDict, dict)):
                for adapter_name, mod_a in lora_A.items():
                    if adapter_name in lora_B:
                        mod_b = lora_B[adapter_name]
                        w_a = mod_a.weight if hasattr(mod_a, "weight") else mod_a
                        w_b = mod_b.weight if hasattr(mod_b, "weight") else mod_b
                        s = float(scaling.get(adapter_name, 1.0)) if isinstance(scaling, dict) else float(scaling)
                        delta = (w_b @ w_a) * s
                        if merged_w.shape == delta.shape:
                            merged_w = merged_w + delta.to(device=merged_w.device, dtype=merged_w.dtype)
                        elif merged_w.ndim == 2 and delta.ndim == 2 and merged_w.shape == delta.t().shape:
                            merged_w = merged_w + delta.t().to(device=merged_w.device, dtype=merged_w.dtype)
            elif isinstance(lora_A, (nn.Module, torch.Tensor)) and isinstance(lora_B, (nn.Module, torch.Tensor)):
                w_a = lora_A.weight if hasattr(lora_A, "weight") else lora_A
                w_b = lora_B.weight if hasattr(lora_B, "weight") else lora_B
                s = float(scaling) if not isinstance(scaling, dict) else 1.0
                delta = (w_b @ w_a) * s
                if merged_w.shape == delta.shape:
                    merged_w = merged_w + delta.to(device=merged_w.device, dtype=merged_w.dtype)
                elif merged_w.ndim == 2 and delta.ndim == 2 and merged_w.shape == delta.t().shape:
                    merged_w = merged_w + delta.t().to(device=merged_w.device, dtype=merged_w.dtype)

        # Handle DoRA (Weight-Decomposed LoRA) magnitude vector if present
        mag_vector = getattr(linear_or_conv, "lora_magnitude_vector", None)
        if mag_vector is not None:
            m = None
            if isinstance(mag_vector, (nn.ParameterDict, nn.ModuleDict, dict)):
                active = getattr(linear_or_conv, "active_adapter", None)
                if isinstance(active, str) and active in mag_vector:
                    m = mag_vector[active]
                elif isinstance(active, list) and active and active[0] in mag_vector:
                    m = mag_vector[active[0]]
                else:
                    for adapter_name in mag_vector:
                        m = mag_vector[adapter_name]
                        break
            elif isinstance(mag_vector, (torch.Tensor, nn.Parameter)):
                m = mag_vector
            if m is not None:
                m_t = m.detach().to(device=merged_w.device, dtype=merged_w.dtype)
                if m_t.numel() == merged_w.shape[0]:
                    v_norm = torch.linalg.norm(merged_w, dim=1, keepdim=True).clamp(min=1e-12)
                    merged_w = m_t.view(-1, 1) * (merged_w / v_norm)
                elif m_t.numel() == merged_w.shape[1]:
                    v_norm = torch.linalg.norm(merged_w, dim=0, keepdim=True).clamp(min=1e-12)
                    merged_w = m_t.view(1, -1) * (merged_w / v_norm)
                logger.info("Applied DoRA magnitude vector normalization to %s", linear_or_conv.__class__.__name__)

        logger.info(
            "Merged LoRA adapter weights for %s projection into base weight (shape %s)",
            linear_or_conv.__class__.__name__,
            tuple(merged_w.shape),
        )
        return merged_w, base_b, adapter_info

    @classmethod
    def _extract_weight_bias(
        cls, linear_or_conv: Union[nn.Module, nn.Parameter, torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extract canonical weight (out_features, in_features) and bias (out_features,) from any projection."""
        if cls._is_lora_module(linear_or_conv):
            w, b, _ = cls._extract_weight_bias_with_lora(linear_or_conv)
            return w, b

        if isinstance(linear_or_conv, (nn.Linear, nn.modules.linear.NonDynamicallyQuantizableLinear)):
            return linear_or_conv.weight, linear_or_conv.bias

        if cls.is_conv1d(linear_or_conv):
            w = linear_or_conv.weight
            b = getattr(linear_or_conv, "bias", None)
            if w.ndim == 2:
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
        selected_prefixes: List[str] = []

        for name, module in model.named_modules():
            if id(module) in visited_ids:
                continue
            if any(name.startswith(p) for p in selected_prefixes):
                continue

            spec = cls._inspect_module(model, name, module)
            if spec is not None:
                targets.append(spec)
                visited_ids.add(id(spec.target_module))
                prefix = name + "." if name else ""
                if prefix:
                    selected_prefixes.append(prefix)
                else:
                    # Root module itself is the target attention block; prune all its descendant submodules
                    break

        return targets

    @classmethod
    def _inspect_module(
        cls, root_model: nn.Module, path: str, module: nn.Module
    ) -> Optional[AttentionTargetSpec]:
        """Inspect a single module and match against universal patterns."""
        cls_name = module.__class__.__name__.lower()

        # If module is wrapped by PEFT/LoRA (e.g. PeftModel or LoraModel wrapper around attention),
        # inspect the unwrapped base attention block while maintaining adapter metadata.
        wrapper_adapters = None
        if cls._is_lora_module(module):
            wrapper_adapters = cls._extract_lora_adapters(module)
            if hasattr(module, "base_layer"):
                module = module.base_layer

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

        if isinstance(module, nn.MultiheadAttention):
            embed_dim = module.embed_dim
            num_heads = module.num_heads
            if num_heads <= 0 or embed_dim % num_heads:
                raise ValueError(
                    f"Invalid nn.MultiheadAttention dimensions at {path!r}: "
                    f"embed_dim={embed_dim}, num_heads={num_heads}"
                )
            head_dim = embed_dim // num_heads
            # ``batch_first`` describes tensor layout, not causal masking.
            # PyTorch MHA receives causality per forward call, so the scanner
            # must not infer it from the module configuration.
            is_causal = False

            in_b = module.in_proj_bias
            out_proj = module.out_proj
            if module.in_proj_weight is not None:
                W_q, W_k, W_v = module.in_proj_weight.chunk(3, dim=0)
                b_q, b_k, b_v = in_b.chunk(3, dim=0) if in_b is not None else (None, None, None)
            else:
                W_q, W_k, W_v = (
                    module.q_proj_weight,
                    module.k_proj_weight,
                    module.v_proj_weight,
                )
                if in_b is not None:
                    b_q, b_k, b_v = in_b.chunk(3, dim=0)
                else:
                    b_q, b_k, b_v = None, None, None
            W_o, b_o = out_proj.weight, out_proj.bias

            extra_meta = {"lora_adapters": {"wrapper": wrapper_adapters}} if wrapper_adapters else None

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
                extra_metadata=extra_meta,
            )

        if hasattr(module, "self") and hasattr(module, "output") and ("bert" in cls_name or "roberta" in cls_name):
            self_attn = module.self
            output_block = module.output
            output_dense = getattr(output_block, "dense", None)
            output_ln = getattr(output_block, "LayerNorm", getattr(output_block, "layer_norm", None))

            q_mod = getattr(self_attn, "query", None)
            k_mod = getattr(self_attn, "key", None)
            v_mod = getattr(self_attn, "value", None)

            if q_mod is not None and k_mod is not None and v_mod is not None and output_dense is not None:
                lora_adapters = {}
                for p_name, p_mod in [("query", q_mod), ("key", k_mod), ("value", v_mod), ("output", output_dense)]:
                    if cls._is_lora_module(p_mod):
                        lora_adapters[p_name] = cls._extract_lora_adapters(p_mod)
                if wrapper_adapters:
                    lora_adapters["wrapper"] = wrapper_adapters

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
                    extra_metadata={"lora_adapters": lora_adapters} if lora_adapters else None,
                )

        if hasattr(module, "c_attn") and hasattr(module, "c_proj"):
            c_attn = module.c_attn
            c_proj = module.c_proj

            lora_adapters = {}
            for p_name, p_mod in [("c_attn", c_attn), ("c_proj", c_proj)]:
                if cls._is_lora_module(p_mod):
                    lora_adapters[p_name] = cls._extract_lora_adapters(p_mod)
            if wrapper_adapters:
                lora_adapters["wrapper"] = wrapper_adapters

            W_attn, b_attn = cls._extract_weight_bias(c_attn)
            W_o, b_o = cls._extract_weight_bias(c_proj)

            embed_dim = W_attn.shape[1]
            total_qkv_out = W_attn.shape[0]
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
            if num_heads <= 0 or split_dim % num_heads:
                raise ValueError(
                    f"Invalid fused QKV dimensions at {path!r}: "
                    f"qkv_width={total_qkv_out}, num_heads={num_heads}"
                )
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
                extra_metadata={"lora_adapters": lora_adapters} if lora_adapters else None,
            )

        qkv_mod = getattr(module, "qkv", getattr(module, "query_key_value", None))
        proj_mod = getattr(module, "proj", getattr(module, "out_proj", getattr(module, "o_proj", None)))

        if qkv_mod is not None and proj_mod is not None:
            lora_adapters = {}
            for p_name, p_mod in [("qkv", qkv_mod), ("proj", proj_mod)]:
                if cls._is_lora_module(p_mod):
                    lora_adapters[p_name] = cls._extract_lora_adapters(p_mod)
            if wrapper_adapters:
                lora_adapters["wrapper"] = wrapper_adapters

            W_qkv, b_qkv = cls._extract_weight_bias(qkv_mod)
            W_o, b_o = cls._extract_weight_bias(proj_mod)

            embed_dim = W_qkv.shape[1]
            num_heads = getattr(module, "num_heads", 12)
            total_out = W_qkv.shape[0]

            if num_heads <= 0:
                raise ValueError(f"Invalid attention head count at {path!r}: {num_heads}")

            explicit_kv_heads = getattr(
                module, "num_key_value_heads", getattr(module, "num_kv_heads", None)
            )
            head_dim = getattr(module, "head_dim", embed_dim // num_heads)
            expected_mha_out = 3 * num_heads * head_dim

            is_symmetric_mha = (
                (explicit_kv_heads is None or explicit_kv_heads == num_heads)
                and (total_out == expected_mha_out or (explicit_kv_heads == num_heads and total_out % 3 == 0))
            )

            if is_symmetric_mha:
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
                q_dim = num_heads * head_dim
                rem_kv = total_out - q_dim
                if rem_kv <= 0 or rem_kv % 2:
                    raise ValueError(
                        f"Cannot split fused QKV projection at {path!r}: "
                        f"projection width {total_out} is incompatible with {num_heads} query heads"
                    )
                kv_dim = rem_kv // 2
                if kv_dim % head_dim:
                    raise ValueError(
                        f"Fused KV width at {path!r} is not divisible by head_dim={head_dim}"
                    )
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
                extra_metadata={"lora_adapters": lora_adapters} if lora_adapters else None,
            )

        q_mod, k_mod, v_mod, o_mod = None, None, None, None

        if hasattr(module, "q_proj") and hasattr(module, "k_proj") and hasattr(module, "v_proj") and hasattr(module, "o_proj"):
            q_mod = module.q_proj
            k_mod = module.k_proj
            v_mod = module.v_proj
            o_mod = module.o_proj
        elif hasattr(module, "q_lin") and hasattr(module, "k_lin") and hasattr(module, "v_lin") and hasattr(module, "out_lin"):
            q_mod = module.q_lin
            k_mod = module.k_lin
            v_mod = module.v_lin
            o_mod = module.out_lin
        elif hasattr(module, "q") and hasattr(module, "k") and hasattr(module, "v") and hasattr(module, "o"):
            q_mod = module.q
            k_mod = module.k
            v_mod = module.v
            o_mod = module.o
        elif hasattr(module, "query") and hasattr(module, "key") and hasattr(module, "value") and hasattr(module, "out_proj"):
            q_mod = module.query
            k_mod = module.key
            v_mod = module.value
            o_mod = module.out_proj

        if q_mod is not None and k_mod is not None and v_mod is not None and o_mod is not None:
            lora_adapters = {}
            for p_name, p_mod in [("q", q_mod), ("k", k_mod), ("v", v_mod), ("o", o_mod)]:
                if cls._is_lora_module(p_mod):
                    lora_adapters[p_name] = cls._extract_lora_adapters(p_mod)
            if wrapper_adapters:
                lora_adapters["wrapper"] = wrapper_adapters

            W_q, b_q = cls._extract_weight_bias(q_mod)
            W_k, b_k = cls._extract_weight_bias(k_mod)
            W_v, b_v = cls._extract_weight_bias(v_mod)
            W_o, b_o = cls._extract_weight_bias(o_mod)

            embed_dim = W_q.shape[1]
            num_heads = getattr(module, "num_heads", getattr(module, "n_heads", getattr(module, "num_attention_heads", 12)))
            if num_heads <= 0:
                raise ValueError(f"Invalid attention head count at {path!r}: {num_heads}")
            head_dim = getattr(module, "head_dim", W_q.shape[0] // num_heads)
            if head_dim <= 0 or W_q.shape[0] % head_dim or W_k.shape[0] % head_dim:
                raise ValueError(
                    f"Incompatible projection dimensions at {path!r}: "
                    f"q_width={W_q.shape[0]}, k_width={W_k.shape[0]}, head_dim={head_dim}"
                )
            num_kv_heads = getattr(module, "num_key_value_heads", getattr(module, "num_kv_heads", W_k.shape[0] // head_dim))
            raw_causal = getattr(module, "is_causal", None)
            if raw_causal is not None:
                is_causal = bool(raw_causal)
            else:
                is_causal = bool(
                    getattr(getattr(module, "config", None), "is_decoder", False)
                    or any(
                        arch in cls_name
                        for arch in (
                            "llama",
                            "mistral",
                            "gemma",
                            "qwen",
                            "gpt",
                            "opt",
                            "phi",
                            "starcoder",
                            "deepseek",
                        )
                    )
                )

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
                extra_metadata={"lora_adapters": lora_adapters} if lora_adapters else None,
            )

        return None


__all__ = [
    "AttentionTargetSpec",
    "Conv1DLinearAdapter",
    "UniversalArchitectureScanner",
]
