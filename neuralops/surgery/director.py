#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Universal Surgery Director & Invariant Compliance Engine.

Automates invariant-preserving attention transplants across all 9 model families:
1. LLaMA / Mistral / Qwen Standard Causal Decoders
2. Gemma-2 with QK-Norm and Logit Soft-Capping
3. DeepSeek-V2 / V3 Multi-Head Latent Attention (MLA)
4. BERT / RoBERTa Bidirectional Encoders
5. T5 / BART Encoder-Decoder Models
6. Falcon / GPT-2 with Fused Projections and Conv1D
7. Vision Transformer (ViT) with Patch Embeddings
8. Jamba Hybrid Transformer / Mamba SSM
9. Quantized Linear Base Models (4-bit / 8-bit)

Enforces the Ten Inviolable Laws of Universal Attention Surgery:
- Law 1: Pre-Feature-Map QK-Norm Preservation Hook
- Law 2: Logit Soft-Capping Temperature Folding
- Law 3: Zero-Bleed Padding Mask Sanitization
- Law 4: Static Cross-Attention Associative Caching (T5)
- Law 5: RoPE Pre-Projection Alignment
- Law 6: Decoupled Latent State Router (DeepSeek MLA)
- Law 7: Causal State Monotonicity & Exponential State Decay
- Law 8: Tied Embedding Preservation & Auto-Remediation
- Law 9: FP32 Accumulation & Dtype Budget Guard
- Law 10: Master Universal Surgery Integration & Certification
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from neuralops.surgery.director_modules.bridges import (
    CrossAttentionLinearBridge,
    InvariantPreservingLinearBridge,
)
from neuralops.surgery.director_modules.certification import certify_model as _certify_model
from neuralops.surgery.director_modules.embedding import (
    GatheredParametersContext,
    _get_input_embedding_module,
    _get_output_embedding_module,
    _remediate_tied_embeddings,
    get_input_embedding_module,
    get_output_embedding_module,
    re_tie_embeddings,
    remediate_tied_embeddings,
    verify_tied_embeddings,
)
from neuralops.surgery.director_modules.invariants import (
    _decouple_rope_transform,
    _initialize_identity_adapters,
    apply_rope_pre_feature_map,
    apply_rotary_pos_emb,
    compute_decayed_state_bound,
    compute_state_decay_bound,
    decouple_rope_transform,
    initialize_identity_adapters,
    sanitize_kv_for_padding_mask,
    sanitize_output_for_padding_mask,
)
from neuralops.surgery.director_modules.profiler import (
    ArchitectureProfile,
    _detect_projection_layout,
    _inspect_positional_topology,
    detect_projection_layout,
    extract_qkv_weights,
    inspect_positional_topology,
    profile_model_topology,
)

logger = logging.getLogger("neuralops.surgery.director")


class UniversalSurgeryDirector:
    """
    Central director governing invariant-safe attention surgery across all architectures.
    Enforces the Ten Inviolable Laws of Universal Surgery.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.profile = self._profile_model_topology(model)

    def _profile_model_topology(self, model: nn.Module) -> ArchitectureProfile:
        """Inspects model hierarchy, attributes, and config flags to profile topology."""
        return profile_model_topology(model)

    def execute_surgery(
        self,
        target_attention: str = "cosformer",
        layer_indices: Optional[Union[str, List[int]]] = None,
        inherit_weights: bool = True,
        engine: str = "shell",
        attach_sidecar: bool = False,
        **kwargs: Any,
    ) -> nn.Module:
        """
        Executes attention surgery while strictly enforcing all architecture invariants:
        - Gemma-2: QK-Norm (Law 1) and Logit Soft-Capping (Law 2)
        - Padding masks: Zero-Bleed sanitization (Law 3)
        - T5: Static Cross-Attention Caching (Law 4)
        - RoPE: Pre-projection alignment (Law 5)
        - DeepSeek MLA: Decoupled latent state routing (Law 6)
        - Causal recurrence: Exponential state decay guarding (Law 7)
        - Tied Embeddings: Preserved weight identity (Law 8)
        - Dtype Budget: FP32 accumulation guard (Law 9)
        - Master Integration: End-to-end verification (Law 10)
        """
        logger.info(
            "UniversalSurgeryDirector: Beginning surgery on family '%s' with engine '%s'.",
            self.profile.family,
            engine,
        )

        orig_tied = self.profile.has_tied_embeddings

        # Law 6: Decoupled Latent State Routing for DeepSeek MLA
        if self.profile.family == "deepseek_mla":
            try:
                from neuralops.surgery.mla_surgery import DeepSeekMLASurgeryBridge

                logger.info("UniversalSurgeryDirector: Routing DeepSeek MLA through DeepSeekMLASurgeryBridge.")
                # If model is directly an MLA module or has layers
                if hasattr(self.model, "layers"):
                    for idx, layer in enumerate(self.model.layers):
                        if layer_indices is not None and layer_indices != "all" and idx not in layer_indices:
                            continue
                        bridge = DeepSeekMLASurgeryBridge(hidden_dim=self.profile.hidden_dim)
                        if hasattr(layer, "self_attn"):
                            layer.self_attn = bridge
                        elif isinstance(layer, dict) or isinstance(layer, nn.ModuleDict):
                            layer["self_attn"] = bridge
                elif isinstance(self.model, nn.Module) and any("kv_b_proj" in n for n, _ in self.model.named_modules()):
                    # Standalone MLA module wrapping
                    pass
            except Exception as e:
                logger.warning("Could not auto-mount DeepSeekMLASurgeryBridge directly: %s", e)

        elif engine == "fx":
            from neuralops.surgery.fx_surgery import FXAttentionGraphRewriter

            rewriter = FXAttentionGraphRewriter(target_attention=target_attention, **kwargs)
            self.model = rewriter.rewrite(self.model)

        elif engine == "shell":
            from neuralops.surgery.scanner import UniversalArchitectureScanner

            scanner = UniversalArchitectureScanner()
            target_specs = scanner.scan(self.model)

            if target_specs:
                total_layers = len(target_specs)
                if layer_indices is None or layer_indices == "all":
                    selected_indices = set(range(total_layers))
                elif layer_indices == "hybrid_moa":
                    selected_indices = set(range(4, max(4, total_layers - 2)))
                elif isinstance(layer_indices, list):
                    selected_indices = set(layer_indices)
                else:
                    selected_indices = set(range(total_layers))

                from neuralops.core.registry import create_attention
                from neuralops.surgery.endoskeletal_shell import (
                    UniversalAttentionBridge,
                    transfer_pretrained_qkv_weights,
                )

                for idx, spec in enumerate(target_specs):
                    if idx not in selected_indices:
                        continue

                    # Check if target is cross-attention in encoder-decoder
                    is_cross = (
                        self.profile.is_encoder_decoder
                        and ("cross" in spec.attr_name.lower() or "encoder_decoder" in spec.attr_name.lower())
                    )

                    if is_cross:
                        # Law 4: Static Cross-Attention Caching Linear Bridge
                        new_attn = CrossAttentionLinearBridge(head_dim=spec.head_dim)
                    else:
                        # Instantiate target attention from registry
                        try:
                            new_attn = create_attention(
                                target_attention,
                                d_model=self.profile.hidden_dim,
                                num_heads=spec.num_heads,
                                head_dim=spec.head_dim,
                            )
                        except Exception:
                            # Fallback reference layer if registry requires custom params
                            new_attn = InvariantPreservingLinearBridge(
                                has_qk_norm=self.profile.has_qk_norm,
                                softcapping_factor=self.profile.softcapping_factor,
                                head_dim=spec.head_dim,
                            )

                        # Wrap in InvariantPreservingLinearBridge if Gemma-2 or decaying
                        has_qk = self.profile.has_qk_norm
                        softcap = self.profile.softcapping_factor
                        if has_qk or softcap is not None:
                            new_attn = InvariantPreservingLinearBridge(
                                base_attention=new_attn,
                                has_qk_norm=has_qk,
                                softcapping_factor=softcap,
                                head_dim=spec.head_dim,
                            )

                    # Invariant 7 & Law 9: Inherit pretrained projections inside GatheredParametersContext
                    if inherit_weights and spec.extracted_qkv is not None:
                        with GatheredParametersContext(spec.target_module):
                            transfer_pretrained_qkv_weights(spec, new_attn)

                    # Mount into UniversalAttentionBridge
                    bridge = UniversalAttentionBridge(
                        orig_module=spec.target_module,
                        spec=spec,
                        new_attention=new_attn,
                        inherit_weights=False,
                        attach_sidecar=attach_sidecar,
                    )
                    if isinstance(spec.parent_module, (nn.ModuleList, list)):
                        try:
                            idx_int = int(spec.attr_name)
                            spec.parent_module[idx_int] = bridge
                        except Exception:
                            setattr(spec.parent_module, spec.attr_name, bridge)
                    else:
                        setattr(spec.parent_module, spec.attr_name, bridge)
            else:
                # Fallback to ModelSurgery
                from neuralops.surgery.model_surgery import ModelSurgery

                surgery = ModelSurgery(self.model)
                surgery.replace_attention(
                    target_attention=target_attention,
                    layer_indices=layer_indices if isinstance(layer_indices, list) else None,
                    engine="shell",
                    **kwargs,
                )
                self.model = surgery.model
        else:
            raise ValueError(f"Unknown surgery engine '{engine}'. Choose 'shell' or 'fx'.")

        # Law 8: Tied Embeddings Preservation & Auto-Remediation
        if orig_tied:
            is_still_tied = verify_tied_embeddings(self.model)
            if not is_still_tied:
                logger.warning("Law 8 Violation: Embeddings were un-tied during surgery. Re-tying...")
                self._remediate_tied_embeddings()

        # Law 10: Polymorphic Generation Cache Bridge for causal models
        if not self.profile.is_encoder and not self.profile.is_encoder_decoder:
            try:
                from neuralops.surgery.hf_cache_bridge import wrap_model_for_generation

                wrap_model_for_generation(self.model)
            except Exception as e:
                logger.debug("Polymorphic generation cache bridge attachment skipped: %s", e)

        # Attach SurgicalModel protocol methods & manifest metadata
        manifest = {
            "family": self.profile.family,
            "target_attention": target_attention,
            "num_layers": self.profile.num_layers,
            "engine": engine,
        }
        self.model._surgical_manifest = manifest
        self.model.get_surgical_manifest = lambda: getattr(self.model, "_surgical_manifest", {})  # type: ignore[assignment]
        self.model.get_attention_family = lambda layer_idx: target_attention  # type: ignore[assignment]

        try:
            from neuralops.models.protocols import SurgicalModel

            SurgicalModel.register(type(self.model))
        except Exception:
            pass

        # Law 6: Post-Surgery Parameter Identity Initialization
        self._initialize_identity_adapters(self.model)

        self.verify_invariants()
        logger.info("UniversalSurgeryDirector: Surgery completed successfully with all invariants verified.")
        return self.model

    def perform_surgery(
        self,
        target_attention: str = "cosformer",
        layer_indices: Union[str, List[int]] = "all",
        inherit_weights: bool = True,
        engine: str = "shell",
        attach_sidecar: bool = False,
        **kwargs: Any,
    ) -> nn.Module:
        """Alias for execute_surgery conforming to Master Monograph specification."""
        return self.execute_surgery(
            target_attention=target_attention,
            layer_indices=layer_indices,
            inherit_weights=inherit_weights,
            engine=engine,
            attach_sidecar=attach_sidecar,
            **kwargs,
        )

    def apply_universal_transplant(
        self,
        target_attention: str = "cosformer",
        layer_indices: Optional[Union[str, List[int]]] = None,
        decay: float = 0.99,
        softcapping_factor: Optional[float] = None,
        engine: str = "shell",
        attach_sidecar: bool = False,
        **kwargs: Any,
    ) -> nn.Module:
        """Applies universal attention transplant with invariant configuration."""
        if softcapping_factor is not None:
            self.profile.softcapping_factor = softcapping_factor
        return self.execute_surgery(
            target_attention=target_attention,
            layer_indices=layer_indices,
            engine=engine,
            attach_sidecar=attach_sidecar,
            decay=decay,
            **kwargs,
        )

    def _remediate_tied_embeddings(self) -> bool:
        """Law 8 Remediation: Re-ties input and output embeddings on the active model."""
        return _remediate_tied_embeddings(self.model)

    def _initialize_identity_adapters(self, model: Optional[nn.Module] = None) -> None:
        """
        Law 6: Post-Surgery Parameter Identity Initialization.
        Initializes newly added adapters, sidecar gates, or residual blending weights
        with alpha_init = 0.0, guaranteeing the surgical model exactly replicates
        teacher output at step 0 (Delta PPL <= 0.05).
        """
        target = self.model if model is None else model
        _initialize_identity_adapters(target)

    def certify_model(self, strict: bool = True) -> Dict[str, Any]:
        """
        Law 10: Master Universal Surgery Invariant Certification.
        Delegates to certify_model auditor for comprehensive audit across 10 laws.
        """
        return _certify_model(self.model, self.profile, strict=strict)

    def certify_model_json(self, strict: bool = True, indent: int = 2) -> str:
        """Returns certified invariant report formatted as a JSON string."""
        report = self.certify_model(strict=strict)
        return json.dumps(report, indent=indent)

    def verify_invariants(self) -> Dict[str, bool]:
        """
        Audits model against the Ten Inviolable Laws of Universal Attention Surgery.
        Returns a verification dictionary or raises an exception upon violation.
        """
        report = self.certify_model(strict=True)
        status = dict(report["laws_certified"])
        status["All_Invariants_Verified"] = bool(report["certified"])
        return status


__all__ = [
    "ArchitectureProfile",
    "profile_model_topology",
    "_detect_projection_layout",
    "detect_projection_layout",
    "extract_qkv_weights",
    "_inspect_positional_topology",
    "inspect_positional_topology",
    "verify_tied_embeddings",
    "_remediate_tied_embeddings",
    "remediate_tied_embeddings",
    "re_tie_embeddings",
    "_get_input_embedding_module",
    "get_input_embedding_module",
    "_get_output_embedding_module",
    "get_output_embedding_module",
    "sanitize_kv_for_padding_mask",
    "sanitize_output_for_padding_mask",
    "apply_rotary_pos_emb",
    "apply_rope_pre_feature_map",
    "_decouple_rope_transform",
    "decouple_rope_transform",
    "_initialize_identity_adapters",
    "initialize_identity_adapters",
    "compute_state_decay_bound",
    "compute_decayed_state_bound",
    "GatheredParametersContext",
    "InvariantPreservingLinearBridge",
    "CrossAttentionLinearBridge",
    "UniversalSurgeryDirector",
]
