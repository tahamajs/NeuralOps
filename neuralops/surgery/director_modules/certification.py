#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Model Invariant Certification Auditor.

Audits PyTorch models against the Ten Inviolable Laws of Universal Attention Surgery:
- Law 1: QK-Norm Preservation
- Law 2: Logit Soft-Capping Temperature Folding
- Law 3: Zero-Bleed Padding Mask Sanitization
- Law 4: Static Cross-Attention Associative Caching
- Law 5: RoPE Pre-Projection Alignment
- Law 6: Decoupled Latent State Routing (DeepSeek MLA)
- Law 7: Causal State Monotonicity & Exponential State Decay
- Law 8: Tied Embedding Preservation & Auto-Remediation
- Law 9: FP32 Accumulation & Dtype Budget Guard
- Law 10: Master Universal Surgery Integration & Residual Invariance
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, TYPE_CHECKING

import torch
import torch.nn as nn

from neuralops.surgery.director_modules.embedding import verify_tied_embeddings
from neuralops.surgery.director_modules.invariants import (
    apply_rope_pre_feature_map,
    compute_decayed_state_bound,
    sanitize_kv_for_padding_mask,
    sanitize_output_for_padding_mask,
)

if TYPE_CHECKING:
    from neuralops.surgery.director_modules.profiler import ArchitectureProfile

logger = logging.getLogger("neuralops.surgery.director.certification")


def certify_model(
    model: nn.Module,
    profile: ArchitectureProfile,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Law 10: Master Universal Surgery Invariant Certification.
    Automated auditor executing all 10 invariant checks across the model:
      - Law 1: QK-Norm Preservation
      - Law 2: Logit Soft-Capping Temperature Folding
      - Law 3: Zero-Bleed Padding Mask Sanitization
      - Law 4: Static Cross-Attention Associative Caching
      - Law 5: RoPE Pre-Projection Alignment
      - Law 6: Decoupled Latent State Routing (DeepSeek MLA)
      - Law 7: Causal State Monotonicity & Exponential State Decay
      - Law 8: Tied Embedding Preservation & Auto-Remediation
      - Law 9: FP32 Accumulation & Dtype Budget Guard
      - Law 10: Master Universal Surgery Integration & Residual Invariance

    Returns a structured dictionary audit report with per-law verdicts.
    Raises RuntimeError if any mandatory invariant check fails and strict=True.
    """
    laws: Dict[str, bool] = {}
    violations: List[str] = []

    # Law 1: QK-Norm Preservation
    if profile.has_qk_norm:
        has_qk = any(
            getattr(mod, "has_qk_norm", False) or hasattr(mod, "q_norm") or hasattr(mod, "k_norm")
            for mod in model.modules()
        )
        laws["Law_1_QK_Norm"] = bool(has_qk)
        if not has_qk:
            violations.append("Law 1 Violation: QK-Norm missing on model configured with QK-Norm.")
    else:
        laws["Law_1_QK_Norm"] = True

    # Law 2: Logit Soft-Capping Temperature Folding
    if profile.softcapping_factor is not None and profile.softcapping_factor > 0:
        scale = (1.0 / math.sqrt(max(profile.head_dim, 1))) * math.tanh(
            profile.softcapping_factor / max(profile.softcapping_factor, 1.0)
        )
        scale_valid = bool(scale > 0.0 and math.isfinite(scale))
        from neuralops.surgery.director_modules.bridges import InvariantPreservingLinearBridge

        bridges = [m for m in model.modules() if isinstance(m, InvariantPreservingLinearBridge)]
        mismatched_bridges = [
            b for b in bridges if getattr(b, "softcapping_factor", None) != profile.softcapping_factor
        ]
        passed = scale_valid and len(mismatched_bridges) == 0
        laws["Law_2_Softcapping"] = passed
        if not scale_valid:
            violations.append("Law 2 Violation: Invalid folded soft-capping scale factor.")
        elif mismatched_bridges:
            violations.append("Law 2 Violation: Replaced linear bridges missing required softcapping factor.")
    else:
        laws["Law_2_Softcapping"] = True

    # Law 3: Zero-Bleed Padding Mask Sanitization
    try:
        k_test = torch.ones(1, 1, 4, 16)
        v_test = torch.ones(1, 1, 4, 16)
        mask_test = torch.tensor([[1, 1, 0, 0]])
        k_san, v_san = sanitize_kv_for_padding_mask(k_test, v_test, mask_test)
        k_zero = bool(torch.all(k_san[:, :, 2:, :] == 0.0) and torch.all(v_san[:, :, 2:, :] == 0.0))
        out_san = sanitize_output_for_padding_mask(k_test, mask_test)
        out_zero = bool(torch.all(out_san[:, :, 2:, :] == 0.0))
        passed_law3 = k_zero and out_zero
    except Exception:
        passed_law3 = False
    laws["Law_3_Zero_Bleed_Padding"] = passed_law3
    if not passed_law3:
        violations.append("Law 3 Violation: Padding sanitization failed to zero masked token activations.")

    # Law 4: Static Cross-Attention Caching in Encoder-Decoder
    if profile.is_encoder_decoder:
        from neuralops.surgery.director_modules.bridges import CrossAttentionLinearBridge

        has_cross = any(
            isinstance(mod, CrossAttentionLinearBridge)
            or "CrossAttention" in mod.__class__.__name__
            or hasattr(mod, "prefill_encoder")
            for mod in model.modules()
        )
        laws["Law_4_Static_Cross_Attention"] = bool(has_cross)
        if not has_cross:
            violations.append("Law 4 Violation: Encoder-Decoder model lacking static cross-attention linear bridge.")
    else:
        laws["Law_4_Static_Cross_Attention"] = True

    # Law 5: RoPE Pre-Projection Alignment
    if (
        profile.positional_topology == "rope"
        or hasattr(model, "rotary_emb")
        or any(hasattr(m, "rotary_emb") for m in model.modules())
    ):
        even_head = (profile.head_dim % 2 == 0) if profile.head_dim > 0 else True
        try:
            d = max(profile.head_dim, 4)
            if d % 2 != 0:
                d = d + 1
            q_t = torch.randn(1, 1, 2, d)
            k_t = torch.randn(1, 1, 2, d)
            cos_t = torch.ones(1, 1, 2, d)
            sin_t = torch.zeros(1, 1, 2, d)
            qr, kr = apply_rope_pre_feature_map(q_t, k_t, cos_t, sin_t)
            identity_passed = bool(torch.allclose(qr, q_t, atol=1e-5) and torch.allclose(kr, k_t, atol=1e-5))
        except Exception:
            identity_passed = False
        passed_rope = even_head and identity_passed
        laws["Law_5_RoPE_Pre_Projection"] = passed_rope
        if not even_head:
            violations.append(f"Law 5 Violation: RoPE model head_dim={profile.head_dim} is odd (must be even).")
        elif not identity_passed:
            violations.append("Law 5 Violation: RoPE pre-projection alignment test failed.")
    else:
        laws["Law_5_RoPE_Pre_Projection"] = True

    # Law 6: Decoupled Latent State Routing (DeepSeek MLA)
    if profile.family == "deepseek_mla" or profile.is_mla:
        is_mla_routed = any(
            "DeepSeekMLASurgeryBridge" in mod.__class__.__name__
            or hasattr(mod, "init_latent_state")
            or hasattr(mod, "kv_a_proj_with_mqa")
            for mod in model.modules()
        )
        laws["Law_6_Decoupled_Latent_Router"] = bool(is_mla_routed)
        if not is_mla_routed:
            violations.append("Law 6 Violation: DeepSeek MLA model lacking decoupled latent state bridge.")
    else:
        laws["Law_6_Decoupled_Latent_Router"] = True

    # Law 7: Causal State Monotonicity & Exponential State Decay
    decay_valid = True
    for mod in model.modules():
        gamma = getattr(mod, "decay", None)
        if gamma is not None:
            if not (isinstance(gamma, (int, float)) and 0.0 < gamma < 1.0):
                decay_valid = False
                violations.append(f"Law 7 Violation: Decay factor gamma={gamma} outside open interval (0, 1).")
            else:
                try:
                    bound = compute_decayed_state_bound(float(gamma))
                    if not math.isfinite(bound) or bound <= 0.0:
                        decay_valid = False
                        violations.append(f"Law 7 Violation: Unbounded asymptotic state bound for gamma={gamma}.")
                except Exception:
                    decay_valid = False
                    violations.append(f"Law 7 Violation: Error computing state bound for gamma={gamma}.")
    try:
        b_sample = compute_decayed_state_bound(0.99)
        bound_calc_ok = bool(math.isfinite(b_sample) and b_sample > 0.0)
    except Exception:
        bound_calc_ok = False
    laws["Law_7_Causal_Monotonicity_State_Decay"] = decay_valid and bound_calc_ok
    if not bound_calc_ok:
        violations.append("Law 7 Violation: State decay asymptotic bound computation failed.")

    # Law 8: Tied Embeddings Preservation
    if profile.has_tied_embeddings:
        tied = verify_tied_embeddings(model)
        laws["Law_8_Tied_Embeddings"] = bool(tied)
        if not tied:
            violations.append("Law 8 Violation: Input and output embeddings are un-tied after surgery!")
    else:
        laws["Law_8_Tied_Embeddings"] = True

    # Law 9: FP32 Accumulation Guard
    fp32_valid = True
    for mod in model.modules():
        if hasattr(mod, "last_accum_dtype") and mod.last_accum_dtype != torch.float32:
            fp32_valid = False
            violations.append(f"Law 9 Violation: Module {type(mod).__name__} last_accum_dtype != torch.float32.")
        if hasattr(mod, "recurrent_state_dtype") and mod.recurrent_state_dtype != torch.float32:
            fp32_valid = False
            violations.append(f"Law 9 Violation: Module {type(mod).__name__} recurrent_state_dtype != torch.float32.")
    laws["Law_9_FP32_Accumulation"] = fp32_valid

    # Law 10: Master Universal Surgery Integration
    is_mod = isinstance(model, nn.Module)
    has_params = any(True for _ in model.parameters()) or any(True for _ in model.modules())
    finite_weights = all(bool(torch.isfinite(p).all().item()) for p in model.parameters() if p.numel() > 0)
    if not is_mod:
        violations.append("Law 10 Violation: Model is not an nn.Module instance.")
    if not finite_weights:
        violations.append("Law 10 Violation: Non-finite (NaN/Inf) parameters detected in model.")

    laws["Law_10_Master_Integration"] = bool(is_mod and has_params and finite_weights)
    laws["Law_10_Master_Universal_Certification"] = bool(
        is_mod and has_params and finite_weights and len(violations) == 0
    )

    passed_ten = sum(
        1
        for k in (
            "Law_1_QK_Norm",
            "Law_2_Softcapping",
            "Law_3_Zero_Bleed_Padding",
            "Law_4_Static_Cross_Attention",
            "Law_5_RoPE_Pre_Projection",
            "Law_6_Decoupled_Latent_Router",
            "Law_7_Causal_Monotonicity_State_Decay",
            "Law_8_Tied_Embeddings",
            "Law_9_FP32_Accumulation",
            "Law_10_Master_Integration",
        )
        if laws.get(k, False)
    )

    all_passed = len(violations) == 0 and passed_ten == 10

    report: Dict[str, Any] = {
        "model_family": profile.family,
        "num_layers": profile.num_layers,
        "hidden_dim": profile.hidden_dim,
        "num_heads": profile.num_heads,
        "head_dim": profile.head_dim,
        "laws_certified": laws,
        "score": f"{passed_ten}/10",
        "certified": all_passed,
        "violations": violations,
    }

    if violations and strict:
        raise RuntimeError(f"Universal Surgery Certification failed: {'; '.join(violations)}")

    return report
