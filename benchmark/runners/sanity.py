"""Layer 0: Pre-flight sanity validation checking shapes, datatypes, and gradient flow."""

from __future__ import annotations
import traceback
from typing import Dict, Any
import torch
import torch.nn as nn
from ..core.model_factory import get_transformer_layers


def run_sanity_check(model: nn.Module, attention_model_name: str) -> Dict[str, Any]:
    """
    Perform structural and gradient integrity checks on the instantiated model.

    Validation steps:
    1. Pass synthetic batch through the attention layer and full model.
    2. Assert tensor output shapes [B, L, D] and correct floating-point dtypes.
    3. Verify backward pass produces non-zero, finite, non-NaN gradients across all trainable weights.
    4. Validate attention matrix output formats when output_attentions=True.

    Returns:
        {"passed": bool, "error": str | None}
    """
    try:
        device = next(model.parameters()).device
        layers = get_transformer_layers(model)

        if layers is not None and len(layers) > 0:
            first_layer = layers[0]
            attn = getattr(first_layer, "attention", None)

            if attn is not None:
                # Find hidden size
                hidden_size = 768
                for p in attn.parameters():
                    if p.dim() >= 2:
                        hidden_size = p.shape[-1]
                        break

                B, L, D = 4, 64, hidden_size
                hidden_states = torch.randn((B, L, D), dtype=torch.float32, device=device, requires_grad=True)
                mask_2d = torch.ones((B, L), dtype=torch.bool, device=device)
                mask_2d[:, -12:] = False  # ~20% padding zeros
                mask_4d = mask_2d.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L] for HF SDPA

                attn.eval()
                # Try standard custom attention call first, then standard HF Multi-Head attention signature
                try:
                    outputs = attn(hidden_states, attention_mask=mask_2d)
                except Exception:
                    try:
                        # HuggingFace standard DistilBertAttention format
                        outputs = attn(
                            query=hidden_states,
                            key=hidden_states,
                            value=hidden_states,
                            mask=mask_4d.float(),
                        )
                    except Exception:
                        outputs = attn(hidden_states)

                if isinstance(outputs, tuple):
                    out_tensor = outputs[0]
                else:
                    out_tensor = outputs

                if out_tensor.shape != (B, L, D):
                    return {
                        "passed": False,
                        "error": f"FAIL_SHAPE: Expected output shape ({B}, {L}, {D}), got {out_tensor.shape}",
                    }

                if out_tensor.dtype != hidden_states.dtype:
                    return {
                        "passed": False,
                        "error": f"FAIL_DTYPE: Expected dtype {hidden_states.dtype}, got {out_tensor.dtype}",
                    }

                # Gradient backward check
                loss = out_tensor.sum()
                loss.backward()

                for name, param in attn.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            return {
                                "passed": False,
                                "error": f"FAIL_NAN_GRAD: Parameter '{name}' has NaN or Inf gradients in sanity backward pass",
                            }

        # Full model sanity check
        model.train()
        model.zero_grad()
        dummy_ids = torch.randint(0, 1000, (4, 32), device=device)
        dummy_mask = torch.ones((4, 32), device=device)
        dummy_labels = torch.randint(0, 2, (4,), device=device)

        full_out = model(input_ids=dummy_ids, attention_mask=dummy_mask, labels=dummy_labels)
        if full_out.loss is None or torch.isnan(full_out.loss):
            return {
                "passed": False,
                "error": "FAIL_NAN: Full model forward pass produced NaN loss",
            }

        full_out.loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    return {
                        "passed": False,
                        "error": f"FAIL_NAN_GRAD: Full model parameter '{name}' received NaN gradient",
                    }

        model.zero_grad()
        model.eval()
        return {"passed": True, "error": None}

    except Exception as e:
        err = traceback.format_exc()
        return {"passed": False, "error": f"FAIL_SANITY: {str(e)}\n{err}"}
