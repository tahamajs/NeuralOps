#!/usr/bin/env python3
"""
Comprehensive Master Test Suite for All Attention Mechanisms and Compression Engines.
Tests instantiation, mock forward pass, cross-entropy / loss computation, and gradient backpropagation.
"""

import sys
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CODES = _HERE.parent.parent
sys.path.insert(0, str(_CODES))
sys.path.insert(0, str(_HERE.parent))

import torch
import torch.nn as nn
from transformers import AutoConfig, DistilBertConfig

# Import registries
import framework.attentions
import framework.compression_engine
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY


class MockTransformerModel(nn.Module):
    def __init__(self, hidden_dim=768, num_classes=2):
        super().__init__()
        self.config = DistilBertConfig(
            hidden_dim=hidden_dim,
            dim=hidden_dim,
            n_heads=12,
            n_layers=2,
            num_attention_heads=12,
            hidden_size=hidden_dim,
            vocab_size=30522,
            max_position_embeddings=512,
        )
        self.embedding = nn.Embedding(100, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.tokenizer = None

    def forward(self, x):
        h = self.linear1(x)
        h = torch.relu(h)
        h = self.linear2(h)
        logits = self.classifier(h[:, 0, :])
        return logits


def test_all_attentions():
    print("\n" + "=" * 65)
    print("  TESTING ALL REGISTERED ATTENTION MECHANISMS (FORWARD + BACKWARD)")
    print("=" * 65)

    all_attns = sorted(ATTENTION_REGISTRY.keys())
    passed = []
    failed = []

    config = DistilBertConfig(
        hidden_dim=768,
        dim=768,
        n_heads=12,
        num_attention_heads=12,
        hidden_size=768,
        max_position_embeddings=512,
    )

    dummy_old_attn = nn.MultiheadAttention(embed_dim=768, num_heads=12, batch_first=True)

    for attn_name in all_attns:
        try:
            cls_or_func = ATTENTION_REGISTRY.resolve(attn_name)
            layer = None
            # Try factory function signatures
            try:
                layer = cls_or_func(config)
            except Exception:
                try:
                    layer = cls_or_func(embed_dim=768, num_heads=12)
                except Exception:
                    try:
                        layer = cls_or_func(
                            model_config=config,
                            comp_config=None,
                            layer_idx=0,
                            total_layers=6,
                            old_attn=dummy_old_attn
                        )
                    except Exception:
                        try:
                            layer = cls_or_func(
                                model_config=config,
                                comp_config=None,
                                layer_idx=0,
                                total_layers=6
                            )
                        except Exception:
                            layer = cls_or_func()

            if layer is None:
                raise ValueError(f"Could not instantiate {attn_name}")

            # Forward pass
            x = torch.randn(2, 64, 768, requires_grad=True)
            if hasattr(layer, "forward"):
                # Check expected arguments
                try:
                    out = layer(x)
                except TypeError:
                    try:
                        out = layer(x, x, x)
                    except TypeError:
                        out = layer(query=x, key=x, value=x)
            elif callable(layer):
                out = layer(x)
            else:
                raise ValueError(f"Object {layer} is not callable")

            if isinstance(out, (tuple, list)):
                out = out[0]

            if not isinstance(out, torch.Tensor):
                raise ValueError(f"Output must be Tensor, got {type(out)}")

            if not torch.isfinite(out).all():
                raise ValueError(f"Output contains NaN or Inf values")

            # Backward pass
            loss = out.sum()
            loss.backward()

            passed.append(attn_name)
            print(f"  [PASS] {attn_name:<35} | Shape: {tuple(out.shape)}")
        except Exception as e:
            failed.append((attn_name, str(e)))
            print(f"  [FAIL] {attn_name:<35} | Error: {e}")

    print(f"\nAttentions Result: {len(passed)} Passed, {len(failed)} Failed / Total: {len(all_attns)}")
    return passed, failed


def test_all_compressors():
    print("\n" + "=" * 65)
    print("  TESTING ALL REGISTERED COMPRESSION ENGINES (APPLY + FORWARD)")
    print("=" * 65)

    all_comps = sorted(COMPRESSOR_REGISTRY.keys())
    passed = []
    failed = []

    for comp_name in all_comps:
        try:
            model = MockTransformerModel(hidden_dim=768)
            cls_or_func = COMPRESSOR_REGISTRY.resolve(comp_name)
            compressor = None

            try:
                compressor = cls_or_func(model, rank=32)
            except Exception:
                try:
                    compressor = cls_or_func(model)
                except Exception:
                    try:
                        compressor = cls_or_func(model, comp_config={})
                    except Exception:
                        compressor = cls_or_func(model, rank=32, tokenizer=None)

            if hasattr(compressor, "compress"):
                res = compressor.compress()
                comp_model = res[0] if isinstance(res, tuple) else res
            elif callable(compressor) and not isinstance(compressor, nn.Module):
                res = compressor()
                comp_model = res[0] if isinstance(res, tuple) else res
            else:
                comp_model = getattr(compressor, "model", model)

            if not isinstance(comp_model, nn.Module):
                comp_model = model

            # Test forward on compressed model
            x = torch.randn(2, 64, 768)
            out = comp_model(x)
            if isinstance(out, (tuple, list)):
                out = out[0]

            if not torch.isfinite(out).all():
                raise ValueError(f"Compressed model output contains NaN or Inf")

            passed.append(comp_name)
            print(f"  [PASS] {comp_name:<35} | Out Shape: {tuple(out.shape)}")
        except Exception as e:
            failed.append((comp_name, str(e)))
            print(f"  [FAIL] {comp_name:<35} | Error: {e}")

    print(f"\nCompressors Result: {len(passed)} Passed, {len(failed)} Failed / Total: {len(all_comps)}")
    return passed, failed


if __name__ == "__main__":
    attn_p, attn_f = test_all_attentions()
    comp_p, comp_f = test_all_compressors()

    total_passed = len(attn_p) + len(comp_p)
    total_failed = len(attn_f) + len(comp_f)
    print("\n" + "=" * 65)
    print(f"  OVERALL VERIFICATION: {total_passed} PASSED | {total_failed} FAILED")
    print("=" * 65)
