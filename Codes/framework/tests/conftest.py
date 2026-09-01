"""
Shared pytest fixtures and helpers for attention module testing.
Auto‑discovers all registered attention variants and provides
standard test inputs (hidden states, masks, configs).
"""

from typing import List

import pytest
import torch
from transformers import AutoModel, DistilBertConfig

import os
import sys

_FRAMEWORK_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _FRAMEWORK_PARENT not in sys.path:
    sys.path.insert(0, _FRAMEWORK_PARENT)

# ----------------------------------------------------------------------
# Ensure the framework registries are populated before tests run.
# Import the top‑level framework so that all @ATTENTION_REGISTRY.register
# decorators are executed.
# ----------------------------------------------------------------------
import framework
import framework.attentions
from framework.core.registry import ATTENTION_REGISTRY


# ----------------------------------------------------------------------
# Fixture: standard DistilBERT config
# ----------------------------------------------------------------------
@pytest.fixture(scope="session")
def model_config():
    """Return a standard DistilBertConfig (768 hidden, 12 heads)."""
    return DistilBertConfig(
        hidden_size=768,
        num_attention_heads=12,
        dropout=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
    )


# ----------------------------------------------------------------------
# Fixture: batch of random hidden states
# ----------------------------------------------------------------------
@pytest.fixture(scope="session")
def hidden_states():
    """Return a random (B=2, L=32, D=768) tensor, reproducible."""
    torch.manual_seed(42)
    return torch.randn(2, 32, 768)


# ----------------------------------------------------------------------
# Fixture: standard 2D binary attention mask
# ----------------------------------------------------------------------
@pytest.fixture(scope="session")
def attention_mask():
    """Return a 2D binary mask (B=2, L=32). First 24 tokens valid, last 8 padding."""
    mask = torch.ones(2, 32)
    mask[:, 24:] = 0
    return mask


# ----------------------------------------------------------------------
# Fixture: small hidden states for quick tests
# ----------------------------------------------------------------------
@pytest.fixture(scope="session")
def small_hidden_states():
    """Return a small (B=2, L=8, D=768) tensor."""
    torch.manual_seed(123)
    return torch.randn(2, 8, 768)


# ----------------------------------------------------------------------
# Fixture: small attention mask
# ----------------------------------------------------------------------
@pytest.fixture(scope="session")
def small_attention_mask():
    """Return a small 2D mask (B=2, L=8). Last 2 tokens padding."""
    mask = torch.ones(2, 8)
    mask[:, 6:] = 0
    return mask


# ----------------------------------------------------------------------
# Fixture: list of all registered attention keys
# ----------------------------------------------------------------------
@pytest.fixture(scope="session")
def registered_attention_keys() -> List[str]:
    """Return all keys currently in ATTENTION_REGISTRY."""
    if ATTENTION_REGISTRY is None:
        return []
    return sorted(ATTENTION_REGISTRY.keys())


# ----------------------------------------------------------------------
# Fixture: original DistilBERT attention module (for zero‑shot tests)
# ----------------------------------------------------------------------
@pytest.fixture(scope="session")
def distilbert_attention(model_config):
    """
    Load a real DistilBERT model or initialized DistilBertModel and return its first layer's attention module.
    This gives us the q_lin, k_lin, v_lin, out_lin weights for zero-shot testing.
    """
    from transformers import DistilBertModel
    model = DistilBertModel(model_config)
    return model.transformer.layer[0].attention


# ----------------------------------------------------------------------
# Utility: instantiate an attention module by registry key
# ----------------------------------------------------------------------
def build_attention(key: str, config, pretrained_attn=None):
    """
    Build a single attention instance from the registry.
    Handles modules that require or don't require pretrained_attn.
    """
    cls = ATTENTION_REGISTRY.get(key)
    if cls is None:
        raise ValueError(f"Unknown attention type: {key}")

    # If it's a factory / injection function hook
    if callable(cls) and not (isinstance(cls, type) and issubclass(cls, nn.Module)):
        dummy_layer = nn.Module()
        dummy_layer.attention = pretrained_attn
        class CompConfig:
            rank = 32
        try:
            res = cls(layer=dummy_layer, model_config=config, comp_config=CompConfig(), layer_idx=0, total_layers=6, old_attn=pretrained_attn)
            if isinstance(res, nn.Module):
                return res
        except Exception:
            pass

    for args, kwargs in [
        ((), {"config": config, "rank": 32, "pretrained_attn": pretrained_attn}),
        ((), {"config": config, "rank": 32}),
        ((), {"hidden_size": getattr(config, "hidden_size", 768), "rank": 32}),
        ((), {"d_model": getattr(config, "hidden_size", 768), "rank": 32}),
        ((config,), {}),
        ((), {"config": config}),
    ]:
        try:
            inst = cls(*args, **kwargs)
            if isinstance(inst, nn.Module):
                return inst
        except Exception:
            continue

    return cls(config)


# ----------------------------------------------------------------------
# Utility: collect actual attention types that can be instantiated
# ----------------------------------------------------------------------
def get_testable_attention_keys(config, pretrained_attn=None):
    """Return the subset of registry keys whose modules can be instantiated."""
    testable = []
    for key in ATTENTION_REGISTRY.keys():
        if key.endswith("_inject") or key.startswith("inject_"):
            continue
        try:
            inst = build_attention(key, config, pretrained_attn)
            if isinstance(inst, nn.Module):
                testable.append(key)
        except Exception:
            continue
    return testable
