"""NOVA attention module – 8 state‑of‑the‑art compression techniques."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

# (Full NOVA code from your message – I'm including a placeholder for brevity;
# in the actual script, this heredoc would contain the entire NOVA implementation.)

# ------------------------------------------------------------
# Framework injection & registration
# ------------------------------------------------------------
from framework.attentions.core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("nova")
def inject_nova(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    """Build NOVA config from CompressionConfig and inject."""
    cfg = NOVAConfig(
        dropout=getattr(comp_config, 'dropout', 0.1),
        armor_enabled=getattr(comp_config, 'armor_enabled', True),
        armor_energy=getattr(comp_config, 'svd_energy', 0.90),
        ast_enabled=True,
        deltakv_enabled=True,
        kvt_enabled=True,
        softpick_enabled=getattr(comp_config, 'softpick_enabled', True),
        subspace_enabled=True,
        subspace_energy=getattr(comp_config, 'svd_energy', 0.95),
    )
    new_attn = NOVAAttention(
        model_config=model_config,
        nova_cfg=cfg,
        layer_idx=layer_idx,
        total_layers=total_layers,
        pretrained_attn=old_attn,
        bert_mode=False,
    )
    layer.attention = new_attn
    return new_attn
