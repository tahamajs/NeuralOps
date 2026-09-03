"""
Native Hybrid Attention (NHA) Compressor.
"""
from typing import Any, Optional, Tuple
import torch
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
import torch.nn as nn
from framework.attentions.nha_attention import NativeHybridAttention

# Alias NHACompressor to NativeHybridAttention for registry and direct instantiation
NHACompressor = NativeHybridAttention
