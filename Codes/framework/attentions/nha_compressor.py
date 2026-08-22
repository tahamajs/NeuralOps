"""
Native Hybrid Attention (NHA) Compressor.
"""
from typing import Any, Optional, Tuple
import torch
import torch.nn as nn
from framework.attentions.nha_attention import NativeHybridAttention

# Alias NHACompressor to NativeHybridAttention for registry and direct instantiation
NHACompressor = NativeHybridAttention
