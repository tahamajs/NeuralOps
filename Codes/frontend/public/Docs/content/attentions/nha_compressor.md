# NHA Compressor

## 1. Architecture & Functionality
Clean alias and wrapper module referencing Native Hybrid Attention (NHA) for sequence-level compression without registry collisions.

## 2. Empirical Performance
Matches Native Hybrid Attention at 88.9% Acc on SST-2.

---

## Complete Source Code Implementation (`nha_compressor.py`)

```python
"""
Native Hybrid Attention (NHA) Compressor.
"""
from typing import Any, Optional, Tuple
import torch
import torch.nn as nn
from framework.attentions.nha_attention import NativeHybridAttention

# Alias NHACompressor to NativeHybridAttention for registry and direct instantiation
NHACompressor = NativeHybridAttention
```
