# 🧩 Auxiliary Components for Hybrid Attention‑SSM Blocks

This package provides three core building blocks used by the hybrid attention‑state‑space model (SSM) layers in the Efficient Transformer Toolkit. All components are registered in `AUX_REGISTRY` (located in `framework.attentions.core.registry`) and can be dynamically instantiated or imported directly.

## 📁 Files

| File | Description |
|------|-------------|
| `unified_rope.py` | Rotary Position Embedding (RoPE) – adds positional information via rotation. |
| `simple_ssm.py`   | A lightweight State‑Space Model (SSM) block with convolution and recurrent scan. |
| `gated_delta_net.py` | Recurrent Gated DeltaNet layer with learnable forget gate. |

## 🚀 Quick Start

```python
from framework.attentions.ssm import UnifiedRoPE, SimpleSSM, GatedDeltaNetLayer
from framework.attentions.core.registry import AUX_REGISTRY

# Direct import
rope = UnifiedRoPE(dim=768)

# Registry lookup
SSMClass = AUX_REGISTRY.get("simple_ssm")
ssm = SSMClass(d_model=768)
```

## 📦 Component Details

### 1. `UnifiedRoPE`

Rotary Position Embedding (RoPE) as described in the RoFormer paper. It pre‑computes cosine and sine tables for a given maximum sequence length and applies rotation to the input tensor.

**Initialisation**:
```python
UnifiedRoPE(dim, max_seq_len=4096, base=10000.0)
```

**Forward signature**:
```python
rope(x, offset=0) -> torch.Tensor
```

**Usage example**:
```python
rope = UnifiedRoPE(dim=768)
hidden_states = rope(hidden_states)   # shape unchanged
```

### 2. `SimpleSSM`

A lightweight SSM block inspired by Mamba and S4. It applies a 1D convolution, projects to an inner dimension, discretises a diagonal state matrix, and runs a recurrent scan.

**Initialisation**:
```python
SimpleSSM(d_model, d_state=16, d_conv=4, expand=2)
```

**Forward signature**:
```python
ssm(x) -> torch.Tensor    # x: (batch, seq_len, d_model)
```

**Usage example**:
```python
ssm = SimpleSSM(d_model=768)
out = ssm(hidden_states)
```

### 3. `GatedDeltaNetLayer`

Recurrent Gated DeltaNet layer with forget and input gates. Can be used as a long‑range memory component in hybrid blocks.

**Initialisation**:
```python
GatedDeltaNetLayer(d_model, use_spectral_norm=False)
```

**Forward signature**:
```python
layer(x, state=None) -> torch.Tensor   # state is optional internal hidden state
```

**Usage example**:
```python
delta = GatedDeltaNetLayer(d_model=768)
out = delta(hidden_states)
```

## 🔗 Integration with Hybrid Blocks

- `GoldFinch‑AHN` uses `GatedDeltaNetLayer` for long‑term memory.
- `HymbaTransX` uses `UnifiedRoPE` and `SimpleSSM`.

These auxiliary components are automatically registered in `AUX_REGISTRY` when the `ssm` package is imported. The top‑level `framework.attentions.__init__` imports `ssm`, so no manual registration is needed.

## 🧪 Testing

All modules can be tested with a simple script:

```python
import torch
from framework.attentions.ssm import UnifiedRoPE, SimpleSSM, GatedDeltaNetLayer

x = torch.randn(2, 128, 768)
rope = UnifiedRoPE(768)
out_rope = rope(x)
print(f"RoPE output shape: {out_rope.shape}")

ssm = SimpleSSM(768)
out_ssm = ssm(x)
print(f"SSM output shape: {out_ssm.shape}")

delta = GatedDeltaNetLayer(768)
out_delta = delta(x)
print(f"DeltaNet output shape: {out_delta.shape}")
```

## 📜 References

- RoPE: Su et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding*. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- Mamba: Gu & Dao (2023). *Mamba: Linear‑Time Sequence Modeling with Selective State Spaces*. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- Gated Delta Networks: Yang et al. (2025). *Gated Delta Networks: Improving Mamba2 with Delta Rule*. ICLR 2025. [arXiv:2412.06464](https://arxiv.org/abs/2412.06464)

## 📄 License

Part of the Efficient Transformer Toolkit – MIT License.
