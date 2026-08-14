# FlashAttention-3: Memory-Efficient Exact Attention

## File Reference
`./flash_attention3.py`

## Overview
**FlashAttention3** (`FlashAttention3`) integrates the third-generation FlashAttention CUDA kernels for Hopper GPU architectures (NVIDIA H100/H200). It utilizes asynchronous Tensor Memory Accelerator (TMA) hardware instructions, warp specialization, and IO-aware tiling to evaluate exact multi-head scaled dot-product attention without writing intermediate $L \times L$ attention matrices to GPU High Bandwidth Memory (HBM).

---

## Mathematical Formulation

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^\top}{\sqrt{d_h}} + M\right) V$$

### Hardware Acceleration Pipeline
1. **Asynchronous Memory Transfers (TMA):** Overlaps memory loading from global HBM to shared memory (SRAM) with Tensor Core FP16/BF16 matrix multiplications.
2. **Warp Specialization:** Partitioning Producer (memory fetching) and Consumer (math execution) GPU warps.
3. **Fallback Execution:** Gracefully falls back to PyTorch's native `F.scaled_dot_product_attention` (SDPA) when executed on CPU or older GPU generations.

---

## Computational Complexity
- **Time Complexity:** $\mathcal{O}(B \cdot H \cdot L^2 \cdot d_h)$
- **Memory Access Cost (MAC):** $\mathcal{O}\left(B \cdot H \cdot L \cdot d_h + \frac{L^2 d_h^2}{M_{\text{SRAM}}}\right)$ (IO-optimal)
- **VRAM Footprint:** Linear $\mathcal{O}(B \cdot L \cdot D)$ activations.

---

## Usage Example

```python
import torch
from attentions.flash_attention3 import FlashAttention3


class ModelConfig:

  def __init__(self):
    self.hidden_size = 4096
    self.num_attention_heads = 32


config = ModelConfig()
attn = FlashAttention3(config=config)

x = torch.randn(2, 2048, 4096, device="cuda", dtype=torch.bfloat16)
output, _ = attn(x)
print(output.shape)  # Output: torch.Size([2, 2048, 4096])
```
