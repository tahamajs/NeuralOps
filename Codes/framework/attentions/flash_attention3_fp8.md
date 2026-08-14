# FlashAttention-3 FP8: Low-Precision 8-bit Float Transformer Attention

## File Reference
`./flash_attention3_fp8.py`

## Overview
**FlashAttention3FP8** (`FlashAttention3FP8`, registered as `flash3_fp8`) executes FlashAttention-3 using FP8 (`torch.float8_e4m3fn`) data format on NVIDIA Hopper WGMMA Tensor Cores. Quantizing Query, Key, and Value activations into FP8 format reduces HBM memory bandwidth consumption by $50\%$ and doubles arithmetic throughput.

---

## Mathematical Formulation

### 1. Dynamic/Static FP8 Quantization
$$Q_{\text{fp8}} = \text{cast}_{\text{fp8}}(Q), \quad K_{\text{fp8}} = \text{cast}_{\text{fp8}}(K), \quad V_{\text{fp8}} = \text{cast}_{\text{fp8}}(V)$$

### 2. Exact Hardware Tiled Reduction
$$O_{\text{fp8}} = \text{FlashAttn3}(Q_{\text{fp8}}, K_{\text{fp8}}, V_{\text{fp8}})$$
$$\text{Output} = \text{cast}_{\text{dtype}}(O_{\text{fp8}}) W_{\text{out}}^\top + b_{\text{out}}$$

---

## Performance & Memory Characteristics
- **HBM Bandwidth Savings:** $50\%$ memory traffic reduction compared to FP16/BF16.
- **Compute Latency:** Up to $2\times$ throughput speedup on H100 SXM5 GPUs using FP8 Tensor Core instructions.
- **Precision Preservation:** Utilizes `E4M3` mantissa distribution to preserve attention dynamic range without accuracy degradation.

---

## Usage Example

```python
import torch
from attentions.flash_attention3_fp8 import FlashAttention3FP8

attn = FlashAttention3FP8(d_model=4096, n_heads=32, use_fp8=True).cuda()
x = torch.randn(2, 1024, 4096, device="cuda", dtype=torch.bfloat16)

out, _ = attn(x)
print(out.shape)  # Output: torch.Size([2, 1024, 4096])
```
