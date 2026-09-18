# 5. Architecture and Implementation


```text
  +-------------------------------------------------+
  |                Input Activations                |
  |                  X: (B, N, D)                   |
  +------------------------+------------------------+
                           |
             +-------------+-------------+
             |                           |
             v                           v
     +---------------+           +---------------+
     |  Base Proj    |           | Residual/Spars|
     |  Factorized   |           | Projection    |
     +-------+-------+           +-------+-------+
             |                           |
             +-------------+-------------+
                           |
                           v
                   Aggregated Context
```

Operates directly on `base_layer.weight` in `LoRALinear` layers, preserving LoRA adapters for recovery fine-tuning.


## 3. Step-by-Step Calibration & Execution Trace

<!-- source: Codes/frontend/dist/Docs/content/compressions/admm_compressor.md -->
### 3.1 Dataflow & Transformation Trace

1. **Tensor Preparation**:
Extract weight matrix $W \in \mathbb{R}^{N \times M}$ in FP32 precision.

2. **Flat Top-K Thresholding**:
Reshape $|W|$ into a 1D tensor of size $N \cdot M$. Furthermore, Retrieve top $k = \lfloor (1 - S) \cdot N \cdot M \rfloor$ values. Concurrently, Construct binary mask matrix $M = (|W| \ge \tau)$.

3. **Quantization Grid Mapping**:
Scale factor $S_q = 2^{b-1} - 1$. Furthermore, Multiply $W_{\text{pruned}}$ by $S_q$ and clamp within $[-S_q, S_q]$. Concurrently, Apply element-wise rounding $\lfloor \cdot \rceil$ and divide by $S_q$.

4. **Precision Conversion**:
Cast resulting matrix back to original module dtype (FP16/BF16) and device.


```
Input Weight W [N, M] ──► $|W|$ [N*M] ──► Top-K Selection ──► Mask M [N, M]
                                                                  │
                                                                  ▼
Output Weight W_q [N, M] ◄── Round & De-scale ◄── Scale & Clamp ◄─ W * M
```

---
