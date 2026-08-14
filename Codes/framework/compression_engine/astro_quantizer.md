# AstroQuantizer: Dynamic Uniform Range Quantization Engine

## 1. Executive Abstract & Theoretical Motivation

Uniform fixed-point quantization provides straightforward software deployment paths for modern deep neural networks. **AstroQuantizer** implements a uniform min-max range mapping engine that quantizes continuous weight and bias tensors into discrete $b$-bit integer levels.

AstroQuantizer extracts dynamic dynamic range bounds $[\min(W), \max(W)]$ per module, computes a linear scale factor $s = \frac{\max(W) - \min(W)}{2^b - 1}$, and projects floating-point parameters into integer bins before de-quantizing back to floating-point representation. This uniform quantization reduces model VRAM footprint and memory bandwidth during inference.

```
       Continuous Parameter Tensor W
                     │
                     ▼
       Min-Max Range Extraction [min_v, max_v]
                     │
                     ▼
       Scale Factor Compute s = (max_v - min_v) / (2^b - 1)
                     │
                     ▼
       Quantize: Round( (W - min_v) / s )
                     │
                     ▼
       De-quantize: Quantized * s + min_v
```

---

## 2. Mathematical Formulations & Matrix Decompositions

### 2.1 Uniform Min-Max Quantization Operators

Let $W \in \mathbb{R}^{N \times M}$ be a weight tensor and $b \in \{4, 8, 16\}$ be the target bit width.

1. **Dynamic Range Extraction**:
   $$v_{\min} = \min_{i, j} W_{i, j}, \quad v_{\max} = \max_{i, j} W_{i, j}$$
   If $v_{\max} = v_{\min}$, the module is bypassed to avoid division by zero.

2. **Quantization Step Scale Factor**:
   $$s = \frac{v_{\max} - v_{\min}}{2^b - 1}$$

3. **Affine Forward Quantization & De-Quantization**:
   The discrete integer index matrix $Q \in \{0, 1, \dots, 2^b - 1\}^{N \times M}$ is:
   $$Q = \left\lfloor \frac{W - v_{\min}}{s} \right\rceil$$
   The de-quantized floating-point tensor $W_q \in \mathbb{R}^{N \times M}$ is:
   $$W_q = Q \cdot s + v_{\min}$$

4. **Bias Quantization**:
   If bias $b_{\text{bias}}$ exists and $v_{\max}^{\text{bias}} > v_{\min}^{\text{bias}}$:
   $$s_{\text{bias}} = \frac{v_{\max}^{\text{bias}} - v_{\min}^{\text{bias}}}{2^b - 1}, \quad b_q = \left\lfloor \frac{b_{\text{bias}} - v_{\min}^{\text{bias}}}{s_{\text{bias}}} \right\rceil \cdot s_{\text{bias}} + v_{\min}^{\text{bias}}$$

---

## 3. Step-by-Step Calibration & Execution Trace

### 3.1 Dataflow & Execution Pipeline

1. **Linear Layer Iteration**:
   - Iterate over all `nn.Linear` modules in target model.
2. **Min-Max Range Search**:
   - Compute `min_v, max_v = W.min(), W.max()`.
   - Validate condition `max_v > min_v`.
3. **Scale Factor & Grid Shift**:
   - Compute scale `scale = (max_v - min_v) / (2**bits - 1)`.
   - Apply element-wise round & scale: `W_q = torch.round((W - min_v) / scale) * scale + min_v`.
4. **Bias Processing**:
   - Apply min-max dynamic scaling independently to layer bias if present.

```
Input Weight W ──► Compute min_v & max_v ──► Scale s = (max_v - min_v) / (2^b - 1)
                                                            │
                                                            ▼
Quantized Model ◄── De-quantize (Q*s + min_v) ◄── Round & Shift Q = Round((W - min_v)/s)
```

---

## 4. Parameter Savings & Memory Footprint Profiling

### 4.1 Profiling Table

| Component | Formula | Example (LLaMA-7B Linear Layer: $N=4096, M=4096$) |
| :--- | :--- | :--- |
| **FP16 Dense Weights** | $N \times M \times 2 \text{ bytes}$ | $33.55 \text{ MB}$ |
| **AstroQuant 8-Bit Storage** | $N \times M \times 1 \text{ byte} + 2 \times 4 \text{ bytes}$ | $16.78 \text{ MB}$ ($50.0\%$ Savings) |
| **AstroQuant 4-Bit Storage** | $N \times M \times 0.5 \text{ bytes} + 2 \times 4 \text{ bytes}$ | $8.39 \text{ MB}$ ($75.0\%$ Savings) |
| **Memory Footprint Reduction** | $1 - \frac{b}{16}$ | $\mathbf{75.0\%}$ Reduction at $b=4$ |

---

## 5. Precision Loss, Perplexity & Sensitivity Bounds

### 5.1 Quantization Rounding Error Bound

The maximum absolute error per weight element under uniform quantization with step size $s$ is strictly bounded by half the step size:

$$\| W - W_q \|_\infty \le \frac{s}{2} = \frac{v_{\max} - v_{\min}}{2 \cdot (2^b - 1)}$$

### 5.2 Failure Cases

- **Extreme Outlier Parameters**: Large outlier values inflate dynamic range $[v_{\min}, v_{\max}]$, increasing step size $s$ and causing quantization errors across non-outlier parameters.

---

## 6. Production CUDA / Triton Execution Blueprint

```python
import torch
import triton
import triton.language as tl

@triton.jit
def astro_quant_kernel(
    W_ptr, Out_ptr,
    Min_val, Scale,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    w = tl.load(W_ptr + offsets, mask=mask, other=0.0)

    # Uniform min-max quantization
    scaled = (w - Min_val) / Scale
    rounded = tl.extra.cuda.libdevice.rint(scaled)
    dequant = rounded * Scale + Min_val

    tl.store(Out_ptr + offsets, dequant, mask=mask)
```

---

## 7. Hybrid Synthesis & Next-Generation Enhancements

1. **Per-Channel AstroQuant**: Replace scalar range $[\min(W), \max(W)]$ with per-channel dynamic ranges $[\min(W_i), \max(W_i)]$.
2. **Outlier-Aware AstroQuant**: Extract top $1\%$ outlier weights into a sparse FP16 matrix before applying uniform quantization to remainder.
3. **AstroQuant + Huffman Entropy Coding**: Apply Huffman entropy coding to discrete quantized indices $Q$ to reach sub-3-bit average storage widths.