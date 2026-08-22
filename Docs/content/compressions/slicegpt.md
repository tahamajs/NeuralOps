# SliceGPT: Post-Training Slicing with Orthogonal Transformation

## 1. Overview & Architecture

SliceGPT is a post-training structural pruning algorithm that compresses transformer architectures by physically deleting full rows and columns from weight matrices. By applying an orthogonal matrix $Q \in \mathbb{R}^{d \times d}$ (where $Q^\top Q = I$) derived from empirical activation covariances, SliceGPT rotates residual stream representations so that parameter variance is concentrated into the first $r$ principal coordinates.

```
                    [ Input Activation X ]
                              │
                              ▼
            [ Empirical Covariance C = X^T X / N ]
                              │
                              ▼
            [ Eigendecomposition C = Q Λ Q^T ]
                              │
                              ▼
     [ Orthogonal Weight Transformation W_tilde = W Q ]
                              │
                              ▼
     [ Non-Uniform Slicing Operator P_r: Keep Top-r ]
                              │
                              ▼
      [ Sliced Weights W_sliced = W_tilde P_r P_r^T Q^T ]
```

---

## 2. Pruning Saliency Metric Formulas (LaTeX)

### 2.1 Empirical Covariance & Orthogonal Rotation
Given calibration activations $X \in \mathbb{R}^{N \times d_{\text{in}}}$, the empirical second-moment covariance matrix $C_X$ is computed as:
$$C_X = \frac{1}{N} X^\top X + \epsilon I \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$$

The orthogonal transformation matrix $Q$ is obtained via spectral eigendecomposition:
$$C_X = Q \Lambda Q^\top, \quad \Lambda = \operatorname{diag}(\lambda_1, \lambda_2, \dots, \lambda_{d_{\text{in}}}), \quad \lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_{d_{\text{in}}}$$

### 2.2 Slicing Projection Operator
The orthogonal projection operator $P_r \in \mathbb{R}^{d_{\text{in}} \times r}$ selects the $r$ coordinates containing the highest cumulative variance:
$$P_r = [e_1, e_2, \dots, e_r] \in \mathbb{R}^{d_{\text{in}} \times r}$$
where $r = \lfloor d_{\text{in}} \cdot (1 - \text{sparsity}) \rfloor$.

### 2.3 Non-Uniform Head Slicing
For multi-head attention projections ($W_Q, W_K, W_V, W_O$), non-uniform slicing retains higher capacity in attention routing heads while aggressively pruning feed-forward MLP projections:
$$r_h = \max\left(1, \left\lfloor d_{\text{head}} \cdot \left(1 - \alpha_{\text{attn}} \cdot \text{sparsity}\right) \right\rfloor\right), \quad \alpha_{\text{attn}} \in [0.5, 0.75]$$

---

## 3. Structural vs. Unstructured Sparsity Comparison

| Metric | Unstructured Sparsity | 2:4 Semi-Structured | SliceGPT Structural Slicing |
| :--- | :--- | :--- | :--- |
| **Non-Zero Pattern** | Irregular (any index) | 2 non-zeros per 4 entries | Dense continuous submatrix |
| **Weight Tensor Shape** | Unchanged $[d_{\text{out}}, d_{\text{in}}]$ | Unchanged $[d_{\text{out}}, d_{\text{in}}]$ | Reduced $[d_{\text{out}}, r]$ |
| **Physical Parameter Reduction** | 0% (requires sparse mask) | 0% (compressed storage) | **Up to 40% True Parameter Deletion** |
| **Standard CuBLAS GEMM Support** | No (requires custom SpMM) | Requires Sparse Tensor Cores | **Yes (Direct Standard GEMM)** |
| **Inference Memory Bandwidth** | Unreduced without sparse runtime | $2\times$ memory compression | **Direct $1.33\times - 1.6\times$ reduction** |

---

## 4. Hardware Matrix Core Speedup Analysis

- **Ampere / Hopper / Blackwell Tensor Cores:** Because SliceGPT changes the physical matrix dimensions from $[M \times K]$ to $[M \times r]$, standard `cublasGemmEx` and CUTLASS kernels operate on smaller tile sizes with $100\%$ arithmetic density.
- **Roofline Model Speedup:** In memory-bound autoregressive decoding (batch size = 1), decoding time scales linearly with parameter footprint:
  $$\text{Speedup} = \frac{T_{\text{dense}}}{T_{\text{sliced}}} \approx \frac{d_{\text{in}}}{r} = \frac{1}{1 - \text{sparsity}} \approx 1.33\times - 1.45\times$$

---

## 5. Usage Example

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from framework.compression_engine.slicegpt import SliceGPTCompressor

# 1. Load model and calibration data
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 2. Instantiate SliceGPT Compressor
compressor = SliceGPTCompressor(
    model=model,
    tokenizer=tokenizer,
    sparsity=0.25,
    non_uniform=True,
)

# 3. Compress model
compressed_model, stats = compressor.compress()

print(f"Original Parameters:   {stats['original_parameters']:,}")
print(f"Compressed Parameters: {stats['compressed_parameters']:,}")
print(f"Reduction:             {stats['reduction_percentage']:.2f}%")
```
