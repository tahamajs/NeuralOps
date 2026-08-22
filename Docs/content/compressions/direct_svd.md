# DirectSVDCompressor: Closed-Form Truncated SVD Low-Rank Factorization

## 1. Executive Abstract & Theoretical Motivation
Dense linear projection layers in transformer architectures ($W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$) account for a significant portion of memory consumption and floating-point operations (FLOPs). Under the Eckart-Young-Mirsky Theorem, truncated Singular Value Decomposition (SVD) provides the optimal low-rank matrix approximation under both Frobenius and spectral matrix norms:

$$\min_{\operatorname{rank}(\hat{W}) \le r} \|W - \hat{W}\|_F = \sqrt{\sum_{i=r+1}^{\min(d_{\text{in}}, d_{\text{out}})} \sigma_i^2}$$

`DirectSVDCompressor` executes closed-form truncated SVD across all target `nn.Linear` layers in a neural network. It decomposes dense weight matrices into rank-$r$ bottleneck factors with symmetric square-root singular value distribution ($U \sqrt{\Sigma}, \sqrt{\Sigma} V^T$) and replaces standard linear layers with factorized `SVDLinear` modules, achieving real physical parameter and FLOP reduction without requiring external calibration data.

---

## 2. Mathematical Formulations & Factorization

### 2.1 Full Singular Value Decomposition
Given a dense weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, full SVD is computed in single-precision (FP32) for numerical stability:

$$W = U \Sigma V^T$$

where:
- $U \in \mathbb{R}^{d_{\text{out}} \times d_{\text{out}}}$ is an orthogonal matrix of left-singular vectors.
- $\Sigma = \operatorname{diag}(\sigma_1, \sigma_2, \dots, \sigma_{\min(d_{\text{in}}, d_{\text{out}})})$ contains singular values sorted in descending order ($\sigma_1 \ge \sigma_2 \ge \dots \ge 0$).
- $V^T \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$ is an orthogonal matrix of right-singular vectors.

### 2.2 Rank Truncation & Symmetric Singular Value Distribution
For a chosen target rank $r \le \min(d_{\text{in}}, d_{\text{out}})$, the dominant $r$ singular components are retained. To maintain balanced gradient scaling and numerical stability during forward propagation and subsequent fine-tuning, the singular value matrix $\Sigma_r = \operatorname{diag}(\sigma_1, \dots, \sigma_r)$ is split symmetrically via its square root:

$$U_{\text{factor}} = U[:, :r] \cdot \operatorname{diag}(\sqrt{\Sigma_r}) \in \mathbb{R}^{d_{\text{out}} \times r}$$

$$V_{\text{factor}} = \operatorname{diag}(\sqrt{\Sigma_r}) \cdot (V^T[:r, :]) \in \mathbb{R}^{r \times d_{\text{in}}}$$

The reconstructed rank-$r$ linear transformation acts on input $x \in \mathbb{R}^{B \times N \times d_{\text{in}}}$ as:

$$y = (x V_{\text{factor}}^T) U_{\text{factor}}^T + b$$

### 2.3 Complexity & Parameter Reduction
- **Original Parameters**: $d_{\text{out}} \times d_{\text{in}}$
- **Compressed Parameters**: $r \times (d_{\text{in}} + d_{\text{out}})$
- **Break-even Rank**: $r_{\text{crit}} = \frac{d_{\text{in}} d_{\text{out}}}{d_{\text{in}} + d_{\text{out}}}$
- **FLOPs Reduction**: From $2 d_{\text{in}} d_{\text{out}}$ to $2 r (d_{\text{in}} + d_{\text{out}})$ per token.

---

## 3. Architecture & Execution Pipeline

```
          Input Activation Tensor x: [B, N, d_in]
                             │
                             ▼
              ┌─────────────────────────────┐
              │ First Stage: V_factor       │
              │ Shape: [r, d_in]            │
              │ Intermediate: [B, N, r]     │
              └──────────────┬──────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │ Second Stage: U_factor      │
              │ Shape: [d_out, r]           │
              │ Output: [B, N, d_out]       │
              └──────────────┬──────────────┘
                             │  + Bias b [d_out]
                             ▼
                    Output Tensor y
```

---

## 4. API Reference & Factory Usage

### 4.1 DirectSVDCompressor Class
```python
from framework.compression_engine.direct_svd import DirectSVDCompressor

compressor = DirectSVDCompressor(
    model=model,
    rank=32,                  # Target rank per layer
    rank_ratio=None,          # Optional fraction of min(d_in, d_out)
    target_modules=None,      # Optional list of substring filters
)

compressed_model, stats = compressor.compress()
```

### 4.2 Factory Function
```python
from framework.compression_engine.direct_svd import direct_svd_factory

# Standard universal factory signature
compressed_model = direct_svd_factory(
    model=model,
    comp_config=None,
    rank=32,
)
```

### 4.3 Parameters
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | `nn.Module` | Required | PyTorch neural network to compress. |
| `rank` | `int` | `32` | Target bottleneck rank $r$. |
| `rank_ratio` | `float` | `None` | Fraction of $\min(d_{\text{in}}, d_{\text{out}})$ (overrides `rank` if set). |
| `target_modules` | `List[str]` | `None` | List of layer name substrings to compress. |
