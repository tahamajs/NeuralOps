# SliceGPT PCA Compressor

## 1. Overview & Architecture

SliceGPT PCA Compressor performs principal component analysis (PCA) directly on the activation covariance manifold of transformer layers. It applies an orthogonal rotation $V \in \mathbb{R}^{d \times d}$ derived from activation covariance $\Sigma$, retaining the top-$k$ principal components that capture $(1 - \text{sparsity})$ cumulative variance.

```
                  [ Input Activations X ]
                             │
                             ▼
              [ Centered Covariance Σ_X ]
                             │
                             ▼
              [ Eigendecomposition Σ_X = V Λ V^T ]
                             │
                             ▼
            [ Principal Dimension Truncation k ]
                             │
                             ▼
            [ Sliced Weight W_pca = (W V)[:, :k] V[:, :k]^T ]
```

---

## 2. Pruning Saliency Metric Formulas (LaTeX)

### 2.1 Activation Centering and Covariance
Given $N$ activation samples $X \in \mathbb{R}^{N \times d_{\text{in}}}$ and mean $\mu = \frac{1}{N} \sum_{i=1}^N X_i$:
$$\Sigma = \frac{1}{N - 1} (X - \mu)^\top (X - \mu) \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$$

### 2.2 Principal Coordinate Eigendecomposition
$$\Sigma = V \Lambda V^\top, \quad \Lambda = \operatorname{diag}(\lambda_1, \lambda_2, \dots, \lambda_{d_{\text{in}}})$$

### 2.3 Cumulative Variance Retention
The retention cutoff $k$ is determined by the target retention energy $(1 - \text{sparsity})$:
$$k = \min \left\{ r \;\middle|\; \frac{\sum_{i=1}^r \lambda_i}{\sum_{i=1}^{d_{\text{in}}} \lambda_i} \ge 1 - \text{sparsity} \right\}$$

### 2.4 Reconstructed Weight Matrix
$$W_{\text{pca}} = (W V P_k) P_k^\top V^\top, \quad P_k = [e_1, \dots, e_k] \in \mathbb{R}^{d_{\text{in}} \times k}$$

---

## 3. Structural vs. Unstructured Sparsity Comparison

| Metric | Unstructured Sparsity | 2:4 Semi-Structured | SliceGPT PCA |
| :--- | :--- | :--- | :--- |
| **Saliency Basis** | Element magnitude | Block-wise Taylor/Hessian | **Activation PCA Spectrum** |
| **Mathematical Guarantee** | None | Local 2-of-4 constraint | **Optimal linear subspace under MSE** |
| **Matrix Core Acceleration** | Requires custom SpMM | 2x speedup on Ampere+ | **Direct CuBLAS tile reduction** |
| **Calibration Requirement** | None or small | Small Hessian dataset | **Forward pass covariance sample** |

---

## 4. Hardware Matrix Core Speedup Analysis

- In memory bandwidth-bound LLM decoding ($b=1$), throughput is inversely proportional to tensor memory traffic:
  $$\text{Speedup} = \frac{d_{\text{in}}}{k} \approx 1.25\times - 1.50\times$$
- Zero padding or physical column pruning maps directly to GPU warp tile sizes ($16 \times 16$).

---

## 5. Usage Example

```python
import torch
from transformers import AutoModelForSequenceClassification
from framework.compression_engine.slicegpt_pca import SliceGPTPCACompressor

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
compressor = SliceGPTPCACompressor(
    model=model,
    sparsity=0.3,
)

compressed_model, stats = compressor.compress()
print(f"Compressed Parameters: {stats['compressed_parameters']:,} ({stats['reduction_percentage']:.2f}% reduction)")
```
