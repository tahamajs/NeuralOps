# DynamicEntropySVDCompressor: Singular Entropy-Budgeted Rank Allocation

## 1. Executive Abstract & Theoretical Motivation
Uniform low-rank factorization assigns identical rank $r$ across all network layers, ignoring that different layers possess starkly different representation capacities and spectral decay rates. Early layers typically capture broad syntactic patterns with heavy-tailed singular spectra, whereas deeper layers specialize in narrow task representations with fast spectral decay.

`DynamicEntropySVDCompressor` (registered as `dynamic_entropy_svd` and `entropy_rank_allocator`) dynamically allocates rank budgets $r_l \in [r_{\min}, r_{\max}]$ across transformer layers proportional to their empirical **Singular Value Shannon Entropy**:

$$H(\Sigma_l) = -\sum_{i=1}^d p_i \ln(p_i), \quad p_i = \frac{\sigma_i}{\sum_{j=1}^d \sigma_j}$$

Layers with flat singular spectra (high entropy, indicating distributed information) receive higher rank budgets, while layers with sharp spectral decay (low entropy) receive lower ranks. Factorized layers are replaced with `DynamicSVDLinear` modules initialized with symmetric square-root singular values ($U \sqrt{\Sigma}, \sqrt{\Sigma} V^T$).

---

## 2. Mathematical Formulations

### 2.1 Empirical Singular Entropy Computation
Given a layer's empirical activation covariance matrix $C_X = \frac{1}{N} X^T X$ or weight matrix $W$, its singular value spectrum $\Sigma = \{\sigma_1, \sigma_2, \dots, \sigma_d\}$ is computed. The singular probability mass function $p_i$ is formed by normalising the singular values:

$$p_i = \frac{\sigma_i}{\sum_{j=1}^d \sigma_j}, \quad \text{where } \sum_{i=1}^d p_i = 1, \quad p_i > 0$$

The singular entropy $H(\Sigma)$ measures the uniformity of information distribution across singular directions:

$$H(\Sigma) = -\sum_{i=1}^d p_i \ln(p_i)$$

- **Maximum Entropy ($H = \ln d$)**: All singular values are equal (spherical spectrum); information cannot be compressed without significant loss.
- **Minimum Entropy ($H \to 0$)**: A single dominant singular value; the matrix can be compressed to rank-1 losslessly.

### 2.2 Bounded Water-Filling Rank Allocation
Given layer entropies $\{H_1, H_2, \dots, H_L\}$ and a global target rank budget $R_{\text{total}}$, the per-layer rank $r_l$ is determined via bounded water-filling:

$$r_l = \operatorname{clip}\left(\left\lfloor \lambda \cdot \frac{H_l}{\sum_{j=1}^L H_j} \right\rceil, r_{\min}, r_{\max}\right)$$

where multiplier $\lambda$ is adjusted via bisection search such that $\sum_{l=1}^L r_l = R_{\text{total}}$.

### 2.3 Symmetric Low-Rank Factorization
For each layer $l$, weights are decomposed into:

$$U_r = U[:, :r_l] \operatorname{diag}(\sqrt{\Sigma_{:r_l}}) \in \mathbb{R}^{d_{\text{out}} \times r_l}$$

$$V_r = \operatorname{diag}(\sqrt{\Sigma_{:r_l}}) V^T[:r_l, :] \in \mathbb{R}^{r_l \times d_{\text{in}}}$$

producing true physical parameter reduction in `DynamicSVDLinear`.

---

## 3. Architecture Pipeline

```
     Model Target Layers {W_1, W_2, ..., W_L}
                         │
                         ▼
        ┌───────────────────────────────────┐
        │ Spectral Analysis & Entropy Calc  │
        │ H(Sigma_l) = -sum(p_i ln p_i)     │
        └─────────────────┬─────────────────┘
                          │  [Layer Entropies]
                          ▼
        ┌───────────────────────────────────┐
        │ Water-Filling Rank Allocator      │
        │ r_l in [r_min, r_max], sum = R    │
        └─────────────────┬─────────────────┘
                          │  [Allocated Ranks r_l]
                          ▼
        ┌───────────────────────────────────┐
        │ Symmetric Factorization           │
        │ U_r = U*sqrt(S), V_r = sqrt(S)*V^T│
        └─────────────────┬─────────────────┘
                          │
                          ▼
        Replace with DynamicSVDLinear Layers
```

---

## 4. API Reference & Factory Usage

### 4.1 Class Reference
```python
from framework.compression_engine.dynamic_rank import DynamicEntropySVDCompressor

compressor = DynamicEntropySVDCompressor(
    model=model,
    min_rank=8,
    max_rank=64,
    target_rank=32,            # Average rank target across layers
    total_rank_budget=None,    # Optional explicit total sum of ranks
    target_param_ratio=None,   # Optional target parameter retention ratio
    calibration_loader=None,   # Optional DataLoader for activation covariance
)

compressed_model, stats = compressor.compress()
```

### 4.2 Factory Functions
```python
from framework.compression_engine.dynamic_rank import (
    dynamic_entropy_svd_factory,
    dynamic_rank_factory,
    entropy_rank_allocator_factory,
)

# Canonical factory invocation
compressed_model = dynamic_entropy_svd_factory(
    model=model,
    comp_config=None,
    rank=32,
)
```

### 4.3 Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | `nn.Module` | Required | Neural network model to compress. |
| `min_rank` | `int` | `16` | Lower bound on per-layer rank. |
| `max_rank` | `int` | `128` | Upper bound on per-layer rank. |
| `target_rank` | `int` | `None` | Baseline target rank per layer. |
| `total_rank_budget` | `int` | `None` | Total rank sum across all factorized layers. |
| `target_param_ratio` | `float` | `None` | Parameter retention ratio constraint. |
| `calibration_loader` | `DataLoader` | `None` | DataLoader for empirical activation covariance. |
