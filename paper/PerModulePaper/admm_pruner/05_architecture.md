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

Pruned layers are wrapped in `ADMMMaskedLinear` modules using `ADMMSTEPrune` autograd functions.


## 3. Step-by-Step Calibration & Execution Trace

<!-- source: Codes/frontend/dist/Docs/content/compressions/admm_pruner.md -->
### 3.1 Dataflow & Transformation Trace

1. **Initialization**:
Extract weight tensor $W \in \mathbb{R}^{N \times M}$ from target `nn.Linear` module. Furthermore, Allocate auxiliary buffers $Z \leftarrow W$ and dual variable $U \leftarrow \mathbf{0}_{N \times M}$.

2. **Iterative ADMM Loop ($k = 1 \dots K$)**:
Compute candidate matrix $V = Z + U \in \mathbb{R}^{N \times M}$. Furthermore, Flatten $|V|$ to 1D tensor of length $N \cdot M$. Concurrently, Compute quantile threshold $\tau = \text{Percentile}(|V|, 100 \cdot S)$. In this setting, Generate sparse tensor $Z = \text{where}(|V| > \tau, V, 0)$. Additionally, Update dual accumulator $U = U + (W - Z)$.

3. **Weight Overwrite**:
Assign final $Z$ tensor back to `module.weight.data` with original precision and device attributes.


```
Input Weight W [N, M]
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
Auxiliary Buffer Z [N, M]      Dual Multiplier U [N, M]
       │                                 │
       └──────────────┬──────────────────┘
                      ▼
            Candidate V = Z + U [N, M]
                      │
                      ▼
           Quantile Threshold τ
                      │
                      ▼
             Masking & Hard Cut
                      │
                      ▼
           Updated Sparse Z [N, M]
                      │
                      ▼
            Dual Update U += W - Z
```

---
