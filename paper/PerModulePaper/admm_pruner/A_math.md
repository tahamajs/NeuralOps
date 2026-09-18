# Appendix A. Dynamic Rho Scheduling
Adjusts penalty based on primal residual $r^{(t+1)} = W^{(t+1)} - Z^{(t+1)}$ and dual residual $s^{(t+1)} = 
ho (Z^{(t+1)} - Z^{(t)})$.


## 1. Executive Abstract & Theoretical Motivation

<!-- source: Codes/frontend/dist/Docs/content/compressions/admm_pruner.md -->
Pruning neural network weight tensors without sacrificing performance is a fundamental challenge in model compression. Standard magnitude pruning suffers from greedy decisions that create unrecoverable optimization bottlenecks. **ADMM Pruner** addresses this problem by formulating model pruning as a constrained non-convex optimization problem solved via the **Alternating Direction Method of Multipliers (ADMM)** with dynamic penalty parameter ($\rho$) scheduling and Straight-Through Estimator (STE) gradient backpropagation.

By decomposing the hard structural sparsity constraint into a sequence of tractable subproblems, ADMM decouples continuous parameter optimization from discrete combinatorial structural pruning. The algorithm alternates between updating continuous weights $W$, projecting onto the sparse constraint set $Z$ (unstructured, block, or 2:4 semi-structured), and updating a dual multiplier tensor $U$ that tracks constraint residual drift.

```
                    [ Original Uncompressed Weight W_0 ∈ ℝ^{N × M} ]
                                          │
                                          ▼
            ┌───────────────────────────────────────────────────────────┐
            │ ADMM Alternating Iteration Loop (t = 1 ... T)             │
            │ 1. Sparse Projection: Z^(t+1) = Π_S(W^(t) + U^(t))        │
            │ 2. Proximal Weight Update:                                │
            │    W^(t+1) = (W_0 + ρ^(t) (Z^(t+1) - U^(t))) / (1 + ρ^(t))│
            │ 3. Dual Accumulator Update:                               │
            │    U^(t+1) = U^(t) + W^(t+1) - Z^(t+1)                    │
            │ 4. Residual Evaluation & Dynamic Rho Scheduling:          │
            │    Primal r^(t+1) = ||W^(t+1) - Z^(t+1)||_F               │
            │    Dual s^(t+1) = ρ^(t) ||Z^(t+1) - Z^(t)||_F             │
            │    Adapt ρ^(t+1) to balance ||r|| and ||s||               │
            └─────────────────────────────┬─────────────────────────────┘
                                          │
                                          ▼
            [ Convert to ADMMMaskedLinear with STE Autograd Function ]
            [ Post-Pruning LayerNorm / RMSNorm Variance Recalibration ]
```

---
