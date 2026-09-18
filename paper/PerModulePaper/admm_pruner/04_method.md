# 4. Method Formulation
1. **$Z$-Update**: $Z^{(t+1)} = \Pi_{\mathcal{S}}(W^{(t)} + U^{(t)})$ <!-- source: Codes/framework/compression_engine/admm_pruner.py line 10 -->
2. **Intermediate $W$**: $W^{(t+1/2)} = rac{1}{1 + 
ho^{(t)}} \left( W_0 + 
ho^{(t)} (Z^{(t+1)} - U^{(t)}) 
ight)$ <!-- source: Codes/framework/compression_engine/admm_pruner.py line 11 -->
3. **Distillation Step**: $W^{(t+1)} = W^{(t+1/2)} - \eta \lambda_{	ext{distill}} 
abla_W (	ext{MSE}_{	ext{distill}})$ <!-- source: Codes/framework/compression_engine/admm_pruner.py line 12 -->
4. **Dual Update**: $U^{(t+1)} = U^{(t)} + W^{(t+1)} - Z^{(t+1)}$ <!-- source: Codes/framework/compression_engine/admm_pruner.py line 13 -->
5. **Dynamic $
ho$**: $
ho^{(t+1)}$ scales up if $\|r^{(t+1)}\| > \mu \|s^{(t+1)}\|$ and scales down if $\|s^{(t+1)}\| > \mu \|r^{(t+1)}\|$. <!-- source: Codes/framework/compression_engine/admm_pruner.py line 15 -->


## 2. Why This Module Works Correctly & Mathematical Proofs

<!-- source: Codes/frontend/dist/Docs/content/compressions/admm_pruner.md -->
### 2.1 Theorem 1: Constrained Splitting & Augmented Lagrangian Formulation
Let $W_0 \in \mathbb{R}^{N \times M}$ denote the initial uncompressed weight matrix. The goal of ADMM pruning is to find weights $W$ close to $W_0$ satisfying a target sparsity constraint set $\mathcal{S}$:
$$\min_{W, Z} \frac{1}{2} \|W - W_0\|_F^2 + g(Z) \quad \text{subject to} \quad W - Z = 0
<!-- source: Codes/framework/compression_engine/admm_pruner.py line 114 -->

<!-- source: Codes/framework/compression_engine/admm_pruner.py line 114 -->
$$
where $g(Z)$ is the indicator function for the non-convex structural sparsity manifold $\mathcal{S}$ (e.g., 2:4 semi-structured, block $4 \times 4$, or top-$k$ percentile sparsity):
$$g(Z) = \begin{cases} 0 & \text{if } Z \in \mathcal{S} \\ +\infty & \text{if } Z \notin \mathcal{S} \end{cases}
<!-- source: Codes/framework/compression_engine/admm_pruner.py line 114 -->

<!-- source: Codes/framework/compression_engine/admm_pruner.py line 114 -->
$$

The Augmented Lagrangian with penalty parameter $\rho > 0$ and scaled dual multiplier $U \in \mathbb{R}^{N \times M}$ is:
$$\mathcal{L}_\rho(W, Z, U) = \frac{1}{2} \|W - W_0\|_F^2 + g(Z) + \frac{\rho}{2} \|W - Z + U\|_F^2 - \frac{\rho}{2} \|U\|_F^2
<!-- source: Codes/framework/compression_engine/admm_pruner.py line 114 -->

<!-- source: Codes/framework/compression_engine/admm_pruner.py line 114 -->
$$

### 2.2 Theorem 2: Closed-Form Solutions to ADMM Subproblems
ADMM minimizes $\mathcal{L}_\rho$ by alternating between three exact closed-form update steps at iteration $t$:

**Step 1: Sparse Constraint Projection ($Z$-Update):**
Fixing $W = W^{(t)}$ and $U = U^{(t)}$, the $Z$-subproblem is:
$$Z^{(t+1)} = \operatorname{arg}\min_Z \left( g(Z) + \frac{\rho^{(t)}}{2} \|W^{(t)} - Z + U^{(t)}\|_F^2 \right) = \Pi_{\mathcal{S}}\left( W^{(t)} + U^{(t)} \right)
<!-- source: Codes/framework/compression_engine/admm_pruner.py line 114 -->

<!-- source: Codes/framework/compression_engine/admm_pruner.py line 114 -->
$$
where $\Pi_{\mathcal{S}}(V)$ projects continuous tensor $V = W^{(t)} + U^{(t)}$ onto the constraint set $\mathcal{S}$:
- **2:4 Semi-Structured:** In every 4-element block, retain the 2 elements with highest absolute magnitude $|V_{ij}|$ and set the other 2 to zero.
- **Block Sparsity ($B_r \times B_c$):** Zero out whole blocks whose Frobenius norm falls below the target sparsity quantile.
- **Percentile / Unstructured:** $Z_{ij} = V_{ij} \cdot \mathbb{I}(|V_{ij}| \ge \tau)$ where $\tau$ is the $(100 \cdot s)$-th percentile of $|V|$.

**Step 2: Proximal Continuous Weight Update ($W$-Update):**
Fixing $Z = Z^{(t+1)}$ and $U = U^{(t)}$, the objective is strictly convex and differentiable with respect to $W$:
$$\nabla_W \mathcal{L}_\rho = (W - W_0) + \rho^{(t)} \left( W - Z^{(t+1)} + U^{(t)} \right) = 0
<!-- source: Codes/framework/compression_engine/admm_pruner.py line 114 -->

<!-- source: Codes/framework/compression_engine/admm_pruner.py line 114 -->
$$
$$(1 + \rho^{(t)}) W = W_0 + \rho^{(t)} \left( Z^{(t+1)} - U^{(t)} \right)
<!-- source: Codes/framework/compression_engine/admm_pruner.py line 114 -->

<!-- source: Codes/framework/compression_engine/admm_pruner.py line 114 -->
$$
$$W^{(t+1)} = \frac{1}{1 + \rho^{(t)}} \left( W_0 + \rho^{(t)} \left( Z^{(t+1)} - U^{(t)} \right) \right)
<!-- source: Codes/framework/compression_engine/admm_pruner.py line 114 -->

<!-- source: Codes/framework/compression_engine/admm_pruner.py line 114 -->
$$
This update acts as a proximal convex combination: when $\rho^{(t)} \to 0$, $W^{(t+1)} \to W_0$; when $\rho^{(t)} \to \infty$, $W^{(t+1)} \to Z^{(t+1)} - U^{(t)}$.

**Step 3: Dual Multiplier Accumulation ($U$-Update):**
$$U^{(t+1)} = U^{(t)} + \left( W^{(t+1)} - Z^{(t+1)} \right)
<!-- source: Codes/framework/compression_engine/admm_pruner.py line 114 -->

<!-- source: Codes/framework/compression_engine/admm_pruner.py line 114 -->
$$
The dual variable $U$ acts as a running integral of the primal constraint violation, penalizing coordinates that repeatedly deviate from the sparse support.

### 2.3 Theorem 3: Convergence & Dynamic Penalty ($\rho$) Scheduling
At each iteration $t+1$, the primal residual $r^{(t+1)}$ and dual residual $s^{(t+1)}$ measure constraint feasibility and optimality:
$$r^{(t+1)} = W^{(t+1)} - Z^{(t+1)} \in \mathbb{R}^{N \times M}
<!-- source: Codes/framework/compression_engine/admm_pruner.py line 114 -->

<!-- source: Codes/framework/compression_engine/admm_pruner.py line 114 -->
$$
$$s^{(t+1)} = \rho^{(t)} \left( Z^{(t+1)} - Z^{(t)} \right) \in \mathbb{R}^{N \times M}
<!-- source: Codes/framework/compression_engine/admm_pruner.py line 114 -->

<!-- source: Codes/framework/compression_engine/admm_pruner.py line 114 -->
$$

Under standard ADMM convergence theory, asymptotic convergence requires balancing primal feasibility ($\|r^{(t+1)}\|_F \to 0$) and dual feasibility ($\|s^{(t+1)}\|_F \to 0$). The dynamic $\rho$-scheduling rule adjusts the penalty parameter adaptively:
$$\rho^{(t+1)} = \begin{cases}
\min(\rho^{(t)} \cdot \tau_{\text{incr}}, \rho_{\max}) & \text{if } \|r^{(t+1)}\|_F > \mu \|s^{(t+1)}\|_F \\
\max(\rho^{(t)} / \tau_{\text{decr}}, \rho_{\min}) & \text{if } \|s^{(t+1)}\|_F > \mu \|r^{(t+1)}\|_F \\
\rho^{(t)} & \text{otherwise}
\end{cases}
<!-- source: Codes/framework/compression_engine/admm_pruner.py line 114 -->

<!-- source: Codes/framework/compression_engine/admm_pruner.py line 114 -->
$$
where typical hyperparameters are $\mu = 10.0, \tau_{\text{incr}} = \tau_{\text{decr}} = 1.5, \rho_{\min} = 10^{-4}, \rho_{\max} = 100.0$. This dynamic adjustment prevents the optimization from becoming trapped in rigid early hard-thresholding regimes.

### 2.4 Theorem 4: Straight-Through Estimator (STE) Gradient Backpropagation
To allow continuous gradient fine-tuning after ADMM pruning without zero-gradient blocking from discrete binary masks, `ADMMMaskedLinear` integrates an STE custom autograd function:
$$\text{Forward: } \quad y = (W \odot \mathcal{M}) x + b
<!-- source: Codes/framework/compression_engine/admm_pruner.py line 114 -->

<!-- source: Codes/framework/compression_engine/admm_pruner.py line 114 -->
$$
$$\text{Backward: } \quad \frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial (W \odot \mathcal{M})} = \frac{\partial \mathcal{L}}{\partial (W \odot \mathcal{M})}
<!-- source: Codes/framework/compression_engine/admm_pruner.py line 114 -->

<!-- source: Codes/framework/compression_engine/admm_pruner.py line 114 -->
$$
This ensures that gradients flow back into continuous unpruned weight tensors, enabling surviving parameters to compensate for pruned connections during subsequent fine-tuning.

### 2.5 Theorem 5: Post-Pruning LayerNorm Recalibration
Structural pruning decreases the expected activation norm $\mathbb{E}[\|y\|_2^2] = \mathbb{E}[\|(W \odot \mathcal{M}) x\|_2^2]$ by a factor roughly proportional to $(1 - \text{sparsity})$. In deep networks, this induces exponential representation decay across layers. Recalibrating LayerNorm scale parameters $\gamma$ by the empirical standard deviation ratio:
$$\gamma_{\text{recal}} = \gamma \cdot \left( 0.8 + 0.2 \cdot \frac{1}{\operatorname{std}(X_{\text{calib}})} \right)
<!-- source: Codes/framework/compression_engine/admm_pruner.py line 114 -->

<!-- source: Codes/framework/compression_engine/admm_pruner.py line 114 -->
$$
restores unit variance across transformer residual streams, preventing vanishing gradients and logits collapse.

---
