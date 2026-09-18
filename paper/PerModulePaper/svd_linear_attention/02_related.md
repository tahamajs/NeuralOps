<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Linear Attention

### 2.1.1 Linear Transformer

Katharopoulos et al. (2020) show that softmax attention can be linearized 
by replacing $\text{softmax}(Q K^\top)$ with $\phi(Q) \phi(K)^\top$:

$$
\text{Attn}(Q, K, V)_t = \frac{\phi(Q_t) \sum_{j \leq t} \phi(K_j) V_j^\top}{\phi(Q_t) \sum_{j \leq t} \phi(K_j)}
\tag{2.1}
$$

with $\phi(x) = \text{ELU}(x) + 1$. This is exactly the recurrence we adopt, 
extended with exponential decay.

### 2.1.2 Performer

Choromanski et al. (2021) use orthogonal random features to approximate 
the softmax kernel:

$$
\exp(q \cdot k) \approx \phi_{\text{PORF}}(q)^\top \phi_{\text{PORF}}(k)
\tag{2.2}
$$

SVD-Linear Attention uses the simpler ELU+1 feature map, which is 
deterministic and parameter-free.

### 2.1.3 RetNet

Sun et al. (2023) propose RetNet, which uses retention with exponential 
decay. Our decay gate is closely related, but we apply it in a 
**chunked parallel scan** for efficiency.

### 2.1.4 GLA (Gated Linear Attention)

Yang et al. (2024) use data-dependent decay for linear attention. Our 
decay mechanism matches theirs but we additionally factorize the projections.

## 2.2 Decay Gating

### 2.2.1 RetNet Decay

RetNet uses fixed decay $\lambda_t = \gamma^{t}$, where $\gamma < 1$ is a 
fixed scalar. Our decay is **input-dependent**, enabling content-based 
selection of what to remember.

### 2.2.2 Mamba

Mamba (Gu & Dao, 2023) uses selective state space with input-dependent 
$\Delta_t, B_t, C_t$. Our decay gate is analogous to $\bar{A}_t = e^{\Delta_t A}$.

### 2.2.3 HGRN

Qin et al. (2024) use lower-bounded decay gates. Our decay uses softplus 
to guarantee $\lambda \in (0, 1)$ without a lower bound, but the log-space 
init ensures the effective decay is bounded.

## 2.3 SVD-Based Compression

### 2.3.1 ASVD

Yuan et al. (2023) use activation-aware SVD for weight compression. 
SVD-Linear Attention applies standard SVD to projections without 
activation weighting, but combines it with linear attention.

### 2.3.2 AASVD / A3

Companion modules that use full-covariance or diagonal-Fisher weighting 
for SVD. SVD-Linear Attention does not use calibration data.

## 2.4 Chunked Parallel Scans

### 2.4.1 Mamba's Parallel Scan

Mamba uses `associative_scan` (Smith et al., 2023) for parallel prefix 
computation. Our chunked scan is similar but uses only `torch.cumsum`.

### 2.4.2 FlashLinearAttention

Yang et al. (2024) provide a Triton kernel for chunked linear attention. 
We use pure PyTorch for portability.

## 2.5 Position of Our Work

| Aspect | Prior work | SVD-Linear Attention |
|---|---|---|
| **Attention** | Softmax or linear | Linear + decay |
| **Projections** | Dense | SVD-factorized |
| **Decay** | Fixed or input-dep | **Input-dependent + log-init** |
| **Streaming** | Some support | **`step_recurrent`** |
| **Multi-architecture** | Usually one family | Four families |
| **Parameter effect** | Varies | **Reduces** |

## 2.6 Comparison to Companion Modules

| Module | Attention | Params | Complexity | Streaming |
|---|---|---|---|---|
| AASVD | Softmax | **−50%** | $\mathcal{O}(N^2)$ | No |
| NHA | Local + RFF | +LoRA | $\mathcal{O}(N \cdot W)$ | No |
| SVD-Mamba | Attention + SSM | +10% | $\mathcal{O}(N)$ | No |
| RPC | Sketched softmax | +LoRA | $\mathcal{O}(N^2)$ | No |
| **SVD-Linear** | **Linear + decay** | **−[D_PARAM_PCT]%** | **$\mathcal{O}(N)$** | **Yes** |

---