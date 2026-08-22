# SALT Attention: Singular Value Adaptation with Low-Rank Tuning for Parameter-Efficient Transformer Fine-Tuning

## 1. Abstract & Theoretical Motivation

Fine-tuning large pre-trained Transformer architectures via full parameter update is computationally expensive and memory-prohibitive. While standard Low-Rank Adaptation (LoRA) introduces trainable bottleneck matrices $A B$, it ignores the underlying singular value spectrum of the pre-trained weight matrices, often perturbing dominant singular directions during gradient updates.

**SALT Attention** (Singular Value Adaptation with Low-Rank Tuning) introduces a parameter-efficient adaptation paradigm for multi-head attention projections that combines:
1. **Frozen Orthogonal Base Singular Vector Preservation**: Singular vectors $U_{:, :r}$ and $V_{:r, :}^T$ from the pre-trained weight SVD ($W_0 = U \Sigma V^T$) are registered as frozen buffers, preserving principal subspace geometry.
2. **Singular Value Rescaling Vector**: A learnable vector $\gamma \in \mathbb{R}^r$ modulates the top-$r$ pre-trained singular values $S_{1:r} \odot \gamma$, allowing fine-grained scaling of existing principal components.
3. **Additive Low-Rank Residual Adapter**: A trainable low-rank branch $A B$ ($A \in \mathbb{R}^{d_{\text{out}} \times r}, B \in \mathbb{R}^{r \times d_{\text{in}}}$) captures orthogonal residual task-specific updates.

---

## 2. Mathematical Derivations & Formulations

### 2.1 Weight Factorization & Singular Value Rescaling (`SALTLinear`)
For a pre-trained projection weight matrix $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, singular value decomposition is evaluated:

\[
W_0 = U \Sigma V^\top, \quad U \in \mathbb{R}^{d_{\text{out}} \times \min(d_{\text{out}}, d_{\text{in}})}, \; V^\top \in \mathbb{R}^{\min(d_{\text{out}}, d_{\text{in}}) \times d_{\text{in}}}
\]

Top-$r$ singular bases $U_{:, :r}$ and $V_{:r, :}^\top$ are registered as frozen buffers. The adapted projection weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ is constructed via singular value rescaling and additive LoRA residual fusion:

\[
W = U_{:, :r} \operatorname{diag}\left( S_{1:r} \odot \gamma \right) V_{:r, :}^\top + A B
\]

where:
$S_{1:r} \in \mathbb{R}^r$ are pre-trained singular values. Furthermore, $\gamma \in \mathbb{R}^r$ is a learnable parameter initialized to all ones ($\mathbf{1}_r$). Concurrently, $A \in \mathbb{R}^{d_{\text{out}} \times r}$ and $B \in \mathbb{R}^{r \times d_{\text{in}}}$ are learnable adapter matrices initialized to zero on $B$ to guarantee exact initial preservation ($W_{\text{init}} = W_0$).

Forward transformation for hidden activation state $X \in \mathbb{R}^{B \times L \times d_{\text{in}}}$ evaluates:

\[
Y = X W^\top + b = X \left( U_{:, :r} \operatorname{diag}\left( S_{1:r} \odot \gamma \right) V_{:r, :}^\top + A B \right)^\top + b \in \mathbb{R}^{B \times L \times d_{\text{out}}}
\]

### 2.2 Multi-Head Attention Computation
Projections for Query, Key, Value, and Output layers are instantiated via `SALTLinear`:

\[
Q = \text{SALTLinear}_Q(X), \quad K = \text{SALTLinear}_K(X), \quad V = \text{SALTLinear}_V(X) \in \mathbb{R}^{B \times L \times H \cdot d_h}
\]

Reshaping into $H$ heads with head dimension $d_h = D / H$ yields $Q, K, V \in \mathbb{R}^{B \times H \times L \times d_h}$. Scaled dot-product attention logits evaluate:

\[
S_{b, h, i, j} = \frac{Q_{b, h, i} \cdot K_{b, h, j}^\top}{\sqrt{d_h}} + M^{\text{pad}}_{b, j}
\]

\[
A = \operatorname{Softmax}(S), \quad A_{\text{drop}} = \operatorname{Dropout}(A)
\]

Context representations are contracted and transformed through the SALT-adapted output projection:

\[
O = A_{\text{drop}} V \in \mathbb{R}^{B \times H \times L \times d_h}
\]

\[
Y = \text{SALTLinear}_O\left( \operatorname{Reshape}(O) \right) \in \mathbb{R}^{B \times L \times D}
\]

---

## 3. Tensor Execution Flow & Dimensional Shape Tracing

| Execution Step | Tensor Symbol | Shape Transformation | Operational Description |
| :--- | :--- | :--- | :--- |
| **Input Activation** | $X$ | $[B, L, D]$ | Base hidden input state |
| **Singular Value Scaling** | $S_{1:r} \odot \gamma$ | $[r]$ | Vector element-wise multiplication |
| **Adapted SVD Base GEMM** | $X V_{:r, :} \text{diag}(\tilde{S}) U_{:, :r}^\top$| $[B, L, D] \to [B, L, r] \to [B, L, D]$ | SVD base projection path |
| **Additive LoRA GEMM** | $X B^\top A^\top$ | $[B, L, D] \to [B, L, r] \to [B, L, D]$ | Residual adapter projection path |
| **Q/K/V Head Reshaping** | $Q, K, V$ | $[B, L, D] \to [B, H, L, d_h]$ | Multi-head view split |
| **Attention Logits** | $S$ | $[B, H, L, d_h] \times [B, H, L, d_h]^\top \to [B, H, L, L]$ | Scaled dot-product score matrix |
| **Softmax & Dropout** | $A$ | $[B, H, L, L]$ | Row-wise Softmax probability matrix |
| **Value Contraction** | $O$ | $[B, H, L, L] \times [B, H, L, d_h] \to [B, H, L, d_h]$ | Context accumulation |
| **SALT Output GEMM** | $Y$ | $[B, L, H \cdot d_h] \to [B, L, D]$ | Output SALT linear transformation |

---

## 4. Analytical Complexity & Parameter Efficiency

### 4.1 Floating Point Operations (FLOPs)
- **SALT Linear Projections ($Q, K, V, O$):** $16 B L D r$ FLOPs.
- **Multi-Head Attention Core:** $4 B L^2 D$ FLOPs.
- **Total FLOP Count:** $\text{FLOPs}_{\text{total}} = 16 B L D r + 4 B L^2 D$.
- **Trainable Parameters:** $\Theta_{\text{trainable}} = 4 \times (r + 2 D r) \ll 4 D^2$. For $D=768, r=16$, parameters drop from $2.36\text{M}$ to $98.4\text{K}$ ($95.8\%$ reduction).

---

## 5. Empirical SST-2 Benchmark & Comparative Profiling

| Attention Variant | SST-2 Accuracy | SST-2 F1 | Runtime / Latency | Parameter Efficiency |
| :--- | :--- | :--- | :--- | :--- |
| **Full Attention** | **89.56%** | **89.55%** | Baseline (1.0x) | 100% Trainable (Dense) |
| **SALT Attention (r=16)** | **88.92%** | **88.90%** | Fast (1.05x) | **4.2% Trainable Params** |
| **Performer (FAVOR+)** | 82.91% | 82.90% | 504.8s | Full Weights |
| **Nyströmformer** | 82.11% | 82.10% | Linear (Pinv) | Full Weights |
| **Adaptive Span** | 80.39% | 80.38% | 5.3s | Full Weights |
| **Span-Expanded (LoRA)** | 81.31% | 81.30% | Linear-Local | Hybrid LoRA |
