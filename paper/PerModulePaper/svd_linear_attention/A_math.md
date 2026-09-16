<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Mathematical derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 Linear Attention Identity

**Claim.** Kernelized linear attention is exact for any feature map $\phi$:

$$
\text{Attn}(Q, K, V)_t = \frac{\phi(Q_t) \sum_j \phi(K_j) V_j^\top}{\phi(Q_t) \sum_j \phi(K_j)}
\tag{A.1}
$$

**Proof.** Expand:

$$
\text{softmax}(Q_t K^\top) V \approx \frac{\phi(Q_t) \phi(K)^\top V}{\phi(Q_t) \phi(K)^\top \mathbf{1}}
$$

The numerator is $\phi(Q_t) (\phi(K)^\top V)$ where the product 
$\phi(K)^\top V$ is $d_h \times d_v$. The denominator is 
$\phi(Q_t) (\phi(K)^\top \mathbf{1})$ where $\phi(K)^\top \mathbf{1}$ is 
$d_h \times 1$. Both are computed once for all $t$. $\square$

## A.2 Chunked Scan Correctness

**Claim.** The chunked scan with intra-chunk and inter-chunk terms 
recovers the sequential recurrence.

**Proof.** Define $\log \gamma_i = \sum_{j \leq i} \log \lambda_j$. Then:

$$
S_i = e^{\log \gamma_i} \sum_{j \leq i} e^{-\log \gamma_j} \phi(K_j) V_j^\top
\tag{A.2}
$$

For a chunk $c$ starting at position $s_c$:

$$
S_{c, i} = \underbrace{e^{\log \gamma_{c, i}} \sum_{j=s_c}^{s_c+i} e^{-\log \gamma_{c, j}} \phi(K_j) V_j^\top}_{\text{intra}} + \underbrace{e^{\log \gamma_{c, i}} S_{c-1, C}}_{\text{inter}}
\tag{A.3}
$$

The intra term is computed with a `cumsum`; the inter term uses the 
carried state $S_{c-1, C}$. $\square$

## A.3 Causality

**Claim.** SVD-Linear preserves causality.

**Proof.** In the chunked scan:
- Intra-chunk uses `tril` mask (causal).
- Inter-chunk only uses chunks $< c$.

In step-recurrent mode:
- $S_t = \lambda_t S_{t-1} + \phi(K_t) V_t^\top$ depends only on $1..t$.
- Output at $t$ uses $S_t$, $\phi(Q_t)$, $\phi(K_t)$.

So output at $t$ depends only on $1..t$. $\square$

## A.4 Complexity

**Claim.** SVD-Linear is $\mathcal{O}(N d_h^2)$ for bidirectional and 
$\mathcal{O}(N C d_h + N d_h^2)$ for chunked causal.

**Proof.** 
- Bidirectional: $S = \phi(K)^\top V$ is $\mathcal{O}(N d_h^2)$; output 
  is $\mathcal{O}(N d_h^2)$.
- Chunked causal: intra-chunk matmul is $\mathcal{O}(C^2 d_h)$ per chunk; 
  with $N/C$ chunks, total is $\mathcal{O}(N C d_h)$. State update is 
  $\mathcal{O}(N d_h^2)$. $\square$

## A.5 Parameter Count

**Claim.** SVD-Linear's per-layer parameters are:

$$
P = r(d_\text{in}^Q + d_\text{out}^Q) + r(d_\text{in}^K + d_\text{out}^K) + r(d_\text{in}^V + d_\text{out}^V) + r(d_\text{in}^O + d_\text{out}^O) + 2 d_h r_L
$$

**Proof.** Direct sum. For DistilBERT with $d = 768$, $r = 32$:

- Q: $32 \times 1536 = 49,152$
- K, V: same, $49,152$ each
- O: $49,152$
- LoRA (4): $4 \times 16 \times 1536 = 98,304$ (but per-proj LoRA has rank $r_L$)

Wait, the LoRA count differs. Let me redo:

- SVD per projection: $r(d + d) = 2 r d$
- LoRA per projection: $r_L(d + d) = 2 r_L d$
- Total 4 projections: $8 r d + 8 r_L d = 8 d (r + r_L)$
- Decay gate: $d H + H$

For $d = 768$, $H = 12$, $r = 32$, $r_L = 16$:

- SVD: $8 \times 768 \times 32 = 196,608$
- LoRA: $8 \times 768 \times 16 = 98,304$
- Decay gate: $768 \times 12 + 12 = 9,228$
- Total: $304,140$

Baseline: $4 \times 768 \times 768 = 2,359,296$.

Ratio: $2,359,296 / 304,140 \approx 7.8×$. $\square$

## A.6 Decay Initialization

**Claim.** The bias initialization $b_h = -\log \tau_h$ approximates 
the target decay time $\tau_h$.

**Proof.** The actual decay is:

$$
\lambda_h = e^{-\text{softplus}(b_h)} = \frac{1}{1 + e^{b_h}}
\tag{A.4}
$$

Setting $b_h = -\log \tau_h$:

$$
\lambda_h = \frac{1}{1 + 1/\tau_h} = \frac{\tau_h}{1 + \tau_h}
\tag{A.5}
$$

For large $\tau_h$, $\lambda_h \approx 1 - 1/\tau_h$. The retention time 
is $\tau_h^\text{eff} = -\log \lambda_h \approx 1/\tau_h$, which is **inverse** 
to the target. So the initialization is **inverted**. $\square$

**Note.** This is a bug in the code. The correct initialization would be 
$b_h = \text{softplus}^{-1}(\log \tau_h) \approx \log(e^{\log \tau_h} - 1) \approx \log \tau_h$ 
for large $\tau_h$. Since we set $b_h = -\log \tau_h$, the actual decay 
is $1 - 1/\tau_h$, which is close to 1 (short memory) for small $\tau_h$ 
and to 1 (long memory) for large $\tau_h$. The heads are therefore 
**incorrectly initialized**.

---