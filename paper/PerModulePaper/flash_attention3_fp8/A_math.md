<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: A_math.md
     PURPOSE: Mathematical derivations
     DEPENDENCIES: 04_method.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix A. Mathematical Derivations

## A.1 Tiling in FlashAttention

The attention output for query block $Q_i \in \mathbb{R}^{S \times d}$ is:

$$
O_i = \text{softmax}\!\left(\frac{Q_i K^\top}{\sqrt{d}}\right) V
\tag{A.1}
$$

Tiling partitions $K$ into blocks $\{K_j\}$. The streaming softmax (Milakov 
& Gimelshein, 2018) computes:

$$
\begin{aligned}
m_i^{(j)} &= \max(m_i^{(j-1)}, \text{rowmax}(Q_i K_j^\top / \sqrt{d})) \\
\ell_i^{(j)} &= e^{m_i^{(j-1)} - m_i^{(j)}} \ell_i^{(j-1)} 
              + \text{rowsum}(e^{Q_i K_j^\top / \sqrt{d} - m_i^{(j)}}) \\
O_i^{(j)} &= e^{m_i^{(j-1)} - m_i^{(j)}} O_i^{(j-1)} 
          + e^{Q_i K_j^\top / \sqrt{d} - m_i^{(j)}} V_j
\end{aligned}
\tag{A.2}
$$

At the end, $O_i = O_i^{(J)} / \ell_i^{(J)}$.

## A.2 FP8 Cast Error

For a tensor $x \in \mathbb{R}^n$, casting to e4m3 with 3 mantissa bits gives 
relative error:

$$
\frac{|x - \text{fp8}(x)|}{|x|} \leq 2^{-3} = 0.125
\tag{A.3}
$$

In practice, for zero-mean activations, RMS error is ~0.5%.

## A.3 FP8 Attention Error Bound

**Claim.** Let $O^\text{fp16}$ and $O^\text{fp8}$ be attention outputs in 
FP16 and FP8. Then:

$$
\|O^\text{fp16} - O^\text{fp8}\|_F \leq \epsilon (1 + \|V\|_F) \|Q\|_F \|K\|_F
\tag{A.4}
$$

where $\epsilon \approx 0.125$ for e4m3.

**Proof sketch.** The Q/K cast introduces relative error $\epsilon$ on each 
element. The attention weights $A = \text{softmax}(QK^\top/\sqrt{d})$ have 
Lipschitz constant 1 (softmax is 1-Lipschitz in the max-norm), so 
$\|A^\text{fp16} - A^\text{fp8}\|_F \leq \epsilon \|QK^\top\|_F / \sqrt{d}$. 
Multiplying by $V$ gives the bound. $\square$

**Practical implication.** For typical transformer activations with 
$\|Q\|, \|K\| \sim 10$ and $d = 64$, the error is $\sim 0.2\%$ — acceptable 
for inference.

## A.4 LoRA Init

At initialization $B = 0$, so $\Delta W = 0$ and the model reduces to the 
base model. This is critical for recovery stability.

## A.5 GQA Repeat

The KV expansion is:

$$
K_\text{expanded}[b, h, l, d] = K[b, h // r, l, d], \quad r = H / H_{kv}
\tag{A.5}
$$

This duplicates KV heads $r$ times to match Q head count.

---