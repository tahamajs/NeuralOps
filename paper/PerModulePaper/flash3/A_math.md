# Appendix A. Mathematical Derivations

## A.1 Online Softmax Derivation

**Claim.** Online softmax produces the same output as standard softmax.

**Proof.** Let $s \in \mathbb{R}^L$ be scores. Standard softmax:

$$
\text{softmax}(s)_i = \frac{e^{s_i}}{\sum_j e^{s_j}}
\tag{A.1}
$$

For numerical stability, subtract $\max_j s_j$:

$$
\text{softmax}(s)_i = \frac{e^{s_i - m}}{\sum_j e^{s_j - m}}, \quad m = \max_j s_j
\tag{A.2}
$$

**Online.** Process scores in blocks. Maintain:
- $m$: running max
- $\ell$: running sum $\sum_j e^{s_j - m}$
- $O$: running weighted sum $\sum_j e^{s_j - m} v_j$

After block $t$ with new scores $s^{(t)}$:

$$
m^{(t)} = \max(m^{(t-1)}, \max(s^{(t)}))
\tag{A.3}
$$

$$
\ell^{(t)} = e^{m^{(t-1)} - m^{(t)}} \ell^{(t-1)} + \sum_j e^{s_j^{(t)} - m^{(t)}}
\tag{A.4}
$$

$$
O^{(t)} = e^{m^{(t-1)} - m^{(t)}} O^{(t-1)} + \sum_j e^{s_j^{(t)} - m^{(t)}} v_j
\tag{A.5}
$$

Final: $O = O^{(T)} / \ell^{(T)}$. This equals the standard formula. $\square$

## A.2 Complexity Analysis

**Memory:** The block size $B$ is chosen so $B \times B$ score blocks fit 
in shared memory. Total HBM: $\mathcal{O}(L d_h)$.

**Time:** Total FLOPs same as naive: $\mathcal{O}(L^2 d_h)$. But HBM 
accesses reduced by factor $B$, giving effective speedup.

## A.3 FP8 Quantization Error

For FP8 e4m3:

$$
\epsilon_{\text{rel}} = \frac{2^{-3}}{2} = \frac{1}{16} = 0.0625
\tag{A.6}
$$

**Propagation.** If $Q, K, V$ are quantized:

$$
\tilde{Q} = Q(1 + \epsilon_Q), \quad |\epsilon_Q| \leq 0.0625
\tag{A.7}
$$

Then scores have error $\epsilon_s \approx \epsilon_Q + \epsilon_K \approx 0.125$.

## A.4 Causality in FlashAttention

FlashAttention supports causal masking via a mask inside the kernel. For 
block $i, j$ with $j > i$, the block is skipped entirely (efficient). For 
$j = i$, a triangular mask is applied.

**Exactness.** The output matches naive causal attention exactly.

## A.5 GQA Repeat Derivation

For GQA with $H_{kv} < H$:

$$
\text{repeat\_kv}(K, n)_{b, h, i, d} = K_{b, h \bmod H_{kv}, i, d}
\tag{A.8}
$$

where $n = H / H_{kv}$. This gives $H$ heads from $H_{kv}$.

## A.6 Partial RoPE

For head_dim = 256 and rotary_dim = 64:

$$
\tilde{Q}_i = [\text{RoPE}(Q_{i, :64}), \ Q_{i, 64:}]
\tag{A.9}
$$

Only the first 64 coordinates are rotated.

---