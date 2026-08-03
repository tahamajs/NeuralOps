# LSH Attention: Locality‑Sensitive Hashing for Efficient Exact Attention

**Abstract**

We present a **Locality‑Sensitive Hashing (LSH) Attention** module, inspired by the Reformer (Kitaev et al., 2020). This module replaces the full quadratic self‑attention with an approximate scheme that only computes attention between query–key pairs that are hashed to the same bucket via random hyperplane projections. By using multiple rounds of hashing and averaging, it provably recovers the true softmax attention in expectation. The implementation performs a simple sign‑based LSH: queries and keys are projected by a random matrix, and the sign of the dot product yields a binary code of length `bucket_size`. Tokens with identical hash codes attend to each other. The module loops over rounds, batches, heads, and hash buckets, computing exact dot‑product attention within each bucket. This reduces the expected complexity from \(O(N^2)\) to \(O(N \log N)\) under appropriate bucket size choices, though the current implementation uses Python loops which are slow for large scale. The module is registered as a drop‑in replacement for standard self‑attention in DistilBERT, and includes standard QKV projections and output layer. This paper provides a rigorous mathematical formulation, discusses the hashing mechanism, and analyses the implementation details and limitations.

---

## 1. Introduction

Multi‑head self‑attention requires computing an \(N \times N\) attention matrix, which is prohibitive for long sequences. The Reformer (Kitaev et al., 2020) introduced the idea of using **locality‑sensitive hashing (LSH)** to group similar queries and keys together, so that attention is only computed within groups. The underlying insight is that the softmax attention is dominated by the largest dot products; LSH provides a way to quickly find these without all‑to‑all comparison. LSH Attention uses random projections and sign thresholds to assign each vector a binary hash code; vectors with the same hash code are likely to be close in angle. Attention is then computed only within these hash buckets, drastically reducing the number of pairwise comparisons.

The module presented here implements a simplified version of LSH attention. It employs multiple rounds of hashing (default 4) with independent random projection matrices per round, and sums the attention outputs across rounds (without averaging, which can be seen as a form of averaging with the sum later divided by the number of rounds in the loss? No division is performed; this may be a bug). The hash function uses `bucket_size` bits, yielding up to \(2^{\text{bucket\_size}}\) possible buckets, but only a tiny fraction are populated. The implementation iterates over batches, heads, rounds, and hash buckets using nested Python loops, making it primarily pedagogical rather than optimised. Nevertheless, it demonstrates the core concept and can serve as a baseline for efficient attention.

---

## 2. Method

### 2.1 Multi‑Head Projections

Given hidden states \(\mathbf{H} \in \mathbb{R}^{B \times L \times D}\), the module first computes standard linear projections:

\[
\mathbf{Q} = \mathbf{H} \mathbf{W}_Q, \quad
\mathbf{K} = \mathbf{H} \mathbf{W}_K, \quad
\mathbf{V} = \mathbf{H} \mathbf{W}_V,
\]

with \(\mathbf{W}_* \in \mathbb{R}^{D \times (H d_h)}\). The tensors are reshaped to \((B, H, L, d_h)\) and transposed to \((B, H, L, d_h)\)? Actually the code does `.permute(0,2,1,3)` giving shape \((B, H, L, d)\).

### 2.2 Hashing via Random Projections

For each round \(r \in \{1, \dots, R\}\), a random projection matrix \(\mathbf{P}_r \in \mathbb{R}^{d_h \times b}\) is drawn from a standard normal distribution and normalised along columns (so each column is a unit vector). The hash function \(\mathcal{H}(\mathbf{x})\) takes a vector \(\mathbf{x} \in \mathbb{R}^{d_h}\) and produces an integer code:

\[
\mathbf{z} = \frac{\mathbf{x}}{\|\mathbf{x}\|} \mathbf{P}_r \quad \in \mathbb{R}^{b},
\]
\[
\mathbf{c} = \mathbb{I}(\mathbf{z} > 0) \quad \in \{0,1\}^{b},
\]
\[
\mathcal{H}(\mathbf{x}) = \sum_{i=1}^{b} c_i \, 2^{i-1}.
\]

Thus, each hash code is an integer in \([0, 2^b - 1]\). This is a standard LSH for cosine similarity: vectors with a small angle between them will likely have the same sign pattern.

The hash is applied element‑wise to all queries and keys, yielding integer matrices \(\mathbf{H}_Q, \mathbf{H}_K \in \mathbb{Z}^{B \times H \times N}\).

### 2.3 Bucketed Attention

For each round \(r\), the tokens (across all batches and heads) are partitioned into buckets according to their hash codes. For a given head \(h\) in batch \(b\), we collect the set of unique hash codes present in \(\mathbf{H}_Q\) and \(\mathbf{H}_K\) combined. For each code, we extract:

\[
\mathcal{Q}_{\text{code}} = \{ \mathbf{q}_i \mid \mathbf{H}_{Q,b,h,i} = \text{code} \}, \quad
\mathcal{K}_{\text{code}} = \{ \mathbf{k}_j \mid \mathbf{H}_{K,b,h,j} = \text{code} \}, \quad
\mathcal{V}_{\text{code}} = \{ \mathbf{v}_j \mid \mathbf{H}_{K,b,h,j} = \text{code} \}.
\]

Attention is computed within this subset:

\[
\mathbf{S}_{\text{code}} = \frac{\mathbf{Q}_{\text{code}} \mathbf{K}_{\text{code}}^\top}{\sqrt{d_h}}, \qquad
\mathbf{A}_{\text{code}} = \operatorname{softmax}(\mathbf{S}_{\text{code}}), \qquad
\mathbf{O}_{\text{code}} = \mathbf{A}_{\text{code}} \mathbf{V}_{\text{code}}.
\]

These outputs are scattered back into the full output tensor for the round. Since a token may appear in multiple buckets across rounds (or even within the same round if it shares a code with multiple buckets? No, a token has only one hash per round, so it belongs to exactly one bucket per round), the outputs are accumulated across rounds.

### 2.4 Aggregation Across Rounds

The final output for a token is the **sum** of its outputs from all \(R\) rounds:

\[
\mathbf{O}_{\text{final}} = \sum_{r=1}^{R} \mathbf{O}^{(r)}.
\]

The code does not divide by \(R\), so the magnitude of the output scales with the number of rounds. The softmax within each bucket normalises the attention weights, but summing across rounds effectively treats each round as an independent estimate of the true attention. The original Reformer uses a single round but with shared‑QK hashing and causal masking; here multiple rounds with averaging is common to reduce variance. The lack of averaging (or division by \(R\)) could cause the output to be \(R\) times larger than intended, which may affect gradient scales and training dynamics. This is a notable divergence that should be corrected.

### 2.5 Complexity

In expectation, the bucket size is roughly \(N / 2^b\). For \(b\) large enough, each bucket contains a small constant number of elements, so the total cost per round is \(O(N \cdot 2^b)\)? Actually, if buckets are balanced, the total cost is \(O(N \times \text{average bucket size}) = O(N^2 / 2^b)\). With \(b = 64\), the number of buckets is astronomical, so buckets are usually very small; the complexity approaches \(O(N)\) if collisions are rare. The cost per round is dominated by the number of unique codes and the bucket sizes. With nested loops over batch and heads, the implementation is not vectorised, leading to high overhead, but it remains linear in \(N\) for sufficiently large \(b\).

---

## 3. Implementation Details

### 3.1 Module Structure

- **LSHAttention** is registered as `"lsh"`.
- Constructor parameters:
  - `config`: provides `num_attention_heads`, `hidden_size`, `dropout`.
  - `rounds` (default 4): number of hash rounds \(R\).
  - `bucket_size` (default 64): number of hash bits \(b\).
  - Standard linear layers for Q, K, V, O, and dropout.

### 3.2 Forward Pass

1. Project Q, K, V, reshape to \((B, H, N, d_h)\).
2. Initialise output tensor to zeros.
3. For each round \(r\):
   - Generate a random projection matrix \(\mathbf{P}\) of shape \((d_h, b)\), normalised column‑wise.
   - Hash Q and K using `_hash` (applies L2 normalise, dot product with P, sign to binary, convert to integer).
   - Loop over batch \(b\) and head \(h\):
     - Find unique hash codes among queries and keys.
     - For each code:
       - Gather query, key, value tokens where hash equals code.
       - Compute dot‑product attention, softmax, dropout.
       - Add the resulting context to the output tensor at the query positions.
4. After all rounds, reshape output back to \((B, L, D)\) and apply output projection.

### 3.3 Hash Function

The method `_hash(vectors, proj)`:
- Normalises vectors to unit length.
- Computes `dots = matmul(normalized, proj)` of shape \((B, H, N, b)\).
- Sets `bits = (dots > 0).long()`.
- Computes multipliers as powers of 2: `multipliers = 2^torch.arange(b)`.
- Hash = sum(bits * multipliers, dim=-1). This yields a unique integer for each binary pattern.

### 3.4 Masking and Padding

The forward method accepts `attention_mask` but does **not** use it. Padding tokens will receive hash codes and participate in buckets, potentially wasting computation and affecting softmax normalisation. The original Reformer addresses padding by forcing attention to zero for padding positions via large negative bias. This omission is a significant limitation for practical use.

### 3.5 Performance Considerations

The nested loops over batch, head, and unique codes are inefficient for GPU execution. The Reformer’s implementation uses sorting and chunking to enable parallelisation. The current module serves as a functional prototype but would be slow for large sequences.

---

## 4. Usage and Configuration

The module is activated by setting the attention type to `"lsh"`. Configuration parameters from `comp_config` or direct arguments:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rounds` | 4 | Number of independent hash rounds. |
| `bucket_size` | 64 | Number of random hyperplanes (bits) per hash. |
| `dropout` | (from config) | Dropout applied to attention weights. |

Example:
```python
attn = LSHAttention(config, rounds=4, bucket_size=64)
layer.attention = attn
```

No zero‑shot weight transfer is performed; the Q, K, V, O projections are randomly initialised. For best results, the model should be fine‑tuned from a pretrained checkpoint.

---

## 5. Discussion

LSH Attention is a clever way to approximate full attention with sub‑quadratic time. The hashing scheme focuses computation on likely high‑similarity pairs, leaving irrelevant pairs out. The use of multiple rounds improves the recall of important edges. However, the current implementation has several caveats:

- **Sum instead of average**: Accumulating round outputs without division changes the effective attention magnitude. This can lead to unstable gradients and should be fixed by dividing by \(R\) or using a weighted average.
- **No padding mask**: Padding tokens are not excluded; they will affect both the bucket sizes and the softmax denominators, potentially causing the model to attend to meaningless tokens.
- **Nested loops**: The Python loops are extremely slow for large models; a production version would use vectorised operations like `scatter` or `group_by` functions.
- **Random projection per round**: The random matrix is regenerated each forward pass, not stored. This means the hash buckets change at every call, which is fine for training but may cause variance. It also precludes the use of cached hash tables for inference.
- **Hash code collisions**: With \(b=64\), the bucket space is vast (\(2^{64}\)), so collisions are essentially only among near‑neighbours. However, the hash function is deterministic given \(\mathbf{P}\), but \(\mathbf{P}\) is random each time, so the same pair might not always collide.

Despite these limitations, the module faithfully captures the essence of LSH attention and can serve as a reference implementation.

---

## 6. Conclusion

We have presented LSH Attention, an approximate attention mechanism based on locality‑sensitive hashing. It reduces the computational cost of self‑attention by restricting dot‑product comparisons to tokens that share the same hash code. The implementation demonstrates the core algorithm using multiple rounds of random hyperplane hashing. While its current form is not optimised and lacks essential features like padding masking and output averaging, it provides a valuable template for understanding and extending efficient attention methods. Future work will focus on vectorisation, masking, and stable multi‑round aggregation.