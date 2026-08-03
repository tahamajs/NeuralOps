# MixAttention‑MoSKA: Batch‑Shared Key‑Value GEMM with FASA Pruning for Efficient Attention

**Abstract**

We introduce **MixAttention‑MoSKA**, an efficient attention module that combines two orthogonal strategies for reducing the cost of self‑attention: (1) the concept of **batch‑shared key‑value (KV) operations via GEMM** to exploit commonalities across sequences, and (2) **FASA (Filtering and Adaptive Sparsity Attention) pruning** of tokens based on a variance‑based importance score. While the current implementation focuses on the FASA token pruning component—selecting a fixed fraction of tokens according to their feature variance across heads—the module is designed as a drop‑in replacement for the standard attention layer in DistilBERT. It dynamically reduces the number of key‑value tokens used in the attention computation, achieving sub‑quadratic complexity with a controlled sparsity ratio. The module also includes placeholders for more advanced masking and the batch‑shared GEMM strategy. This paper provides a rigorous mathematical description of the FASA pruning mechanism, the attention computation, and the integration into the compression framework.

---

## 1. Introduction

Standard multi‑head self‑attention scales quadratically with the sequence length \(N\) because every query interacts with every key. Many methods have been proposed to alleviate this by pruning or selecting a subset of key‑value tokens. The **MixAttention‑MoSKA** module incorporates a simple yet effective token pruning technique called **FASA (Filtering and Adaptive Sparsity Attention)**, which dynamically identifies and retains the most informative tokens based on their variance across the head dimension. By keeping only a fraction \(\rho\) (e.g., \(\rho = 0.189\)) of the tokens for key‑value computation, the attention cost is reduced to \(O(N \times (\rho N)) = O(\rho N^2)\), which is still quadratic but with a constant factor reduction, and can be further enhanced by combining with other strategies.

The name “MoSKA” suggests a multi‑strategy approach: **Mo** (maybe “mixture” or “multiple”), **S** (sparse), **KA** (key‑value attention). The module also hints at **batch‑shared KV as GEMM**, a technique where common key‑value pairs across batch elements are merged and computed once, but this component is not implemented in the provided code and is left for future work.

The FASA pruning is simple: for each token, an importance score is computed as the variance of its representation across the head dimension, averaged over all heads. Tokens with higher variance are deemed more informative and are retained. The top‑\(\lfloor \rho N \rfloor\) tokens are kept for key‑value access, while the rest are discarded. The attention output is then computed using only the pruned keys and values. This method assumes that tokens with higher cross‑head variance carry more diverse information and are more critical for context.

In this paper, we detail the mathematical formulation of FASA pruning, the overall attention module, and discuss its limitations and potential extensions.

---

## 2. Method

### 2.1 Multi‑Head Projections

Given an input hidden state \(\mathbf{X} \in \mathbb{R}^{B \times N \times D}\), the module first applies standard linear projections for all heads:

\[
\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad
\mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad
\mathbf{V} = \mathbf{X} \mathbf{W}_V,
\]

with weight matrices \(\mathbf{W}_* \in \mathbb{R}^{D \times (H d_h)}\), where \(H\) is the number of heads and \(d_h = D/H\). The tensors are reshaped and transposed to \((B, H, N, d_h)\).

### 2.2 FASA Token Importance

For each token across all heads, we compute an importance score that captures how much the token’s representation varies across heads. Specifically, for key tensor \(\mathbf{K}\) of shape \((B, H, N, d_h)\), we first compute the variance along the last dimension (the feature dimension within each head), giving a tensor of shape \((B, H, N)\). This variance captures the spread of the vector’s components within each head. Then, we average this variance across heads, yielding a per‑token importance score for each batch element:

\[
s_{b,i} = \frac{1}{H} \sum_{h=1}^{H} \operatorname{Var}\!\big[ \mathbf{K}_{b,h,i} \big].
\]

The variance is computed as \(\operatorname{Var}(\mathbf{x}) = \frac{1}{d_h} \sum_{j=1}^{d_h} (x_j - \bar{x})^2\). Because the tensors are large, the code uses `k.var(dim=-1).mean(dim=1)` which gives exactly this quantity.

We then select the top‑\(K\) tokens with the highest importance, where \(K = \max(1, \lfloor N \times \rho \rfloor)\), and \(\rho = \text{fasa\_keep\_ratio}\) (default 0.189). This selection is performed across the sequence dimension for each batch element independently. The indices of the selected tokens are gathered, and we extract the corresponding key and value slices:

\[
\mathbf{K}_{\text{pruned}} = \operatorname{gather}(\mathbf{K}, \text{indices}), \qquad
\mathbf{V}_{\text{pruned}} = \operatorname{gather}(\mathbf{V}, \text{indices}),
\]

both of shape \((B, H, K, d_h)\). The indices are expanded along the head and feature dimensions for gathering.

### 2.3 Attention with Pruned Keys and Values

The attention scores are computed between the full queries \(\mathbf{Q}\) (shape \((B, H, N, d_h)\)) and the pruned keys \(\mathbf{K}_{\text{pruned}}\):

\[
\mathbf{S} = \frac{\mathbf{Q} \mathbf{K}_{\text{pruned}}^\top}{\sqrt{d_h}} \in \mathbb{R}^{B \times H \times N \times K}.
\]

No masking is applied in the current implementation; the `mask` argument, if provided, raises a `NotImplementedError` because adapting the mask to the pruned token set is non‑trivial and not handled. Softmax is then applied to obtain attention weights \(\mathbf{A} = \operatorname{softmax}(\mathbf{S})\), and the output is computed as:

\[
\mathbf{O} = \mathbf{A} \mathbf{V}_{\text{pruned}} \in \mathbb{R}^{B \times H \times N \times d_h}.
\]

The heads are then transposed, concatenated, and projected:

\[
\mathbf{O}_{\text{out}} = \mathbf{O}_{\text{flat}} \mathbf{W}_O.
\]

### 2.4 Complexity

The key‑value length is reduced from \(N\) to \(\rho N\). The attention computation cost is \(O(B H N (\rho N) d_h) = O(B D N^2 \rho)\). For small \(\rho\), this yields a significant reduction (e.g., \(\rho = 0.189\) gives ~5× less key‑value tokens). However, the sorting and gathering operations add a moderate overhead. The method remains quadratic in \(N\) but with a reduced constant.

### 2.5 Batch‑Shared KV GEMM (Placeholder)

The docstring mentions “batch‑shared KV as GEMM”, which suggests that in a batch, some key‑value pairs could be shared across sequences (e.g., in a multi‑turn conversation or retrieval‑augmented generation). A shared KV cache could be represented as a single GEMM operation, amortising computation. This feature is not implemented in the current code and is left as a future extension.

---

## 3. Implementation Details

- **`MixAttentionMoSKA`** inherits from `BaseAttention` and is registered as `"mixattention_moska"`.
- Constructor parameters:
  - `config`: provides `hidden_size`, `num_attention_heads`.
  - `rank`: ignored (interface compatibility).
  - `fasa_keep_ratio`: the fraction \(\rho\) of tokens to keep (default 0.189).
  - Standard linear layers `W_q`, `W_k`, `W_v` (no bias) and `out_proj` (with bias).
- **`_fasa_prune(k, v)`**:
  - Computes importance as `k.var(dim=-1).mean(dim=1)` (variance within each head, averaged over heads).
  - Determines `keep = max(1, int(L * self.fasa_keep_ratio))`.
  - Uses `torch.topk` to get indices of top‑\(K\) importance scores.
  - Expands indices to gather along the sequence dimension for both keys and values.
  - Returns pruned key and value tensors.
- **Forward**:
  1. Projects input to Q, K, V.
  2. Calls `_fasa_prune` on K, V.
  3. Computes attention scores using Q and pruned K.
  4. If `mask` is provided, raises `NotImplementedError`.
  5. Softmax, matrix multiply with pruned V.
  6. Reshape, output projection, return `(out, None)`.

- **Masking**: Not supported; the module is intended for sequences without padding or for causal contexts where masking can be omitted. Future work could incorporate mask gathering analogous to token pruning.

---

## 4. Usage and Configuration

The module is activated by setting the attention type to `"mixattention_moska"`. The only configurable parameter (via `comp_config`) is `fasa_keep_ratio`. Example:

```python
comp_config.attention_type = "mixattention_moska"
comp_config.fasa_keep_ratio = 0.2
```

Manual instantiation:
```python
attn = MixAttentionMoSKA(config=model_config, fasa_keep_ratio=0.2)
layer.attention = attn
```

No zero‑shot weight transfer is performed; the linear projections are randomly initialised. To preserve pretrained knowledge, one would need to copy the original weights into `W_q`, etc., which is not handled by the current module. An injection function could be added for that.

---

## 5. Discussion

FASA pruning is a lightweight way to reduce the key‑value footprint, motivated by the observation that some tokens are more “informative” than others. The variance‑based importance score is intuitive: tokens whose representations vary more across heads likely contain more diverse information and are worth attending to. However, the method has several limitations:

- **No mask support**: Handling padding masks with variable token counts is tricky; tokens that are padding might still be selected if they have high variance. A correct implementation would need to mask out padding tokens before top‑k selection.
- **Fixed keep ratio**: The fraction is constant regardless of sequence length or task. An adaptive mechanism could be beneficial.
- **Importance metric simplicity**: Variance might not always correlate with usefulness. Learned importance scoring (e.g., using a small MLP) could improve performance.
- **Batch‑shared KV not implemented**: The GEMM sharing concept is only hinted at, leaving potential performance on the table.

Despite these, the module provides a clear template for pruning‑based efficient attention and can serve as a baseline for more sophisticated token selection methods.

---

## 6. Conclusion

We have presented MixAttention‑MoSKA, an attention module that applies FASA token pruning to reduce the key‑value length and thus the attention computation cost. The method selects tokens based on cross‑head variance and keeps a fixed fraction. While the current implementation lacks masking and the advertised batch‑shared GEMM component, it demonstrates a simple pruning strategy that can be integrated into the DistilBERT compression framework. Future work will add mask support, explore learned importance, and implement the batch‑shared KV GEMM for additional efficiency gains.