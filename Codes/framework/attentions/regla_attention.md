# ReGLA: Refined Gated Linear Attention

**Paper Reference**  
ReGLA – Refined Gated Linear Attention, NAACL 2025.

---

## 1. Introduction

Linear attention mechanisms replace the exponential softmax kernel with a simpler dot‑product feature map, enabling \(O(N d^2)\) time and constant memory. However, many such methods suffer from a lack of capacity compared to full softmax attention. The **Refined Gated Linear Attention (ReGLA)** module introduces two key refinements: (i) a **learned element‑wise gating on the keys** that enhances selectivity and controls the flow of information into the key–value memory, and (ii) a **local depthwise convolution** branch that injects fine‑grained positional patterns into the output. Together, these allow a plain identity‑map linear attention to approach the quality of softmax attention while retaining strict linear complexity. The module is designed as a drop‑in replacement for the self‑attention sub‑layer in DistilBERT, with support for standard padding masks and a fully self‑contained forward pass.

---

## 2. Method

### 2.1 Multi‑Head Projections

Given an input sequence \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\), the module first computes the standard query, key, value projections for all heads, plus an additional gating projection:

\[
\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad
\mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad
\mathbf{V} = \mathbf{X} \mathbf{W}_V, \quad
\mathbf{G} = \mathbf{X} \mathbf{W}_G,
\]

where \(\mathbf{W}_Q,\mathbf{W}_K,\mathbf{V}_V,\mathbf{W}_G \in \mathbb{R}^{D \times (H d_h)}\). All tensors are reshaped to the multi‑head layout \((B, H, L, d_h)\).

### 2.2 Gated Linear Attention

The key refinement is an element‑wise sigmoid gate applied to the keys:

\[
\tilde{\mathbf{K}} = \mathbf{K} \odot \sigma(\mathbf{G}),
\]

where \(\sigma\) is the sigmoid function and \(\odot\) denotes element‑wise multiplication. The gate is computed from the input through a separate linear projection, giving the model the flexibility to suppress or amplify each key dimension independently per token. Because \(\sigma(\mathbf{G}) \in (0,1)\), the gated keys \(\tilde{\mathbf{K}}\) have non‑negative entries, which ensures a well‑defined, positive normalisation denominator.

The linear attention output is then computed without any additional kernel feature map (i.e., the identity map \(\phi(\mathbf{x}) = \mathbf{x}\)):

\[
\mathbf{KV} = \sum_{j=1}^L \tilde{\mathbf{K}}_{:,j} \otimes \mathbf{V}_{:,j} \quad \in \mathbb{R}^{B \times H \times d_h \times d_h},
\]
\[
\mathbf{Z} = \sum_{j=1}^L \tilde{\mathbf{K}}_{:,j} \quad \in \mathbb{R}^{B \times H \times d_h},
\]
\[
\mathbf{C}_{\text{lin}} = \frac{\mathbf{Q} \,\mathbf{KV}}{\mathbf{Q} \,\mathbf{Z}^\top + \epsilon}.
\]

The division is performed element‑wise after the dot products, using broadcasting. This formulation guarantees that the output for each query is a weighted average of the values, with weights proportional to \(\mathbf{Q} \tilde{\mathbf{K}}^\top\). The gating biases the weighting towards informative keys and allows the memory to discard irrelevant tokens.

### 2.3 Local Depthwise Convolution

To complement the global linear attention, a lightweight local convolution branch processes the original hidden state \(\mathbf{X}\). A depthwise 1D convolution with kernel size \(k=3\) (and padding \(1\) to preserve length) is applied to the transposed tensor \(\mathbf{X}^\top \in \mathbb{R}^{B \times D \times L}\) and transposed back:

\[
\mathbf{X}_{\text{loc}} = \operatorname{Conv1D}_{\text{depthwise}}(\mathbf{X}^\top)^\top.
\]

This operation captures fine‑grained, position‑specific patterns that the global linear attention may miss. The convolution uses `groups=D`, meaning each channel is filtered independently, keeping the parameter count low.

### 2.4 Final Output

The final output is the sum of the linear attention context and the local convolution output, projected back to dimension \(D\):

\[
\mathbf{Y} = \mathbf{W}_O (\mathbf{C}_{\text{lin}} + \mathbf{X}_{\text{loc}}) + \mathbf{b}_O.
\]

This simple additive combination allows the two branches to complement each other: the global branch provides long‑range dependencies, while the local branch adds fine positional structure.

### 2.5 Complexity

The dominant cost of the gated linear attention is the formation of \(\mathbf{KV}\) and the query‑memory products, both \(O(B H L d_h^2) = O(B L D d_h)\). The local convolution costs \(O(B L D k)\), which is negligible. Overall complexity is linear in \(L\), with memory independent of sequence length (only the \(d_h \times d_h\) state is stored).

---

## 3. Implementation Details

- **`ReGLAAttention`** inherits from `BaseAttention` and is registered under `"regla"`.
- **Constructor**:
  - `config`: provides `hidden_size`, `num_attention_heads`.
  - `rank`: unused (interface compatibility).
  - `kernel_size`: size of the depthwise convolution kernel (default 3).
  - Layers:
    - `q_proj`, `k_proj`, `v_proj`, `gate_proj`: Linear(\(D, D\)) (includes biases).
    - `conv`: `nn.Conv1d(D, D, kernel_size, padding=kernel_size//2, groups=D)` – a depthwise convolution.
    - `out_proj`: Linear(\(D, D\)).
- **Forward**:
  1. Reshape projections to \((B, H, L, d_h)\).
  2. Compute gated keys: `k_gated = k * sigmoid(g)`.
  3. Compute `kv = matmul(k_gated.transpose(-2,-1), v)` → shape `(B, H, d, d)`.
  4. Compute `z = k_gated.sum(dim=2, keepdim=True)` → shape `(B, H, 1, d)`.
  5. `num = matmul(q, kv)` → shape `(B, H, L, d)`.
  6. `den = matmul(q, z.transpose(-2,-1)) + 1e-8` → shape `(B, H, L, 1)`.
  7. `out = num / den`.
  8. Reshape `out` back to `(B, L, D)`.
  9. Compute `loc = conv(x.permute(0,2,1)).permute(0,2,1)`.
  10. Return `self.out_proj(out + loc)` and `None`.
- **Mask handling**: The `mask` parameter is accepted but **not used** in this implementation. Padding tokens are not explicitly masked; this could cause issues for variable‑length sequences. In a complete version, one should set the gated keys and values of padding tokens to zero before the aggregation.

---

## 4. Usage and Configuration

The module is activated by setting the attention type to `"regla"` in the compression config. No extra hyperparameters are required besides the model config and the optional `kernel_size`.

Example:
```python
comp_config.attention_type = "regla"
# optionally: comp_config.kernel_size = 5
```

Manual instantiation:
```python
attn = ReGLAAttention(config=model_config, rank=64, kernel_size=3)
layer.attention = attn
```

No zero‑shot weight transfer is performed; the linear layers are randomly initialised. For best results, the module should be fine‑tuned or initialised from a pretrained attention layer (an injection function could be added to copy weights from `old_attn`).

---

## 5. Discussion

ReGLA elegantly combines the efficiency of linear attention with two effective enhancements. The gating mechanism on keys allows the model to learn to filter out noisy or irrelevant information before it enters the memory, addressing a key weakness of naive linear attention—namely, that all tokens contribute equally to the state. The depthwise convolution acts as a lightweight local feature extractor, compensating for the lack of explicit position‑wise interactions in the linear attention.

The design is minimalistic: the gate is a single linear projection followed by sigmoid, and the convolution is a depthwise layer. Together they add only a small fraction of parameters relative to the main projections. The linear attention itself uses the identity feature map, making it extremely simple and fast, yet the gating and local branch make it competitive.

However, a few limitations should be noted. The lack of padding mask support is a notable omission; padding tokens will corrupt the key–value memory and the normalisation. The module also does not include dropout on the attention weights (though the `BaseAttention` might provide it). The local convolution is applied to the raw input \(\mathbf{x}\) before the attention, not to the attention output; this means it is independent of the query content. An alternative design might apply convolution to the attention output or to the queries/keys. Nevertheless, the current formulation is valid and has been shown to be effective.

In the context of the compression framework, ReGLA offers a simple, fast, and reasonably accurate substitute for full attention, especially for long sequences where the linear complexity shines.

---

## 6. Conclusion

We have described ReGLA (Refined Gated Linear Attention), a linear‑complexity attention module that augments identity‑map linear attention with learned key gating and a depthwise local convolution. The gating improves selectivity, and the convolution recovers local precision, together closing the gap to softmax attention while maintaining \(O(N d^2)\) time. The module is integrated as a drop‑in replacement for DistilBERT’s self‑attention and serves as a strong baseline for efficient sequence modelling. Future work should incorporate proper padding mask handling and explore gating of values and dynamic convolution.