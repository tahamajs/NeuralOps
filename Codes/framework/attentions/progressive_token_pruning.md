# Progressive Token Pruning: Learnable Dynamic Token Reduction for Efficient Transformers

**Abstract**

We present **Progressive Token Pruning**, a lightweight, learnable module that dynamically reduces the number of tokens in a Transformer sequence by removing those deemed least important at each layer. A simple linear scorer, followed by a sigmoid activation, assigns an importance score to each token based on its current hidden representation. A fixed keep ratio determines how many tokens are retained, and the top‑\(k\) scoring tokens are gathered while preserving their original sequence order. The module can be inserted between Transformer layers (or after the feed‑forward sub‑layer) to progressively compress the sequence, leading to significant computational savings in deeper layers while maintaining task performance. The method requires minimal additional parameters—just a single linear layer per pruning stage—and is trained end‑to‑end with the main model. This paper provides a rigorous mathematical description of the token scoring and selection mechanism, the training considerations, and the integration of the module into a Transformer pipeline.

---

## 1. Introduction

Transformers process sequences of tokens with a computational cost that scales quadratically with the sequence length in the self‑attention sub‑layers. For many tasks, not all tokens are equally informative; some tokens (e.g., punctuation, articles, or redundant context words) contribute less to the final representation. Token pruning methods aim to identify and remove such tokens on‑the‑fly, reducing the effective sequence length for subsequent layers and thereby lowering the overall computational burden.

**Progressive Token Pruning** implements a simple, parameter‑efficient token removal strategy. For each position in the sequence, a **scoring network**—a single linear transformation followed by a sigmoid—computes an importance score. The scores are used to select the top \(k = \lfloor \rho \cdot L \rfloor\) tokens, where \(\rho \in (0,1)\) is a predefined keep ratio. The selected tokens are then gathered in the original order, so that positional information is preserved (the relative ordering of tokens is retained, though absolute positions may change; a positional embedding update may be needed in practice). The pruned sequence is passed to the next layer, while the removed tokens are discarded.

This module is designed to be inserted after a Transformer layer’s feed‑forward network (or between layers), progressively reducing the sequence length. Because the scorer is trained jointly with the model, the pruning becomes task‑aware and can adapt to the specific needs of downstream tasks. The method adds only \(d_{\text{model}}\) additional parameters per pruning stage (a single weight matrix of size \(d_{\text{model}} \times 1\) plus bias), making it extremely lightweight.

In this paper, we formalise the token importance scoring, the top‑\(k\) selection, and the reordering process. We also discuss the training dynamics, the integration with the DistilBERT compression framework, and the trade‑offs of progressive pruning.

---

## 2. Method

### 2.1 Token Importance Scoring

Given an input tensor \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\), where \(B\) is the batch size, \(L\) the sequence length, and \(D\) the hidden dimension, a scoring network computes a scalar importance for each token:

\[
s_{b,i} = \sigma(\mathbf{W}_s \mathbf{x}_{b,i} + b_s),
\]

where \(\mathbf{W}_s \in \mathbb{R}^{1 \times D}\) is a weight vector (or equivalently a linear layer with output size 1), \(b_s\) is a bias, and \(\sigma\) is the sigmoid function. The score \(s_{b,i} \in (0,1)\) reflects the predicted relevance of token \(i\) in batch element \(b\). The sigmoid ensures scores are normalised between 0 and 1, though they are not required to sum to 1.

### 2.2 Top‑\(k\) Selection with Keep Ratio

A global keep ratio \(\rho \in (0,1]\) (e.g., 0.8) determines the number of tokens to retain:

\[
k = \max(1, \lfloor \rho \cdot L \rfloor).
\]

For each batch element, the indices of the \(k\) tokens with the highest scores are obtained:

\[
\mathcal{I}_b = \operatorname{TopK}(\mathbf{s}_b, k),
\]

where \(\mathbf{s}_b \in \mathbb{R}^L\) is the vector of scores for batch element \(b\). The `torch.topk` function returns both the values and the indices, and we discard the values.

### 2.3 Order Preservation

To maintain the original sequential order of the tokens (which is important for positional encodings and for the model to interpret the sequence correctly), the selected indices are sorted in ascending order:

\[
\tilde{\mathcal{I}}_b = \operatorname{sort}(\mathcal{I}_b).
\]

Sorting ensures that even though the tokens were selected by their scores, they appear in the same relative order as in the original sequence. For example, if the original positions were 2, 5, 1, 8, sorting gives 1, 2, 5, 8. This is crucial because the positional embeddings (if not updated) would otherwise be inconsistent. However, after pruning, the absolute positions have changed; typically, the model re‑embeds the pruned sequence with new positional indices corresponding to the new positions 1,2,...,k. In a progressive pruning pipeline, this re‑indexing must be handled by the layer that calls the pruner (e.g., by passing new token indices).

### 2.4 Token Gathering

The pruned sequence is formed by gathering the token representations at the selected positions:

\[
\tilde{\mathbf{X}}_{b} = \operatorname{gather}(\mathbf{X}_b, \tilde{\mathcal{I}}_b),
\]

resulting in a tensor of shape \((B, k, D)\). The module returns both the pruned sequence and the sorted indices (for potential use in updating token‑level masks or positional information).

### 2.5 Training and Gradient Flow

The top‑\(k\) selection is a discrete operation that is not differentiable with respect to the scores. However, because the selected tokens are gathered and the rest are discarded, gradients can still flow through the selected tokens back to the parameters that produced those tokens (including the scorer weights). The straight‑through estimator is implicitly used: the scorer receives gradients only from the tokens that were retained, providing a signal to increase the scores of useful tokens and decrease the scores of discarded ones. This is effective in practice, as the scorer learns to assign higher importance to tokens that are necessary for the task.

---

## 3. Implementation Details

- **Class `ProgressiveTokenPruning`** inherits from `nn.Module` and is registered under `"progressive_token_pruning"` in the `ATTENTION_REGISTRY`. This registration suggests that the module can be used as a replacement for an attention layer (though it is a pruning module, not an attention module). In the context of the compression framework, it might be intended to replace the entire attention block with a pruner (perhaps as an extreme compression) or to be used as a sub‑layer within a larger block.
- **Constructor**:
  - `d_model`: the hidden dimension \(D\).
  - `keep_ratio`: the fraction \(\rho\) of tokens to keep (default 0.8).
  - `self.score_network`: a linear layer mapping \(D \to 1\) (so weight shape \(1 \times D\)) with a bias.
- **Forward**:
  1. Compute scores using sigmoid: `scores = sigmoid(score_network(x)).squeeze(-1)` → shape \((B, L)\).
  2. Compute \(k = \max(1, \text{int}(L * \text{keep\_ratio}))\).
  3. Get top‑\(k\) indices via `torch.topk(scores, k, dim=-1)` → indices shape \((B, k)\).
  4. Sort the indices along the last dimension to preserve order.
  5. Gather the tokens: `pruned_x = x.gather(1, indices.unsqueeze(-1).expand(-1, -1, D))`.
  6. Return `(pruned_x, indices)`.

The module does not modify positional embeddings; the calling code is responsible for updating the position IDs if necessary.

---

## 4. Usage and Configuration

The module can be injected into a model by setting the attention type to `"progressive_token_pruning"` in the compression configuration. The relevant hyperparameter is `keep_ratio`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `keep_ratio` | 0.8 | Fraction of tokens to retain after pruning. |

Example injection via framework:
```python
comp_config.attention_type = "progressive_token_pruning"
comp_config.keep_ratio = 0.9
```

Manual instantiation:
```python
pruner = ProgressiveTokenPruning(d_model=768, keep_ratio=0.9)
pruned_x, indices = pruner(hidden_states)
```

Since this module replaces an attention layer in the registry, it is expected that the surrounding code (e.g., `replace_attention`) will handle the wiring correctly. In a typical Transformer, one would insert this pruner after the feed‑forward network, not as a replacement for the entire attention mechanism. The registration as an attention replacement may be a misuse or a shorthand for a broader block replacement. Care must be taken to ensure that the pruner is integrated at the correct position (e.g., after the FFN and before the next layer).

---

## 5. Discussion

Progressive Token Pruning offers a simple yet effective approach to sequence compression. Its key advantages are:
- **Minimal parameters**: only \(D\) additional weights per pruning stage.
- **Speed**: the scoring, top‑\(k\), and gathering operations are efficient on GPUs.
- **Task‑adaptive**: the scorer is trained end‑to‑end, learning which tokens are dispensable for the specific task.
- **Order preservation**: sorting ensures the sequence order is maintained, which is critical for language understanding.

However, there are several limitations and considerations:
- **Fixed keep ratio**: The fraction is constant across all layers and samples, which may not be optimal. Dynamic or learned ratios could improve flexibility.
- **Non‑differentiable selection**: The top‑\(k\) operation blocks gradient flow to discarded tokens. This can cause the scorer to receive no feedback from tokens that were dropped, possibly leading to suboptimal assignment if important tokens are initially dropped and never recovered. Warm‑up training with no pruning or a gradually decreasing keep ratio is often necessary.
- **Positional encoding update**: The pruner returns new indices, but the positional encodings must be adjusted accordingly. If the pruner is inserted inside a layer, the subsequent layers must recalculate positional embeddings for the shortened sequence. In standard DistilBERT, positional embeddings are added only at the first layer; pruning would require either re‑adding positional embeddings or using relative positional encoding (e.g., RoPE) that handles varying positions naturally. If absolute positions are used, the model would misinterpret the new positions. This module does not handle that; the caller must manage it.
- **Usage as attention replacement**: Registering a token pruning module in the attention registry is unconventional. It suggests that the module might be used in an extreme compression scenario where the entire attention and FFN are replaced by a simple pruning step (perhaps followed by a tiny network). However, in the provided snippet, there is no attention at all—only pruning. This would cripple the model unless the pruning is intended to be part of a larger block that still includes attention (e.g., as in EATBlock). This misuse may be specific to the testing or modular design of the framework.

Despite these caveats, the core pruning logic is sound and widely used in the literature (e.g., Learned Token Pruning, DynamicViT). It serves as a building block for more complex efficient Transformer designs.

---

## 6. Conclusion

We have described Progressive Token Pruning, a lightweight, learnable module that dynamically reduces the sequence length by retaining the most important tokens. The scoring network, top‑\(k\) selection, and order‑preserving gathering form a simple yet effective pipeline. When integrated appropriately into a Transformer model, it can significantly reduce computation in deeper layers with minimal loss in accuracy. Future improvements could include dynamic keep ratios, better handling of positional encodings, and integration with attention modules that natively support variable sequence lengths. The module stands as a clear example of how a small amount of task‑specific learning can lead to substantial efficiency gains in Transformer inference.