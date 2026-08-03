# EATBlock: Efficient Attention Transformer Block with Progressive Token Pruning

**Abstract**

We introduce the **EATBlock** (Efficient Attention Transformer block), a standard Transformer block augmented with an optional **progressive token pruning** module. The block consists of multi‑head self‑attention and a feed‑forward network, each wrapped with residual connections and layer normalisation, following the classic Transformer architecture. After the feed‑forward sub‑layer, a pruning layer can be applied to dynamically reduce the number of tokens, discarding less informative ones according to a learned or heuristic scoring function. This design allows the model to gradually compress the sequence length across layers, significantly reducing the computational cost of self‑attention in deeper layers while preserving critical information. The module is intended for long‑sequence processing where early layers operate on a larger number of tokens and later layers on a progressively pruned set. It integrates seamlessly with standard Transformer frameworks and can be used as a drop‑in replacement for a full attention layer in models such as DistilBERT.

---

## 1. Introduction

The quadratic complexity of self‑attention with respect to sequence length makes it challenging to apply Transformers to very long documents. One popular strategy to mitigate this cost is **token pruning**—removing tokens that are deemed unimportant for the task at later stages of the network. The EATBlock formalises this idea by coupling a standard Transformer block with a progressive pruning mechanism. Unlike methods that prune only after the final layer or in a single step, progressive pruning reduces the sequence length gradually over multiple layers, enabling a smooth trade‑off between accuracy and efficiency.

In this paper, we detail the architecture of the EATBlock, which consists of a multi‑head self‑attention sub‑layer, a position‑wise feed‑forward network, and an optional progressive token pruning module applied after the feed‑forward transformation. The pruning module is designed to be modular; any token pruning algorithm (e.g., based on attention scores, token importance, or a learned policy) can be plugged in. The default implementation uses `ProgressiveTokenPruning`, a companion module that implements progressive removal of tokens using a learnable scoring network and a scheduling strategy.

---

## 2. Method

### 2.1 Standard Transformer Block

The core of the EATBlock is a classic Transformer block. Given an input sequence of hidden states \(\mathbf{X} \in \mathbb{R}^{B \times L \times d_{\text{model}}}\), the block computes:

\[
\mathbf{A} = \text{MultiHeadAttention}(\mathbf{X}, \mathbf{X}, \mathbf{X}),
\]
\[
\mathbf{X}_1 = \text{LayerNorm}(\mathbf{X} + \mathbf{A}),
\]
\[
\mathbf{F} = \text{FFN}(\mathbf{X}_1) = \mathbf{W}_2 \, \text{GELU}(\mathbf{W}_1 \mathbf{X}_1 + \mathbf{b}_1) + \mathbf{b}_2,
\]
\[
\mathbf{X}_2 = \text{LayerNorm}(\mathbf{X}_1 + \mathbf{F}).
\]

The multi‑head attention uses `nn.MultiheadAttention` with `batch_first=True`, performing self‑attention without an external mask (padding masks can be added if needed). The feed‑forward network is a two‑layer MLP with an expansion factor of 4 and GELU activation.

### 2.2 Progressive Token Pruning

If a pruning module is provided (`prune_layer` is not `None`), it is applied to the output of the FFN sub‑layer:

\[
\mathbf{X}_{\text{pruned}}, \mathbf{I}_{\text{new}} = \text{Prune}(\mathbf{X}_2),
\]

where \(\mathbf{I}_{\text{new}} \in \mathbb{N}^{B \times L_{\text{new}}}\) are the indices of the surviving tokens. The module is expected to return a reduced sequence length \(L_{\text{new}} \le L\). If the original token indices (e.g., word piece IDs) are provided via `token_indices`, they are gathered using the new indices to keep alignment:

\[
\mathbf{T}_{\text{new}} = \mathbf{T}_{\text{old}}.\text{gather}(1, \mathbf{I}_{\text{new}}).
\]

The pruning mechanism is **progressive**: at each layer that has pruning enabled, a certain fraction of tokens are removed. The `ProgressiveTokenPruning` module (imported but not detailed in this snippet) typically includes:

- A **scoring function** that assigns an importance score to each token (e.g., using the attention weights or a lightweight linear layer).
- A **pruning schedule** that determines how many tokens to keep at each layer (e.g., linear decay of sequence length).
- A **masking** strategy that sets the outputs of pruned tokens to zero or simply removes them, as done here.

By pruning tokens after the FFN sub‑layer, the subsequent layer receives a shorter sequence, thereby reducing the cost of the following multi‑head attention.

### 2.3 Overall Forward Pass

The forward method accepts two inputs:

- `x`: tensor of shape \((B, L, d_{\text{model}})\).
- `token_indices`: optional tensor of shape \((B, L)\) containing original token IDs, used to track which tokens survive after pruning.

If pruning is not active, the block simply returns \(\mathbf{X}_2\) and the unchanged `token_indices`. If pruning is active, it returns the pruned sequence and the updated token indices (or `None` if token indices were not provided).

---

## 3. Implementation Details

The `EATBlock` class is defined as follows:

- **Constructor**:
  - `d_model`: hidden dimension.
  - `n_heads`: number of attention heads.
  - `prune_layer`: an instance of `ProgressiveTokenPruning` (or any compatible module) that implements the pruning logic. If `None`, no pruning occurs.
  - It creates `self.attn` (a PyTorch `MultiheadAttention` with `batch_first=True`), `self.ffn` (a Sequential with Linear→GELU→Linear), and two `LayerNorm` layers. The FFN uses an inner dimension of \(4 \times d_{\text{model}}\).

- **Forward**:
  - Computes attention with `self.attn(x, x, x)`, which returns `attn_output, attn_output_weights`.
  - Adds the residual, applies norm1.
  - Computes FFN, adds residual, applies norm2.
  - If `self.prune_layer` is not None, calls it on `x` to get pruned hidden states and new indices; then optionally gathers `token_indices` using the new indices.
  - Returns the (pruned or full) hidden states and the updated token indices.

The module does **not** include dropout explicitly, but dropout can be injected via the `MultiheadAttention`’s built‑in dropout or by adding dropout layers in the FFN. In the provided code, dropout is not configured, but in a full implementation it would be.

---

## 4. Usage and Configuration

The block can be integrated into any Transformer architecture. To enable progressive pruning, instantiate a `ProgressiveTokenPruning` module (not shown, but presumably from the same framework) with parameters like initial sequence length, target sequence length, pruning schedule, etc., and pass it to the `EATBlock`.

Example:

```python
pruner = ProgressiveTokenPruning(
    d_model=d_model,
    prune_ratio=0.1,            # fraction of tokens to remove per pruning step
    min_tokens=16,
    start_layer=2,              # layer index when pruning begins
    pruning_schedule='linear'
)
block = EATBlock(d_model=768, n_heads=12, prune_layer=pruner)
```

If no pruning is desired, omit the `prune_layer` argument, and the block behaves identically to a standard Transformer block.

**Configuration parameters** (indirect, via the pruning module) may include:

| Parameter | Description |
|-----------|-------------|
| `prune_ratio` | Fraction of tokens pruned at each pruning layer. |
| `min_tokens` | Minimum number of tokens to retain. |
| `start_layer` | The first layer where pruning becomes active. |
| `pruning_schedule` | How the number of tokens decays (e.g., linear, exponential). |

The block itself has no extra hyperparameters beyond those of the standard attention and FFN.

---

## 5. Discussion

The EATBlock provides a simple, modular way to incorporate token pruning into a Transformer. By decoupling the pruning operation from the attention mechanism, it can be combined with any attention variant and any token pruning algorithm. Progressive pruning offers a better accuracy–efficiency trade‑off than one‑shot pruning because it allows the model to gradually refine the selection of important tokens as the representations become more abstract.

However, the current implementation uses a separate pruning module that is executed sequentially after the FFN. This adds a small overhead per layer. Moreover, the pruning decision is based on the current layer’s output; if the scoring function is not well‑aligned with the task, important tokens might be discarded prematurely. The module also assumes that the pruning layer returns indices and that the subsequent layer can handle a variable sequence length (which is typically possible if the subsequent attention layer uses the new, shorter sequence). When integrating with standard Transformer libraries that expect a fixed sequence length, the pruned tokens may need to be masked rather than physically removed, which could require modifications to the attention code.

Another consideration is that the `nn.MultiheadAttention` used here does not accept a causal mask for autoregressive tasks; for decoder models, a causal mask must be added. The current design focuses on encoder‑style (bidirectional) processing.

---

## 6. Conclusion

The EATBlock is a versatile Transformer block that combines standard self‑attention and feed‑forward layers with an optional progressive token pruning stage. This design enables efficient processing of long sequences by gradually removing redundant tokens, thereby reducing the computational burden of later layers. Its modular architecture allows easy integration with existing Transformer codebases and various pruning strategies. Future work will explore learned pruning policies, dynamic scheduling, and integration with other efficient attention mechanisms to further enhance performance and scalability.