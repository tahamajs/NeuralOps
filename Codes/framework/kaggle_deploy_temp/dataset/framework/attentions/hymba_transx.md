# Hymba‑TransX: A Hybrid Block with Differential Attention, State‑Space Model, and Unified Rotary Position Embeddings

**Abstract**

We introduce the **Hymba‑TransX** block, a hybrid sequence mixing module that fuses two complementary mechanisms: **differential self‑attention** and a **state‑space model (SSM)**. The attention component uses a novel “differential attention” scheme, where two sets of queries and keys are computed from the rotary‑position‑encoded input, and the final attention weights are obtained by subtracting half of one attention map from the other, followed by a ReLU activation. This operation is intended to enhance sparsity and emphasise salient token interactions. The SSM component (referred to as “Mamba” or `SimpleSSM`) provides a lightweight linear‑time recurrence that captures long‑range dependencies with state‑space dynamics. The outputs of both branches are blended using a learned scalar gate, and the result is combined with the original input via a residual connection and a final linear projection. Additionally, a **Unified Rotary Position Embedding (RoPE)** is applied to the full hidden states before any branching, enabling the model to incorporate relative positional information across both pathways. The Hymba‑TransX block is designed as a drop‑in replacement for the standard self‑attention layer in Transformer models and is integrated into the DistilBERT compression framework through a registry‑based injection function. This paper provides a rigorous mathematical description of the Unified RoPE, the differential attention mechanism, the integration with the SSM, and the overall architecture.

---

## 1. Introduction

The standard Transformer architecture relies on multi‑head self‑attention, which captures pairwise token dependencies with quadratic complexity. To improve efficiency and capacity, recent research has explored hybrid models that combine attention with other sequence mixing layers, such as state‑space models (SSMs) like Mamba. These hybrids aim to leverage the global receptive field of attention while benefiting from the linear complexity of SSMs. Furthermore, modifications to the attention mechanism itself, such as differential attention, can enhance sparsity and focus.

The **Hymba‑TransX** block implements such a hybrid. It first applies a unified rotary position embedding to the entire hidden state, ensuring that both the attention and the SSM branches have access to explicit relative positional information. The attention branch then uses a **differential attention** formulation: two distinct sets of queries and keys produce two independent attention maps, and the final weights are given by the element‑wise ReLU of the difference between the first map and half of the second map. This non‑linear combination encourages the model to attend only where the first map sufficiently dominates the second, yielding sparser attention. The SSM branch processes the same position‑aware input through a simple state‑space model (denoted `SimpleSSM`), which is assumed to be a linear recurrence akin to the HGRN or Mamba‑style sequence model. The outputs of the two branches are then combined via a learned scalar gate \(\alpha\), and the input is added back (as a residual) before a final output projection.

The block is highly modular and can be injected directly into a Transformer layer, replacing its attention sub‑layer. The name “Hymba” is likely a portmanteau of “Hybrid” and “Mamba”, and “TransX” indicates a Transformer extension. This paper details the mathematical formulation of each component, the implementation considerations, and the integration into the compression framework.

---

## 2. Method

### 2.1 Unified Rotary Position Embedding (Unified RoPE)

Rotary Position Embedding (Su et al., 2024) encodes position information by rotating the feature vectors in a pairwise manner. The **UnifiedRoPE** class generalises this to the full hidden dimension \(D\) before the features are split into heads. It precomputes sinusoidal frequencies and applies the rotation to the entire sequence.

Let \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\) be the input. The module precomputes inverse frequencies:

\[
\theta_i = \text{base}^{-2i/D}, \quad i = 0, 1, \dots, \frac{D}{2}-1.
\]

For positions \(t = 0, \dots, L-1\), the frequencies are \(\Theta_{t,i} = t \cdot \theta_i\). Then the rotation matrices are defined by:

\[
\mathbf{C}_{t,i} = \cos(\Theta_{t,i}), \quad \mathbf{S}_{t,i} = \sin(\Theta_{t,i}).
\]

For a vector \(\mathbf{x} \in \mathbb{R}^D\), the RoPE transformation pairs consecutive dimensions: for each pair \((x_{2i}, x_{2i+1})\), the rotated pair is:

\[
\begin{bmatrix}
x'_{2i} \\
x'_{2i+1}
\end{bmatrix}
=
\begin{bmatrix}
x_{2i} \cos\Theta_{t,i} - x_{2i+1} \sin\Theta_{t,i} \\
x_{2i+1} \cos\Theta_{t,i} + x_{2i} \sin\Theta_{t,i}
\end{bmatrix}.
\]

In the implementation, the cos and sin tables are cached with shape \((1, L_{\text{max}}, 1, D)\). During the forward pass, for a given offset (for incremental decoding), the relevant slice is taken. The input’s even and odd channels are extracted, scaled by cos and sin accordingly, and interleaved back. This unified RoPE ensures that both the attention and SSM pathways receive the same position‑aware representations.

### 2.2 Differential Attention

After applying RoPE, we obtain \(\mathbf{X}_{\text{rope}} \in \mathbb{R}^{B \times L \times D}\). The attention branch computes two separate sets of queries and keys, while the values are shared:

\[
\mathbf{Q}_1 = \mathbf{X}_{\text{rope}} \mathbf{W}_{Q1}, \quad
\mathbf{K}_1 = \mathbf{X}_{\text{rope}} \mathbf{W}_{K1},
\]
\[
\mathbf{Q}_2 = \mathbf{X}_{\text{rope}} \mathbf{W}_{Q2}, \quad
\mathbf{K}_2 = \mathbf{X}_{\text{rope}} \mathbf{W}_{K2},
\]
\[
\mathbf{V} = \mathbf{X}_{\text{rope}} \mathbf{W}_{V},
\]

where \(\mathbf{W}_{Q1}, \mathbf{W}_{K1}, \mathbf{W}_{Q2}, \mathbf{W}_{K2}, \mathbf{W}_{V} \in \mathbb{R}^{D \times D}\). These are then reshaped to multi‑head format \((B, H, L, d_h)\) with \(d_h = D/H\).

Two attention maps are computed via scaled dot‑products:

\[
\mathbf{A}^{(1)} = \operatorname{softmax}\!\left( \frac{\mathbf{Q}_1 \mathbf{K}_1^\top}{\sqrt{d_h}} \right), \qquad
\mathbf{A}^{(2)} = \operatorname{softmax}\!\left( \frac{\mathbf{Q}_2 \mathbf{K}_2^\top}{\sqrt{d_h}} \right).
\]

The **differential attention** weights are then defined as:

\[
\tilde{\mathbf{A}} = \operatorname{ReLU}\!\left( \mathbf{A}^{(1)} - \frac{1}{2} \mathbf{A}^{(2)} \right).
\]

The subtraction of half of the second attention map acts as a form of competition: only those positions where the first attention score significantly exceeds half the second score yield positive values; all others are set to zero by the ReLU. This results in a sparse attention pattern that can be interpreted as the network learning to suppress “uninteresting” or “distractor” keys. The differential attention is multiplied by the values to produce the attention output:

\[
\mathbf{O}_{\text{attn}} = \tilde{\mathbf{A}} \mathbf{V}.
\]

This output is then rearranged back to \((B, L, D)\). The differential mechanism is related to the “Differential Attention” recently proposed for enhancing focus and sparsity in transformers, though simplified here.

### 2.3 State‑Space Model (SimpleSSM)

The second branch processes \(\mathbf{X}_{\text{rope}}\) through a `SimpleSSM` module. While the internal details of `SimpleSSM` are not provided in this snippet, it is expected to be a lightweight state‑space sequence model, such as a gated linear recurrence (HGRN2) or a diagonal state‑space model (like a simplified Mamba). In general, an SSM transforms an input sequence \(\mathbf{X}\) into an output sequence \(\mathbf{Y}\) of the same shape, using a recurrent state that evolves with time and captures long‑range dependencies. The complexity is typically \(O(L D^2)\) and does not require a pairwise attention matrix. Let this output be denoted by \(\mathbf{O}_{\text{ssm}} \in \mathbb{R}^{B \times L \times D}\).

### 2.4 Hybrid Fusion with Learned Gate

The two outputs are fused using a scalar learnable parameter \(\alpha\) (initialised to 0.5). The fused representation is:

\[
\mathbf{F} = \alpha \cdot \mathbf{O}_{\text{attn}} + (1 - \alpha) \cdot \mathbf{O}_{\text{ssm}}.
\]

To ensure the gate remains in a valid range, \(\alpha\) is passed through a sigmoid during the forward pass (the code shows `alpha = torch.sigmoid(self.alpha)`). Thus, the effective weight is \(\sigma(\alpha) \in (0,1)\).

The fused features are then added to the original input (residual connection) and projected:

\[
\mathbf{O} = \mathbf{W}_O \big( \mathbf{X}_{\text{rope}} + \mathbf{F} \big) + \mathbf{b}_O,
\]

where \(\mathbf{W}_O \in \mathbb{R}^{D \times D}\) and \(\mathbf{b}_O\) is optional (the code uses `nn.Linear` with default bias). The block thus employs two residual paths: the input is added to the fused output, and then this sum is projected; note that the input \(\mathbf{X}_{\text{rope}}\) already contains positional information and the original content. The final output is returned as a tuple `(output, None)`.

### 2.5 Overall Forward Pass Summary

Given \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\):
1. Apply Unified RoPE: \(\mathbf{X}_{\text{rope}} = \text{RoPE}(\mathbf{X})\).
2. Compute differential attention output \(\mathbf{O}_{\text{attn}}\) from \(\mathbf{X}_{\text{rope}}\).
3. Compute SSM output \(\mathbf{O}_{\text{ssm}}\) from \(\mathbf{X}_{\text{rope}}\).
4. Fuse: \(\mathbf{F} = \sigma(\alpha) \mathbf{O}_{\text{attn}} + (1 - \sigma(\alpha)) \mathbf{O}_{\text{ssm}}\).
5. Output: \(\mathbf{Y} = \text{out\_proj}(\mathbf{X}_{\text{rope}} + \mathbf{F})\).

The module returns \((\mathbf{Y}, \text{None})\).

---

## 3. Implementation Details

### 3.1 UnifiedRoPE

- **Constructor**: takes `dim`, `max_seq_len` (default 4096), and `base` (default 10000.0). Precomputes `inv_freq` and builds cos/sin caches of shape `(1, max_seq_len, 1, dim)`.
- **`forward(x, offset=0)`**: Asserts that the last dimension of `x` matches `dim`. If the sequence length exceeds the cache, the cache is rebuilt. Slices the cos/sin tensors for the current positions, extracts even/odd channels, applies the rotation, and interleaves them back.
- **Caching**: The cache is stored as a buffer, so it is moved with the model to the correct device. Rebuilding on the fly allows dynamic extension for very long sequences.

### 3.2 HymbaTransXBlock

- Inherits from `BaseAttention`.
- **Constructor**:
  - `config`: provides `hidden_size` and optionally `num_attention_heads`.
  - `rank`: ignored; for interface compatibility.
  - `pretrained_attn`: accepted but not used for initialisation; the block is trained from scratch.
  - Creates:
    - `UnifiedRoPE` for the full hidden dimension.
    - Two sets of Q,K projections: `W_q1, W_k1, W_q2, W_k2` (no bias).
    - One value projection `W_v` (no bias).
    - `SimpleSSM(self.dim)` – the SSM module.
    - `out_proj` (with bias).
    - A learnable scalar `alpha` (initialised to 0.5).
- **Forward pass**:
  - Renames `hidden_states` to `x`.
  - Applies RoPE to get `x_rope`.
  - Computes Q1, K1, Q2, K2, V and reshapes to multi‑head.
  - Computes two attention maps via softmax, subtracts half of the second from the first, applies ReLU, and multiplies by V.
  - Reshapes attention output to `(B, L, D)`.
  - Computes SSM output from `x_rope` via `self.mamba(x_rope)`.
  - Fuses with sigmoid‑gated alpha: `alpha * A_out + (1-alpha) * M_out`.
  - Adds `x_rope`, then projects via `out_proj`.
  - Returns `(output, None)`.
- **Masking**: The `attention_mask` parameter is accepted but not used. Padding tokens would need to be handled by the SSM and attention if required; the current implementation assumes no padding or that the downstream training loop masks the loss accordingly.

### 3.3 Injection Function

The `inject_hymba_transx` function is (meant to be) registered, although the code shows it as a regular function (the decorator `@ATTENTION_REGISTRY.register("hymba_transx")` is applied to the class, not the function. There is a potential registry conflict—the class is registered, and the injection function is defined but not decorated; the framework likely expects the class to be used directly, or a separate injection should be registered. We'll note that the class itself is registered as the attention type.

The injection function provided can be used to manually inject the block:
```python
new_attn = HymbaTransXBlock(model_config, rank=..., pretrained_attn=old_attn)
layer.attention = new_attn
```

---

## 4. Usage and Configuration

The module is invoked by setting the attention type to `"hymba_transx"` in the compression configuration. No additional hyperparameters are strictly required beyond the model config. The `SimpleSSM` module must be available in the same package (imported as `from .simple_ssm import SimpleSSM`).

Example configuration:
```python
comp_config.attention_type = "hymba_transx"
# Optionally:
comp_config.rank = 64   # (not used)
```

The outer Transformer layer applies its residual and LayerNorm as usual, meaning the block’s internal residual `x_rope + F` is additional.

---

## 5. Discussion

The Hymba‑TransX block integrates several recent advances into one compact module. The differential attention mechanism introduces a built‑in competition between attention maps, potentially leading to sparser, more focused attention without explicit top‑\(k\) pruning. The subtraction and ReLU are differentiable, allowing end‑to‑end training. The inclusion of RoPE at the full‑hidden‑state level (before branching) ensures that both the attention and the SSM pathways are aware of relative positions, which is crucial for tasks where word order matters.

The hybrid fusion with a learned scalar \(\alpha\) allows the model to dynamically balance the two information sources. The SSM branch provides a globally consistent memory at linear cost, while the attention branch provides pairwise interactions. This combination is particularly suited for long sequences where full attention is expensive; the SSM can handle long‑range dependencies efficiently, and attention can be used with a limited window or even globally (as here) but mitigated by the sparsity from differential attention.

However, several limitations exist. The module currently ignores the attention mask, so padding would corrupt the SSM state and the attention maps. The SSM (`SimpleSSM`) is unspecified—it could be a basic linear recurrence or a more sophisticated model; its capacity and speed will directly impact the block’s overall performance. The differential attention uses a fixed coefficient of 0.5 for the subtraction; a learnable coefficient could provide more flexibility. Finally, the block adds the `x_rope` residual before the output projection, which means the input signal is mixed with the fused output before projection, potentially making the network less sensitive to the gate.

Despite these issues, the Hymba‑TransX block provides a clean and innovative hybrid design that can serve as a foundation for further research and engineering.

---

## 6. Conclusion

We have presented the Hymba‑TransX block, a hybrid sequence mixer that combines differential attention and a state‑space model under a unified rotary position embedding. The differential attention mechanism enhances sparsity and focus, while the SSM branch offers linear‑time global processing. A learned gate dynamically fuses the two paths. The block is implemented as a drop‑in replacement for Transformer attention and is ready for integration into the DistilBERT compression framework. Future work will address masking, explore more advanced SSM backbones, and investigate dynamic fusion strategies to further improve efficiency and accuracy.