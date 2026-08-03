# Gated DeltaNet Block: A Recurrent Alternative to Self-Attention

**Abstract**

We introduce the **Gated DeltaNet Block**, a lightweight drop‑in replacement for the standard multi‑head self‑attention module in Transformer models. At its core lies the **HGRN2 (Hierarchically Gated Recurrent Network v2)** layer, a linear recurrent unit that processes tokens sequentially using learned forget and input gates, along with a bounded hidden state. The full block wraps either the HGRN2 layer or, optionally, standard full attention, within a residual connection and layer normalisation. The design follows the principle of the DeltaNet architecture: replacing the expensive quadratic self‑attention with a fast, linear‑complexity recurrent state update. This paper details the mathematical formulation of the HGRN2 recurrence, the architecture of the Gated DeltaNet Block, and its integration into the DistilBERT compression framework via a registry‑based injection function. The module offers a highly efficient alternative to attention for long sequences while preserving compatibility with existing Transformer pipelines.

---

## 1. Introduction

Self‑attention in Transformers enables parallel computation and captures long‑range dependencies but scales quadratically with sequence length. Recurrent models, particularly gated variants, offer linear complexity at the cost of limited parallelisation during training. Recent work on linear attention, state‑space models, and gated recurrent units has shown that recurrence can match or approach the performance of attention on many language tasks. The **Gated DeltaNet Block** adopts this paradigm, replacing the entire attention sub‑layer with a simple yet effective recurrent unit called HGRN2. This recurrent unit is a direct descendant of the HGRN (Hierarchically Gated Recurrent Network) and the earlier DeltaNet designs, which use a single recurrent state with gating to control information flow. The block is designed to be plug‑compatible: it outputs a tensor of the same shape as the input and can be injected into any DistilBERT layer via the framework’s injection mechanism.

In this paper, we provide a comprehensive description of the HGRN2 recurrence and the GatedDeltaNetBlock. We discuss the mathematical roles of the forget and input gates, the clipping of the hidden state, and the optional fallback to full attention. We also detail the implementation and configuration options, and analyse the computational benefits.

---

## 2. Method

### 2.1 HGRN2 Recurrent Layer

The HGRN2 layer processes a sequence \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\) token by token, maintaining a hidden state \(\mathbf{h}_t \in \mathbb{R}^D\) for each batch element. At step \(t\), the input \(\mathbf{x}_t = \mathbf{X}_{:,t,:}\) is used to compute two gating values:

\[
\mathbf{f}_t = \sigma(\mathbf{W}_f \mathbf{x}_t + \mathbf{b}_f), \qquad
\mathbf{i}_t = \sigma(\mathbf{W}_i \mathbf{x}_t + \mathbf{b}_i),
\]

where \(\sigma\) is the sigmoid function, and \(\mathbf{W}_f, \mathbf{W}_i \in \mathbb{R}^{D \times D}\). The forget gate \(\mathbf{f}_t\) determines how much of the previous state is retained, and the input gate \(\mathbf{i}_t\) scales the new input.

The state is updated as:

\[
\mathbf{h}_t = \mathbf{f}_t \odot \mathbf{h}_{t-1} + \mathbf{i}_t \odot \mathbf{x}_t,
\]

where \(\odot\) denotes element‑wise multiplication. The new state is then clamped to the range \([-1, 1]\) via \(\mathbf{h}_t = \operatorname{clamp}(\mathbf{h}_t, -1, 1)\). This clipping prevents unbounded growth of the state, acting as a simple regularisation and stabilising training.

The output of the layer at step \(t\) is a linear transformation of the clamped state:

\[
\mathbf{y}_t = \mathbf{W}_o \mathbf{h}_t + \mathbf{b}_o.
\]

The outputs are stacked to form the final sequence \(\mathbf{Y} \in \mathbb{R}^{B \times L \times D}\). This recurrence is linear in the state update (apart from the element‑wise gating), making it computationally efficient. The complexity is \(O(L D^2)\) due to the full‑rank linear transformations, but the recurrence itself is \(O(L D)\) per layer—much cheaper than \(O(L^2 D)\) for attention.

### 2.2 GatedDeltaNetBlock

The GatedDeltaNetBlock consists of a recurrent or attention sub‑layer followed by a residual connection and layer normalisation, mirroring the pre‑norm Transformer architecture. Given an input sequence \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\), the block computes:

\[
\mathbf{Y} = \operatorname{LayerNorm}(\mathbf{X} + \operatorname{SubLayer}(\mathbf{X})),
\]

where \(\operatorname{SubLayer}\) is either the HGRN2 layer or standard multi‑head attention, depending on the flag `use_full_attention`. If attention is used, it is the standard `nn.MultiheadAttention` with `batch_first=True`. The HGRN2 path is used by default; the attention path is provided for ablation or comparison.

The block returns a tuple \((\mathbf{Y}, \text{None})\) to match the interface expected by the framework (which often asks for attention weights). This allows the block to be swapped into any Transformer layer without modifying the calling code.

### 2.3 Relationship to DeltaNet and HGRN

The original DeltaNet proposed a gated recurrent unit where the state update uses a delta (difference) between the new input and the previous state, often with a low‑rank correction. HGRN simplifies the recurrence to a plain gated linear unit with a clamping mechanism. HGRN2 retains the same gating formula but adds the input gate directly to the input embedding, providing a soft reset and update mechanism akin to the LSTM’s input and forget gates. The absence of a recurrent weight matrix \(\mathbf{W}_h\) (the state only interacts via element‑wise gates with the input) keeps the recurrence simple and fast, while the per‑token linear transformations still provide representational power.

---

## 3. Implementation Details

### 3.1 HGRN2Layer

- **Constructor**: Takes `d_model`, the hidden dimension. Creates three linear layers: `forget_gate`, `input_gate`, and `output`, all of shape \((D, D)\) (with biases). A clipping threshold `res_clip` is set to 1.0.
- **Forward**: Iterates over the time dimension with a Python loop. At each step, computes gates and updates the state \(\mathbf{h}\). This sequential loop cannot be parallelised over time, limiting training speed but not inference (since inference is inherently sequential for autoregressive tasks). However, for bidirectional encoding, the sequential nature is a drawback; the module is better suited for causal (left‑to‑right) processing.

### 3.2 GatedDeltaNetBlock

- **Constructor**: Accepts `d_model`, `n_heads`, and `use_full_attention`. Instantiates `HGRN2Layer` and, if `use_full_attention` is True, a `nn.MultiheadAttention` module.
- **Forward**: Applies the selected sub‑layer, adds residual, applies LayerNorm, and returns `(output, None)`.

### 3.3 Injection

The injection function `inject_gated_deltanet_block` is registered under `"gated_deltanet_block"` in `ATTENTION_REGISTRY`. It reads `model_config`, `comp_config`, and `old_attn` (the original attention layer). It uses `comp_config.use_full_attention` to decide whether to enable full attention mode. It then creates a `GatedDeltaNetBlock` and assigns it to `layer.attention`.

---

## 4. Usage and Configuration

The block can be injected into a DistilBERT model by specifying the attention type `"gated_deltanet_block"` in the compression configuration. The relevant configuration key is:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_full_attention` | `False` | If `True`, the block uses standard multi‑head attention instead of the recurrent HGRN2. |

Example:
```python
comp_config.use_full_attention = False
```

No other hyperparameters are needed. The block uses the model’s `hidden_size` and `num_attention_heads` from the configuration.

---

## 5. Discussion

The Gated DeltaNet Block is extremely lightweight and fast. It replaces the entire self‑attention mechanism with a simple gated recurrence, reducing the per‑layer complexity from \(O(L^2 D)\) to \(O(L D^2)\). For moderate to large \(L\) (e.g., > 512), this yields significant speedups. The gating mechanism enables the model to dynamically decide how much past information to retain and how much new input to incorporate, similar to an LSTM but with a single vector state rather than a separate cell state.

However, the current implementation has several limitations:
- **Sequential processing**: The `for` loop over time steps hampers training parallelism for bidirectional tasks. Batched recurrent kernels (e.g., `torch.nn.functional.rnn`) could mitigate this but are not used here.
- **No head‑wise parallelism**: Unlike multi‑head attention, the state is a single vector per token, not split into multiple independent heads. This may limit the model’s ability to attend to different representation subspaces.
- **Lack of masking**: The module ignores any `attention_mask`; padding tokens are processed equally, which could introduce noise. A proper implementation would reset the state or skip updates for padded positions.
- **Fixed clipping**: The hard clamp at ±1 may be too restrictive and could be replaced by a learnable gain or parameterised activation.

Despite these, the module provides a strong baseline for exploring recurrence‑based alternatives to attention, especially in resource‑constrained scenarios or for online/streaming inference.

---

## 6. Conclusion

We have presented the Gated DeltaNet Block, a modular substitute for Transformer self‑attention built around the HGRN2 recurrent layer. Its gated linear state update and optional full attention mode provide flexibility and efficiency. The straightforward injection mechanism allows rapid experimentation within the DistilBERT framework. Future work will focus on parallelising the recurrence, incorporating masking, and exploring more sophisticated state‑update rules inspired by state‑space models.