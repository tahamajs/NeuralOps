# HGRN2‑GatedDeltaNet: A Hybrid Recurrent‑Attention Network with Configurable Linear‑to‑Full Ratio

**Abstract**

We introduce **HGRN2‑GatedDeltaNet**, a hybrid architecture that interleaves lightweight recurrent layers (GatedDeltaNet blocks) with standard multi‑head self‑attention blocks according to a fixed cyclic schedule. Each recurrent block uses the HGRN2 recurrence—a gated linear state update with input and forget gates and state clipping—while each attention block employs full quadratic self‑attention. The ratio of recurrent to attention layers is controlled by a single integer parameter \(R\) (default 3): for every \(R\) recurrent layers, one attention layer is inserted. This design aims to combine the linear‑complexity efficiency of the recurrent layers with the global representational power of occasional full attention, providing a tunable trade‑off between speed and accuracy. The module is implemented as a drop‑in replacement for the standard attention stack in DistilBERT, integrated via the compression framework’s registry. In this paper, we detail the mathematical formulation of the HGRN2 recurrence, the hybrid scheduling strategy, the overall architecture, and the injection mechanism. We also discuss the computational properties, limitations, and potential extensions of the model.

---

## 1. Introduction

Transformer models rely on self‑attention to capture contextual dependencies, but the quadratic cost of attention motivates the search for more efficient alternatives. Gated recurrent units, such as the HGRN2 (Hierarchically Gated Recurrent Network v2), offer linear time complexity and have shown promise in sequence modelling. However, purely recurrent models sometimes lack the ability to directly access long‑range dependencies with the same precision as attention. A hybrid approach that combines both can harness the efficiency of recurrence for the majority of the processing while periodically refreshing the global context with a full attention step.

The **HGRN2‑GatedDeltaNet** architecture implements this hybrid philosophy. It constructs an entire multi‑layer network (or a sub‑network replacing the attention module of a single Transformer layer) using a sequence of `GatedDeltaNetBlock` modules. Each block internally can be either a recurrent HGRN2 layer or a standard multi‑head attention layer, determined by a configurable **ratio \(R\)**. The schedule is deterministic: blocks at positions \(i = 0, 1, \dots, L-1\) (where \(L\) is the total number of layers) use full attention if \(i \bmod (R+1) = R\), and use the recurrent HGRN2 layer otherwise. Thus, one out of every \(R+1\) layers is a full attention layer.

The core recurrent layer, HGRN2, employs element‑wise gating and a bounded hidden state, making it fast and stable. The hybrid network is designed to be injected into a DistilBERT model via the framework’s registry, replacing the original attention mechanism entirely. This paper provides a comprehensive description of the architecture, its mathematical underpinnings, and its implementation.

---

## 2. Method

### 2.1 GatedDeltaNetBlock

The building block of the network is the `GatedDeltaNetBlock`, described in a companion paper. It takes an input \(\mathbf{X} \in \mathbb{R}^{B \times L \times d_{\text{model}}}\) and applies a sub‑layer followed by a residual connection and layer normalisation:

\[
\mathbf{Y} = \text{LayerNorm}\!\big(\mathbf{X} + \text{SubLayer}(\mathbf{X})\big).
\]

The `SubLayer` is either the HGRN2 recurrence or standard multi‑head self‑attention, depending on a boolean flag `use_full_attention`.

**HGRN2 Recurrence** (when `use_full_attention=False`):
The HGRN2 layer processes the sequence sequentially. At each time step \(t\), it computes forget and input gates from the current input \(\mathbf{x}_t\):

\[
\mathbf{f}_t = \sigma(\mathbf{W}_f \mathbf{x}_t + \mathbf{b}_f), \quad
\mathbf{i}_t = \sigma(\mathbf{W}_i \mathbf{x}_t + \mathbf{b}_i),
\]

updates a hidden state \(\mathbf{h}_t \in \mathbb{R}^{d_{\text{model}}}\):

\[
\mathbf{h}_t = \mathbf{f}_t \odot \mathbf{h}_{t-1} + \mathbf{i}_t \odot \mathbf{x}_t,
\]

clamps the state to \([-1, 1]\):

\[
\mathbf{h}_t = \text{clamp}(\mathbf{h}_t, -1, 1),
\]

and produces an output via a linear transformation:

\[
\mathbf{y}_t = \mathbf{W}_o \mathbf{h}_t + \mathbf{b}_o.
\]

The sequential nature of this layer yields \(O(L d_{\text{model}}^2)\) complexity, dominated by the matrix multiplications \(\mathbf{W}_f, \mathbf{W}_i, \mathbf{W}_o\). No pairwise token interactions are computed; the only mixing across time occurs through the single vector hidden state.

**Multi‑Head Attention** (when `use_full_attention=True`):
This is the standard `nn.MultiheadAttention` with `batch_first=True`. It computes scaled dot‑product attention across all token pairs, resulting in \(O(L^2 d_{\text{model}})\) complexity. The block applies the attention, adds the residual, and normalises.

### 2.2 Hybrid Scheduling with Ratio \(R\)

The `HGRN2GatedDeltaNet` module stacks \(L\) blocks, one for each logical layer of the original Transformer. For layer index \(i\) (0‑based), the condition is:

\[
\text{use\_full} = \big( i \bmod (R+1) = R \big).
\]

This means:
- If \(R = 0\): every layer is full attention.
- If \(R = 1\): alternating: HGRN2, attention, HGRN2, attention, …
- If \(R = 3\) (default): pattern HGRN2, HGRN2, HGRN2, attention, HGRN2, HGRN2, HGRN2, attention, …

The parameter \(R\) therefore controls the density of attention layers. A higher \(R\) reduces the total number of attention computations, thereby lowering the overall complexity for long sequences.

### 2.3 Overall Forward Pass

The module’s forward method processes the input sequentially through all blocks:

\[
\mathbf{X}^{(0)} = \mathbf{X},
\]
\[
\mathbf{X}^{(i+1)} = \text{Block}_i(\mathbf{X}^{(i)}), \quad i = 0, \dots, L-1.
\]

Each block returns a tuple `(output, None)`, from which the output is extracted. The final output is \(\mathbf{X}^{(L)}\), and the module returns \((\mathbf{X}^{(L)}, \text{None})\) for interface compatibility.

---

## 3. Implementation Details

### 3.1 HGRN2GatedDeltaNet Class

- **Constructor**: 
  - `d_model`: hidden dimension.
  - `n_heads`: number of attention heads for attention blocks.
  - `n_layers`: total number of blocks (equals the number of Transformer layers in the original model).
  - `ratio`: integer \(R\) controlling attention frequency (default 3).
  - Creates a `nn.ModuleList` of `GatedDeltaNetBlock` instances, configuring `use_full_attention` based on the ratio condition.

- **Forward**: Iterates over the layers, feeding the output of one to the next. The expected input shape is \((B, L, d_{\text{model}})\).

### 3.2 Injection Function

The function `inject_hgrn2_gated_deltanet` is registered under `"hgrn2_gated_deltanet"` in the `ATTENTION_REGISTRY`. It receives the model configuration, compression configuration, and the original attention layer (which is ignored). It constructs a new `HGRN2GatedDeltaNet` using:
- `d_model` from `model_config.hidden_size`.
- `n_heads` from `model_config.num_attention_heads`.
- `n_layers` from `total_layers` (the total number of Transformer layers).
- `ratio` from `comp_config.ratio` (default 3).

It then assigns this module to `layer.attention`. This injection replaces the per‑layer attention with the entire hybrid network, effectively treating the whole encoder as a single block. This design is unusual but consistent with the framework’s ability to replace the attention sub‑layer with any module; here, the replacement is a multi‑layer network that internally performs all the attention/recurrence operations. The outer Transformer layer would still apply its own residual connection and LayerNorm (in DistilBERT) before and after this module, so the total transformation becomes:

\[
\text{output} = \text{LayerNorm}\!\big(\mathbf{X} + \text{HybridNet}(\mathbf{X})\big).
\]

Thus, the HybridNet acts as a drop‑in replacement for the original attention layer, but it processes the entire sequence through many internal sub‑layers, each with its own residual connections. Care must be taken that this does not lead to excessively deep transformation without proper initialisation; however, the HGRN2 layers are relatively lightweight, and the occasional full attention steps help maintain coherence.

### 3.3 Computational Complexity

For a sequence of length \(L\) and hidden dimension \(D\), the per‑layer cost of an HGRN2 block is \(O(L D^2)\) (dominated by the linear transformations), while a full attention block costs \(O(L^2 D)\). With ratio \(R\), the fraction of attention layers is \(1/(R+1)\). The total complexity over \(N\) layers is approximately:

\[
O\!\left( N \cdot \frac{R}{R+1} L D^2 + N \cdot \frac{1}{R+1} L^2 D \right).
\]

For long sequences where \(L \gg D\), the attention term dominates. Increasing \(R\) effectively reduces the number of attention layers and brings the overall complexity closer to linear.

---

## 4. Usage and Configuration

The module is invoked by setting the attention type to `"hgrn2_gated_deltanet"` in the compression configuration. The relevant hyperparameter is:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ratio` | 3 | Number of recurrent layers per attention layer. \(R=0\) means all attention; \(R=3\) gives 3 HGRN2 then 1 attention, etc. |

Example configuration:

```python
comp_config.ratio = 5   # 5 recurrent layers, then 1 attention layer
```

No other special parameters are needed. The injection function uses the model’s `hidden_size`, `num_attention_heads`, and the total number of layers.

---

## 5. Discussion

The HGRN2‑GatedDeltaNet architecture provides a simple yet effective way to balance efficiency and capacity. By interleaving fast recurrent layers with occasional full attention, it avoids the quadratic bottleneck for the majority of processing while preserving the ability to capture long‑range dependencies accurately when needed. This hybrid scheme is reminiscent of the Griffin architecture and the cyclic attention patterns in some hybrid models.

However, there are several limitations. First, the injection mechanism replaces the attention module of a **single** Transformer layer with an entire multi‑layer network. This means the original residual connection and LayerNorm are applied around this whole sub‑network, which may lead to unstable training if the internal layers are not properly initialised or scaled. A more granular integration would inject a single HGRN2 or attention block per Transformer layer, letting the framework handle the interleaving across layers, but the current design opts for a self‑contained module. Second, the HGRN2 layers do not support parallelisation over time (due to the sequential loop), which can slow training. Batched recurrence or a parallel scan implementation would be beneficial. Third, the module ignores any `attention_mask`, so padding tokens are processed like regular tokens, potentially introducing noise; this should be addressed by masking the input or gating the updates for padding positions.

Despite these issues, the module demonstrates the power of hybrid recurrence‑attention architectures and provides a clear, modular implementation that can be easily extended. The ratio parameter offers a convenient knob for practitioners to tune the efficiency‑quality trade‑off for their specific task.

---

## 6. Conclusion

We have presented HGRN2‑GatedDeltaNet, a hybrid architecture that interleaves HGRN2 recurrent blocks with full attention blocks according to a configurable ratio. The design combines the linear complexity of gated recurrence with the global representational power of attention, all within a drop‑in replacement for a DistilBERT attention layer. The module’s injection into the compression framework via the registry makes it easy to experiment with different ratios and sequence lengths. Future work will address the training parallelisation, padding mask handling, and explore more fine‑grained scheduling strategies.