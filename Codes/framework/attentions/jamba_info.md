# Jamba‑Info Block: Information Bottleneck meets Differential Attention with Head Sparsity

**Abstract**

We introduce the **Jamba‑Info Block**, a dual‑mode sequence mixing module that integrates two distinct paradigms: a **differential attention mechanism with head sparsity** for retrieval‑oriented processing, and a **concept bottleneck plus state‑space model (SSM)** for compression‑driven processing. The mode is selected at construction and remains fixed. In retrieval mode, the block computes two sets of query‑key attention maps, subtracts a scaled version of the second map from the first, and enforces a fixed head‑wise sparsity by zeroing out a fraction of attention heads. The resulting sparse, differential attention weights are used to aggregate values. In bottleneck mode, the input is passed through a low‑rank concept bottleneck (down‑project, GELU, up‑project) and then processed by an SSM, with a residual connection. Both modes end with a final linear output projection. The block serves as a drop‑in replacement for standard self‑attention in Transformer models and is registered within the DistilBERT compression framework. This paper provides a rigorous mathematical formulation of the two modes, discusses their motivation, and details the implementation.

---

## 1. Introduction

Modern Transformer architectures explore a wide variety of sequence mixing modules beyond standard softmax attention. Two notable trends are **differential attention**—where the network learns to focus by comparing two attention distributions—and **information bottleneck** techniques that compress representations before processing them with a lightweight recurrent or state‑space model. The **Jamba‑Info Block** unifies these ideas into a single module with two distinct operating modes, selectable via a configuration flag.

In **retrieval mode**, the block aims to perform sparse, content‑focused attention by computing two separate attention maps from independent query‑key pairs and then taking their difference. This design is inspired by the “differential transformer” concept, where subtracting a baseline attention map can sharpen the focus and suppress irrelevant tokens. Additionally, a sparsity parameter determines how many heads are kept active; the remaining heads are forced to have zero attention, which can act as a form of structured pruning, encouraging the model to rely on a subset of heads for retrieval.

In **bottleneck mode**, the block adopts a different philosophy: it first compresses the input into a lower‑dimensional concept space via a bottleneck (Linear → GELU → Linear), then processes this compact representation with a state‑space model (`SimpleSSM`), and finally adds the original input as a residual. This design is reminiscent of “InfoMamba” or “concept bottleneck” approaches, where the model learns a compressed, abstract representation before applying a lightweight sequence model.

By integrating both modes into a single configurable block, the Jamba‑Info Block provides a flexible testbed for exploring these two types of efficient sequence mixing within the same codebase. The module is intended to be injected into DistilBERT layers via the framework’s registry.

---

## 2. Method

### 2.1 Retrieval Mode: Differential Attention with Head Sparsity

When `mode = "retrieval"`, the block computes multi‑head attention using two independent query‑key projections and a shared value projection.

Given an input \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\), the following linear transformations (without bias) are applied:

\[
\mathbf{Q}^{(1)} = \mathbf{X} \mathbf{W}_{Q1}, \quad
\mathbf{K}^{(1)} = \mathbf{X} \mathbf{W}_{K1},
\]
\[
\mathbf{Q}^{(2)} = \mathbf{X} \mathbf{W}_{Q2}, \quad
\mathbf{K}^{(2)} = \mathbf{X} \mathbf{W}_{K2},
\]
\[
\mathbf{V} = \mathbf{X} \mathbf{W}_{V}.
\]

All weight matrices are square \(D \times D\). The tensors are reshaped to multi‑head format \((B, H, L, d_h)\), where \(H\) is the total number of attention heads and \(d_h = D / H\).

Two attention maps are computed via scaled dot‑product softmax:

\[
\mathbf{A}^{(1)} = \operatorname{softmax}\!\left( \frac{\mathbf{Q}^{(1)} (\mathbf{K}^{(1)})^\top}{\sqrt{d_h}} \right), \qquad
\mathbf{A}^{(2)} = \operatorname{softmax}\!\left( \frac{\mathbf{Q}^{(2)} (\mathbf{K}^{(2)})^\top}{\sqrt{d_h}} \right).
\]

The **differential attention** is defined as:

\[
\tilde{\mathbf{A}} = \mathbf{A}^{(1)} - \frac{1}{2} \mathbf{A}^{(2)}.
\]

This operation can be interpreted as computing how much the first attention map exceeds half of the second. Unlike the ReLU‑gated variant (as in Hymba‑TransX), here no non‑linearity is applied after subtraction; negative values are kept, allowing the differential map to be signed. This signed attention map then directly weights the values, potentially enabling the network to both enhance and suppress certain key‑value pairs.

**Head Sparsity**: The module also enforces sparsity at the head level. A parameter `sparsity` (default 0.15) determines the fraction of heads that are **inactive**. Specifically, the number of active heads is:

\[
H_{\text{active}} = \max(1, \lfloor H \times \text{sparsity} \rfloor).
\]

In the forward pass, after computing the differential attention \(\tilde{\mathbf{A}}\) (shape \((B, H, L, L)\)), the attention maps for heads indexed \(h \ge H_{\text{active}}\) are set to zero:

\[
\tilde{\mathbf{A}}_{[:, h \ge H_{\text{active}}, :, :]} = 0.
\]

This forces the inactive heads to contribute nothing to the output, which is equivalent to a structured head pruning during training, potentially encouraging the remaining heads to specialise. The sparsity is fixed per layer and can be tuned.

The final output of the attention branch is:

\[
\mathbf{O}_{\text{attn}} = \tilde{\mathbf{A}} \mathbf{V}.
\]

This is reshaped back to \((B, L, D)\).

### 2.2 Bottleneck Mode: Concept Compression + State‑Space Model

When `mode = "bottleneck"`, the block utilises a completely different architecture. It first passes the input through an **information bottleneck** layer, `InfoMambaConceptBottleneck`, defined as:

\[
\mathbf{Z} = \mathbf{W}_{\text{down}} \mathbf{X} \quad (\text{no bias}), \qquad
\tilde{\mathbf{Z}} = \operatorname{GELU}(\mathbf{Z}), \qquad
\mathbf{X}_{\text{bottle}} = \mathbf{W}_{\text{up}} \tilde{\mathbf{Z}},
\]

where \(\mathbf{W}_{\text{down}} \in \mathbb{R}^{d_{\text{bottleneck}} \times D}\), \(\mathbf{W}_{\text{up}} \in \mathbb{R}^{D \times d_{\text{bottleneck}}}\), and \(d_{\text{bottleneck}}\) is typically 64. This low‑rank bottleneck forces the representation to capture the most salient features before being processed further.

The compressed sequence \(\mathbf{X}_{\text{bottle}} \in \mathbb{R}^{B \times L \times D}\) is then processed by a `SimpleSSM` module—a state‑space model similar to HGRN or a simplified Mamba. The exact operation of `SimpleSSM` is not detailed here, but it is expected to be a recurrent or parallel scan‑based sequence transformation that captures long‑range dependencies with \(O(L D^2)\) complexity.

Finally, a **residual connection** is added: the output of the SSM is summed with the original input \(\mathbf{X}\):

\[
\mathbf{O}_{\text{bottle}} = \mathbf{X} + \text{SimpleSSM}(\mathbf{X}_{\text{bottle}}).
\]

This design follows the pattern of first compressing the input to a meaningful bottleneck, applying a lightweight sequence model, and then adding back the original signal to ease gradient flow and retain positional information.

### 2.3 Output Projection

In both modes, the result (either \(\mathbf{O}_{\text{attn}}\) or \(\mathbf{O}_{\text{bottle}}\)) is passed through a final linear output projection:

\[
\mathbf{Y} = \mathbf{W}_O \mathbf{O} + \mathbf{b}_O,
\]

with \(\mathbf{W}_O \in \mathbb{R}^{D \times D}\). The block returns a tuple \((\mathbf{Y}, \text{None})\) for compatibility.

---

## 3. Implementation Details

### 3.1 JambaInfoBlock Class

- **Constructor**:
  - `config`: provides `hidden_size` and `num_attention_heads`.
  - `rank`: ignored (interface compatibility).
  - `mode`: `"retrieval"` or `"bottleneck"`.
  - `sparsity`: fraction of heads kept active (used only in retrieval mode, default 0.15).
  - In retrieval mode, creates:
    - `W_q1, W_k1, W_q2, W_k2` (no bias) – two sets of Q, K.
    - `W_v` (no bias) – shared value projection.
    - `out_proj` (with bias).
    - Computes `active_heads = max(1, int(num_heads * sparsity))`.
  - In bottleneck mode, creates:
    - `bottleneck = InfoMambaConceptBottleneck(dim)` (default bottleneck_dim=64 hardcoded inside that class).
    - `ssm = SimpleSSM(dim)`.
    - `out_proj`.

- **Forward**:
  - Retrieval mode:
    1. Project Q1, K1, Q2, K2, V.
    2. Compute two attention maps.
    3. Subtract: `diff_attn = attn1 - 0.5 * attn2`.
    4. Zero out heads from `active_heads` onwards.
    5. Multiply by V, reshape, project.
  - Bottleneck mode:
    1. `x_bottleneck = self.bottleneck(x)`.
    2. `ssm_out = self.ssm(x_bottleneck)`.
    3. `out = x + ssm_out`.
    4. Project.

- **Mask handling**: The `mask` parameter is accepted but **not used**; thus, padding tokens are not masked, which is a significant limitation. In retrieval mode, padding tokens would be included in the attention, and in bottleneck mode they'd be processed normally.

### 3.2 InfoMambaConceptBottleneck

A simple two‑layer bottleneck:
- `down`: Linear(dim, bottleneck_dim, bias=False)
- `up`: Linear(bottleneck_dim, dim, bias=False)
- `forward`: `up(GELU(down(x)))`.
The bottleneck dimension is hardcoded to 64 in the class definition, which may be suboptimal for all model sizes.

### 3.3 Integration

The module is registered under the key `"jamba_info"` in `ATTENTION_REGISTRY`. A separate injection function (not shown) would be needed to pass parameters like `mode` and `sparsity` from the compression config. The current code only defines the class with the registry decorator, expecting the framework to instantiate it with appropriate arguments (perhaps via reflection). In practice, an injection function should be added to extract `mode` and `sparsity` from `comp_config`.

---

## 4. Usage and Configuration

The block can be selected by setting the attention type to `"jamba_info"` and providing the additional parameters (through a custom injection function or via the framework's parameter extraction). The expected configuration parameters are:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `"retrieval"` | Operating mode: `"retrieval"` or `"bottleneck"`. |
| `sparsity` | 0.15 | Fraction of heads active in retrieval mode (e.g., 0.15 means only 15% of heads are used). |
| `bottleneck_dim` | 64 | (Hardcoded in InfoMambaConceptBottleneck) The bottleneck dimension. |

Example usage (assuming an injection function exists):
```python
comp_config.attention_type = "jamba_info"
comp_config.attn_mode = "retrieval"
comp_config.sparsity = 0.2
```

Manual instantiation:
```python
attn = JambaInfoBlock(config=model_config, mode="retrieval", sparsity=0.2)
layer.attention = attn
```

---

## 5. Discussion

The Jamba‑Info Block offers an interesting contrast between two highly different sequence mixing strategies. The retrieval mode’s differential attention encourages the model to focus on relevant tokens by canceling a baseline attention map, while the head sparsity forces a compact use of attention heads. This can be seen as a form of learnable head pruning that may improve interpretability and efficiency (since fewer heads actively participate). However, the subtraction without a ReLU can lead to negative attention weights, which, while mathematically valid, could cause instability because the resulting output is not a convex combination of values. Adding a ReLU or a softmax after subtraction might stabilise training.

The bottleneck mode emulates a two‑stage process: compress to a conceptual representation, then apply a lightweight sequence model. The residual connection helps preserve the raw input signal, which may be important because the bottleneck (with dimension 64) is a very aggressive compression (for a typical hidden size 768, 64 is ~12× reduction). The SSM then operates on this compressed signal, which could be efficient but might lose detailed token‑level information. Increasing the bottleneck dimension or adding a gating mechanism could improve capacity.

Both modes suffer from the lack of attention mask handling, which is a common issue in many of these experimental modules. For production, explicit padding mask support must be added. Also, the `SimpleSSM` class is not defined in this snippet; its performance and speed will heavily influence the bottleneck mode. If `SimpleSSM` is itself a placeholder or a naive implementation, the overall module may not yield competitive results.

Overall, the Jamba‑Info Block is a promising example of a modular hybrid design that allows rapid experimentation with different sequence mixing paradigms within a unified interface. Future work should incorporate masking, explore dynamic sparsity instead of fixed head pruning, and add learnable gating for the differential attention.

---

## 6. Conclusion

We have presented the Jamba‑Info Block, a dual‑mode sequence mixing module featuring differential sparse attention and a concept‑bottleneck + SSM pipeline. The module’s flexibility and integration into the compression framework make it a useful building block for exploring efficient Transformer variants. Further refinements in masking, stability, and parameterisation will unlock its full potential for long‑sequence modeling.