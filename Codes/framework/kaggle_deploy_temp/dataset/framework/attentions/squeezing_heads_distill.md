# SqueezingHeadsDistiller: A Placeholder for Head‑Pruning Distillation with Standard Self‑Attention

**Abstract**

We present **SqueezingHeadsDistiller**, a module that is intended to serve as a compressor for multi‑head attention by pruning or distilling redundant attention heads. In its current form, however, the implementation is a simple, unaltered **full standard multi‑head self‑attention** layer, with no head‑squeezing or distillation logic. It is registered in the compression framework as a drop‑in replacement for the attention sub‑layer of DistilBERT, accepting and correctly handling 2D padding masks. This paper describes the intended role of the module, details the mathematical operations it currently performs (which are identical to the original attention), and discusses the gap between its name and its implementation. The module serves as a baseline placeholder for future integration of head‑pruning distillation techniques.

---

## 1. Introduction

Large Transformer models often contain many attention heads, some of which may be redundant or unnecessary for the target task. Several methods have been proposed to compress models by removing or merging such heads, either through structured pruning, head‑wise knowledge distillation, or a combination of both. The **SqueezingHeadsDistiller** module is named after this concept: “squeezing” heads out of the model via distillation. In a typical head‑distillation pipeline, a teacher model with full heads would guide a student model with fewer heads, or a subset of heads would be trained to mimic the behavior of the full set.

However, the current implementation of `SqueezingHeadsDistiller` does **not** perform any head squeezing or distillation. Instead, it implements a standard, full‑capacity multi‑head self‑attention mechanism. The module is essentially a wrapper that replicates the original DistilBERT attention forward pass. Its purpose in the framework is likely as a test placeholder, allowing the injection mechanism to be verified without requiring a fully implemented distillation‑based compressor. Once the actual head‑pruning logic is developed, this module can be replaced or extended.

In this paper, we describe the mathematical formulation of the standard attention that is currently implemented, the handling of the attention mask, and the integration into the compression framework via the registry. We also discuss the intended design of a SqueezingHeadsDistiller and the discrepancy between the name and the code.

---

## 2. Method (Current Implementation)

### 2.1 Multi‑Head Projections

Given an input sequence \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\), the module first applies standard linear projections for all heads:

\[
\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V,
\]

with \(\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{D \times D}\) (the full attention size, as in DistilBERT). The tensors are reshaped to the multi‑head layout \((B, H, L, d_h)\), where \(H\) is the number of heads and \(d_h = D/H\). The heads are then transposed to shape \((B, H, L, d_h)\).

### 2.2 Scaled Dot‑Product Attention

The attention scores are computed as:

\[
\mathbf{S} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_h}} \quad \in \mathbb{R}^{B \times H \times L \times L}.
\]

A 2D padding mask of shape \((B, L)\) (with 1 for valid tokens and 0 for padding) is supported. The mask is converted to an additive form by expanding it to \((B, 1, 1, L)\) and adding \((1 - \text{mask}) \cdot (-\infty)\) to the scores:

\[
\mathbf{S} \leftarrow \mathbf{S} + (1 - \text{mask}) \cdot (-10^9).
\]

This ensures padding tokens do not affect the softmax.

### 2.3 Softmax and Aggregation

The attention weights are obtained via softmax:

\[
\mathbf{A} = \operatorname{softmax}(\mathbf{S}).
\]

Dropout is applied to \(\mathbf{A}\). The context for each head is then:

\[
\mathbf{C}_h = \mathbf{A}_h \mathbf{V}_h.
\]

The heads are concatenated by transposing and reshaping back to \((B, L, D)\). Finally, an output projection \(\mathbf{W}_O \in \mathbb{R}^{D \times D}\) is applied:

\[
\mathbf{Y} = \mathbf{C}_{\text{flat}} \mathbf{W}_O + \mathbf{b}_O.
\]

The module returns a tuple \((\mathbf{Y}, \text{None})\) to conform to the framework’s attention interface.

### 2.4 Head Squeezing / Distillation (Intended)

The module’s name implies that it should reduce the number of effective heads or distill information from many heads into fewer ones. Possible implementations could include:

- A learned gating mechanism that zeroes out entire heads (structured head pruning).
- A low‑rank factorisation that merges heads into a smaller set.
- A distillation loss applied during training that encourages a subset of heads to mimic the full attention distribution.
- A combination of pruning and fine‑tuning guided by a teacher model.

In the current code, none of these mechanisms are present. The module merely performs the same operations as the original attention layer.

---

## 3. Implementation Details

- **Class `SqueezingHeadsDistiller`** inherits from `nn.Module` and is registered under the key `"squeezing_heads_distiller"` in the `ATTENTION_REGISTRY`.
- **Constructor**:
  - `config`: provides `num_attention_heads`, `hidden_size`, and `dropout`.
  - `rank`: ignored (for interface compatibility).
  - Creates four linear layers (`q_proj`, `k_proj`, `v_proj`, `out_proj`) and a dropout layer.
- **Forward**:
  - Accepts `x` (hidden states), `attention_mask` (optional), and `**kwargs`.
  - Projects Q, K, V, reshapes to multi‑head.
  - Computes scores, applies mask (converts 2D mask), softmax, dropout.
  - Aggregates values, reshapes, output projection.
  - Returns `(out, None)`.

The module does **not** include any parameter that reduces the number of heads, nor any distillation loss term. It is functionally identical to the `FullAttention` module described in a companion paper.

---

## 4. Usage and Configuration

The module is invoked by setting the attention type to `"squeezing_heads_distiller"` in the compression configuration. No additional hyperparameters are required. For example:

```python
comp_config.attention_type = "squeezing_heads_distiller"
```

Since the module contains randomly initialised linear layers, it would normally need to be initialised from a pretrained model for zero‑shot performance. In the compression framework, an injection function (not shown in this snippet) would typically copy the weights from the original attention layer (`old_attn`) into `q_proj`, `k_proj`, `v_proj`, `out_proj`. Without that, the module would behave like a freshly initialised attention layer and require full fine‑tuning.

---

## 5. Discussion

The `SqueezingHeadsDistiller` in its current form is a placeholder. It provides a standard attention forward pass, which allows the framework’s replacement mechanism to be tested end‑to‑end without introducing errors from an incomplete compression method. The name suggests that the final implementation will incorporate techniques to reduce head redundancy, but this has not yet been implemented. Potential reasons for this placeholder include:

- The module may be a scaffold for ongoing research, where the compression logic is being developed in parallel.
- It might serve as a baseline to compare actual head‑squeezing methods against (i.e., “no compression”).
- The injection infrastructure might expect a module with this name, and the placeholder ensures the pipeline does not break.

For practical deployment, one would replace this module with a genuine head‑pruning or head‑distillation module, such as one that learns head importance scores and removes low‑importance heads, or one that uses attention transfer loss to mimic the full attention distribution with fewer heads.

The mask handling correctly supports the 2D mask format common in DistilBERT. However, it assumes that the mask contains 1 for valid tokens and 0 for padding. If the mask is provided in a different format (e.g., additive), the conversion would be incorrect. The code also ignores the `output_attentions` flag, always returning `None` for the attention weights.

---

## 6. Conclusion

We have documented the `SqueezingHeadsDistiller` module, which is currently a standard full multi‑head attention implementation, acting as a placeholder for future head‑compression distillation techniques. Its presence in the registry allows the compression framework to function without disruption, and it provides a clear point where a real head‑squeezing algorithm can be integrated. Once the intended distillation logic is added, this module will enable effective reduction of attention heads with minimal impact on model performance. Future work will focus on implementing and evaluating actual head‑squeezing strategies within this module’s framework.