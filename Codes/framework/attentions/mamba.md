# Mamba Placeholder: A Simplified Drop‑In Replacement for Attention

**Abstract**

We present a minimal placeholder module named **Mamba**, intended to serve as a drop‑in replacement for the standard self‑attention layer in Transformers. The module implements a simple position‑wise feed‑forward transformation: a linear projection expanding the hidden dimension by a factor of 2, followed by the SiLU (Sigmoid Linear Unit) activation, and a linear projection back to the original dimension. This is **not** the true state‑space model (Mamba) but a lightweight stub that allows the framework to test integration and routing without requiring the official Mamba implementation. The module is registered in the compression framework as `"mamba"` and can be injected into DistilBERT layers via the provided injection function. This paper describes the placeholder’s architecture, its mathematical definition, and its role within the broader experimentation pipeline. We also discuss the expected behaviour of the real Mamba model and the limitations of the current stub.

---

## 1. Introduction

The Mamba state‑space model (Gu & Dao, 2023) has recently emerged as a powerful alternative to attention for sequence modelling, achieving linear time complexity and strong performance on long‑range tasks. However, its implementation requires careful design of structured state‑space kernels and parallel scans. For rapid prototyping and framework integration, a **placeholder** module is often used to verify the injection mechanism and to provide a functional (though not performant) drop‑in replacement that can be swapped later for the full model.

The provided `Mamba` class serves this purpose. It mimics the interface of a sequence mixing module: it takes a tensor of shape \((B, L, D)\) and returns a tensor of the same shape, using only position‑wise operations. Internally, it applies a simple MLP with SiLU activation—a design that is common for lightweight non‑linear transformations and bears no relation to the actual Mamba recurrence. The module is registered in the attention registry under the key `"mamba"`, allowing the compression framework to inject it into any DistilBERT layer in place of the original self‑attention.

This paper describes the placeholder’s mathematical formulation, its implementation, and the injection mechanism. We also discuss the gap between this stub and the true Mamba model, and outline the path to a full integration.

---

## 2. Method

### 2.1 Placeholder Architecture

Given an input sequence \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\), the module computes:

\[
\mathbf{Z} = \operatorname{SiLU}(\mathbf{X} \mathbf{W}_1 + \mathbf{b}_1), \qquad
\mathbf{Y} = \mathbf{Z} \mathbf{W}_2 + \mathbf{b}_2,
\]

where \(\mathbf{W}_1 \in \mathbb{R}^{D \times 2D}\), \(\mathbf{W}_2 \in \mathbb{R}^{2D \times D}\), and \(\operatorname{SiLU}(x) = x \cdot \sigma(x)\) (also known as Swish). This is exactly a two‑layer MLP with an expansion factor of 2, applied independently to each position. There is no token mixing; each position is transformed in isolation. Therefore, the module does not capture any contextual information; it relies entirely on the input features (which already contain positional embeddings and previous layer context) to produce the output.

The forward pass also accepts `**kwargs` (e.g., `attention_mask`) but ignores them. This means padding tokens are processed identically to valid tokens.

### 2.2 Injection Function

The function `inject_mamba` is registered under the same key `"mamba"`. It receives the layer, model configuration, compression configuration, layer index, total layers, and the original attention module (`old_attn`). It creates a new `Mamba` instance with `d_model` equal to `model_config.hidden_size`, and assigns it to `layer.attention`. The original attention module is discarded; no weight transfer occurs because the placeholder has no corresponding weights to copy. The module is therefore randomly initialised and must be trained from scratch.

### 2.3 Intended Full Mamba Model (Background)

The actual Mamba model (Gu & Dao, 2023) is a selective state‑space sequence model. It processes the input by projecting it to a higher dimension, applying a depthwise convolution, computing a time‑varying state transition matrix and step size, and performing a recurrent scan along the sequence dimension. The output is gated with a SiLU activation and projected back. The recurrence captures long‑range dependencies efficiently, with complexity \(O(L D^2)\). The provided placeholder omits all these components and retains only the initial projection and gating idea in a static, non‑recurrent form.

---

## 3. Implementation Details

- **`Mamba` class**:
  - Inherits from `nn.Module`.
  - `__init__` takes `d_model` and optional `**kwargs` (ignored). Creates `in_proj`: Linear(\(D\), \(2D\)) and `out_proj`: Linear(\(2D\), \(D\)).
  - `forward(x, **kwargs)`: applies `in_proj`, SiLU, then `out_proj`. Returns the transformed tensor.
- **Injection function**:
  - Registered as `"mamba"`.
  - Extracts `d_model` from `model_config.hidden_size`.
  - Constructs `Mamba(d_model)`.
  - Replaces `layer.attention` with the new module.
  - Returns the new module.

---

## 4. Usage and Configuration

The module is activated by setting the attention type to `"mamba"` in the compression configuration. No additional hyperparameters are required, as the placeholder only needs the hidden size.

Example:
```python
comp_config.attention_type = "mamba"
```

Manual injection:
```python
new_module = Mamba(d_model=768)
layer.attention = new_module
```

The module will be trained from scratch; no pretrained weights are transferred.

---

## 5. Discussion

The Mamba placeholder serves a purely structural role: it allows the framework to instantiate and run the model without the full Mamba implementation, ensuring that the injection mechanism and training loop are functional. Its position‑wise MLP design means it will not perform any sequence mixing; the output at each position depends only on that position’s input. Consequently, the model’s ability to capture dependencies will be severely limited unless the input already encodes rich context (e.g., through bidirectional layers or positional embeddings). The placeholder is not intended for production use and will underperform even a basic linear attention baseline.

To realise the benefits of the Mamba architecture, the placeholder must be replaced with a proper SSM implementation (e.g., from the official `mamba_ssm` package or a custom parallel scan). The injection function and registry key can be retained; only the module implementation needs to change. This modular design exemplifies the flexibility of the compression framework.

---

## 6. Conclusion

We have documented the Mamba placeholder module, a simple MLP acting as a stand‑in for the true Mamba state‑space model. It provides a drop‑in replacement for self‑attention in DistilBERT for framework testing purposes. The module highlights the ease of integrating new attention types into the compression registry and paves the way for the eventual integration of the full Mamba model. Future work will replace this stub with an efficient, trainable state‑space implementation to unlock linear‑complexity sequence modelling.