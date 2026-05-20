# YourCustomAttention: A Modular Template for Implementing Novel Attention Mechanisms

**Abstract**

We present a **template module** designed to streamline the implementation of custom attention mechanisms within the DistilBERT compression framework. The template provides a clean, extensible base that already implements a full standard multi‑head self‑attention layer, including query, key, value, and output projections, multi‑head reshaping, scaled dot‑product attention, masking, dropout, and output projection. It further includes an **optional low‑rank modulation** pathway, demonstrating how to inject a custom transformation into the attention scores. The module is designed as a drop‑in replacement for DistilBERT’s self‑attention sub‑layer and is registered in the `ATTENTION_REGISTRY` under a placeholder name `"your_attention_name"`. Users are intended to copy this template, rename the class and registry key, and implement their own attention logic by modifying the `forward` method. This paper describes the structure of the template, the mathematics of the provided baseline attention and modulation, and guidelines for customisation. The template serves as a pedagogical and practical starting point for rapid experimentation with new efficient attention variants.

---

## 1. Introduction

The compression framework provides a large library of pre‑implemented attention modules, but researchers often need to design and test their own mechanisms. The **YourCustomAttention** template offers a robust starting point that eliminates the boilerplate code required to integrate a new attention module into the DistilBERT model. It includes all the necessary components of a standard attention layer—projection matrices, multi‑head reshaping, scaled dot‑product, masking, and output projection—as well as a hook for custom modifications. As an illustrative example, the template also demonstrates an **optional rank‑modulation** feature: the attention scores are globally scaled by a scalar derived from a low‑rank projection of the pooled hidden states. This shows how additional computations can be inserted into the attention pipeline.

The template is registered with the framework’s registry using a placeholder key (`"your_attention_name"`), which the developer must change to a unique identifier. By following the template, new attention modules can be created with minimal effort and can be immediately injected into DistilBERT for training or evaluation.

In this paper, we document the architecture of the template, explain the mathematics of the default behaviour, describe the rank‑modulation example, and provide a guide for customisation. We also discuss best practices for integrating new modules into the compression pipeline.

---

## 2. Method

### 2.1 Standard Multi‑Head Self‑Attention (Baseline)

The template implements a full standard scaled dot‑product multi‑head self‑attention mechanism, which serves as the default behaviour when no custom modifications are applied. Given an input sequence \(\mathbf{X} \in \mathbb{R}^{B \times L \times d_{\text{model}}}\), the module first projects it into queries, keys, and values:

\[
\mathbf{Q} = \mathbf{X} \mathbf{W}_Q,\quad \mathbf{K} = \mathbf{X} \mathbf{W}_K,\quad \mathbf{V} = \mathbf{X} \mathbf{W}_V,
\]

where \(\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}\) (since the projections output the full hidden size, which equals \(H d_h\)). The tensors are reshaped to the multi‑head layout \((B, H, L, d_h)\) using the helper method `_reshape`. The attention scores are computed as:

\[
\mathbf{S} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_h}}.
\]

If an attention mask is provided, it is added to the scores; the mask is expected to be a tensor broadcastable to \((B, H, L, L)\) and should contain 0 for valid positions and \(-\infty\) for padding. Softmax is applied to obtain attention weights:

\[
\mathbf{A} = \operatorname{softmax}(\mathbf{S} + \mathbf{M}).
\]

Dropout is applied to the weights, and the context for each head is computed as \(\mathbf{C}_h = \mathbf{A}_h \mathbf{V}_h\). The heads are then concatenated and projected back to \(d_{\text{model}}\) via \(\mathbf{W}_O\):

\[
\mathbf{Y} = \mathbf{C}_{\text{flat}} \mathbf{W}_O.
\]

The module returns \((\mathbf{Y}, \mathbf{A})\) if `output_attentions` is `True`, otherwise \((\mathbf{Y}, \text{None})\).

This baseline is identical to the standard attention mechanism and serves as the foundation upon which custom modifications can be built.

### 2.2 Optional Low‑Rank Modulation (Example Customisation)

To illustrate how to inject a novel computation into the attention pipeline, the template includes an optional **rank‑modulation** block, activated by providing a positive integer `rank` to the constructor. The idea is to globally modulate the attention scores based on a summary of the input sequence.

The modulation proceeds as follows:

1. **Global pooling**: The hidden states \(\mathbf{X} \in \mathbb{R}^{B \times L \times d_{\text{model}}}\) are averaged over the sequence length, yielding a single vector per batch element:
   \[
   \mathbf{h}_{\text{pool}} = \frac{1}{L} \sum_{i=1}^{L} \mathbf{X}_{:,i,:} \in \mathbb{R}^{B \times d_{\text{model}}}.
   \]

2. **Low‑rank projection**: This pooled vector is projected to a lower‑dimensional space (size \(r\)) using a linear layer without bias:
   \[
   \mathbf{z} = \mathbf{W}_{\text{rank}} \mathbf{h}_{\text{pool}} \in \mathbb{R}^{B \times r}.
   \]

3. **Modulation scalar**: The elements of \(\mathbf{z}\) are passed through a sigmoid activation to obtain values in \((0,1)\). Then, the mean over the rank dimension is taken, resulting in a scalar per batch element:
   \[
   \beta = \frac{1}{r} \sum_{j=1}^{r} \sigma(z_j) \in (0,1).
   \]

4. **Score scaling**: This scalar is reshaped to \((B, 1, 1, 1)\) and multiplied element‑wise with the entire attention score tensor:
   \[
   \mathbf{S}' = \beta \cdot \mathbf{S}.
   \]

Thus, the modulation acts as a global gating of the attention strengths, controlled by the aggregated sequence information. This is a simplistic demonstration; real custom mechanisms would replace this with more sophisticated operations (e.g., dynamic span masks, sparsification, local windowing, etc.).

### 2.3 Integration and Customisation Guide

To create a new attention module using this template:

1. **Copy the template file** and rename the class from `YourCustomAttention` to the desired name.
2. **Change the registry key** from `"your_attention_name"` to a unique string that will identify the module in the compression configuration (e.g., `"my_sparse_attention"`).
3. **Modify the constructor** to add any new layers, parameters, or hyperparameters.
4. **Implement the custom attention logic** inside the `forward` method. The template provides a clearly marked section (“Your custom modification”) where the new code should be placed. The rest of the pipeline—projection, reshaping, mask handling, softmax, and output—can be left unchanged or adapted as needed.
5. **Optionally**, remove the example rank‑modulation block if it is not relevant.
6. **Update docstrings** and metadata methods (`complexity`, `extra_repr`) to reflect the new mechanism.

The template also provides hooks such as `_init_weights` (custom weight initialisation) and `complexity` / `extra_repr` for reporting. The module inherits from `BaseAttention`, which may add features like orthogonal loss; if not needed, the custom class can simply inherit from `nn.Module` and omit the `BaseAttention` interface, though then it must implement the expected `forward` signature.

### 2.4 Example Use Case

A researcher wishing to implement a “Local‑Global Hybrid Attention” could use this template to set up the standard projections, then in the custom section:

- Compute a local window mask and apply it to the scores (setting distant positions to \(-\infty\)).
- Compute a global summary via low‑rank projection or a separate set of global tokens.
- Combine the two branches with a learned gate.
- The remaining softmax and output steps remain unchanged.

The template reduces the development overhead by providing the complete attention infrastructure.

---

## 3. Implementation Details

- **Class `YourCustomAttention`**: Inherits from `BaseAttention`. Registered as `"your_attention_name"`.
- **Constructor**:
  - `config`: standard model configuration object.
  - `rank`: if provided and >0, creates the rank‑modulation path (a linear layer from `hidden_size` to `rank`); otherwise, `rank_proj` is `None`.
  - `custom_param`: an example of an additional hyperparameter passed via `**kwargs`.
  - Initialises Q, K, V, O linear layers, and an optional `rank_proj`.
  - Calls `_init_weights` to set Xavier uniform initialisation on all linear layers (biases zero).
- **`_reshape`**: Splits the last dimension into `(num_heads, head_dim)` and permutes to `(B, H, L, d_h)`.
- **Forward**:
  1. Compute Q, K, V via projections and `_reshape`.
  2. Compute scaled dot‑product scores.
  3. If `rank_proj` is not None, compute modulation scalar and multiply scores.
  4. Add attention mask (unsqueezed to `(B, 1, L, L)` – note: this assumes mask is already in the correct shape; typical HuggingFace 2D mask should be converted to additive form beforehand, which is not done here, a limitation).
  5. Softmax, dropout.
  6. Aggregate values, concatenate heads, output projection.
  7. Return output and optionally attention weights.

---

## 4. Usage and Configuration

As a template, this module is not intended to be used directly with the placeholder name. After renaming, it can be injected by setting the attention type to the new registry key in the compression configuration. Additional parameters can be passed via `comp_config` or `**kwargs`. For the example modulation, one would set:

```python
comp_config.attention_type = "your_attention_name"   # after renaming
comp_config.rank = 16
```

The module can then be injected into each DistilBERT layer by the framework’s replacement logic. The developer must ensure that the injection function (if needed) properly instantiates the custom class with the required parameters. A standard injection function can be modelled after those in other modules, or the class can be injected directly if the framework supports it.

---

## 5. Discussion

The template embodies the philosophy of the compression framework: provide reusable infrastructure so that researchers can focus on the novel aspects of their attention mechanisms. The standard attention backbone ensures that the new module is immediately compatible with the existing training loops, data pipelines, and evaluation procedures. The included rank‑modulation example is intentionally simple, serving more as a syntax reference than as a competitive method.

However, developers should be aware of a few points:

- **Mask format**: The template does not convert a standard 2D binary mask (1 for valid, 0 for padding) into additive form. When replacing DistilBERT’s attention, the incoming mask may be 2D and must be converted (e.g., `(1.0 - mask) * -1e9`) before adding to scores. This conversion can be added in the custom `forward` method or handled by a wrapper.
- **Performance**: The rank‑modulation example uses a global average pool, which is a strong baseline but may not be optimal for all tasks. The template encourages experimentation but also requires careful design to avoid introducing bottlenecks.
- **Inheritance**: `BaseAttention` may impose certain expectations (e.g., `get_ortho_loss`). If the custom module does not need those, it can inherit from `nn.Module` and define the forward pass independently; the framework’s injection functions can still handle it if they expect a generic `nn.Module`.

Overall, the template successfully lowers the barrier to entry for developing new attention mechanisms, accelerating the research cycle in efficient Transformers.

---

## 6. Conclusion

We have described the `YourCustomAttention` template, a fully functional and extensible self‑attention module for the DistilBERT compression framework. By providing a clean baseline and a simple customisation example, it enables rapid prototyping of novel attention variants. Researchers can clone the template, modify the core attention computation, and immediately evaluate their ideas on standard NLP tasks. The template is a testament to the modularity and extensibility of the framework, and we encourage its use for future innovations in efficient attention. After renaming and implementing the desired logic, the module can be deployed as a drop‑in replacement for the original attention layers, seamlessly integrating with the rest of the compression pipeline.