# UPIMambaBlock: A State‑Space Model with Unified Positional Interaction and Stuffed Gating

**Abstract**

We present **UPIMambaBlock**, a lightweight state‑space sequence model designed as a drop‑in replacement for the self‑attention sub‑layer in Transformer architectures. The block follows the core structure of the Mamba architecture—input projection, depthwise convolution, input‑dependent discretisation, diagonal state‑space recurrence, and output gating—while introducing two distinctive modifications. First, a **stuffed gating** mechanism applies a learnable global decay factor \(\gamma\) to the hidden state at each time step, implementing a simple but effective form of state regularisation. Second, only the first \(d_{\text{model}}\) channels of the expanded inner dimension are used for the state‑space dynamics, with the remaining channels zero‑padded after the recurrence to match the inner dimension before gating. This design reduces the computational cost of the state update while retaining compatibility with the expansion factor. The module is integrated into the compression framework via a registry‑based injection function (`"upi_stuffed_mamba"`), enabling its use as an efficient alternative to attention in DistilBERT layers. This paper provides a rigorous mathematical formulation of the block, highlights the role of the \(\gamma\) gating and the channel truncation strategy, and discusses the implementation details and potential benefits.

---

## 1. Introduction

State‑space models (SSMs) have emerged as powerful alternatives to self‑attention for sequence processing, offering linear time complexity and strong long‑range modelling capabilities. The Mamba architecture (Gu & Dao, 2023) combines a diagonal state matrix with input‑dependent step sizes and an output gate, achieving state‑of‑the‑art performance. The **UPIMambaBlock** presented here is a variant that modifies the original Mamba design in two ways aimed at simplifying and regularising the recurrent state.

First, a **stuffed gating** term is introduced: after the usual additive state update, a fraction \(\gamma\) of the state is subtracted, which is equivalent to scaling the state by \((1-\gamma)\). This introduces a learnable, global forgetting factor that can suppress the state uniformly, potentially improving stability and preventing unbounded growth.

Second, the state dynamics are applied only to the first \(d_{\text{model}}\) channels of the expanded inner representation. The remaining channels are kept zero, and after the recurrent scan, the output tensor is padded with zeros to recover the full inner dimension before being multiplied element‑wise with the gating branch. This truncation reduces the dimensionality of the state matrix, making the recurrence cheaper (the state size is \(d_{\text{model}} \times d_{\text{state}}\) instead of \(d_{\text{inner}} \times d_{\text{state}}\)), while the gating pathway still benefits from the full expanded representation.

These modifications result in a more compact and potentially faster SSM block while preserving the essential components of Mamba. The block is registered as an attention replacement in the compression framework, allowing it to be swapped into DistilBERT layers. In this paper, we provide a complete mathematical description of the UPIMambaBlock, detail its implementation, and analyse its computational properties and design choices.

---

## 2. Method

### 2.1 Input Projection and Expansion

Given an input sequence \(\mathbf{X} \in \mathbb{R}^{B \times L \times d_{\text{model}}}\), the block first projects the input to an inner dimension \(d_{\text{inner}} = E \cdot d_{\text{model}}\), where the expansion factor \(E\) defaults to 2. The projection is split into two branches:

\[
\mathbf{X}_{\text{proj}} = \mathbf{X} \mathbf{W}_{\text{in}} \in \mathbb{R}^{B \times L \times 2 d_{\text{inner}}},
\]
\[
\mathbf{X}_{\text{ssm}}, \mathbf{Z} = \operatorname{chunk}(\mathbf{X}_{\text{proj}}, 2, \text{dim}=-1).
\]

The branch \(\mathbf{X}_{\text{ssm}}\) (shape \(B \times L \times d_{\text{inner}}\)) is destined for the state‑space dynamics, while \(\mathbf{Z}\) is reserved for the output gate.

### 2.2 Depthwise Convolution

The SSM branch is processed by a depthwise 1D convolution to capture local temporal features. The convolution has kernel size \(d_{\text{conv}}\) (default 4) and uses \(d_{\text{inner}}\) groups (i.e., each channel is filtered independently). Causal padding is applied on the left, and the output is truncated to the original length \(L\):

\[
\tilde{\mathbf{X}}_{\text{ssm}} = \operatorname{Conv1D}(\mathbf{X}_{\text{ssm}}^\top)^\top \in \mathbb{R}^{B \times L \times d_{\text{inner}}}.
\]

### 2.3 Discretisation Step

The continuous‑time state‑space model requires a discretisation step size \(\Delta_t\). This is computed from the convolved features via a linear projection followed by a softplus activation (to ensure positivity):

\[
\Delta_t = \operatorname{softplus}(\tilde{\mathbf{X}}_{\text{ssm}, t} \mathbf{W}_\Delta + \mathbf{b}_\Delta) \in \mathbb{R}^{B \times d_{\text{model}}}.
\]

Note that \(\Delta_t\) is a vector of size \(d_{\text{model}}\) (not \(d_{\text{inner}}\)), matching the truncated state dimension. The weight \(\mathbf{W}_\Delta \in \mathbb{R}^{d_{\text{inner}} \times d_{\text{model}}}\) projects from the inner dimension.

### 2.4 Truncated State Matrix

The state‑space model uses a diagonal state matrix \(\mathbf{A} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{state}}}\), parameterised in log space for stability:

\[
\mathbf{A} = -\exp(\mathbf{A}_{\text{log}}), \quad \mathbf{A}_{\text{log}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{state}}}.
\]

The state dimension \(d_{\text{state}}\) is a hyperparameter (default 16). Crucially, the input to the state update is taken only from the first \(d_{\text{model}}\) channels of \(\tilde{\mathbf{X}}_{\text{ssm}}\):

\[
\mathbf{x}^{\text{ssm}}_t = \tilde{\mathbf{X}}_{\text{ssm}, t, :d_{\text{model}}} \in \mathbb{R}^{B \times d_{\text{model}}}.
\]

Thus, the state update operates on a reduced representation of size \(d_{\text{model}}\) instead of the full \(d_{\text{inner}}\).

### 2.5 State Update with Stuffed Gating

The hidden state \(\mathbf{h}_t \in \mathbb{R}^{B \times d_{\text{model}} \times d_{\text{state}}}\) evolves according to:

\[
\mathbf{h}_t = \mathbf{h}_{t-1} \odot \exp(\Delta_t \otimes \mathbf{A}) + \mathbf{x}^{\text{ssm}}_t \otimes \Delta_t,
\]

where \(\otimes\) denotes broadcasting over the appropriate dimensions. After this standard Mamba update, the **stuffed gating** is applied:

\[
\mathbf{h}_t \leftarrow \mathbf{h}_t - \gamma \cdot \mathbf{h}_t = (1 - \gamma) \mathbf{h}_t.
\]

Here \(\gamma \in \mathbb{R}\) is a learnable scalar parameter initialised to zero, ensuring that initially the state update is unchanged. The term subtracts a fraction of the state magnitude at every time step, acting as a uniform decay that complements the input‑dependent decay. Because \(\gamma\) can become positive during training, the effective decay factor becomes \(1-\gamma\), which may be less than 1, providing additional regularisation.

### 2.6 Output Readout and Padding

The output at time \(t\) is obtained by collapsing the state dimension with a learnable skip vector \(\mathbf{D} \in \mathbb{R}^{d_{\text{model}}}\) (initialised to ones):

\[
\mathbf{y}_t = (\mathbf{h}_t \cdot \mathbf{D}) = \sum_{j=1}^{d_{\text{state}}} \mathbf{h}_{t,:,j} \cdot \mathbf{D}_j \quad \in \mathbb{R}^{B \times d_{\text{model}}}.
\]

The resulting tensor \(\mathbf{Y}_{\text{ssm}} \in \mathbb{R}^{B \times L \times d_{\text{model}}}\) is then **padded** with zeros along the last dimension to match the size \(d_{\text{inner}}\):

\[
\tilde{\mathbf{Y}}_{\text{ssm}} = \operatorname{pad}(\mathbf{Y}_{\text{ssm}}, (0, d_{\text{inner}} - d_{\text{model}})) \quad \in \mathbb{R}^{B \times L \times d_{\text{inner}}}.
\]

This padding ensures compatibility with the gating branch \(\mathbf{Z}\), which is of shape \(B \times L \times d_{\text{inner}}\). The final output is obtained by element‑wise multiplication of the padded SSM output with a SiLU‑activated gate:

\[
\mathbf{Y}_{\text{out}} = \tilde{\mathbf{Y}}_{\text{ssm}} \odot \operatorname{SiLU}(\mathbf{Z}).
\]

Finally, a linear output projection \(\mathbf{W}_{\text{out}} \in \mathbb{R}^{d_{\text{inner}} \times d_{\text{model}}}\) maps back to the original model dimension:

\[
\mathbf{O} = \mathbf{Y}_{\text{out}} \mathbf{W}_{\text{out}}.
\]

The module returns a tuple \((\mathbf{O}, \text{None})\) for compatibility.

### 2.7 Complexity

By restricting the state dynamics to \(d_{\text{model}}\) channels, the recurrent state has size \(B \times d_{\text{model}} \times d_{\text{state}}\) instead of \(B \times d_{\text{inner}} \times d_{\text{state}}\). The cost per step is reduced from \(O(d_{\text{inner}} d_{\text{state}})\) to \(O(d_{\text{model}} d_{\text{state}})\). The overall complexity remains linear in \(L\). The convolution and projections still operate on the full \(d_{\text{inner}}\) dimension, but those are independent of the state size.

---

## 3. Implementation Details

- **Class `UPIMambaBlock`** is defined as a `nn.Module` and is registered under `"upi_stuffed_mamba"` in the `ATTENTION_REGISTRY` via the injection function `inject_upi_mamba`. However, note that the class itself is not decorated; only the injection function is registered. The injection function creates an instance of `UPIMambaBlock` and assigns it to `layer.attention`.
- **Constructor parameters**:
  - `dim`: the model dimension \(d_{\text{model}}\).
  - `d_state`: state dimension (default 16).
  - `d_conv`: convolution kernel size (default 4).
  - `expand`: expansion factor \(E\) (default 2).
- **Layers and parameters**:
  - `in_proj`: Linear layer mapping \(d_{\text{model}} \to 2 d_{\text{inner}}\) (without bias).
  - `conv1d`: depthwise Conv1d with kernel `d_conv`, groups = \(d_{\text{inner}}\), padding = `d_conv-1`.
  - `A_log`: parameter of shape \((d_{\text{model}}, d_{\text{state}})\) initialised as \(\log(\text{uniform}(0,0.1))\).
  - `D`: parameter of shape \((d_{\text{model}},)\) initialised to ones.
  - `dt_proj`: Linear layer \(d_{\text{inner}} \to d_{\text{model}}\) with bias.
  - `gamma`: learnable scalar initialised to zero.
  - `out_proj`: Linear layer \(d_{\text{inner}} \to d_{\text{model}}\) with bias (default).
- **Forward pass**:
  1. Project input to obtain `x_ssm` and `z`.
  2. Apply convolution to `x_ssm`.
  3. Compute `dt` via softplus.
  4. Compute `A = -exp(A_log)`.
  5. Truncate `x_ssm` to first `dim` channels.
  6. Initialise hidden state `h` as zeros.
  7. Loop over time steps: update `h`, apply gamma decay, compute output using `D`.
  8. Stack outputs, pad to `inner_dim`, gate with SiLU of `z`, project, return.

- **Mask handling**: The forward method accepts `**kwargs` but ignores them; there is no masking of padding tokens. This is a limitation for variable‑length sequences.

---

## 4. Usage and Configuration

The module is injected by setting the attention type to `"upi_stuffed_mamba"` in the compression configuration. It relies on the model config to provide `hidden_size`; no additional parameters are required via `comp_config` in the provided injection (the defaults for `d_state`, `d_conv`, `expand` are used). Example:

```python
comp_config.attention_type = "upi_stuffed_mamba"
```

Manual instantiation:
```python
block = UPIMambaBlock(dim=768, d_state=16, d_conv=4, expand=2)
layer.attention = block
```

The block does not initialise from pretrained attention weights; it is randomly initialised and must be fine‑tuned. The injection function discards the original `old_attn`.

---

## 5. Discussion

The UPIMambaBlock offers an interesting variation of the Mamba architecture. The stuffed gating parameter \(\gamma\) provides a simple, global control over the state magnitude, which may act as a regulariser or help the model learn a suitable forgetting rate when the input‑dependent decay is insufficient. Because \(\gamma\) is initialised to zero, the block initially behaves exactly like a standard Mamba (apart from the truncation), and can learn to increase \(\gamma\) if beneficial. The channel truncation is a pragmatic way to reduce the state size, especially when the expansion factor is large; however, it discards half of the expanded features from the state dynamics, which could limit capacity. The padding with zeros ensures the gating path still uses the full expansion, which may partially compensate.

Several limitations are apparent:
- **Sequential loop**: The recurrence uses a Python `for` loop, which is inefficient for long sequences. A parallel scan would be necessary for practical speed.
- **No padding mask**: The block processes all tokens equally, including padding, which can distort the state and output. Masking should be added to skip padding positions in the recurrence and convolution.
- **Truncation and padding**: The truncation strategy is a heuristic; it may be more effective to project the SSM output to the inner dimension with a linear layer rather than padding with zeros, but that would increase parameters.
- **Gamma interpretation**: The term \(h = h - \gamma h\) is applied after the state update, which makes it a post‑update decay rather than a pre‑update gating. This could also be absorbed into the discretisation, but keeping it separate may allow it to learn a data‑independent decay.

Despite these issues, the module demonstrates a creative modification of Mamba and integrates cleanly into the compression framework. It can serve as a baseline for exploring state‑size reduction and additional gating in SSMs.

---

## 6. Conclusion

We have presented UPIMambaBlock, a state‑space sequence model featuring a stuffed gating scalar and a truncated state dimension. The design aims to reduce computational cost while preserving the essential character of Mamba. The block is registered as an attention replacement and can be dropped into DistilBERT layers. Future improvements will address parallelisation, masking, and more sophisticated methods for handling the expanded inner dimension. The module adds to the growing family of efficient SSM‑based alternatives in the compression framework.