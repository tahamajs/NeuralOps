# SimpleSSM: A Lightweight State‑Space Model Block for Hybrid Transformers

**Abstract**

We present **SimpleSSM**, a compact, self‑contained state‑space sequence model (SSM) block designed as a drop‑in replacement for self‑attention or as a component in hybrid architectures. Inspired by the Mamba and S4 architectures, the block integrates a **1D depthwise convolution**, a **diagonal state‑space model with input‑dependent discretisation**, a **skip connection**, and a **gating mechanism**. The module processes sequences in linear time with respect to sequence length, using a recurrent scan that is suitable for long‑context processing. It serves as a foundational building block for many of the efficient attention modules in the compression framework. This paper provides a rigorous mathematical description of the SimpleSSM block, including its projection, convolution, state transition, discretisation, and output gating, along with implementation details and usage.

---

## 1. Introduction

State‑space models (SSMs) have emerged as powerful alternatives to self‑attention for sequence modelling, offering linear computational complexity and strong performance on long‑range tasks. Architectures such as S4, S5, and Mamba demonstrate that carefully structured SSMs can match or exceed Transformers in quality while being far more efficient during inference. The **SimpleSSM** module implements a streamlined version of these ideas, suitable for rapid experimentation and integration into hybrid Transformer blocks.

The module follows the core design of Mamba: an input is projected to a higher‑dimensional space, processed by a depthwise convolution, and then passed through a discretised diagonal state‑space model. The state update uses a learned diagonal transition matrix \(\mathbf{A}\) and an input‑dependent time step \(\Delta_t\). The output is gated by a SiLU‑activated branch and projected back to the original dimension. While the implementation is simplified—it uses a sequential Python loop for the recurrent scan and a truncation of the expanded dimension to match the model size—it retains the essential characteristics of a modern SSM.

SimpleSSM is registered under the key `"simple_ssm"` in the `ATTENTION_REGISTRY` and is used by several hybrid attention blocks (e.g., Hymba‑TransX, Jamba‑Info, SAMBA‑M3) as a lightweight global processing branch. This paper details the architecture, mathematical formulation, and practical usage of SimpleSSM.

---

## 2. Method

### 2.1 Input Projection

Given an input sequence \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\), the module first projects it to an inner dimension \(D_{\text{inner}} = E \cdot D\) where the expansion factor \(E\) defaults to 2:

\[
\mathbf{X}_{\text{proj}} = \mathbf{X} \mathbf{W}_{\text{in}} \in \mathbb{R}^{B \times L \times 2 D_{\text{inner}}}.
\]

The weight matrix \(\mathbf{W}_{\text{in}} \in \mathbb{R}^{D \times 2 D_{\text{inner}}}\) maps the input to **two** copies of the inner dimension, which are then split into two branches:

\[
\mathbf{X}_{\text{ssm}}, \mathbf{Z} = \operatorname{chunk}(\mathbf{X}_{\text{proj}}, 2, \text{dim}=-1),
\]

each of shape \((B, L, D_{\text{inner}})\). The first branch \(\mathbf{X}_{\text{ssm}}\) will undergo the SSM transformation, while the second branch \(\mathbf{Z}\) will serve as a multiplicative gate.

### 2.2 Depthwise Convolution

Before entering the state‑space recurrence, the SSM branch \(\mathbf{X}_{\text{ssm}}\) is processed by a 1D depthwise convolution. The convolution uses a kernel size \(d_{\text{conv}}\) (default 4) with `groups = D_{\text{inner}}`, meaning each channel is filtered independently:

\[
\mathbf{X}_{\text{ssm}}^{(t)} \;\xrightarrow{\text{Conv1D}}\; \tilde{\mathbf{X}}_{\text{ssm}}^{(t)}.
\]

The convolution is causal: padding is applied only on the left, and the output is truncated to the original sequence length \(L\). This introduces local temporal smoothing and is essential for the SSM to capture short‑term patterns. The operation is performed after transposing to \((B, D_{\text{inner}}, L)\) and back.

### 2.3 Discretisation of the State‑Space Model

The core of the SSM is a continuous‑time linear state‑space model defined by the state equation:

\[
\dot{\mathbf{h}}(t) = \mathbf{A} \mathbf{h}(t) + \mathbf{B} \mathbf{x}(t).
\]

For computational efficiency, the system is discretised with a zero‑order hold (ZOH) using a time‑varying step size \(\Delta_t\). The discrete state update becomes:

\[
\mathbf{h}_t = \exp(\Delta_t \mathbf{A}) \mathbf{h}_{t-1} + \Delta_t \mathbf{B} \mathbf{x}_t,
\]

where \(\Delta_t\) is a scalar that can vary per token and per channel. In SimpleSSM, the matrix \(\mathbf{A}\) is restricted to be **diagonal**. Its eigenvalues are parameterised in log space as \(\mathbf{A}_{\text{log}} \in \mathbb{R}^{D \times d_{\text{state}}}\) and then exponentiated and negated to obtain stable, decaying dynamics:

\[
\mathbf{A} = -\exp(\mathbf{A}_{\text{log}}).
\]

The step size \(\Delta_t\) is computed from the convolved signal via a linear projection and a softplus activation:

\[
\Delta_t = \operatorname{softplus}(\tilde{\mathbf{X}}_{\text{ssm},t} \mathbf{W}_{\Delta} + \mathbf{b}_{\Delta}) \in \mathbb{R}^{B \times D}.
\]

The weight \(\mathbf{W}_{\Delta} \in \mathbb{R}^{D_{\text{inner}} \times D}\) maps the inner dimension to the model dimension \(D\). Importantly, \(\Delta_t\) is a vector of size \(D\), providing channel‑wise adaptation of the time scale.

### 2.4 Recurrent State Update

The hidden state \(\mathbf{h}_t \in \mathbb{R}^{B \times D \times d_{\text{state}}}\) evolves according to:

\[
\mathbf{h}_t = \exp(\Delta_t \otimes \mathbf{A}) \odot \mathbf{h}_{t-1} + \Delta_t \otimes \mathbf{x}_t,
\]

where \(\otimes\) broadcasts \(\Delta_t\) over the state dimension \(d_{\text{state}}\), and \(\mathbf{x}_t\) is the input at time \(t\) (taken as the first \(D\) channels of \(\tilde{\mathbf{X}}_{\text{ssm},t}\) — a simplification for dimension compatibility). The output at time \(t\) is then:

\[
\mathbf{y}_t = (\mathbf{h}_t \cdot \mathbf{D}) \in \mathbb{R}^{B \times D},
\]

where \(\mathbf{D} \in \mathbb{R}^{D}\) is a learnable skip connection vector (initialised to ones). The term \(\mathbf{h}_t \cdot \mathbf{D}\) performs a weighted sum over the state dimension, effectively collapsing the state back to the model dimension. The output sequence \(\mathbf{Y}_{\text{ssm}} \in \mathbb{R}^{B \times L \times D}\) is collected by stacking \(\mathbf{y}_t\) across time.

In the current implementation, the recurrence is computed via a simple Python `for` loop over the sequence length. This is not optimised but is straightforward and sufficient for moderate lengths. A parallel scan algorithm would be used in a production‑grade implementation.

### 2.5 Gating and Output Projection

The second branch \(\mathbf{Z}\) from the input projection is activated with the SiLU (Sigmoid Linear Unit) function:

\[
\mathbf{G} = \mathbf{Z} \odot \sigma(\mathbf{Z}) \quad (\text{SiLU}).
\]

To match dimensions, only the first \(D\) channels of \(\mathbf{G}\) are used (as \(\mathbf{G}_{[:,:,:D]}\)). The SSM output is then gated element‑wise:

\[
\mathbf{Y}_{\text{out}} = \mathbf{Y}_{\text{ssm}} \odot \mathbf{G}_{[:,:,:D]}.
\]

Finally, a linear output projection \(\mathbf{W}_{\text{out}} \in \mathbb{R}^{D_{\text{inner}} \times D}\) maps back to the model dimension (though in the code, it is applied after the gating? Actually the code does `out = out_ssm * self.act(z[:,:,:d_model])` and then `return self.out_proj(out)`. But `out_ssm` is of shape `(B, L, D)` and `z` is of shape `(B, L, inner)`. The gating uses `z[:,:,:d_model]`, which is a subset of channels. Then the `out_proj` expects input of size `inner_dim`? Wait, the code says `self.out_proj = nn.Linear(inner_dim, d_model)`. But `out = out_ssm * self.act(z[:,:,:d_model])` produces shape `(B, L, d_model)`, and then `self.out_proj(out)` expects input of size `inner_dim`, which is `2*d_model`. There is a dimensionality mismatch in the code! The output projection should take either `inner_dim` or `d_model` as input; currently it takes `inner_dim`, but the gating produces `d_model`. This is a bug. We'll point it out in the Discussion. For the sake of explanation, we assume the intended design is to project the gated output back to `d_model` after gating with a linear layer of input size `d_model`. But the code as written will throw a shape error. We'll note this in the discussion.

### 2.6 Complexity

The dominant costs are:
- Projection: \(O(B L D D_{\text{inner}})\).
- Convolution: \(O(B L D_{\text{inner}} d_{\text{conv}})\).
- Discretisation: \(O(B L D_{\text{inner}} D)\).
- Recurrent state update: \(O(B L D d_{\text{state}})\).
Total time is linear in \(L\). Memory for the state is \(O(B D d_{\text{state}})\), independent of sequence length.

---

## 3. Implementation Details

- **`SimpleSSM`** inherits from `nn.Module` and is registered under `"simple_ssm"`.
- **Constructor parameters**:
  - `d_model`: the model dimension \(D\) (e.g., 768).
  - `d_state`: state dimension \(N\) (default 16).
  - `d_conv`: convolution kernel size (default 4).
  - `expand`: expansion factor \(E\) (default 2).
- **Layers**:
  - `in_proj`: Linear(\(D\), \(2 E D\)) without bias.
  - `conv1d`: depthwise Conv1d with kernel `d_conv`, padding `d_conv-1`, groups = inner_dim.
  - `dt_proj`: Linear(\(E D\), \(D\)) with bias.
  - `A_log`: learnable parameter of shape \((D, N)\), initialised as \(\log(\text{Uniform}(0,0.1))\)? Actually `torch.log(torch.rand(D, N) * 0.1)` gives log of values in (0, 0.1), so A_log is negative and after exp and negation yields small positive values. The range is small to ensure stable dynamics.
  - `D`: learnable vector of size \(D\), initialised to 1.0.
  - `out_proj`: Linear(\(E D\), \(D\)) (but as noted, the input size may be mismatched with gating output).
  - `act`: SiLU activation.
- **Forward**:
  1. Project input.
  2. Split into `x_ssm` and `z`.
  3. Apply convolution to `x_ssm`.
  4. Compute `dt` via softplus on `dt_proj(x_ssm)`.
  5. Compute `A = -exp(A_log)`.
  6. Initialise hidden state `h` of zeros.
  7. Loop over time steps:
     - Take slice of `x_ssm` (first `d_model` channels) as input.
     - Update state: `h = h * exp(dt * A) + (x_slice.unsqueeze(-1) * dt.unsqueeze(-1))`.
     - Compute output: `out_t = (h * D.unsqueeze(-1)).sum(dim=-1)`.
  8. Stack outputs.
  9. Gate with `z` subset (first `d_model` channels).
  10. Apply output projection (bug: dimension mismatch likely).
  11. Return result.

- **Bug**: The gating produces a tensor of shape `(B, L, D)`, but `out_proj` expects input of shape `(B, L, inner_dim)` where `inner_dim = E * D`. This will raise a runtime error when `E != 1`. For the default \(E=2\), `inner_dim = 2D`, so the linear layer expects 1536‑dim input, but receives 768. This must be fixed either by adjusting the gating to use all `inner_dim` channels or by changing the output projection to accept `D`-dim input. In many implementations, the gating uses the full expanded dimension and then a linear projection back, or the SSM output is first projected to `inner_dim` before gating. We'll note this as a critical bug.

---

## 4. Usage and Configuration

SimpleSSM can be instantiated directly and used as a sequence mixer:

```python
ssm = SimpleSSM(d_model=768, d_state=16, d_conv=4, expand=2)
output = ssm(input_tensor)  # input: (B, L, 768)
```

It can also be injected as an attention replacement via the registry key `"simple_ssm"` if an injection function is provided (not shown). The hyperparameters are:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | required | Hidden dimension \(D\). |
| `d_state` | 16 | State dimension \(N\). |
| `d_conv` | 4 | Convolution kernel size. |
| `expand` | 2 | Expansion factor for inner dimension. |

The module is often used within hybrid blocks that require a lightweight SSM component.

---

## 5. Discussion

SimpleSSM captures the essential ingredients of a modern state‑space model: a diagonal state matrix, input‑dependent time steps, a convolutional front‑end, and gating. It serves as a useful baseline and a drop‑in component for hybrid architectures. The use of a diagonal \(\mathbf{A}\) simplifies the state update to an element‑wise operation, making it memory‑efficient and easy to implement. The softplus discretisation ensures positivity of \(\Delta_t\), which is crucial for stable state evolution.

However, the current implementation has several notable limitations:

1. **Dimensionality bug**: The gating and output projection shapes are incompatible when `expand != 1`. This prevents the module from running with the default settings and must be corrected. A typical fix is to either:
   - Gate with the full expanded dimension (i.e., use all channels of \(\mathbf{Z}\)) and then project back, which matches `out_proj` expectation.
   - Or, project the SSM output to the expanded dimension before gating, then project back.
   The intended design likely follows the first approach, where the SSM output is \(\mathbf{Y}_{\text{ssm}}\) of shape \((B, L, D)\), and the gate \(\mathbf{G}\) should also be of shape \((B, L, D)\); thus the `out_proj` should take input size \(D\), not \(E D\). The current `out_proj = nn.Linear(inner_dim, d_model)` is inconsistent.

2. **Sequential loop**: The recurrent scan is implemented with a Python `for` loop, which is inefficient on GPUs for long sequences. A parallel scan (e.g., `torch.cumsum` or a custom kernel) would greatly improve speed.

3. **Truncation of input**: The SSM input is taken as only the first \(D\) channels of the expanded representation. This discards half of the projected features, wasting capacity. A full implementation would use all \(E D\) channels by reshaping the state dimension accordingly (e.g., treating each channel as an independent SSM with a single state, akin to Mamba’s independent channels).

4. **No padding mask support**: The module ignores any `attention_mask`, meaning padding tokens will be processed and affect the state. This can be problematic for variable‑length sequences.

Despite these issues, the module is pedagogically clear and demonstrates the core principles of SSMs. It is actively used in the framework’s hybrid blocks (e.g., Hymba‑TransX, Jamba‑Info, SAMBA‑M3) as a placeholder that can be replaced with a more efficient SSM kernel once the bugs are fixed.

---

## 6. Conclusion

We have described SimpleSSM, a lightweight state‑space model block that combines a diagonal state transition, input‑dependent discretisation, a depthwise convolution, and gating. The module offers linear‑time sequence processing and serves as a foundational building block for hybrid Transformer architectures within the compression framework. While the current implementation has some dimensional mismatches and performance bottlenecks, it provides a clear and extensible starting point for SSM‑based efficient attention. Future work will resolve the gating dimension bug, implement a parallel scan, and incorporate padding mask handling to make SimpleSSM fully production‑ready.