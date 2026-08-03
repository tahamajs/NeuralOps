# HGRN2 Layer: A Gated Linear Recurrence with Controlled Forgetting

**Abstract**

We present the **HGRN2 Layer**, a lightweight, linear‑time sequence mixing module based on a gated recurrent state update. The layer processes tokens sequentially, maintaining a single vector hidden state that is updated via element‑wise forget and input gates. The state is clipped to a fixed range to prevent unbounded growth, and the output at each step is a linear projection of the clamped state. This recurrence is inspired by the Hierarchically Gated Recurrent Network (HGRN) and the DeltaNet family, but simplifies the design to two independent gates and a direct additive input. The HGRN2 Layer serves as a drop‑in replacement for the multi‑head self‑attention sub‑layer in Transformer models, offering linear complexity in sequence length and extremely low parameter overhead. The module is integrated into the DistilBERT compression framework via a registry‑based injection function. This paper details the mathematical formulation of the recurrence, the implementation, and its usage.

---

## 1. Introduction

Standard self‑attention in Transformers scales quadratically with the sequence length, making it costly for long sequences. Recurrent models, on the other hand, process tokens one by one with constant per‑step computational cost, giving linear total complexity. However, simple recurrent networks like vanilla RNNs suffer from vanishing and exploding gradients and limited expressiveness. Gated recurrent architectures (e.g., LSTM, GRU) mitigate these issues by learning to retain or discard information from the past.

The **HGRN2 Layer** is a minimal gated recurrence designed for efficiency. It maintains a hidden state \(\mathbf{h}_t \in \mathbb{R}^D\) that is updated at each time step using a forget gate \(\mathbf{f}_t\) and an input gate \(\mathbf{i}_t\), both computed from the current input. The new input is added directly to the state (without an internal transformation), and the state is clamped to \([-1, 1]\) to ensure stability. The output is a linear readout of the clamped state. This design avoids the quadratic cost of attention and the complex parameterisations of LSTMs, while still providing a learnable gating mechanism for selective memory.

In this paper, we provide a rigorous description of the HGRN2 recurrence, its implementation as a PyTorch module, and its integration into the compression framework. We also discuss its computational properties, limitations, and possible extensions.

---

## 2. Method

### 2.1 Notation

Let \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\) be a batch of \(B\) sequences of length \(L\) with hidden dimension \(D\). For each time step \(t \in \{1, \dots, L\}\), the input token is \(\mathbf{x}_t = \mathbf{X}_{[:, t, :]} \in \mathbb{R}^{B \times D}\). The layer maintains a hidden state \(\mathbf{h}_t \in \mathbb{R}^{B \times D}\), initialised to the zero vector: \(\mathbf{h}_0 = \mathbf{0}\).

### 2.2 Gate Computation

At each step, two gates are computed from the input using linear transformations followed by sigmoid activation:

\[
\mathbf{f}_t = \sigma\!\big(\mathbf{W}_f \mathbf{x}_t + \mathbf{b}_f\big), \qquad
\mathbf{i}_t = \sigma\!\big(\mathbf{W}_i \mathbf{x}_t + \mathbf{b}_i\big),
\]

where \(\mathbf{W}_f, \mathbf{W}_i \in \mathbb{R}^{D \times D}\) and \(\mathbf{b}_f, \mathbf{b}_i \in \mathbb{R}^D\) are learnable parameters. The sigmoid \(\sigma(x) = 1/(1+e^{-x})\) ensures each gate value lies in \((0, 1)\).

### 2.3 State Update with Clipping

The hidden state is updated as a convex combination of the previous state and the current input, weighted by the gates:

\[
\mathbf{h}_t = \mathbf{f}_t \odot \mathbf{h}_{t-1} + \mathbf{i}_t \odot \mathbf{x}_t,
\]

where \(\odot\) denotes element‑wise multiplication. The forget gate controls how much of the past state is retained, while the input gate scales the new information to be injected. After the update, the state is clamped to the range \([-1, 1]\):

\[
\mathbf{h}_t \leftarrow \operatorname{clamp}\!\big(\mathbf{h}_t,\ -1,\ 1\big).
\]

Clipping acts as a simple, hard form of regularisation that prevents the state from growing arbitrarily large over long sequences, stabilising training and inference. The threshold is fixed at 1.0 (stored as `res_clip`).

### 2.4 Output Readout

The output at time \(t\) is a linear projection of the clamped state:

\[
\mathbf{y}_t = \mathbf{W}_o \mathbf{h}_t + \mathbf{b}_o,
\]

with \(\mathbf{W}_o \in \mathbb{R}^{D \times D}\) and \(\mathbf{b}_o \in \mathbb{R}^D\). The final output of the layer is the sequence of \(\mathbf{y}_t\) stacked along the time dimension: \(\mathbf{Y} \in \mathbb{R}^{B \times L \times D}\).

### 2.5 Complexity

The per‑step computation involves two linear transforms of size \(D \times D\) (for gates) and one for the output, plus element‑wise operations. Thus, the total complexity for a sequence of length \(L\) is \(O(L D^2)\). This compares favourably to the \(O(L^2 D)\) of full attention when \(L \gg D\). The hidden state is a single vector, so memory is \(O(D)\) per token, independent of sequence length.

---

## 3. Implementation Details

### 3.1 Module Definition

The `HGRN2Layer` class inherits from `nn.Module` and is registered under the key `"hgrn2_layer"` in `ATTENTION_REGISTRY`. Its constructor takes only `d_model` (the hidden dimension \(D\)) and initialises three linear layers:

- `forget_gate`: maps \(D \to D\).
- `input_gate`: maps \(D \to D\).
- `output`: maps \(D \to D\).

All linear layers include biases by default. The clipping threshold is stored as `res_clip = 1.0`.

### 3.2 Forward Pass

The `forward` method iterates over the time dimension with a Python loop. At each step:
- Extract \(\mathbf{x}_t\).
- Compute \(\mathbf{f}_t\) and \(\mathbf{i}_t\) via sigmoid.
- Update \(\mathbf{h}_t\) using element‑wise operations.
- Clamp \(\mathbf{h}_t\).
- Compute \(\mathbf{y}_t = \text{output}(\mathbf{h}_t)\).
- Append \(\mathbf{y}_t\) to a list.

After the loop, the outputs are stacked into a tensor of shape \((B, L, D)\). The function returns a tuple `(out, None)` to mimic the standard attention interface (which sometimes returns attention weights). The `**kwargs` argument is accepted but not used; any `attention_mask` is ignored.

### 3.3 Injection Function

The injection function `inject_hgrn2_layer` is also registered under `"hgrn2_layer"` (overriding the class? Actually both the class and function are registered separately; the registry likely distinguishes by name, but both have the same key. This might be an oversight; typically, one would register only the class or the injection function, not both. In the code, the injection function is defined after the class and also registered with the same key. In many frameworks, the injection function is the one used by the outer framework to instantiate and replace the attention module. The class definition might be used if someone directly instantiates it. For the purpose of this explanation, we describe the injection function as the entry point for the framework.)

The injection function:
- Takes the usual parameters: `layer` (the DistilBERT layer), `model_config`, `comp_config`, `layer_idx`, `total_layers`, `old_attn`.
- Extracts `hidden_size` from `model_config`.
- Creates a new `HGRN2Layer(d_model)`.
- Assigns it to `layer.attention`.
- Returns the new layer.

The function does **not** utilise `old_attn`; thus, no pretrained weight transfer is performed. The HGRN2 layer is trained from scratch.

---

## 4. Usage and Configuration

The module can be injected into a DistilBERT model by setting the attention type to `"hgrn2_layer"` in the compression configuration. No additional hyperparameters are needed, as the layer only requires the hidden size, which comes from the model config.

Example of an injection via the framework:
```python
comp_config.attention_type = "hgrn2_layer"
# The framework calls inject_hgrn2_layer automatically.
```

Manual instantiation:
```python
layer = HGRN2Layer(d_model=768)
old_attn = original_transformer_layer.attention
original_transformer_layer.attention = layer
```

**Compatibility note**: The HGRN2 layer returns a tuple `(output, None)`, which is compatible with the expected output of the attention sub‑layer in DistilBERT. The outer Transformer layer (which already applies a residual connection and layer normalisation) will add the residual and normalise as usual:

\[
\mathbf{H}_{\text{out}} = \text{LayerNorm}(\mathbf{H} + \text{HGRN2Layer}(\mathbf{H})).
\]

---

## 5. Discussion

The HGRN2 Layer offers an extremely efficient alternative to self‑attention. It is reminiscent of the Linear Transformer’s recurrent formulation with identity feature map, but with explicit gating and a clipping mechanism. Its advantages include:

- **Linear complexity**: \(O(L D^2)\) time, independent of sequence length.
- **Small state**: Only a single vector per batch element, making it memory friendly for very long sequences.
- **Simplicity**: Three linear layers and a handful of element‑wise operations; easy to implement and optimise.

However, several limitations must be noted:

1. **Sequential bottleneck**: The Python `for` loop over time steps prevents parallelisation during training. For GPUs, this results in poor utilisation. A vectorised or parallel scan implementation would be necessary for practical training on long sequences.

2. **No head‑wise parallelism**: Unlike multi‑head attention, all information is compressed into a single vector state. This may limit the model’s capacity to represent different types of relationships simultaneously. Multi‑head extensions are possible (e.g., having multiple independent state vectors) but are not implemented.

3. **No padding mask handling**: The layer processes every token identically, including padding tokens. For padded sequences, the hidden state would become contaminated by meaningless positions. A proper implementation would skip state updates or zero out the state for padding tokens, or the caller should pack sequences without padding.

4. **Fixed clipping**: The clamping threshold of 1.0 is hard‑coded and may not be optimal for all datasets or scales. A learnable threshold or parameterised normalisation (e.g., RMSNorm on the state) could improve flexibility.

5. **Initialisation and training**: Since the module does not initialise from pretrained attention weights, it requires full fine‑tuning from scratch. The gating biases are initialised to zero (default `nn.Linear` bias init), which gives \(\sigma(0)=0.5\), a neutral forgetting rate. This might require careful learning rate scheduling.

Despite these drawbacks, the HGRN2 layer serves as a valuable baseline for exploring lightweight recurrent alternatives to attention, especially in resource‑constrained settings or for streaming applications where tokens arrive one by one.

---

## 6. Conclusion

We have described the HGRN2 Layer, a minimal gated linear recurrence for sequence mixing. Its forget‑and‑input gate design with state clipping provides a stable and efficient method to process tokens sequentially. Integrated into the DistilBERT compression framework, it offers a drop‑in replacement for self‑attention with linear time complexity. Future work will focus on parallelising the recurrence, incorporating masking, and exploring multi‑head variants to enhance capacity while retaining efficiency.Thank you for providing the complete code and the request to explain it comprehensively. Below is a paper‑writer style explanation of the HGRN2 Layer module, covering the method, mathematics, implementation, and discussion, as if it were a section of a research paper.

---

# HGRN2 Layer: A Gated Linear Recurrence with Clipped State

## Abstract

We present the **HGRN2 Layer**, a minimalist recurrent module designed to replace the self‑attention sub‑layer in Transformer models. The layer processes tokens sequentially, maintaining a single hidden vector that is updated through element‑wise forget and input gates and then clamped to a fixed range. The output at each time step is a linear readout of the clamped state. The recurrence is inspired by the Hierarchically Gated Recurrent Network (HGRN) and the DeltaNet family but removes all internal non‑linearities except the gates and the clipping. The resulting computational complexity is linear in sequence length \(O(L d^2)\), making it a highly efficient alternative to quadratic self‑attention. The module is integrated into the DistilBERT compression framework via a registry‑based injection function.

---

## 1. Introduction

The quadratic cost of multi‑head self‑attention with respect to sequence length \(L\) motivates the search for efficient alternatives. Gated recurrent units (e.g., LSTM, GRU) process sequences in \(O(L d^2)\) time but often underperform attention‑based models on long‑range tasks. Recently, simplified linear recurrences, such as the Linear Transformer and the HGRN, have shown that a well‑designed recurrent state can approach the performance of full attention while remaining computationally cheap. The **HGRN2 Layer** adopts this philosophy: it uses a **gated, clipped linear update** to maintain a single hidden vector that summarizes all past tokens, and a direct linear readout to produce contextualized outputs.

In this paper, we formalize the HGRN2 recurrence, detail its implementation, and describe its integration as a drop‑in attention replacement within a compression‑aware Transformer framework.

---

## 2. Method

### 2.1 Notation

Let \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\) be a batch of sequences with length \(L\) and hidden dimension \(D\). For each time step \(t\), let \(\mathbf{x}_t = \mathbf{X}_{[:,t,:]} \in \mathbb{R}^{B \times D}\). The layer maintains a hidden state \(\mathbf{h}_t \in \mathbb{R}^{B \times D}\) initialized to zero: \(\mathbf{h}_0 = \mathbf{0}\).

### 2.2 Gating

At each step, two gating vectors are computed from the input via linear transformations followed by the sigmoid function \(\sigma(x) = 1/(1 + e^{-x})\):

\[
\mathbf{f}_t = \sigma(\mathbf{W}_f \mathbf{x}_t + \mathbf{b}_f), \qquad
\mathbf{i}_t = \sigma(\mathbf{W}_i \mathbf{x}_t + \mathbf{b}_i),
\]

where \(\mathbf{W}_f, \mathbf{W}_i \in \mathbb{R}^{D \times D}\) and \(\mathbf{b}_f, \mathbf{b}_i \in \mathbb{R}^D\) are learnable parameters. The **forget gate** \(\mathbf{f}_t\) controls how much of the previous state is retained, and the **input gate** \(\mathbf{i}_t\) scales the new information.

### 2.3 State Update and Clipping

The hidden state is updated as a convex combination of the previous state and the current input:

\[
\mathbf{h}_t = \mathbf{f}_t \odot \mathbf{h}_{t-1} + \mathbf{i}_t \odot \mathbf{x}_t,
\]

where \(\odot\) denotes element‑wise multiplication. Immediately after this update, the state is **clamped** to the interval \([-1, 1]\):

\[
\mathbf{h}_t \leftarrow \operatorname{clip}(\mathbf{h}_t,\; -1,\; 1).
\]

This clipping serves as a simple form of regularization, preventing the state’s magnitude from growing without bound over long sequences and stabilising training.

### 2.4 Output Readout

The output at time \(t\) is a linear projection of the clamped state:

\[
\mathbf{y}_t = \mathbf{W}_o \mathbf{h}_t + \mathbf{b}_o,
\]

with \(\mathbf{W}_o \in \mathbb{R}^{D \times D}\) and \(\mathbf{b}_o \in \mathbb{R}^D\). The final output of the layer is the sequence \(\mathbf{Y} = [\mathbf{y}_1, \mathbf{y}_2, \dots, \mathbf{y}_L] \in \mathbb{R}^{B \times L \times D}\).

### 2.5 Relation to Other Recurrences

The HGRN2 update can be seen as a gated version of the DeltaNet’s linear recurrence or as an LSTM without an output gate and with a single (cell) state. Unlike the original HGRN, which may involve hierarchical gating across multiple scales, this version uses only a single scale of gating and a hard clamp. The absence of a recurrent weight matrix for the state makes the update strictly input‑driven, while the gates provide adaptability.

---

## 3. Implementation

### 3.1 Module

The `HGRN2Layer` class is defined as a `nn.Module` and is registered under the key `"hgrn2_layer"` in the `ATTENTION_REGISTRY`. The constructor accepts `d_model` (the hidden size) and creates three linear layers:

- `self.forget_gate`: \(D \to D\)
- `self.input_gate`: \(D \to D\)
- `self.output`: \(D \to D\)

All layers include biases (default in PyTorch). The clipping threshold is stored as `self.res_clip = 1.0`.

### 3.2 Forward Pass

The `forward` method iterates over the time dimension using a Python loop. At each step:
1. Extract \(\mathbf{x}_t\).
2. Compute \(\mathbf{f}_t\) and \(\mathbf{i}_t\) via `torch.sigmoid` on the respective linear outputs.
3. Update the state: \(\mathbf{h} = \mathbf{f} * \mathbf{h} + \mathbf{i} * \mathbf{x}_t\).
4. Clamp the state: `h = torch.clamp(h, -self.res_clip, self.res_clip)`.
5. Compute output: `out_t = self.output(h)`.
6. Append to a list.
After the loop, the list is stacked into a tensor of shape \((B, L, D)\). The method returns a tuple `(out, None)` to conform to the framework’s attention interface (where the second element is usually attention weights, ignored here).

### 3.3 Injection Function

The function `inject_hgrn2_layer` is also registered under `"hgrn2_layer"`. It receives the layer, model configuration, compression configuration, and the original attention module. It creates a new `HGRN2Layer` using `model_config.hidden_size` and assigns it to `layer.attention`. No pretrained weights are transferred; the layer is trained from scratch.

---

## 4. Usage and Complexity

The module is invoked by setting the attention type to `"hgrn2_layer"` in the compression configuration. It requires no additional hyperparameters. The computational complexity is \(O(L D^2)\), dominated by the three linear transformations per token. Memory usage is \(O(D)\) per batch element, independent of sequence length.

---

## 5. Discussion

The HGRN2 Layer offers a very simple and efficient alternative to self‑attention. Its gated clipped state can learn to retain or forget information adaptively. The linear complexity makes it suitable for long sequences, and the single‑vector state is extremely memory efficient.

However, several limitations should be noted. The Python loop over time steps prevents training‑time parallelisation; a parallel scan or vectorised implementation would be required for practical speed on GPUs. The layer also **ignores any attention mask**, meaning padding tokens will contaminate the state—a production version must zero out or skip updates for padded positions. The clipping threshold is fixed; a learnable gain could improve flexibility. Finally, because only one state vector is maintained, the capacity is lower than multi‑head attention; extending to multiple independent state vectors (multi‑head HGRN2) could bridge this gap.

Despite these issues, the HGRN2 Layer provides a clean, pedagogically useful baseline for recurrent‑based sequence mixing and can be easily extended or combined with other efficient attention mechanisms.

---

## 6. Conclusion

We have described the HGRN2 Layer, a minimal gated linear recurrence that replaces self‑attention with a simple clipped‑state update. Its design is mathematically clean, implementation straightforward, and integration into the compression framework seamless. Future work will address the sequential bottleneck, masking, and capacity enhancements. The module stands as a testament to the power of simple gating in building efficient sequence models.