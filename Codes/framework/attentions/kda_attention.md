# KDA Attention: Kernelized Decay Attention with Learnable Gating

**Abstract**

We introduce **KDA Attention**, a lightweight linear attention module that replaces standard softmax self‑attention with a recurrent outer‑product memory augmented by a learnable per‑head decay gate. Keys are L2‑normalised before the outer product update, while a sigmoid‑gated scalar \(\beta_h\) controls the retention of past information independently per head. The memory state is updated as a decaying sum of normalized key‑value outer products, and the output at each time step is the matrix‑vector product of the current memory with the query. This yields a causal, linear‑time sequence mixer with a soft forgetting mechanism. The module is designed as a drop‑in replacement for the self‑attention layer in Transformer models and is integrated into the DistilBERT compression framework. In this paper, we detail the mathematical formulation, the rationale behind key normalisation and learned decay, and the implementation specifics.

---

## 1. Introduction

Standard multi‑head self‑attention in Transformers scales quadratically with sequence length, prompting the search for efficient alternatives. Linear attention methods (Katharopoulos et al., 2020) replace the softmax kernel with a feature map \(\phi\), rearranging computation to achieve linear complexity. The recurrent form of linear attention stores a matrix memory \(\mathbf{M}_t = \sum_{i=1}^t \phi(\mathbf{k}_i) \mathbf{v}_i^\top\). Without any forgetting, this memory grows indefinitely, potentially leading to saturation and loss of precision over very long sequences.

The **KDA Attention** module addresses this by incorporating a **learned decay factor** \(\beta_h\) per head, which controls the rate at which the memory discards old information. Concretely, the memory update is:

\[
\mathbf{H}_t = \sigma(\beta_h) \cdot \mathbf{H}_{t-1} + \mathbf{k}_t \mathbf{v}_t^\top,
\]

where \(\sigma\) is the sigmoid function, ensuring the decay lies in \((0,1)\). Additionally, the keys are **L2‑normalized** before the update. This normalisation stabilises the dynamics and makes the decay more interpretable, as the contribution of each new key‑value pair has a bounded norm.

The module also follows a **causal** pattern, updating the memory step by step and producing outputs sequentially, making it suitable for autoregressive tasks. The entire computation has \(O(L d^2)\) time complexity and \(O(d^2)\) memory, independent of the sequence length \(L\). KDA Attention can be seen as a simplified, gated variant of the recurrent linear attention used in models like Mamba or DeltaNet, but with a single scalar gate per head rather than a vector gate.

In this paper, we provide a rigorous mathematical description of the KDA recurrence, discuss the role of key normalisation and the decay gate, and present the implementation as a PyTorch module registered in the compression framework.

---

## 2. Method

### 2.1 Multi‑Head Projections

Given an input sequence \(\mathbf{X} \in \mathbb{R}^{B \times L \times d_{\text{model}}}\), the module first projects to queries, keys, and values for \(H\) heads, each of dimension \(d_h = d_{\text{model}} / H\):

\[
\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad
\mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad
\mathbf{V} = \mathbf{X} \mathbf{W}_V,
\]

with \(\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}\) (using `nn.Linear` with bias). The tensors are reshaped to \((B, H, L, d_h)\).

### 2.2 Key Normalisation

Before being used in the memory update, the keys are L2‑normalised along the head‑dimension:

\[
\hat{\mathbf{k}}_{t,h} = \frac{\mathbf{k}_{t,h}}{\|\mathbf{k}_{t,h}\|_2}.
\]

This ensures \(\|\hat{\mathbf{k}}_{t,h}\| = 1\). Normalisation prevents the outer product \(\hat{\mathbf{k}} \mathbf{v}^\top\) from growing arbitrarily large and makes the decay’s effect on the memory more uniform across different keys. It also removes scale differences that could destabilise the recurrent state.

### 2.3 Memory State and Decay

Each head \(h\) maintains a matrix memory \(\mathbf{H}_h \in \mathbb{R}^{d_h \times d_h}\), initialised to zeros. At each time step \(t\), the memory is updated using a **learnable per‑head decay** \(\beta_h \in \mathbb{R}\). The decay factor is obtained by applying the sigmoid function:

\[
\gamma_h = \sigma(\beta_h).
\]

Because \(\gamma_h \in (0,1)\), the update:

\[
\mathbf{H}_{t,h} = \gamma_h \cdot \mathbf{H}_{t-1,h} + \hat{\mathbf{k}}_{t,h} \mathbf{v}_{t,h}^\top
\]

describes a weighted moving average of the outer products of past key‑value pairs, where older pairs are exponentially suppressed. The parameter \(\beta_h\) is initialised so that \(\gamma_h \approx 0.73\) (since \(\sigma(1) \approx 0.73\)), giving a moderate initial forgetting rate. During training, \(\beta_h\) can adjust to retain more or less context.

The update is performed in batch: the code reshapes \(\gamma\) to \((1, H, 1, 1)\) to broadcast over the batch and the two matrix dimensions, and computes `h = decay * h + (kt.unsqueeze(-1) @ vt.unsqueeze(1))`.

### 2.4 Output Computation

At time step \(t\), the output for head \(h\) is the matrix‑vector product of the current memory with the query \(\mathbf{q}_{t,h}\):

\[
\mathbf{o}_{t,h} = \mathbf{H}_{t,h} \, \mathbf{q}_{t,h}.
\]

In the code, this is `(h @ qt.unsqueeze(-1)).squeeze(-1)`. Unfolding the recurrence, we have:

\[
\mathbf{o}_{t} = \sum_{i=1}^{t} \left( \prod_{j=i+1}^{t} \gamma \right) (\mathbf{q}_t^\top \hat{\mathbf{k}}_i) \mathbf{v}_i.
\]

Thus, the output is a weighted sum of past values, where the weight for a past token depends on the product of decay factors from that token to the present and the dot‑product similarity between the current query and that past normalised key. This is analogous to causal linear attention with an exponential decay factor and a normalized key feature map.

### 2.5 Output Projection and Head Combination

The per‑head outputs are stacked along the time dimension, resulting in a tensor of shape \((B, H, L, d_h)\). This is transposed to \((B, L, H, d_h)\), flattened to \((B, L, d_{\text{model}})\), and passed through a final linear layer \(\mathbf{W}_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}\):

\[
\mathbf{Y} = \mathbf{O}_{\text{flat}} \mathbf{W}_O^\top + \mathbf{b}_O.
\]

The module returns \((\mathbf{Y}, \text{None})\) to comply with the framework’s attention interface.

---

## 3. Implementation Details

### 3.1 Module Structure

- `KDAAttention` inherits from `nn.Module` and is registered under `"kda"`.
- **Constructor**:
  - `config`: provides `num_attention_heads`, `hidden_size` (as `dim`).
  - `rank`: ignored (for interface compatibility).
  - Creates `q_proj`, `k_proj`, `v_proj`, `out_proj` as `nn.Linear` layers with bias.
  - `self.beta`: a learnable parameter of shape \((H, 1)\), initialised to 1.0 (so initial \(\gamma \approx 0.73\)).
- **Forward**:
  - Input: `x` of shape \((B, L, D)\).
  - Projects to Q, K, V and reshapes to \((B, H, L, d)\).
  - Normalises K via `F.normalize(..., dim=-1)`.
  - Initialises memory `h` as zeros of shape \((B, H, d, d)\).
  - Iterates over the time dimension:
    - Extracts \(q_t, k_t, v_t\).
    - Computes `decay = sigmoid(beta)` reshaped to \((1, H, 1, 1)\).
    - Updates memory: `h = decay * h + outer(k_t, v_t)`.
    - Computes output: `out_t = squeeze(h @ q_t.unsqueeze(-1))`.
  - Stacks the outputs, reshapes, and projects.
  - Returns `(out, None)`.

### 3.2 Mask Handling

The forward method does **not** accept or use an attention mask. Padding tokens are not masked in the recurrence, which means they will contribute to the memory. This is a limitation; in a production setting, one should either pack sequences without padding or modify the update to skip padded positions by using a mask to zero out the key/value contribution.

### 3.3 Computational Complexity

Per time step, the outer product and matrix‑vector multiplication are \(O(d_h^2)\) per head. With \(H\) heads, total per‑step cost is \(O(H d_h^2) = O(d_{\text{model}} d_h)\). Over the sequence length, total cost is \(O(L d_{\text{model}} d_h) \approx O(L D^2)\). Memory for the state is \(O(B H d_h^2)\), independent of \(L\). The Python loop over time is not parallelised, which may slow training on GPUs; however, it is simple and functional.

---

## 4. Usage and Configuration

The module is injected by setting the attention type to `"kda"`. No additional configuration parameters are needed beyond the model config. Example:

```python
comp_config.attention_type = "kda"
```

Manual instantiation:
```python
attn = KDAAttention(config=model_config)
layer.attention = attn
```

The outer Transformer layer’s residual and norm remain in place (the block returns the output; the layer typically adds the residual: `x = norm(x + attn(x))`). This is compatible.

---

## 5. Discussion

KDA Attention provides an elegant combination of simplicity and expressiveness. The use of normalised keys and a learned scalar decay per head introduces a minimal number of additional parameters (just \(H\) extra scalars) while offering a meaningful inductive bias: the memory can learn to discard old information at a task‑optimal rate. The sigmoid ensures stability (no unbounded growth), and normalisation prevents the keys from dominating the memory update.

This module can be seen as a bridge between linear attention and gated state‑space models. Like linear attention, it stores a matrix memory and uses query‑key dot products for retrieval. But unlike the vanilla linear attention that sums keys without forgetting, KDA uses a gated decay that resembles the forget gate in an LSTM or the state‑space transition in Mamba (which also has a diagonal decay). The scalar per‑head decay is akin to a head‑specific time constant.

However, there are notable limitations. The scalar decay treats all dimensions of the memory equally; a per‑channel decay (vector) could offer finer control, as in HGRN‑Outer. The Python loop over time is inefficient for training; a parallel scan implementation would be necessary for long sequences. The lack of padding mask handling also needs to be addressed. Additionally, the initial decay of \(\gamma=0.73\) means that the model forgets about 27% of its memory at each step, which might be too aggressive for tasks requiring very long‑range memory; the parameter \(\beta\) can learn to adjust this.

Despite these limitations, KDA Attention is a clean and instructive example of how a simple gating mechanism can be incorporated into linear attention to control memory retention. It is well‑suited for moderate‑length sequences and resource‑constrained settings where full attention is too expensive.

---

## 6. Conclusion

We have presented KDA Attention, a recurrent linear attention module with L2‑normalised keys and a per‑head sigmoid decay gate. The model maintains a matrix memory that is updated as a weighted moving average of outer products and produces outputs via query‑memory multiplication. The design yields linear time and constant memory per step, making it a promising drop‑in replacement for standard self‑attention in Transformers. Integrated into the DistilBERT compression framework, KDA Attention offers a simple yet effective tool for efficient sequence modelling. Future work will focus on parallelising the recurrence, adding mask support, and exploring per‑channel decay for enhanced capacity.