# Granite Hybrid Block: Cyclic Integration of Mamba, Attention, and Mixture of Experts

**Abstract**

We introduce the **Granite Hybrid Block**, a modular Transformer block that cyclically alternates between two distinct sequence mixing layers—a state‑space model (Mamba‑2) and standard multi‑head self‑attention—and follows each mixing step with a token‑choice Mixture of Experts (MoE) feed‑forward network. The block employs a fixed schedule: for \(R\) consecutive steps, a Mamba‑2 layer processes the sequence with linear time complexity, and at every \((R+1)\)-th step, a full attention layer captures global interactions, thereby re‑sharpening long‑range dependencies. The MoE layer, applied after each mixing stage, enables conditional computation by activating only a small subset of expert modules per token. This design blends the efficiency of state‑space models, the global coherence of attention, and the capacity of conditional computation, making it suitable for long‑sequence and resource‑constrained scenarios. The module is implemented as a drop‑in replacement for the standard Transformer attention layer, and integrates with the DistilBERT compression framework via a registry‑based injection function. In this paper, we detail the architectural components, the underlying mathematics of the Mamba‑2 state‑space model and the top‑\(k\) gated MoE, and the cyclic scheduling mechanism that gives the block its hybrid nature.

---

## 1. Introduction

Modern Transformer models rely heavily on two core operations: token mixing (typically self‑attention) and channel mixing (feed‑forward networks). While attention excels at modeling long‑range dependencies, its quadratic complexity motivates alternative mixing layers. State‑space models (SSMs) like Mamba (Gu & Dao, 2023) offer linear‑time sequence processing and have shown competitive performance. Moreover, Mixture of Experts (MoE) layers increase model capacity without proportionally increasing computational cost by activating only a subset of parameters per token. The **Granite Hybrid Block** unifies these three paradigms—SSMs, attention, and MoE—into a single, cyclic architecture.

The key idea is to run the sequence mostly through a fast Mamba‑2 layer to handle local and medium‑range dependencies efficiently, while periodically injecting a global attention step to prevent the state‑space model from “forgetting” distant information. The MoE layer, which follows every mixing step, further enriches the representations with a large pool of experts. The ratio \(R\) (default 9) controls the frequency of attention: for every \(R\) Mamba steps, one attention step is performed. This cyclic design is inspired by the “Granite” series of models that combine various layer types, but here it is condensed into a single block with an internal counter, enabling a simple drop‑in replacement for any Transformer layer.

In this paper, we present the full mathematical description of the Mamba‑2 SSM, the top‑\(k\) MoE router and expert network, and the Granite Hybrid Block’s overall forward pass. We also describe the implementation details, including the stub Mamba‑2 layer, the MoE implementation, and the block’s internal state management.

---

## 2. Method

### 2.1 Mamba‑2 State‑Space Model

The Mamba‑2 layer implements a simplified version of the structured state‑space sequence model, inspired by the original Mamba architecture. Given an input sequence \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\), the layer first projects \(\mathbf{X}\) to an expanded inner dimension \(D_{\text{inner}} = E \cdot D\) (with expansion factor \(E = 2\) by default) and splits the result into two branches: a “sequence transformation” branch \(\mathbf{x}_{\text{ssm}}\) and a “gating” branch \(\mathbf{z}\):

\[
\mathbf{X}_{\text{proj}} = \mathbf{X} \mathbf{W}_{\text{in}} \in \mathbb{R}^{B \times L \times 2 D_{\text{inner}}},
\]
\[
\mathbf{X}_{\text{ssm}}, \mathbf{Z} = \operatorname{chunk}(\mathbf{X}_{\text{proj}}, 2, \text{dim}=-1),
\]

where \(\mathbf{W}_{\text{in}} \in \mathbb{R}^{D \times (2 D_{\text{inner}})}\). The sequence branch \(\mathbf{X}_{\text{ssm}}\) is then processed by a depthwise 1D convolution with kernel size \(d_{\text{conv}}=4\):

\[
\tilde{\mathbf{X}}_{\text{ssm}} = \text{Conv1D}(\mathbf{X}_{\text{ssm}}^\top) \in \mathbb{R}^{B \times D_{\text{inner}} \times L},
\]

and then truncated to length \(L\) and transposed back to \((B, L, D_{\text{inner}})\).

The core of the SSM is a diagonal recurrent state update. A time‑varying step size \(\Delta_t \in \mathbb{R}^{B \times D}\) is computed from the convolved signal:

\[
\Delta = \operatorname{softplus}(\tilde{\mathbf{X}}_{\text{ssm}} \mathbf{W}_\Delta + \mathbf{b}_\Delta),
\]

where \(\mathbf{W}_\Delta \in \mathbb{R}^{D_{\text{inner}} \times D}\) and \(\operatorname{softplus}(x) = \log(1+e^x)\). The state transition matrix \(\mathbf{A} \in \mathbb{R}^{D \times d_{\text{state}}}\) is parameterised in log space and then negated:

\[
\mathbf{A} = -\exp(\mathbf{A}_{\text{log}}), \quad \mathbf{A}_{\text{log}} \in \mathbb{R}^{D \times d_{\text{state}}}.
\]

Here \(d_{\text{state}}\) is a hyperparameter (default 16). For each time step \(t\), the hidden state \(\mathbf{h}_t \in \mathbb{R}^{B \times D \times d_{\text{state}}}\) is updated as:

\[
\mathbf{h}_t = \mathbf{h}_{t-1} \odot \exp(\Delta_t \otimes \mathbf{A}) + \mathbf{x}_t^{\text{(in)}} \otimes \Delta_t,
\]

where \(\mathbf{x}_t^{\text{(in)}}\) is taken as the first \(D\) channels of \(\tilde{\mathbf{X}}_{\text{ssm}}\) at time \(t\) (reshaped appropriately), and \(\otimes\) denotes element‑wise multiplication with broadcasting. The output at time \(t\) is then:

\[
\mathbf{y}_t = (\mathbf{h}_t \cdot \mathbf{D}) \in \mathbb{R}^{B \times D},
\]

where \(\mathbf{D} \in \mathbb{R}^D\) is a learnable vector. The final output of the SSM branch is multiplied element‑wise by the gating branch \(\mathbf{Z}\) after applying the SiLU activation:

\[
\mathbf{O}_{\text{Mamba}} = \mathbf{y} \odot \operatorname{SiLU}(\mathbf{Z}),
\]

and then linearly projected back to dimension \(D\) using \(\mathbf{W}_{\text{out}} \in \mathbb{R}^{D_{\text{inner}} \times D}\).

This recurrence is computed sequentially in the provided code, resulting in \(O(L \cdot D \cdot d_{\text{state}})\) time complexity, which is linear in \(L\) and far cheaper than attention for long sequences.

### 2.2 Mixture of Experts (MoE)

The MoE layer is a token‑choice top‑\(k\) gated mixture. Given an input \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\), a router computes logits:

\[
\mathbf{R} = \mathbf{X} \mathbf{W}_r \in \mathbb{R}^{B \times L \times E},
\]

where \(E\) is the number of experts. The top‑\(k\) logits are selected, and softmax normalisation is applied to these chosen logits:

\[
\mathcal{T} = \operatorname{TopK}(\mathbf{R}_{b,\ell}, k),
\]
\[
g_{e} = \frac{\exp(R_{b,\ell,e})}{\sum_{j \in \mathcal{T}} \exp(R_{b,\ell,j})}, \quad e \in \mathcal{T}.
\]

Each expert is a feed‑forward network with one hidden layer and GELU activation:

\[
E_e(\mathbf{x}) = \mathbf{W}_2^{(e)} \operatorname{GELU}(\mathbf{W}_1^{(e)} \mathbf{x}),
\]

where \(\mathbf{W}_1^{(e)} \in \mathbb{R}^{D_{\text{ff}} \times D}\) and \(\mathbf{W}_2^{(e)} \in \mathbb{R}^{D \times D_{\text{ff}}}\), with \(D_{\text{ff}} = 4D\) (the expansion factor). The final output for a token is the weighted sum of the selected expert outputs:

\[
\mathbf{y} = \sum_{e \in \mathcal{T}} g_e \, E_e(\mathbf{x}).
\]

The implementation in the provided code iterates over experts, applies a mask, and accumulates the weighted contributions. While this is not the most efficient batched implementation, it is functionally correct.

### 2.3 Granite Hybrid Block

The Granite Hybrid Block integrates the two mixing layers and the MoE layer into a single module with a cyclic schedule. The block maintains an internal counter `counter` that increments on every forward call. The forward pass is defined as:

1. **Mixing step**:  
   - If `counter % (ratio + 1) < ratio`, apply the Mamba‑2 layer: \(\mathbf{X} \gets \text{Mamba2}(\mathbf{X})\).  
   - Otherwise, apply multi‑head attention: \(\mathbf{X} \gets \text{MultiheadAttention}(\mathbf{X}, \mathbf{X}, \mathbf{X})\).

2. **MoE step**:  
   - \(\mathbf{X} \gets \mathbf{X} + \text{MoE}(\mathbf{X})\).

The module does not use residual connections internally (the addition of the MoE output to \(\mathbf{X}\) implies that the mixing step’s output is directly fed to the MoE, and the residual around the mixing layer is missing; however, the block is meant to be placed in a Transformer layer that already provides residual connections and layer normalization, so the block itself can be seen as a “sub‑layer” within an outer residual wrapper). In practice, the injection function replaces the standard attention module with this block; the outer layer’s residual connection and LayerNorm (as in DistilBERT) still apply before and after this block.

The ratio \(R\) (default 9) controls the fraction of steps that use attention. A higher ratio saves compute; setting ratio to 0 would mean every step is attention. The design ensures that most of the time the sequence is processed by the efficient Mamba layer, but periodically the attention “rectifies” the representation with a global view.

The block uses standard `nn.MultiheadAttention` with `batch_first=True` and 8 heads (hard‑coded in the code, though it could be made configurable). The Mamba‑2 stub is a simplified implementation; in a production setting, the official `mamba_ssm` package would be preferred for speed and correctness.

---

## 3. Implementation Details

### 3.1 Mamba2 Stub

The `Mamba2` class implements a simplified SSM. Key points:
- **`dt_proj`**: projects the inner dimension to \(D\) to produce \(\Delta_t\).
- **`A_log`**: parameter of shape \((D, d_{\text{state}})\); exponentiated and negated to obtain \(\mathbf{A}\).
- **`D`**: a learnable vector of size \(D\).
- **Forward**: after convolution and projection, iterates over time, updating the hidden state \(h\) and accumulating output. Only the first \(D\) channels of \(\mathbf{X}_{\text{ssm}}\) are used as input to the state, which may be a bug or intentional design. The gating branch \(\mathbf{Z}\) modulates the output. The final linear projection maps back to \(D\).

### 3.2 MixtureOfExperts

The `MixtureOfExperts` class implements token‑choice top‑\(k\) MoE. It instantiates `num_experts` expert networks (each a two‑layer MLP with GELU) and a router linear layer. The forward uses a loop over experts; for each expert, it identifies tokens that selected it in the top‑\(k\) (via the `indices` tensor) and computes a weighted sum of expert outputs. The mask and weight handling ensure that tokens not selecting an expert are unaffected.

### 3.3 GraniteHybridBlock

- **Constructor**: takes `dim`, `num_experts`, `top_k`, `ratio`. Creates a `Mamba2` instance, an `nn.MultiheadAttention` (8 heads), a `MixtureOfExperts` with hidden expansion `4*dim`, and a buffer `counter` initialised to 0. The attention’s number of heads is hard‑coded (8), which assumes `dim` is divisible by 8; this should be updated to match the model configuration.
- **Forward**: increments counter, applies the cyclic mixing, then MoE, returns the output. There is no explicit residual; as noted, this block is intended to be embedded in a standard Transformer layer that already has residual connections.

### 3.4 Injection

The injection function `inject_granite_hybrid` is registered under `"granite_hybrid"`. It reads `num_experts`, `top_k`, and `ratio` from `comp_config`, constructs a `GraniteHybridBlock`, and assigns it to `layer.attention`. This makes the block drop‑in compatible with the framework.

---

## 4. Usage and Configuration

The module can be injected by setting the attention type to `"granite_hybrid"` and providing the following parameters via the compression configuration:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_experts` | 8 | Number of experts in the MoE layer. |
| `top_k` | 2 | Number of experts activated per token in MoE. |
| `ratio` | 9 | Attention frequency: 1 attention step every (ratio+1) forward calls. |

Example:
```python
comp_config.num_experts = 16
comp_config.top_k = 2
comp_config.ratio = 9
```

The block is stateful (counter), so it must be used in a setting where layers are traversed sequentially, and each instance tracks its own step count. This works correctly when a single `GraniteHybridBlock` is placed as the attention module for one Transformer layer; the counter increments each time that layer is called. In a multi‑layer model, each layer has its own counter, which may be desirable (different layers may be at different phases of the cycle). If a global cycle across layers is desired, a different mechanism would be needed.

---

## 5. Discussion

The Granite Hybrid Block is a powerful and flexible architecture that combines three strong paradigms. The periodic attention steps ensure that the model does not lose long‑range coherence, while the majority of processing steps remain cheap. The MoE layer adds immense capacity with minimal flops per token. This makes the block well‑suited for scaling to very large models (many experts, deep layers) while keeping inference cost manageable.

The provided implementation, however, has some limitations. The Mamba‑2 stub is a pedagogical implementation; its sequential loop is slow and lacks the parallel scan algorithm that makes Mamba fast on GPUs. The MoE uses a python loop over experts, which is acceptable for small numbers of experts but not for hundreds. The hard‑coded number of attention heads (8) limits flexibility. Moreover, the block’s lack of internal residual connections may hinder training if the outer layer does not apply the standard residual‑and‑norm. Typically, a DistilBERT layer applies a residual connection around the attention output and another around the FFN; replacing the attention module with this block means that the block’s output goes through the outer residual and LayerNorm. However, since the block already includes a MoE (which functions similarly to the original FFN), there might be a double counting of MoE‑like transformations, potentially degrading performance. A more refined integration would place the MoE after the attention/Mamba output within the block, and rely on the outer residual only once.

Despite these issues, the module exemplifies the concept of cyclic hybridisation and can be refined with efficient Mamba kernels and batched MoE routing.

---

## 6. Conclusion

We have presented the Granite Hybrid Block, a cyclic combination of Mamba‑2, attention, and Mixture of Experts. The block processes tokens mostly with an efficient state‑space model, interspersed with full attention for global rectification, and enriches representations via conditional computation. Its drop‑in design facilitates experimentation with hybrid architectures in the DistilBERT compression framework. Future work will replace the stub components with highly optimised implementations and refine the integration to fully exploit the benefits of each paradigm.