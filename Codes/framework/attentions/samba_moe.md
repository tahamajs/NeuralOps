# Samba‑MoE: Hybrid Attention‑State‑Space Block with Mixture of Experts

**Abstract**

We introduce **Samba‑MoE**, a multi‑pathway sequence mixing block that integrates three complementary mechanisms: (1) a **sliding window multi‑head self‑attention** for fine‑grained local context, (2) a **spectral state‑space model (Mamba)** for efficient global sequence processing, and (3) a **token‑choice top‑2 mixture of experts (MoE)** for high‑capacity, conditional feed‑forward transformations. The attention and state‑space branches are fused within a pre‑normalised residual block (the SAMBA‑M3 architecture), whose output is then passed through the MoE layer with its own residual connection and layer normalisation. The MoE activates only a subset of expert networks per token, increasing model capacity without proportionally increasing the computational cost. The block serves as a drop‑in replacement for the self‑attention sub‑layer in DistilBERT, offering a rich combination of local precision, global context, and sparse expert capacity. This paper provides a detailed mathematical description of each component, the overall architecture, and the integration into the compression framework.

---

## 1. Introduction

Transformer architectures can benefit from hybrid designs that combine different sequence mixing strategies. The **Samba‑MoE** block combines three powerful paradigms:

- A **sliding window attention** captures short‑range dependencies with exact dot‑product attention.
- A **state‑space model (Mamba)** with learnable spectral gating processes the entire sequence in linear time, providing global context.
- A **mixture of experts (MoE)** applies a sparse, token‑conditional feed‑forward network, greatly expanding the model’s capacity while keeping the average per‑token computation limited.

The first two components are fused in the **SAMBA‑M3** block, which employs sequential residual connections with layer norms. The output of this hybrid block is then refined by the MoE layer, which uses a top‑2 gating mechanism to select two experts per token from a pool of (typically 8) expert MLPs. The MoE output is added via a residual connection, and a final layer norm is applied. This modular design allows each component to specialise: the attention captures local syntax, the Mamba captures long‑range dependencies, and the MoE provides task‑specific non‑linear transformations.

The block is implemented as a drop‑in replacement for the self‑attention layer in DistilBERT and is registered in the compression framework under the key `"samba_moe"`. In this paper, we detail the mathematical formulation of the REAPMoE and the overall Samba‑MoE architecture, describe the implementation, and discuss its advantages and limitations.

---

## 2. Method

### 2.1 Sliding Window Attention + Spectral Mamba (SAMBA‑M3)

The first part of the block is the **SAMBA‑M3** module, described in a companion paper. It takes an input \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\) and computes:

\[
\mathbf{X}_{\text{attn}} = \operatorname{MultiHeadAttn}(\mathbf{X}, \mathbf{X}, \mathbf{X}, \text{mask}=\mathbf{M}_{\text{causal, window}}),
\]
\[
\mathbf{X}_1 = \operatorname{LayerNorm}(\mathbf{X} + \mathbf{X}_{\text{attn}}),
\]
\[
\mathbf{X}_{\text{mamba}} = \operatorname{SpectralMamba}(\mathbf{X}_1),
\]
\[
\mathbf{X}_2 = \operatorname{LayerNorm}(\mathbf{X}_1 + \mathbf{X}_{\text{mamba}}).
\]

The attention uses a causal sliding window of size \(W\) (default 128), where each token attends only to the previous \(W\) tokens. The SpectralMamba applies a state‑space model (Mamba) whose output is scaled by a per‑channel sigmoid gate \(\sigma(\mathbf{s})\) with learnable parameter \(\mathbf{s} \in \mathbb{R}^D\). This yields the intermediate representation \(\mathbf{X}_2 \in \mathbb{R}^{B \times L \times D}\).

### 2.2 REAPMoE: Token‑Choice Top‑2 Mixture of Experts

The **REAPMoE** (Routed Expert Activation with top‑2 gating) module enhances the representation with a sparse mixture of experts. Given an input \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\) (which will be \(\mathbf{X}_2\) from the previous stage), a router linear layer computes logits over \(E\) experts:

\[
\mathbf{R} = \mathbf{X} \mathbf{W}_{\text{router}} + \mathbf{b}_{\text{router}} \quad \in \mathbb{R}^{B \times L \times E}.
\]

For each token position, the top‑\(k\) logits are selected (with \(k=2\)):

\[
\mathcal{T}_{b,l} = \operatorname{TopK}(\mathbf{R}_{b,l}, k), \qquad
g_{i} = \frac{\exp(R_{b,l,i})}{\sum_{j \in \mathcal{T}_{b,l}} \exp(R_{b,l,j})}, \quad i \in \mathcal{T}_{b,l}.
\]

Each expert \(e\) is a two‑layer MLP with a GELU activation and an expansion factor of 4:

\[
E_e(\mathbf{x}) = \mathbf{W}_2^{(e)} \operatorname{GELU}(\mathbf{W}_1^{(e)} \mathbf{x} + \mathbf{b}_1^{(e)}) + \mathbf{b}_2^{(e)}.
\]

The output for token \((b,l)\) is the weighted sum of the selected experts’ outputs:

\[
\mathbf{Y}_{b,l} = \sum_{e \in \mathcal{T}_{b,l}} g_e \cdot E_e(\mathbf{x}_{b,l}).
\]

The implementation iterates over experts, applying a mask for tokens that selected that expert, and accumulates the weighted outputs. This restricts each token to using only \(k\) of the \(E\) experts, resulting in a conditional computation pattern that increases total model capacity while keeping the average FLOPs per token roughly that of a single FFN.

### 2.3 Samba‑MoE: Full Block

The `SambaMoEBlock` combines the SAMBA‑M3 hybrid and the REAPMoE. Given an input \(\mathbf{X}\),

\[
\tilde{\mathbf{X}} = \operatorname{SAMBAM3Block}(\mathbf{X}),
\]
\[
\mathbf{X}_{\text{moe}} = \operatorname{REAPMoE}(\tilde{\mathbf{X}}),
\]
\[
\mathbf{Y} = \operatorname{LayerNorm}(\tilde{\mathbf{X}} + \mathbf{X}_{\text{moe}}).
\]

Thus, the block has two residual sub‑layers inside SAMBA‑M3 (attention + Mamba) and a third residual sub‑layer for the MoE. Each sub‑layer has its own layer normalisation. The final output \(\mathbf{Y}\) is returned directly (not as a tuple), which may differ from the framework’s expected `(output, None)` return. This should be noted for compatibility.

### 2.4 Complexity

The attention branch costs \(O(L W d)\). The spectral Mamba, if implemented as a true SSM, costs \(O(L D^2)\), but with the placeholder it is just an MLP of cost \(O(L D^2)\) as well. The MoE costs \(O(L \cdot k \cdot E_{\text{ff}})\) where \(E_{\text{ff}}\) is the FFN cost per token per expert; with top‑2, it is roughly twice the cost of a single FFN per token. Overall, the block remains linear in \(L\) for the non‑attention parts.

---

## 3. Implementation Details

- **`REAPMoE`** (registered as `"reap_moe"`): 
  - `d_model`, `n_experts` (default 8), `top_k` (default 2).
  - `self.experts`: `nn.ModuleList` of `nn.Sequential(Linear→GELU→Linear)`.
  - `self.router`: Linear layer from `d_model` to `n_experts`.
  - Forward: as described, loops over experts.

- **`SambaMoEBlock`** (registered as `"samba_moe"`):
  - `d_model`, `n_heads`, `n_experts` (default 8), `window` (default 128).
  - `self.swa_mamba`: `SAMBAM3Block(d_model, n_heads, window)` – the hybrid block.
  - `self.moe`: `REAPMoE(d_model, n_experts)` – the MoE layer.
  - `self.norm`: `nn.LayerNorm(d_model)` – norm after MoE residual.
  - Forward: `x = self.swa_mamba(x)`, then `x = self.norm(x + self.moe(x))`, return `x`.

- **Injection**: The block is registered directly as a class; the framework can instantiate it using the model config and additional parameters. No separate injection function is shown, but the class decorator registers it.

- **Placeholder Mamba**: The `SAMBAM3Block` uses the `Mamba` placeholder (an MLP), not a true SSM. This will limit the global processing capability until a real Mamba kernel is substituted.

- **Output format**: The block returns a single tensor, which might conflict with the expected `(output, None)` in some framework pipelines. A wrapper or modification to return a tuple may be necessary.

- **Mask handling**: No padding mask is passed to the attention or Mamba/MoE, meaning padded sequences will not be handled correctly. This is a critical limitation for variable‑length inputs.

---

## 4. Usage and Configuration

The block is activated by setting the attention type to `"samba_moe"`. The configuration parameters can be passed via `comp_config` or constructor:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_experts` | 8 | Number of experts in the MoE. |
| `window` | 128 | Window size for the sliding attention. |
| `n_heads` | from model config | Number of attention heads. |
| `d_model` | from model config | Hidden size. |

Example injection via framework:
```python
comp_config.attention_type = "samba_moe"
comp_config.n_experts = 16
comp_config.window = 256
```

The block will be created and assigned to `layer.attention`. Because no pretrained weight transfer is implemented, the model must be fine‑tuned from scratch or the weights of the attention projections copied separately (not done in this snippet).

---

## 5. Discussion

Samba‑MoE is a feature‑rich hybrid that pushes the envelope of what can be packed into a single Transformer sub‑layer. The combination of local attention, state‑space modelling, and expert mixing addresses several dimensions of efficiency and capacity:

- The sliding window ensures low‑cost, high‑precision local interactions.
- The spectral Mamba (when fully implemented) brings global context at linear cost.
- The MoE adds a large amount of trainable parameters but only activates a fraction per token, offering a high capacity ceiling without blowing up inference FLOPs.

The sequential residual design (pre‑norm) follows established practices for training stability. The per‑channel spectral scaling of the Mamba output adds a fine‑grained control over the SSM’s contribution.

However, several issues remain:

1. **Placeholder Mamba**: The current Mamba is a stub; the block’s global processing is essentially a gated MLP, which is unlikely to match a real SSM or full attention. Swapping in the official Mamba implementation is essential for performance.
2. **Output tuple**: Many attention modules in the framework return `(output, None)`. This block returns just `output`, which could break training scripts that expect a two‑element tuple. It should be wrapped to return `(output, None)`.
3. **Padding masks**: Neither the attention, Mamba, nor MoE receive an attention mask. Padding tokens will be treated as real tokens, corrupting the context. Adding mask support is critical for practical use.
4. **No weight transfer**: The block does not initialise its attention projections from a pretrained model. For zero‑shot behaviour, an injection function should copy the original `q_lin`, `k_lin`, `v_lin`, `out_lin` weights into the `MultiheadAttention` module inside SAMBA‑M3. The MoE and Mamba parameters are newly initialised.

Despite these limitations, the Samba‑MoE block demonstrates an ambitious integration of multiple advanced techniques into a single cohesive module. With the fixes above, it could become a powerful tool for scaling Transformers to long sequences with high capacity.

---

## 6. Conclusion

We have presented Samba‑MoE, a hybrid Transformer block that integrates sliding window attention, a spectral state‑space model, and a top‑2 mixture of experts. The block leverages local precision, global context, and conditional computation to offer a rich representational capacity while maintaining linear complexity. The design is registered in the compression framework as a drop‑in replacement for DistilBERT’s self‑attention. Future work will address the placeholder Mamba, output format, mask support, and pretrained weight initialisation to bring the module to a production‑ready state. Samba‑MoE exemplifies the power of combining complementary efficient attention paradigms.