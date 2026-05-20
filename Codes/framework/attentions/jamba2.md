# Jamba2 Block: Hybrid Sequence Mixing with Mamba, Attention, and Mixture of Experts

**Abstract**

We present the **Jamba2 Block**, a hybrid Transformer sub‑layer that interleaves three distinct sequence mixing and transformation stages: a state‑space model (Mamba), a standard multi‑head self‑attention, and a token‑choice Mixture of Experts (MoE) feed‑forward network. Each stage is wrapped with a residual connection and layer normalisation, following the pre‑norm Transformer convention. The block processes the input sequentially—first through Mamba, then through attention, and finally through MoE—each time adding the input of that stage to the output. This design combines the linear‑time recurrent processing of Mamba with the global modelling of attention and the conditional capacity of MoE, all within a single drop‑in replacement for the standard attention layer in DistilBERT. The module is registered in the compression framework and includes an injection function (`"jamba2"`). This paper provides a rigorous mathematical description of each component, the block’s architecture, and its implementation.

---

## 1. Introduction

Transformer architectures have evolved to incorporate diverse sequence mixing mechanisms beyond standard self‑attention. State‑space models (SSMs) like Mamba (Gu & Dao, 2023) offer linear‑time sequence processing with competitive language modelling performance. Mixture of Experts (MoE) layers add capacity without proportional computational cost by activating only a subset of parameters per token. The **Jamba2 Block** integrates these three powerful paradigms—**Mamba**, **multi‑head attention**, and **MoE**—into a single modular block that replaces the standard attention sub‑layer. By stacking Mamba, attention, and MoE sequentially with residual connections, the block aims to combine the efficiency of SSMs, the global pairwise interaction of attention, and the conditional computation of MoE.

The name “Jamba” likely derives from a hybrid of “JAM” (Just Another Model?) and “Mamba”. The “2” indicates an improved or alternate version. In this implementation, the block is self‑contained: it does not rely on an external FFN; the MoE acts as the feed‑forward expansion. The block is designed to be inserted into a DistilBERT layer, where it will process the hidden states and return a transformed sequence.

In this paper, we detail the mathematics of each component (Mamba, attention, MoE) and their composition into the Jamba2 Block. We also discuss the implementation specifics, including the injection function for the framework registry.

---

## 2. Method

The Jamba2 Block receives a hidden state \(\mathbf{X} \in \mathbb{R}^{B \times L \times d_{\text{model}}}\) and produces an output of the same shape. It applies three sequential sub‑layers:

1. **Mamba sub‑layer**: \(\mathbf{X}_1 = \text{LayerNorm}(\mathbf{X} + \text{Mamba}(\mathbf{X}))\)
2. **Attention sub‑layer**: \(\mathbf{X}_2 = \text{LayerNorm}(\mathbf{X}_1 + \text{MultiHeadAttn}(\mathbf{X}_1, \mathbf{X}_1, \mathbf{X}_1))\)
3. **MoE sub‑layer**: \(\mathbf{X}_3 = \text{LayerNorm}(\mathbf{X}_2 + \text{MoE}(\mathbf{X}_2))\)

The final output is \(\mathbf{X}_3\) (returned as a tuple with `None` for compatibility). We now describe each component.

### 2.1 Mamba State‑Space Model

The `Mamba` module (imported from `.mamba`) is a structured state‑space sequence model. While its full implementation is not given in this snippet, a typical Mamba block (simplified) operates as follows. Given an input \(\mathbf{U} \in \mathbb{R}^{B \times L \times D}\), the module projects it into a higher inner dimension (expansion factor 2), applies a 1D depthwise convolution, computes a time‑varying state transition matrix \(\mathbf{A}\) and step size \(\Delta\), and performs a recurrent scan along the sequence using a diagonal state space. The output is then gated with a SiLU activation and projected back to \(D\). The overall complexity is \(O(L D^2)\), linear in \(L\). Mamba captures long‑range dependencies efficiently without pairwise attention.

### 2.2 Multi‑Head Self‑Attention

The standard multi‑head attention (MHA) with `batch_first=True` computes:

\[
\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_h}}\right) \mathbf{V},
\]

where \(\mathbf{Q}, \mathbf{K}, \mathbf{V}\) are linear projections of the input. Here, self‑attention is used (\(\mathbf{Q}=\mathbf{K}=\mathbf{V}=\mathbf{X}_1\)). The quadratic cost \(O(L^2 d_{\text{model}})\) provides a global view of the sequence, complementing the Mamba’s linear recurrence which may focus more on recent context. The attention sub‑layer does **not** use a causal mask (it is bidirectional), making it suitable for encoder tasks. Dropout is included inside the `nn.MultiheadAttention` module if configured.

### 2.3 Top‑2 Token‑Choice Mixture of Experts (MoE)

The `ExpertChoiceMoE` module implements a classic **token‑choice top‑\(k\) MoE** with \(k=2\) (configurable). Given an input \(\mathbf{X}\), a linear router computes logits \(\mathbf{R} \in \mathbb{R}^{B \times L \times E}\), where \(E\) is the number of experts. For each token, the top‑\(k\) largest logits are selected:

\[
\mathcal{T} = \operatorname{TopK}(\mathbf{R}, k).
\]

Softmax is applied to the selected logits to obtain weights \(g_i\). Each expert \(e\) is a two‑layer MLP with GELU and an expansion factor 4:

\[
E_e(\mathbf{x}) = \mathbf{W}_2^{(e)} \operatorname{GELU}(\mathbf{W}_1^{(e)} \mathbf{x}).
\]

The output for a token is the weighted sum of the selected experts’ outputs:

\[
\mathbf{y} = \sum_{e \in \mathcal{T}} g_e \cdot E_e(\mathbf{x}).
\]

Tokens that do not select an expert are not directly processed by that expert. The implementation loops over experts, computes a mask for tokens that selected that expert, and accumulates the weighted outputs. This enables sparse activation: only a fraction \(k/E\) of the total FFN parameters are used per token, greatly increasing capacity with minimal additional flops.

### 2.4 Overall Block

By stacking Mamba, attention, and MoE, the Jamba2 block creates a multi‑stage pipeline. The Mamba sub‑layer provides an efficient initial mixing, the attention sub‑layer then captures pairwise global interactions, and the MoE sub‑layer enhances the representation with sparse conditional computation. Each sub‑layer is equipped with its own residual connection and layer normalisation, which is crucial for training stability. The block does **not** rely on an external feed‑forward network (as in a typical Transformer), because the MoE effectively serves that role.

---

## 3. Implementation Details

### 3.1 Jamba2Block Class

- **Constructor**:
  - `d_model`: hidden dimension.
  - `n_heads`: number of attention heads.
  - `n_experts`: number of experts in MoE (default 8).
  - `top_k`: top‑\(k\) for MoE (default 2).
  - `**kwargs`: extra arguments (ignored).
  - Creates:
    - `self.mamba = Mamba(d_model)` – SSM module.
    - `self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)` – MHA.
    - `self.moe = ExpertChoiceMoE(d_model, n_experts, top_k)` – MoE.
    - `self.norm1`, `self.norm2`, `self.norm3`: three `nn.LayerNorm` layers (no element‑wise affine? By default `LayerNorm` has `elementwise_affine=True`).
- **Forward**:
  1. `x = self.norm1(x + self.mamba(x))`
  2. `attn_out, _ = self.attn(x, x, x)`
  3. `x = self.norm2(x + attn_out)`
  4. `x = self.norm3(x + self.moe(x))`
  5. Return `(x, None)`.

### 3.2 ExpertChoiceMoE

- Constructor:
  - `d_model`, `n_experts`, `top_k`.
  - `self.experts`: `nn.ModuleList` of `nn.Sequential(Linear→GELU→Linear)` each with expansion 4.
  - `self.router`: `nn.Linear(d_model, n_experts)`.
- Forward:
  - Compute router logits.
  - Get top‑\(k\) weights and indices.
  - For each expert, compute mask of tokens that selected it, and accumulate weighted outputs.

### 3.3 Integration

The class is registered under `"jamba2"` in the `ATTENTION_REGISTRY`. The framework will call the class constructor with parameters from the model config and compression config. No dedicated injection function is shown; the class itself serves as the attention replacement when instantiated by the framework’s injection logic (which typically passes `model_config` and other parameters). In DistilBERT, this block replaces the entire attention module; the outer residual and LayerNorm around the attention (in the original layer) are handled by the block internally, so the outer layer should **not** apply another residual. In practice, the injection should probably replace the whole attention sub‑layer with this block, and the outer layer’s own residual and norm should be removed or adapted to avoid double residuals. The current design (triple residual inside the block) assumes that the block is placed at the point where the attention sub‑layer would be, and the original layer’s “attention + norm + residual” is replaced entirely. This is consistent with the pattern used in other blocks like GraniteHybridBlock.

### 3.4 Mamba Module

The `Mamba` class is imported from `.mamba`. We assume it implements a simplified Mamba block similar to the `Mamba2` placeholder seen in earlier modules, but likely with a parallel scan for efficiency. The exact details are omitted but its interface is `forward(x) -> x`.

---

## 4. Usage and Configuration

The Jamba2 block is activated by setting the attention type to `"jamba2"` in the compression configuration. Parameters can be passed via the config or directly to the constructor. Expected configuration keys:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_experts` | 8 | Number of MoE experts. |
| `top_k` | 2 | Top‑\(k\) experts per token. |
| `d_model` | (from model config) | Hidden size. |
| `n_heads` | (from model config) | Number of attention heads. |

Example injection via framework:
```python
comp_config.attention_type = "jamba2"
comp_config.n_experts = 16
comp_config.top_k = 2
```

Manual instantiation:
```python
block = Jamba2Block(d_model=768, n_heads=12, n_experts=16, top_k=2)
layer.attention = block
```

---

## 5. Discussion

The Jamba2 block leverages the complementary strengths of its three components. Mamba provides a fast, global context summarisation via state‑space dynamics; attention refines the representation with exact pairwise interactions; MoE injects high‑capacity non‑linear transformations sparsely. The three residual norms help trainability of this deep‑within‑a‑layer structure.

However, there are several considerations. The block contains **three** sub‑layers internally, making it significantly deeper and heavier than a standard attention layer. This could lead to overfitting or training instability if not carefully regularised. The MoE, while sparse, adds a large number of parameters (8 experts × (4× expansion) = ~8× more FFN parameters than a single FFN). The Mamba module may have its own set of parameters. The total parameter count per block is much higher than vanilla attention, which could be an issue for memory but not for FLOPs if sparsity is exploited.

The block does **not** apply any attention mask (padding mask) in the MHA call; `nn.MultiheadAttention` accepts an optional `key_padding_mask` or `attn_mask` but it is not passed. Thus, padding tokens would be attended to, which is problematic for batch processing with variable lengths. The Mamba and MoE also do not mask padding. A proper implementation should pass the padding mask through all stages.

Another nuance is the sequential stacking: Mamba → Attention → MoE. This order is fixed. It might be beneficial to learn routing weights or to permute the order depending on the task. Additionally, the block’s output is `(x, None)`, which is consistent with the interface. The block could be extended to return attention weights from the attention sub‑layer if needed.

---

## 6. Conclusion

We have introduced the Jamba2 Block, a hybrid Transformer sub‑layer combining Mamba, multi‑head attention, and top‑2 Mixture of Experts. The sequential residual design integrates efficient state‑space processing, global attention, and sparse conditional computation into a single compact module. The block is integrated into the DistilBERT compression framework and provides a powerful alternative to standard self‑attention for tasks requiring both efficiency and capacity. Future work will address masking, exploration of sub‑layer order, and dynamic expert routing.