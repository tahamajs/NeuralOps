# SAMBA‑M3: Hybrid Sliding Window Attention and Spectral Mamba Block

**Abstract**

We introduce **SAMBA‑M3**, a hybrid sequence mixing block that fuses a **causal sliding window multi‑head self‑attention** with a **state‑space model (Mamba) enhanced by learnable spectral scaling**. The attention branch captures fine‑grained local dependencies within a fixed window, while the Mamba branch provides efficient global sequence processing via a state‑space recurrence. The output of the Mamba module is gated by a per‑channel sigmoid scaling that adaptively modulates its contribution. Both branches are wrapped with residual connections and layer normalisation, forming a drop‑in replacement for the self‑attention sub‑layer in DistilBERT. The block integrates seamlessly into the compression framework via the `ATTENTION_REGISTRY`. This paper provides a rigorous mathematical description of the sliding window attention mask, the spectral Mamba augmentation, and the overall hybrid architecture, along with implementation details.

---

## 1. Introduction

Transformer self‑attention provides powerful sequence modelling but suffers from quadratic complexity. Efficient variants often combine a local attention window with a global linear‑time mechanism. State‑space models (SSMs), particularly the Mamba architecture, have emerged as strong alternatives to attention, offering linear complexity and competitive language modelling performance. The **SAMBA‑M3** block (Sliding window Attention + **S**pectral **M**amba **3**? maybe “M3” stands for “Mamba with 3 components” or just a version) unifies these two paradigms: a standard multi‑head attention restricted to a causal sliding window, and a Mamba‑based sequence mixer whose output is scaled by a learned per‑channel gate. The gate, parameterised by a single vector with sigmoid activation, allows the network to selectively emphasise or suppress the Mamba output, providing a lightweight form of spectral conditioning.

The name “SAMBA” likely draws from the popular Brazilian dance, evoking the rhythmic interplay between attention and SSM. The block is designed as a drop‑in replacement for the self‑attention layer in DistilBERT, with each branch followed by its own residual and layer norm, akin to the “pre‑norm” Transformer architecture. This modular design allows the block to be injected into any layer of the model while preserving the outer residual connection of the original Transformer layer? Actually, the block has its own internal norms and residuals, so when placed inside a Transformer layer that already has an outer residual, there might be double residuals. However, the typical injection replaces the whole attention sub‑layer with this block, and the outer layer’s residual is still applied. The block’s internal residuals thus create a deeper pathway. This is a common pattern in many of the custom blocks we’ve seen.

In this paper, we detail the mathematics of the sliding window mask, the spectral Mamba module, and the overall block structure. We also discuss the implementation of the Mamba placeholder used herein.

---

## 2. Method

### 2.1 Sliding Window Attention

The attention branch operates with a **causal sliding window**. Given an input \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\), we compute queries, keys, and values using standard linear projections (inside `nn.MultiheadAttention`). Let the window size be \(W\) (default 128). A mask \(\mathbf{M} \in \mathbb{R}^{1 \times 1 \times L \times L}\) is constructed as follows:

For each position \(i\) and \(j\),

\[
M_{ij} = \begin{cases}
0 & \text{if } j \le i \text{ and } i - j < W, \\
-\infty & \text{otherwise}.
\end{cases}
\]

This mask ensures that token \(i\) can attend only to previous tokens (causal) and only within a distance of \(W\). The scores are computed as \(\mathbf{S} = \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_h}} + \mathbf{M}\), and the rest of the attention proceeds with softmax and weighted sum of values.

The windowed attention captures local syntactic and positional patterns while keeping the computational cost at \(O(L W d)\) instead of \(O(L^2 d)\).

### 2.2 Spectral Mamba

The second branch processes the **same input \(\mathbf{X}\)** through a **Mamba** state‑space model. The Mamba module (imported from `jamba2` as a placeholder) is expected to perform a linear‑time recurrence that captures long‑range dependencies. In the provided implementation, `Mamba` is a simple two‑layer MLP with SiLU, but the architecture is meant to be replaced with a true SSM (e.g., the official Mamba block). For the purpose of this explanation, we treat Mamba as a generic sequence‑to‑sequence transformation \(\mathbf{Y}_{\text{mamba}} = \text{Mamba}(\mathbf{X})\).

To enhance the Mamba output, a **spectral scaling** is applied. A learnable vector \(\mathbf{s} \in \mathbb{R}^D\) (initially zero) is passed through a sigmoid:

\[
\gamma = \sigma(\mathbf{s}) \in (0,1)^D.
\]

The Mamba output is then scaled element‑wise (broadcast over batch and length) as:

\[
\tilde{\mathbf{Y}}_{\text{mamba}} = \gamma \odot \mathbf{Y}_{\text{mamba}}.
\]

Because \(\gamma\) is near 0.5 when \(\mathbf{s}=0\), the Mamba output is initially attenuated, allowing the network to gradually increase its reliance on the SSM pathway. The per‑channel scaling can be interpreted as a learned equalisation of the Mamba features, enabling the model to amplify or suppress specific dimensions of the state‑space output.

### 2.3 Overall Block Structure

The block is organised as follows (similar to a pre‑norm Transformer sub‑layer):

1. **Attention sub‑layer**:
   \[
   \mathbf{X}_{\text{attn}} = \operatorname{MultiHeadAttn}(\mathbf{X}, \mathbf{X}, \mathbf{X}, \text{attn\_mask}=\mathbf{M}),
   \]
   \[
   \mathbf{X}_1 = \operatorname{LayerNorm}(\mathbf{X} + \mathbf{X}_{\text{attn}}).
   \]

2. **Mamba sub‑layer**:
   \[
   \mathbf{X}_{\text{mamba}} = \operatorname{SpectralMamba}(\mathbf{X}_1),
   \]
   \[
   \mathbf{X}_2 = \operatorname{LayerNorm}(\mathbf{X}_1 + \mathbf{X}_{\text{mamba}}).
   \]

The final output is \(\mathbf{X}_2\). The block thus has two sequential residual connections, each with its own normalisation. When injected into a Transformer layer, this block replaces the original attention sub‑layer; the outer layer may still add a third residual after the FFN, but that is outside the scope of this module.

### 2.4 Complexity

The attention branch costs \(O(L W d)\) per head, while the Mamba branch (with a true SSM) would cost \(O(L D^2)\). For long sequences, the Mamba branch can become the bottleneck, but it is still linear in \(L\). The spectral scaling adds negligible overhead.

---

## 3. Implementation Details

### 3.1 SpectralMamba

- **Constructor**: Takes `d_model`. Creates a `Mamba(d_model)` instance and a trainable parameter `dt_scale` of size `d_model`, initialised to zeros.
- **Forward**: Computes `out = self.mamba(x)`, then scales by `torch.sigmoid(self.dt_scale)` broadcast to shape `(1, 1, D)`. Returns the scaled output.

The name `dt_scale` is a misnomer; it’s not related to the delta parameter in Mamba but rather a general channel‑wise gate.

### 3.2 SAMBAM3Block

- **Constructor**:
  - `d_model`: hidden dimension.
  - `n_heads`: number of attention heads.
  - `window` (default 128): the local window size.
  - Creates:
    - `self.attn`: `nn.MultiheadAttention(d_model, n_heads, batch_first=True)`
    - `self.mamba`: `SpectralMamba(d_model)`
    - `self.norm1`, `self.norm2`: `nn.LayerNorm(d_model)`
- **Mask generation**: `sliding_window_mask(L)` returns a mask of shape `(1, 1, L, L)` where valid positions are 0 and masked positions are `-inf`. It enforces causality and the window bound.
- **Forward**:
  1. Generate mask.
  2. Compute attention: `attn_out, _ = self.attn(x, x, x, attn_mask=mask)`.
  3. Add residual, apply norm1: `x = self.norm1(x + attn_out)`.
  4. Pass through spectral Mamba, add residual, apply norm2: `x = self.norm2(x + self.mamba(x))`.
  5. Return `x` (no tuple, unlike some attention modules that return `(out, None)`; this might cause compatibility issues with the framework’s expected output format of `(output, None)`. The block returns only the tensor, which could be a problem. In the injection function context, the framework may expect the attention module’s forward to return a tuple; the SAMBAM3Block returns just a tensor. This is a bug that needs attention.)

### 3.3 Injection Function (Redundant)

The code includes a second registration of `inject_mamba` under `"mamba"`, which simply replaces the attention layer with a plain `Mamba` module (placeholder). This is separate from the `samba_m3` block. The `SAMBA` block itself is registered under `"samba_m3"` via the class decorator? Actually, the class is decorated with `@ATTENTION_REGISTRY.register("samba_m3")`? In the snippet, the `SAMBAM3Block` is defined without a decorator? Checking: The provided code has `@ATTENTION_REGISTRY.register("samba_m3")`? No, the decorator is not shown before the class; there is only `@ATTENTION_REGISTRY.register("spectral_mamba")` on the SpectralMamba class, and then `@ATTENTION_REGISTRY.register("samba_m3")` might be missing. The snippet shows:

```python
@ATTENTION_REGISTRY.register("spectral_mamba")
class SpectralMamba(nn.Module):
    ...

@ATTENTION_REGISTRY.register("samba_m3")
class SAMBAM3Block(nn.Module):
    ...
```

Yes, there is a decorator before `SAMBAM3Block`. So the block is registered under `"samba_m3"`. The separate `inject_mamba` function is unrelated.

---

## 4. Usage and Configuration

To use the SAMBA‑M3 block, the compression config should set `attention_type = "samba_m3"`. The block will need the model’s `hidden_size` and `num_attention_heads`. The window size can be passed via `comp_config.window_size` (or `block_size`). The injection function (if not automatically generated from the class) would need to instantiate the block and assign it to `layer.attention`. The provided snippet does not show a dedicated injection function for SAMBA‑M3; the class itself is registered, so the framework may instantiate it directly (using the model config and extra kwargs). However, the constructor expects `d_model` and `n_heads`; the framework would need to map these from the model config. The `window` parameter could be passed as a kwarg. This should work if the registry supports direct class instantiation.

Example configuration:
```python
comp_config.attention_type = "samba_m3"
comp_config.window_size = 256
```

The placeholder `Mamba` will be used; for performance, one should replace it with the official Mamba implementation.

---

## 5. Discussion

SAMBA‑M3 is a clean hybrid design that marries the strengths of local attention and state‑space models. The causal sliding window ensures that short‑range dependencies are captured with high precision, while the spectral Mamba provides a global context without quadratic cost. The channel‑wise scaling of the Mamba output is a simple yet effective way to control information flow, acting as a soft gating mechanism that can be tuned during training.

However, several issues should be noted:

- **Output format**: The block returns a single tensor, not a tuple. The framework’s training loops might expect `(output, None)`. This discrepancy could cause errors when integrating with the compression framework. It should be fixed to return `(x, None)`.
- **Placeholder Mamba**: The Mamba module is a stub MLP, not a true SSM. The block’s effectiveness heavily depends on a proper Mamba implementation; with the stub, the Mamba branch is essentially a position‑wise feed‑forward, and the block reduces to sliding‑window attention plus a gated MLP, which is far less powerful.
- **Double residual**: The block has its own norm and residual for each sub‑layer. If the outer Transformer layer also applies a residual around the entire attention block, there will be an extra residual pathway, which may be beneficial or could cause training instability. It's common in many hybrid blocks we've seen.
- **No padding mask**: The multi‑head attention call does not receive a padding mask, so padded sequences will be processed incorrectly. The Mamba branch also lacks masking. This is a critical omission for real‑world use.

Despite these limitations, the SAMBA‑M3 block provides a well‑motivated template for hybrid attention‑SSM architectures and can be refined into a fully functional module.

---

## 6. Conclusion

We have presented SAMBA‑M3, a hybrid block that combines a causal sliding window attention with a spectrally gated state‑space model. The design leverages local attention for precise short‑range modelling and a linear‑complexity SSM for global context, all integrated with residual norms. The module is registered in the compression framework as a drop‑in replacement for DistilBERT’s self‑attention. Future work will replace the placeholder Mamba with an efficient SSM kernel, fix the output format, and incorporate padding masks. SAMBA‑M3 stands as a promising step toward hybrid efficient Transformers.