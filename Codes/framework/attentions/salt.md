# SALT: Singular Value Adaptation for Linear Layers (Incomplete Attention Module)

**Abstract**

We present **SALT** (Singular Value Adaptation of Linear Transformations), a lightweight module intended to compress and adapt the linear projections within Transformer attention. It replaces a single linear layer by freezing its original weight matrix and factorising it via a truncated SVD, then adding a trainable low‑rank update and a per‑singular‑value scaling. The module is designed to reduce trainable parameters while preserving a strong initialisation from pretrained weights. However, the provided code does **not** implement a complete attention mechanism—it is only a linear layer substitute—and the injection function incorrectly replaces the entire attention sub‑layer with this single linear layer, which would break the Transformer architecture. This paper describes the SALT linear layer, its mathematical formulation, its intended role within a larger attention block, and the critical integration flaws.

---

## 1. Introduction

Large pre‑trained Transformer models contain millions of parameters in their attention projections. Parameter‑efficient fine‑tuning methods such as LoRA and SVD‑factorisation aim to reduce the number of trainable parameters while retaining the knowledge embedded in the original weights. The **SALT** module applies a hybrid approach: it takes a pretrained weight matrix \(\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}\), computes its full SVD, freezes the singular vectors \(\mathbf{U}\) and \(\mathbf{V}^\top\), and makes the singular values \(\mathbf{S}\) trainable. Additionally, it introduces a scaling factor \(\boldsymbol{\gamma}\) to adjust the top singular values, and a low‑rank correction \(\mathbf{A}\mathbf{B}\) with rank \(r\). This combination allows the model to adapt the spectrum of the original weight while learning a task‑specific residual.

However, SALT as provided is **not** an attention module; it is a building block for a linear layer. The injection function mistakenly replaces the entire DistilBERT attention layer with a single `SALTLinear` instance, which would eliminate the multi‑head attention, cross‑token interactions, and output projections. This paper therefore focuses on the SALT linear layer itself, its mathematics, and its correct usage as a component within attention, while highlighting the integration errors.

---

## 2. Method

### 2.1 Original Weight Freezing and SVD

Given a pretrained weight matrix \(\mathbf{W}_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}\), we first store it as a frozen buffer. Its compact SVD is computed once:

\[
\mathbf{W}_0 = \mathbf{U} \mathbf{S} \mathbf{V}^\top,
\]

where \(\mathbf{U} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{out}}}\), \(\mathbf{V} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}\) (or truncated to full rank), and \(\mathbf{S} = \operatorname{diag}(\sigma_1, \sigma_2, \dots)\) with \(\sigma_1 \ge \sigma_2 \ge \cdots\). The matrices \(\mathbf{U}\) and \(\mathbf{V}\) are frozen (registered as buffers), while the singular values are replaced by a trainable copy `self.S`, initialised with the original \(\sigma_i\).

### 2.2 Top‑Singular‑Value Scaling

A learnable parameter \(\boldsymbol{\gamma} \in \mathbb{R}^r\) (where \(r\) is the `rank` hyperparameter) is introduced to scale the top \(r\) singular values. The adapted singular values become:

\[
\tilde{\sigma}_i = \gamma_i \cdot \sigma_i \quad \text{for } i = 1, \dots, r.
\]

The remaining singular values \(i > r\) are kept fixed at their original values. The adapted diagonal matrix for the top‑\(r\) is \(\tilde{\mathbf{S}}_r = \operatorname{diag}(\tilde{\sigma}_1, \dots, \tilde{\sigma}_r)\). The core adapted weight from the spectral part is then:

\[
\mathbf{W}_{\text{spectral}} = \mathbf{U}_{[:,:r]} \tilde{\mathbf{S}}_r \mathbf{V}_{[:r,:]}^\top + \mathbf{U}_{[:,r:]} \mathbf{S}_{[r:]} \mathbf{V}_{[r:,:]}^\top.
\]

In the code, the implementation is simplified: `adapted_S = self.S[:self.gamma.shape[0]] * self.gamma`, and only the top‑\(r\) singular vectors and values are used to form `adapted_weight = U[:,:r] @ diag(adapted_S) @ Vt[:r,:]`. The rest of the singular components are **discarded**, not added back. This means the low‑rank part is expected to compensate for the dropped components, which is a design choice—it reduces the spectral part to rank \(r\) and relies on the low‑rank update \(\mathbf{A}\mathbf{B}\) to capture the remainder.

### 2.3 Low‑Rank Update

A trainable low‑rank update is added as \(\mathbf{A}\mathbf{B}\), where \(\mathbf{A} \in \mathbb{R}^{d_{\text{out}} \times r}\) and \(\mathbf{B} \in \mathbb{R}^{r \times d_{\text{in}}}\). They are initialised with small random values. The final effective weight is:

\[
\mathbf{W}_{\text{eff}} = \mathbf{U}_{[:,:r]} \operatorname{diag}(\tilde{\boldsymbol{\sigma}}_r) \mathbf{V}_{[:r,:]}^\top + \mathbf{A} \mathbf{B}.
\]

The forward pass computes \(\mathbf{y} = \mathbf{W}_{\text{eff}} \mathbf{x}\) (plus bias? Bias is not included in this version). This formulation reduces the number of trainable parameters from \(d_{\text{out}} d_{\text{in}}\) to \(r(d_{\text{out}} + d_{\text{in}} + 2 + 1?)\) — approximately \(2 r d + r + r\) (for A, B, gamma, and the top‑r singular values S[:r] are also trainable; note S is a full parameter but only top‑r are scaled, yet all singular values could be trained if they are a parameter; in code, S is an nn.Parameter so all are trainable, but gamma only scales top‑r; the rest of S is also trainable? Actually S is a parameter, so it's updated, but only the top‑r are scaled by gamma; the lower ones could change as well, which might lead to overfitting. The current code updates the full S, but only the first r are used in the adapted weight? No, the code only uses the first r singular values and vectors to build `adapted_weight`. It ignores the rest. So the remaining singular values are not used at all. That's a drastic truncation.)

### 2.4 Intended Integration

SALT is intended to replace each **linear projection** within a multi‑head attention module (i.e., the Q, K, V, and output projections) individually, much like LoRA. The complete attention block would use four `SALTLinear` instances, one for each projection, while leaving the rest of the attention mechanism intact. This would provide a parameter‑efficient adaptation of the attention sub‑layer. However, the provided code does not construct such an attention module; it only defines the linear layer and an injection function that incorrectly substitutes the entire `layer.attention` with a single `SALTLinear` object. This would destroy the attention computation entirely, making the model non‑functional.

---

## 3. Implementation Details

- **`SALTLinear` class**:
  - Constructor: `in_features`, `out_features`, `rank` (default 16).
  - Creates a buffer `original_weight` randomly initialised (a placeholder; in a real scenario, it should be initialised from the pretrained weight).
  - Computes SVD of this random matrix, stores `U`, `Vt` as buffers.
  - `S`: trainable copy of the singular values (initially all of them).
  - `gamma`: trainable parameter of shape `(rank,)` initialised to 1.
  - `A`, `B`: low‑rank matrices, random normal initialisation with std=0.01.
  - Forward: reconstructs the effective weight as described and applies `F.linear(x, W)`.
- **Injection function**: `inject_salt_linear` registered under `"salt_linear"`. It creates a single `SALTLinear` with `in_features = out_features = hidden_size`, and replaces `layer.attention` with it. This is obviously wrong for a Transformer, as it discards the multi‑head attention mechanism.

---

## 4. Usage and Configuration (Intended)

In a correct setup, SALT would be used by replacing each of the four linear layers inside an attention block. The registry key `"salt_linear"` might be used in a custom injection pipeline that knows how to wrap the original Q, K, V, O projections with SALT while preserving the rest of the attention logic. The configuration would specify the rank \(r\). Example:

```python
comp_config.attention_type = "salt_linear"   # if properly implemented as an attention wrapper
comp_config.rank = 8
```

But as given, this would break the model.

---

## 5. Discussion

SALT offers an interesting alternative to LoRA: instead of adding a low‑rank update to a frozen weight, it factorises the weight itself via SVD and makes the singular values and a low‑rank correction trainable. The scaling of top singular values allows the model to adjust the importance of the dominant directions. This can be more expressive than LoRA because it not only adds a correction but also modally distorts the original weight’s spectrum. The low‑rank update \(\mathbf{A}\mathbf{B}\) further compensates for the dropped tail of the spectrum.

However, the current implementation has severe flaws:
- It uses a random `original_weight` instead of allowing injection of the actual pretrained weight. The SVD is performed on a random matrix, which makes the initialisation meaningless. A proper initialisation must use the pretrained weight.
- It truncates the SVD to rank \(r\) completely, discarding the lower singular components without adding them back. The low‑rank update may not fully recover the lost information, especially at small ranks.
- The injection as a linear layer replacement demonstrates a fundamental misunderstanding of the Transformer architecture. The code should instead define a wrapper attention class that creates four `SALTLinear` instances for Q, K, V, O, and implements the full multi‑head attention forward pass.
- The `forward` method accepts `**kwargs` but does nothing with them; no attention mask handling is present because it’s not an attention module.

Thus, SALT as provided is a **component prototype**, not a deployable attention module. It requires integration into a proper attention class before it can be used.

---

## 6. Conclusion

We have described SALT (Singular Value Adaptation of Linear Transformations), a method for parameter‑efficient adaptation of linear layers via spectral scaling and a low‑rank residual. The mathematical formulation is promising, but the current implementation is incomplete and incorrectly wired. To realise its potential, future work must embed SALT into a full multi‑head attention module, initialise the SVD from pretrained weights, and ensure the correct handling of the full singular value spectrum. Once these steps are completed, SALT could become a powerful tool for compressing and fine‑tuning Transformer attention layers.