# VSpec++: Variational Spectral Token‑Adaptive Attention

**Abstract**  
We present **VSpec++**, a highly adaptive attention module for Transformer models that integrates three dynamic compression mechanisms: (1) **variational spectral gating**, which learns a per‑batch scaling of the singular values of the pretrained projection matrices, allowing the rank to vary on the fly; (2) **head‑wise gating**, which modulates the contribution of each attention head based on a pooled summary of the input; and (3) **token routing**, which selects a subset of the most salient tokens to attend to, reducing the effective sequence length. The module factorises each projection matrix (Q, K, V, O) into a frozen low‑rank basis derived from truncated SVD of the pretrained weights, plus a trainable diagonal scaling that is dynamically conditioned on the input via a variational network (reparameterisation trick). The token router uses a lightweight scorer to keep a fixed fraction of tokens, and the head gate further reweights the attention scores. The result is a highly parameter‑efficient, content‑adaptive attention mechanism that can be initialised from a pretrained DistilBERT model and fine‑tuned on downstream tasks with a fraction of the original parameters. The module is complemented by a global Lagrangian rank allocation optimiser that distributes a parameter budget across layers and projections. A complete training script integrates VSpec++ with GLUE tasks and the HuggingFace `Trainer`. This paper provides a rigorous mathematical description of each component, the dynamic factorisation, the variational gate, the router, and the overall architecture, as well as the budget‑aware rank allocation. We also discuss potential inconsistencies (such as the reuse of the query gate for output projection) and limitations.

---

## 1. Introduction

The multi‑head self‑attention layers in pretrained Transformers contribute a large share of the model’s parameters, yet they are often over‑parameterised for specific tasks. Static compression techniques, such as a fixed low‑rank factorisation via SVD, can reduce the parameter count but lack the ability to adapt to varying input complexity. The **VSpec++** module introduces a suite of **dynamic, input‑dependent adaptivity** into the compressed attention layer. Building upon the Lagrangian budget allocation of SVD‑Compressed Attention, VSpec++ adds three dynamic mechanisms:

1. **Variational spectral gates** – For each projection (Q, K, V, O), a small neural network observes a pooled representation of the input sequence and predicts a scaling vector for the singular values of that projection. This allows the effective rank to be modulated per batch, effectively performing input‑aware compression. The gate uses a variational reparameterisation (mean and log‑variance) to inject stochasticity, encouraging robust representations.

2. **Head gating** – A separate head gate predicts a scalar per attention head, which is multiplied into the attention scores, enabling the network to dynamically de‑emphasise or emphasise certain heads.

3. **Token routing** – A learned token scorer selects the top‑\(k\) most important tokens before the attention computation. The attention is performed only on these routed tokens, reducing the sequence length and thus the computational cost. The output is then scattered back to the original token positions (keeping the pruned tokens unchanged). This acts as a form of dynamic token pruning.

The combination yields a highly flexible attention layer that can be initialised from a pretrained DistilBERT model and then fine‑tuned with a small number of additional parameters (the gating networks and router). To determine the base ranks for the factorisation, the module employs the same Lagrangian optimisation as in SVD‑Compressed Attention, allocating a global parameter budget across all 24 projection matrices (6 layers × 4 projections). The resulting rank assignment ensures that the most informative subspaces are preserved.

The module is named VSpec++ (“Variational Spectral …”), indicating an evolution of earlier spectral compression methods. It is implemented as a drop‑in replacement for the DistilBERT attention layer and is accompanied by a full training script that applies the allocation, injects the modules, and fine‑tunes on GLUE tasks.

In this paper, we provide a rigorous mathematical formulation of the dynamic low‑rank projection, the variational spectral gate, the head gate, the token router, and the overall VSpecAttention forward pass. We also detail the rank allocation and training pipeline, and we critically examine the design choices and potential pitfalls (such as sharing the query gate for the output projection).

---

## 2. Method

### 2.1 Dynamic Low‑Rank Projection with Variational Spectral Gating

The core of VSpec++ is a **Dynamic Low‑Rank Projector** that factorises a pretrained weight matrix \(\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}\). Instead of a fixed truncation, the projector stores the top‑\(r\) singular components as **frozen** parameters:

\[
\mathbf{U} \in \mathbb{R}^{d_{\text{out}} \times r}, \quad \mathbf{S} \in \mathbb{R}^{r}, \quad \mathbf{V}^\top \in \mathbb{R}^{r \times d_{\text{in}}},
\]

obtained from the truncated SVD of \(\mathbf{W}\). Here \(\mathbf{U}\) and \(\mathbf{V}^\top\) are frozen, while the singular values \(\mathbf{S}\) are **trainable** (but independent of the input). To make the projection input‑dependent, a **variational spectral gate** is introduced. For a given input batch \(\mathbf{X} \in \mathbb{R}^{B \times N \times D}\), the gate network \(G_{\text{spec}}\) first pools the hidden states to a single vector per batch element:

\[
\mathbf{h} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{X}_{:,i,:} \in \mathbb{R}^{B \times D}.
\]

This pooled vector is passed through a small MLP that outputs parameters \(\boldsymbol{\mu} \in \mathbb{R}^{r}\) and \(\boldsymbol{\log\sigma} \in \mathbb{R}^{r}\). Using the reparameterisation trick, a latent variable \(\mathbf{z}\) is sampled:

\[
\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\epsilon} \odot \exp(\boldsymbol{\log\sigma}), \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}).
\]

The final spectral gate is obtained via a sigmoid:

\[
\mathbf{g} = \sigma(\mathbf{z}) \in (0,1)^{r}.
\]

This gate is then used to scale the singular values dynamically:

\[
\mathbf{S}_{\text{eff}} = \mathbf{S} \odot \mathbf{g} \quad (\text{broadcast across batch}).
\]

The forward pass of the projector for an input \(\mathbf{X}\) (which may be the routed token set) is:

\[
\mathbf{Y} = \underbrace{\big( \mathbf{X} \mathbf{V}^\top \big)}_{\mathbb{R}^{B \times N \times r}} \; \odot \; \mathbf{S}_{\text{eff}} \;\; \underbrace{\mathbf{U}^\top}_{\text{shape } (r, d_{\text{out}})} \quad \Rightarrow \quad \mathbf{Y} \in \mathbb{R}^{B \times N \times d_{\text{out}}}.
\]

In practice, the code computes `mid = x @ Vh.t()`, then `mid = mid * S_eff.unsqueeze(1)`, then `out = mid @ U.t()`. This implements a low‑rank transformation whose singular values are modulated per batch.

For each of the four projections (Q, K, V, O), a separate projector and spectral gate are instantiated. The original weight matrix for the output projection is used to initialise the corresponding projector; notably, the output projector uses the **same gate as the query** (`gq` is passed to `Po`), which is a design choice that may be suboptimal but simplifies the architecture.

### 2.2 Head Gating

A separate **Head Gate** module takes the same pooled input \(\mathbf{h}\) and predicts a scalar per attention head:

\[
\boldsymbol{\beta} = \sigma( \operatorname{MLP}_{\text{head}}(\mathbf{h}) ) \in \mathbb{R}^{B \times H},
\]

where \(H\) is the number of heads. This vector is reshaped to \((B, H, 1, 1)\) and multiplied element‑wise with the attention scores \(\mathbf{S}\) before softmax:

\[
\mathbf{S}' = \boldsymbol{\beta} \odot \mathbf{S}.
\]

This allows the model to dynamically weight the contribution of each head based on the input content. It is similar in spirit to learned head importance but applied per batch, not globally.

### 2.3 Token Routing

The **Token Router** reduces the sequence length before the heavy attention computation. A lightweight scorer network processes each token independently:

\[
s_{i} = \sigma( \operatorname{MLP}_{\text{token}}(\mathbf{x}_i) ) \in (0,1).
\]

The top‑\(k\) scoring tokens are selected, where \(k = \max(k_{\min}, \lfloor \rho N \rfloor)\) with a configurable keep ratio \(\rho\) (default 0.5) and a minimum token count. The selected tokens are gathered into a new tensor \(\mathbf{X}_{\text{routed}} \in \mathbb{R}^{B \times k \times D}\), and the original indices are stored. The attention is then performed only on this reduced set of tokens. After the attention output \(\mathbf{Y}_{\text{routed}} \in \mathbb{R}^{B \times k \times D}\) is obtained, it is scattered back to the original positions using the stored indices, while the unselected tokens retain their original hidden states. This implements a hard token pruning, which is not differentiable through the selection step but receives gradient signal through the selected tokens.

### 2.4 Overall VSpecAttention Block

Given the original input \(\mathbf{X} \in \mathbb{R}^{B \times L \times D}\):

1. **Token routing**: \(\mathbf{X}_{\text{routed}}, \mathcal{I} = \operatorname{TokenRouter}(\mathbf{X})\).
2. **Compute spectral gates**: \((\mathbf{g}_q, \mathbf{g}_k, \mathbf{g}_v)\) from \(\mathbf{X}\) (using the original full sequence) via variational spectral gates.
3. **Project Q, K, V** using the dynamic low‑rank projectors with their respective gates (e.g., \(\mathbf{Q} = \operatorname{Proj}_Q(\mathbf{X}_{\text{routed}}, \mathbf{g}_q)\)).
4. **Reshape to multi‑head**: separate heads, yielding \((B, H, k, d_h)\).
5. **Compute attention scores**: \(\mathbf{S} = \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_h}}\).
6. **Apply head gate**: \(\mathbf{S} \leftarrow \boldsymbol{\beta} \odot \mathbf{S}\).
7. **Mask with attention mask** if provided (the mask is assumed to be already in the appropriate format for the routed tokens? Actually the code applies the original mask directly; this could be problematic because the sequence length changed. The code does `if attention_mask is not None: scores = scores + mask`, where `mask` might not correspond to the routed tokens. This is a known issue).
8. **Softmax, dropout**, optional head_mask.
9. **Aggregate values**: \(\mathbf{C} = \operatorname{softmax}(\mathbf{S}) \mathbf{V}\).
10. **Concatenate heads**: reshape to \((B, k, D)\).
11. **Output projection**: \(\mathbf{Y}_{\text{routed}} = \operatorname{Proj}_O(\mathbf{C}, \mathbf{g}_q)\) (note: reusing \(\mathbf{g}_q\)).
12. **Reverse routing**: \(\mathbf{Y}_{\text{final}} = \text{scatter}(\mathbf{X}, \mathcal{I}, \mathbf{Y}_{\text{routed}})\), i.e., the output for selected positions is replaced by the attention output, while unselected positions keep the original hidden state.

### 2.5 Budget‑Aware Rank Allocation

As in SVD‑Compressed Attention, a global Lagrangian optimisation is performed once before training. It takes all attention weight matrices (24 of them), computes their singular values, and determines the optimal rank for each matrix such that the total parameter count stays within a budget (e.g., 20% of original) while minimising the sum of discarded singular‑value energies. The resulting ranks are used to initialise the truncated SVD factors in the dynamic projectors.

---

## 3. Implementation Details

- **`VariationalSpectralGate`**: A two‑layer MLP (hidden size 64) with GELU. Outputs 2× rank values (mu, log_sigma). Applies reparameterisation and sigmoid.
- **`HeadGate`**: A two‑layer MLP (32 hidden) → sigmoid.
- **`TokenRouter`**: A two‑layer MLP (32 hidden) → sigmoid; keeps top‑k tokens, returns routed tensor and indices.
- **`DynamicLowRankProjector`**: Stores `U`, `S`, `Vh` as parameters, but `U` and `Vh` are frozen (requires_grad=False) while `S` is trainable. The gate `g` is applied to `S`.
- **`VSpecAttention`**: Constructor takes `hidden_size`, `num_heads`, `rank`, `pretrained_attn`. It transposes the original linear weights (since the projectors expect shape (in, out)?? Actually, the code does `.t()` for Q/K/V weights and not for output. This might introduce dimension mismatches. The forward pass then uses `x @ Vh.t()` where `Vh` is \((r, \text{in\_features})\). This assumes the input is \(d_{\text{in}}\). In DistilBERT, the input to Q projection is hidden_size, output is all_head_size. So `Vh` should be of shape `(r, hidden_size)`. The initialisation uses `U,S,Vh = torch.linalg.svd(W)` where `W` is the original weight matrix. That is correct. The transposition in the constructor might be a mistake: `pretrained_attn.q_lin.weight.data.clone().t()` would give shape `(hidden_size, all_head_size)`, then SVD would be on that matrix, yielding `U` of shape `(hidden_size, ...)`. That seems inverted. This could cause shape errors. We'll note this as a potential bug.)
- **Training script**: The script `VSpecAttention.py` performs data loading, rank allocation, injection via `inject_all_layers_vspec`, and training. The injection function creates `VSpecAttention` for each layer with the allocated rank (actually it uses only the query rank as the global rank for that layer? The code uses `rank=rank_dict[f"layer{layer_idx}_q"]` for the constructor, ignoring the separate ranks for K/V/O. This is inconsistent with the allocation that gives per‑projection ranks. The constructor of `VSpecAttention` takes a single `rank` parameter, so all projectors get the same rank. The injection script does not pass the individual ranks to the module. Thus the careful Lagrangian allocation is largely wasted. We'll point this out.

---

## 4. Usage and Configuration

The training script is executed as:
```bash
python VSpecAttention.py --task sst2 --budget_ratio 0.2
```
Key hyperparameters:
- `--budget_ratio`: Fraction of original attention parameters to retain for the base ranks.
- `--epochs`, `--lr`, `--batch_size`: standard fine‑tuning settings.
- The script automatically loads the GLUE task, computes rank allocation, injects VSpec++, and trains.

The module can also be used standalone by instantiating `VSpecAttention` with appropriate parameters and replacing the attention layer.

---

## 5. Discussion

VSpec++ combines several orthogonal adaptation mechanisms into a single attention layer, resulting in a highly dynamic yet parameter‑efficient module. The variational spectral gate introduces input‑dependent rank scaling, which can be interpreted as a soft attention over singular components, potentially making the low‑rank approximation more expressive. The head gate adds further per‑example adaptivity. The token router reduces sequence length, which directly reduces the computational cost of attention (which remains quadratic in the routed length). The reverse routing preserves the original token positions, making the pruning transparent to later layers.

However, the current implementation contains several inconsistencies and limitations:

- **Rank allocation not utilised**: The injection function uses only the query rank for all projectors, ignoring the per‑projection ranks computed by the Lagrangian optimiser. This defeats the purpose of the global allocation; the module would need to accept separate ranks for Q, K, V, O to fully benefit from the allocation.
- **Dimension transposition**: The constructor transposes the original Q/K/V weights before SVD, possibly causing shape mismatches. The code `pretrained_attn.q_lin.weight.data.clone().t()` yields a matrix of shape `(hidden_size, all_head_size)`, while the forward pass expects `x @ Vh.t()` where `Vh` is `(rank, hidden_size)`. That would be correct if the input `x` is of dimension `hidden_size`. But then `U` would be of shape `(hidden_size, rank)`, and the output would be `(B, N, hidden_size)`, not `all_head_size`. This is clearly wrong. The proper initialisation should not transpose the weight; it should factor `W` directly as `(all_head_size, hidden_size)`. So the current code would produce incorrect output dimensions. This is a critical bug.
- **Mask handling with token router**: The attention mask is applied to scores after routing, but the mask still has the original sequence length, not the reduced length. This will cause broadcasting errors or incorrect masking. The mask should be gathered along with the tokens.
- **Reusing query gate for output**: The output projection uses the query spectral gate `gq`. There is no dedicated output spectral gate, even though one could be beneficial. The output gate might not align with the output projection’s needs.
- **Gradient flow for token router**: The router’s selection is based on a sigmoid score and a top‑k, which is non‑differentiable. The scorer can still be trained via the retained tokens, but the pruned tokens do not receive gradient from the attention layer. This may be acceptable.
- **Variational aspect**: The use of reparameterisation with random noise is unusual for a deterministic attention mechanism; it may inject beneficial regularisation but also variance during inference. The Kullback–Leibler divergence term is not added to the loss, so the variational aspect is incomplete.

Despite these flaws, the concept of VSpec++—input‑dependent singular value scaling combined with head gating and token pruning—is compelling. With the technical bugs fixed, it could represent a powerful adaptive compression framework.

---

## 6. Conclusion

We have presented VSpec++, a variational spectral token‑adaptive attention module that combines dynamic low‑rank projections, head gating, and token routing. The module aims to achieve high parameter efficiency and adaptivity through input‑dependent singular value modulation, while reducing sequence length via learned token selection. The accompanying Lagrangian rank allocator provides a principled way to set the base ranks. However, the current implementation contains several critical bugs and design oversights (weight transposition, rank allocation underutilisation, mask handling) that prevent it from functioning as intended. With corrections, VSpec++ could become a valuable addition to the compression framework, offering a unified approach to adaptive, budget‑aware attention compression. Future work will focus on resolving these issues and evaluating the approach on a broader set of tasks.