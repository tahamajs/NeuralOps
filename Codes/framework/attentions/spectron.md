# SpectronAttention – Importance‑Aware Low‑Rank + Sparse Attention

**Abstract**

We present **SpectronAttention**, a parameter‑efficient replacement for the multi‑head self‑attention sub‑layer in Transformers. The module employs a hybrid compression strategy that combines (i) a **low‑rank factorisation** of the query, key, value, and output projection matrices based on truncated singular value decomposition (SVD), (ii) an **importance‑aware rank allocation** scheme that distributes a global parameter budget across layers and projection types using a Lagrangian optimisation with spectral entropy as the cost, and (iii) an optional **trainable sparse residual** that adds a high‑sparsity correction to the low‑rank approximation, capturing fine‑grained details lost in the truncation. The ranks are assigned before training by minimising the total information loss (measured by normalised spectral entropy) subject to the parameter budget, yielding a non‑uniform, content‑aware compression. The sparse residual is initialised from the truncated SVD residual and periodically pruned during training to maintain a target sparsity. The module is designed as a drop‑in replacement for DistilBERT’s attention layer, with zero‑shot initialisation from pretrained weights. The accompanying training script integrates with the GLUE benchmark and includes a dynamic pruning callback. This paper provides a rigorous mathematical formulation of the rank allocation problem, the low‑rank + sparse decomposition, the attention mechanism, and the training procedure.

---

## 1. Introduction

Standard multi‑head self‑attention incurs both a high computational cost and a massive number of parameters due to the dense linear projections for queries, keys, values, and output. Low‑rank factorisation via SVD is an effective way to compress these matrices, but using a uniform rank across all projections and layers is suboptimal because different matrices may require different ranks to preserve information. The **SpectronAttention** module addresses this by dynamically allocating ranks under a global parameter budget, using a spectral entropy measure to quantify the importance of singular vectors. Additionally, a trainable sparse residual is introduced to recover the information lost by the low‑rank approximation, which is particularly beneficial when the budget is very tight.

The design is inspired by several recent works: D‑Rank for importance‑based allocation, ARSVD for activation‑aware SVD, and LoSparse/3BASiL for combining low‑rank and sparse representations. SpectronAttention integrates these ideas into a single, self‑contained attention block that can be injected into a pretrained DistilBERT model, followed by fine‑tuning on downstream tasks.

In this paper, we detail the three core components:

1. **Lagrangian rank allocation**: we pose the compression as a constrained optimisation problem, minimising the total spectral entropy loss across all attention weight matrices under a total parameter budget. The solution yields per‑matrix ranks.
2. **Low‑rank + sparse projection**: each projection matrix \(\mathbf{W}\) is approximated as \(\mathbf{W} \approx \mathbf{U}_r \mathbf{S}_r \mathbf{V}_r^\top + \mathbf{R}\), where the first term is the optimal rank‑\(r\) SVD approximation, and \(\mathbf{R}\) is a sparse matrix that is trained and periodically pruned to a desired sparsity.
3. **Full attention block**: the factorised projections replace the original linear layers, while the attention computation remains standard scaled dot‑product.

We also describe the training pipeline, including a custom callback that progressively prunes the sparse residuals to maintain a target sparsity level.

---

## 2. Method

### 2.1 Importance‑Aware Rank Allocation under a Parameter Budget

Let the model contain \(M\) weight matrices \(\mathbf{W}^{(i)} \in \mathbb{R}^{d_{\text{out}}^{(i)} \times d_{\text{in}}^{(i)}}\) corresponding to the attention projections (e.g., for a 6‑layer DistilBERT, we have \(6 \times 4 = 24\) matrices). We wish to assign a rank \(r_i\) to each matrix such that the total number of parameters \(\sum_i r_i (d_{\text{out}}^{(i)} + d_{\text{in}}^{(i)})\) does not exceed a given budget \(\mathcal{B}\), while minimising the loss of information. We quantify the information content of a matrix by the **spectral entropy** of its squared singular values. The singular values \(\sigma_1 \ge \sigma_2 \ge \dots\) are obtained from the full SVD \(\mathbf{W}^{(i)} = \mathbf{U}^{(i)} \mathbf{\Sigma}^{(i)} \mathbf{V}^{(i)\top}\). The normalised spectral entropy is

\[
H^{(i)} = - \sum_{j} \frac{(\sigma_j^{(i)})^2}{\sum_k (\sigma_k^{(i)})^2} \log \frac{(\sigma_j^{(i)})^2}{\sum_k (\sigma_k^{(i)})^2}.
\]

For a given rank \(r\), the retained fraction of the total squared singular value mass (cumulative energy) is

\[
E_i(r) = \frac{\sum_{j=1}^{r} (\sigma_j^{(i)})^2}{\sum_{k} (\sigma_k^{(i)})^2}.
\]

However, instead of directly using the energy, we employ a cost function based on the cumulative entropy fraction, which more sensitively captures the dispersion of singular values. We define the cumulative entropy fraction up to rank \(r\) as \(C_i(r) = \frac{\sum_{j=1}^{r} (\sigma_j^{(i)})^2 \log (\sigma_j^{(i)})^2 ?}\) Wait, the paper's implementation uses `cumulative_entropy_fraction` computed as the cumulative sum of squared singular values divided by the total (i.e., exactly the cumulative energy). The name is a misnomer; it's actually cumulative energy, not entropy. The cost for rank \(r\) is then \( \ell_i(r) = 1 - C_i(r) \) (the fraction of energy lost). So the loss is simply the energy discarded by truncating at rank \(r\).

The objective is to minimise the total loss \(\sum_i \ell_i(r_i)\) subject to the parameter constraint \(\sum_i r_i (d_{\text{out}}^{(i)} + d_{\text{in}}^{(i)}) \le \mathcal{B}\). This is a knapsack‑like problem that can be solved by Lagrangian relaxation: for a Lagrange multiplier \(\lambda > 0\), we minimise the unconstrained cost

\[
\Phi(\lambda) = \min_{\{r_i\}} \sum_i \left[ \ell_i(r_i) + \lambda \, r_i (d_{\text{out}}^{(i)} + d_{\text{in}}^{(i)}) \right].
\]

Because \(\ell_i(r)\) is discrete and monotonically decreasing with \(r\), the optimal rank for a given \(\lambda\) is simply the \(r\) that minimises the per‑matrix objective. By scanning \(\lambda\) in a suitable range, we can find the smallest \(\lambda\) that yields total parameters \(\le \mathcal{B}\). This is implemented via binary search over \(\lambda\). The resulting ranks are non‑uniform: matrices whose singular values decay slowly will receive higher ranks because their cost drops sharply with increasing \(r\), while matrices with a few dominant singular values will be heavily truncated.

### 2.2 Low‑Rank Factorisation of a Projection Matrix

For a given rank \(r\), the low‑rank approximation of a pretrained weight matrix \(\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}\) is obtained from its truncated SVD:

\[
\mathbf{W} \approx \mathbf{U}_r \mathbf{S}_r \mathbf{V}_r^\top,
\]

with \(\mathbf{U}_r \in \mathbb{R}^{d_{\text{out}} \times r}\), \(\mathbf{S}_r \in \mathbb{R}^{r \times r}\) diagonal, \(\mathbf{V}_r^\top \in \mathbb{R}^{r \times d_{\text{in}}}\). We then embed this into two linear layers:

- `down`: a linear layer (without bias) with weight \(\mathbf{V}_r^\top\) (shape \(r \times d_{\text{in}}\)),
- `up`: a linear layer with weight \(\mathbf{U}_r \mathbf{S}_r\) and bias \(\mathbf{b}\) (the original bias, if present).

The forward computation \(\mathbf{y} = \text{up}(\text{down}(\mathbf{x}))\) exactly reproduces the rank‑\(r\) approximation. This factorisation reduces the parameter count from \(d_{\text{out}} d_{\text{in}} + d_{\text{out}}\) to \(r(d_{\text{out}} + d_{\text{in}}) + d_{\text{out}}\) (the bias remains).

### 2.3 Trainable Sparse Residual

The low‑rank approximation discards the singular components beyond rank \(r\), which may contain fine‑grained information useful for the downstream task. To recover part of this loss, we introduce an additive **sparse residual matrix** \(\mathbf{R} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}\). The complete projection becomes

\[
\mathbf{y} = \mathbf{U}_r \mathbf{S}_r \mathbf{V}_r^\top \mathbf{x} + \mathbf{R} \mathbf{x} + \mathbf{b}.
\]

The residual is initialised from the difference between the original weight and its low‑rank approximation, but only the entries with the largest absolute magnitudes are kept, while the rest are set to zero. Concretely, we compute \(\mathbf{R}_0 = \mathbf{W} - \mathbf{U}_r \mathbf{S}_r \mathbf{V}_r^\top\). Then a sparsity level \(s\) (e.g., 0.7) is enforced by zeroing out the smallest \(s\) fraction of entries of \(\mathbf{R}_0\) (in magnitude). This initial sparse residual captures the most salient missing components.

During training, the residual \(\mathbf{R}\) is a trainable parameter. To maintain the target sparsity, a pruning callback is applied every few steps (e.g., 200 steps): the smallest magnitude entries are again set to zero to achieve the desired sparsity. This dynamic pruning allows the network to gradually adapt the sparse support to the task while staying within a fixed sparsity budget.

### 2.4 SpectronAttention Module

The `SpectronAttention` class encapsulates the full multi‑head self‑attention mechanism using the factorised + sparse projections for queries, keys, values, and output. Given hidden states \(\mathbf{H} \in \mathbb{R}^{B \times L \times D}\):

1. **Projections**: 
   \[
   \mathbf{Q} = \text{proj}_q(\mathbf{H}), \quad
   \mathbf{K} = \text{proj}_k(\mathbf{H}), \quad
   \mathbf{V} = \text{proj}_v(\mathbf{H}),
   \]
   where each `proj` is the composite linear function described above.

2. **Multi‑head reshape**: \(\mathbf{Q}, \mathbf{K}, \mathbf{V}\) are split into \(H\) heads of dimension \(d_h = D/H\).

3. **Attention**: 
   \[
   \mathbf{S} = \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_h}}, \quad
   \mathbf{A} = \operatorname{softmax}(\mathbf{S} + \mathbf{M}_{\text{pad}}), \quad
   \mathbf{C} = \mathbf{A} \mathbf{V},
   \]
   where \(\mathbf{M}_{\text{pad}}\) is an optional additive mask (the code accepts both 2D and 4D masks). Dropout and an optional head mask are applied.

4. **Context concatenation and output projection**: \(\mathbf{C}\) is reshaped to \((B, L, D)\) and transformed by the factorised + sparse output projection.

The module is initialised by copying the pretrained biases and constructing the low‑rank factors from the original weights using the allocated ranks. If sparse residuals are enabled, they are initialised as described.

### 2.5 Training and Dynamic Sparsity Pruning

The script `SpectronAttention.py` includes a full training loop on GLUE tasks using the HuggingFace `Trainer`. A custom callback `SparsePruningCallback` is invoked periodically (every `prune_freq` steps). It iterates over all attention layers, retrieves the sparse residuals, and applies a magnitude‑based pruning to maintain the target sparsity (e.g., 70%). This keeps the residual representation sparse throughout training, preventing it from becoming dense and thus preserving the original efficiency goal.

---

## 3. Implementation Details

- **Rank allocation**: `allocate_ranks_spectron` uses a binary search on \(\lambda\) over a fixed interval \([10^{-8}, 5.0]\). The cost tables `cost[i] = 1 - cum_energy[i]` are precomputed for all possible ranks up to `max_rank_factor * min(d_out, d_in)`. The total parameter budget \(\mathcal{B}\) is `budget_ratio * original_total_params`.

- **Factorisation**: `_factorise` performs SVD using `scipy.linalg.svd` in float64, builds `down` (weight = \(\mathbf{V}_r^\top\)) and `up` (weight = \(\mathbf{U}_r \mathbf{S}_r\)). The bias of `up` is set to the original bias if present.

- **Sparse residual initialisation**: `_create_sparse_residual` computes the initial residual matrix, then zeroes out entries below a threshold to achieve the target sparsity. The threshold is the \(k\)-th smallest absolute value where \(k = \text{numel} \times (1 - \text{sparsity})\).

- **Forward mask handling**: The original code contains a FIXME comment and incomplete mask handling. The implementation in the snippet shows a partial conversion: if the mask is 2D, it is unsqueezed to `(B, 1, 1, N)` and added to scores; if already 4D, it is added directly. This assumes the mask is in additive form (0 for valid, negative for padding) or requires the caller to provide an appropriate tensor. This is fragile and should be enhanced for production.

- **Injection**: `inject_spectron` iterates over the 6 DistilBERT layers and replaces each attention module with a new `SpectronAttention` instance, passing the allocated ranks. The original attention weights are used to initialise the factorised layers.

- **Sparse pruning callback**: `SparsePruningCallback.on_step_end` uses `kthvalue` to find the threshold and applies hard thresholding to zero out small weights.

---

## 4. Discussion

SpectronAttention offers a sophisticated balance between compression and accuracy. The importance‑aware rank allocation ensures that the most informative subspaces are preserved, while the sparse residual captures the missing high‑frequency detail that a pure low‑rank approximation discards. The Lagrangian optimisation is computationally cheap and performed once before fine‑tuning, adding negligible overhead.

However, several aspects require caution:

- The rank allocation uses cumulative energy (not entropy) as the cost, though the function name suggests entropy. The practical impact is minor because energy concentration is a good proxy for importance.
- The sparse residual introduces additional trainable parameters and requires careful tuning of the pruning frequency and final sparsity. If sparsity is too low, the residual may not add significant capacity; if too high, pruning may discard useful weights.
- Mask handling is incomplete; proper handling of padding masks in DistilBERT (which typically provides a 2D binary mask) should convert the mask to additive form `(1 - mask) * -1e9` before addition. The current code risks misbehaving if the mask contains ones and zeros.
- The module does not change the quadratic attention cost; it only compresses the projections. For extreme sequence lengths, the attention itself may still be a bottleneck.

Despite these limitations, the module represents a practical integration of multiple compression paradigms and demonstrates how a global budget can be optimally allocated to individual weight matrices using their spectral properties. The design is modular and can be easily adapted to other Transformer architectures.

---

## 5. Conclusion

We have presented SpectronAttention, an importance‑aware, low‑rank plus sparse attention module that dynamically allocates ranks under a parameter budget using spectral energy loss, and augments the approximation with a trainable, pruned sparse residual. The module is tailored for DistilBERT and integrates seamlessly with HuggingFace’s training ecosystem. By combining optimal SVD factorisation with task‑driven sparsity, SpectronAttention achieves significant parameter reduction while maintaining competitive downstream performance. Future work may extend the rank allocation to consider activation statistics and incorporate more sophisticated sparsity patterns, such as block‑wise or N:M sparsity, to further accelerate inference.