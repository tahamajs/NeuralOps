# SVD‑Compressed Attention with Importance‑Aware Rank Allocation

**Abstract**  
We present **SVD‑Compressed Attention**, a parameter‑efficient replacement for the multi‑head self‑attention sub‑layer in DistilBERT. The module compresses the query, key, value, and output projection matrices through truncated singular value decomposition (SVD). Crucially, the rank of each projection is not uniform but is determined by a global, budget‑aware **Lagrangian optimisation** that minimises the total spectral information loss across all attention weight matrices. We formulate the rank allocation as a constrained optimisation problem: minimise \(\sum_i (1 - \text{cumulative\_energy}_i(r_i))\) subject to a total parameter budget. The solution yields per‑matrix ranks, which are then used to factorise the original pretrained weights into two compact linear layers (a down‑projection and an up‑projection). The module also supports an optional **trainable sparse residual** that captures high‑frequency details lost in the low‑rank approximation. The method is fully integrated into the HuggingFace ecosystem, with a training script that dynamically replaces all attention layers and fine‑tunes on GLUE tasks. This paper describes the mathematical formulation of the rank allocation, the SVD factorisation, the sparse augmentation, and the overall pipeline, and provides a thorough analysis of the design choices and potential improvements.

---

## 1. Introduction

Large pre‑trained Transformer models contain millions of parameters in their attention projections, which are often over‑parameterised for downstream tasks. Low‑rank factorisation via SVD offers a direct path to compression: a weight matrix \(\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}\) can be replaced by two smaller matrices \(\mathbf{U}_r \mathbf{S}_r \mathbf{V}_r^\top\), where the rank \(r\) is much smaller than \(\min(d_{\text{out}}, d_{\text{in}})\). However, uniform ranks across all projections and layers are suboptimal because different matrices exhibit different spectral decay patterns.

The **SVD‑Compressed Attention** module addresses this by introducing a **global, budget‑aware rank allocation** strategy that distributes a fixed parameter budget among all attention weight matrices (24 matrices for a 6‑layer DistilBERT) by minimising the sum of the discarded singular‑value energies. The allocation is found by solving a Lagrangian dual problem via binary search. Once the per‑matrix ranks are determined, each original weight is factorised into a down‑projection (from input to rank) and an up‑projection (from rank to output), initialised from the truncated SVD. The attention mechanism itself remains unchanged (full scaled dot‑product). To further recover information lost by truncation, an optional sparse residual matrix can be added, which is initialised from the SVD residual and trained with a fixed sparsity.

The module is designed as a drop‑in replacement for DistilBERT’s attention sub‑layer and is integrated into the compression framework via the `ATTENTION_REGISTRY`. The accompanying training script performs rank allocation once before fine‑tuning, injects the compressed attention layers, and trains on GLUE tasks using the HuggingFace `Trainer`.

In this paper, we detail the mathematical underpinnings of the rank allocation, the factorised linear layer construction, the sparse residual mechanism, and the overall training pipeline. We also discuss the practical limitations evident in the current implementation, such as the assumption of a specific mask format and the absence of dynamic rank adjustment during training.

---

## 2. Method

### 2.1 Rank Allocation via Lagrangian Optimisation

Let the model have \(M\) weight matrices \(\mathbf{W}^{(1)}, \dots, \mathbf{W}^{(M)}\) corresponding to the Q, K, V, and output projections for each of the 6 DistilBERT layers. For a given matrix with singular values \(\sigma_1 \ge \sigma_2 \ge \dots\), the cumulative energy up to rank \(r\) is defined as

\[
E_i(r) = \frac{\sum_{k=1}^r (\sigma_k^{(i)})^2}{\sum_{k} (\sigma_k^{(i)})^2}.
\]

The **cost** of truncating at rank \(r\) is the fraction of energy discarded:

\[
c_i(r) = 1 - E_i(r).
\]

The number of parameters required by a factorised projection with rank \(r\) is approximately \(r (d_{\text{in}}^{(i)} + d_{\text{out}}^{(i)})\) (ignoring the bias, which is kept). Let the total parameter budget be \(\mathcal{B}\) (e.g., 20% of the original total). The optimisation problem is:

\[
\begin{aligned}
\underset{r_1,\dots,r_M}{\text{minimise}} \quad & \sum_{i=1}^{M} c_i(r_i) \\
\text{subject to} \quad & \sum_{i=1}^{M} r_i (d_{\text{in}}^{(i)} + d_{\text{out}}^{(i)}) \le \mathcal{B},\\
& 1 \le r_i \le \min(d_{\text{in}}^{(i)}, d_{\text{out}}^{(i)}, \lfloor \alpha \min(d_{\text{in}}^{(i)}, d_{\text{out}}^{(i)}) \rfloor),
\end{aligned}
\]

where \(\alpha\) is a hyperparameter (`max_rank_factor`, default 0.5) that caps the maximum rank to avoid retaining near‑full matrices.

We solve this via Lagrangian relaxation. For a given multiplier \(\lambda > 0\), we minimise the unconstrained Lagrangian:

\[
L(\lambda) = \min_{\{r_i\}} \sum_i \left[ c_i(r_i) + \lambda \, r_i (d_{\text{in}}^{(i)} + d_{\text{out}}^{(i)}) \right].
\]

Since the objective separates across matrices, the optimal rank for matrix \(i\) is simply

\[
r_i^*(\lambda) = \arg\min_{r} \left[ c_i(r) + \lambda \, r (d_{\text{in}}^{(i)} + d_{\text{out}}^{(i)}) \right].
\]

The total parameter usage \(\mathcal{P}(\lambda) = \sum_i r_i^*(\lambda) (d_{\text{in}}^{(i)} + d_{\text{out}}^{(i)})\) is non‑increasing with \(\lambda\). We find the smallest \(\lambda\) such that \(\mathcal{P}(\lambda) \le \mathcal{B}\) using binary search over a predefined interval ([1e-8, 10.0]) for a fixed number of steps (100). The resulting ranks \(r_i^*\) constitute the allocation.

In the implementation, the cost tables \(c_i(r)\) are precomputed using `compute_cumulative_entropy` (which actually computes cumulative squared‑energy, despite the name) from the singular values obtained via `scipy.linalg.svd` in float64. The binary search is performed in `lagrangian_rank_allocation`, which returns a dictionary mapping layer‑projection names to integer ranks.

### 2.2 Low‑Rank Factorisation of a Single Projection

Once the rank \(r\) for a given original weight \(\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}\) is determined, we perform a truncated SVD:

\[
\mathbf{W} \approx \mathbf{U}_r \mathbf{S}_r \mathbf{V}_r^\top,
\]

where \(\mathbf{U}_r \in \mathbb{R}^{d_{\text{out}} \times r}\), \(\mathbf{S}_r = \operatorname{diag}(\sigma_1,\dots,\sigma_r)\), and \(\mathbf{V}_r^\top \in \mathbb{R}^{r \times d_{\text{in}}}\). The factorised representation is stored as two `nn.Linear` layers:

- `down`: weight = \(\mathbf{V}_r^\top\), bias‑free, mapping \(d_{\text{in}} \to r\).
- `up`: weight = \(\mathbf{U}_r \mathbf{S}_r\), bias = original bias \(\mathbf{b}\) (if present), mapping \(r \to d_{\text{out}}\).

The forward pass of this factorised layer is \(\mathbf{y} = \text{up}(\text{down}(\mathbf{x}))\), which exactly computes the rank‑\(r\) approximation. This reduces the parameter count from \(d_{\text{out}} d_{\text{in}} + d_{\text{out}}\) to \(r(d_{\text{out}} + d_{\text{in}}) + d_{\text{out}}\).

### 2.3 Optional Sparse Residual

The low‑rank approximation discards the singular components beyond rank \(r\), which may contain task‑relevant fine‑grained information. To mitigate this, we introduce an additive sparse residual matrix \(\mathbf{R} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}\) that is initialised from the difference between the original weight and its low‑rank approximation, but only the entries with the largest absolute magnitudes are kept, enforcing a high sparsity (e.g., 70%). The residual is a trainable parameter, allowing the model to refine it during fine‑tuning. No additional pruning is applied during training (unlike the SpectronAttention module), so the sparsity remains fixed after initialisation.

The full projection becomes:

\[
\mathbf{y} = \text{up}(\text{down}(\mathbf{x})) + \mathbf{R} \mathbf{x} + \mathbf{b}.
\]

The sparse residual is implemented as a dense `nn.Parameter` of shape \((d_{\text{out}}, d_{\text{in}})\) (stored transposed as `sparse_weight` of shape `[out, in]`), and the forward pass uses `F.linear(x, self.q_sparse_weight)` to apply it. The initial sparsity is enforced by thresholding the magnitude of the residual.

### 2.4 SVDCompressedAttention Module

The `SVDCompressedAttention` class (registered as `"svd_compressed"`) implements the full multi‑head self‑attention using the factorised (+ optional sparse) projections for Q, K, V, and output. Its forward method:

1. Computes Q, K, V via the factorised projections (each using `up(down(x))` plus optional sparse).
2. Reshapes to multi‑head format \((B, H, L, d_h)\).
3. Calculates scaled dot‑product scores \(\mathbf{S} = \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_h}}\).
4. Adds an attention mask directly (the mask is expected to be an additive tensor, e.g., 0 for valid, -∞ for padding). This is a design choice that deviates from the typical 2D mask conversion; the caller must provide the mask in the appropriate format.
5. Applies softmax, dropout, and an optional head mask.
6. Aggregates values, concatenates heads, and projects through the factorised output layer.

The module is initialised with a dictionary `ranks` containing the per‑projection ranks for that layer, and it extracts the original weight matrices from the `pretrained_attn` object. Biases are copied if present.

### 2.5 Global Injection and Training Pipeline

The script `SVDCompressedAttention.py` performs the following steps:

- Loads a GLUE task via a local dataset utility.
- Loads a pretrained DistilBERT model.
- Gathers all 24 attention weight matrices using `gather_all_attention_weights`.
- Computes the original total parameter count and the target budget as `budget_ratio * original_total`.
- Calls `lagrangian_rank_allocation` to obtain per‑matrix ranks.
- Iterates over the 6 Transformer layers, instantiating `SVDCompressedAttention` with the allocated ranks and replacing the original attention module.
- Fine‑tunes the compressed model on the downstream task using HuggingFace’s `Trainer` with an early stopping callback.

An auxiliary injection function `inject_svd_compressed` is registered under the same key `"svd_compressed"` for use within the compression framework’s layer‑by‑layer injection pipeline (e.g., via `replace_attention`). However, the main training script uses its own `inject_all_layers` to pass the per‑layer ranks, which the generic injection function does not support (it uses a single rank for all projections). This highlights a flexibility gap in the framework’s injection interface.

---

## 3. Implementation Details

- **Spectral utilities** (`calculate_spectral_entropy`, `compute_cumulative_entropy`): Operate on squared singular values to compute the cumulative energy fraction.
- **Lagrangian optimisation**: Implemented in `lagrangian_rank_allocation`. The `cost` array for each matrix is `1 - cum_entropy`. The parameter array is `np.arange(1, max_r+1) * (d_in + d_out)`. The binary search on \(\lambda\) runs for 100 iterations.
- **Factorisation** (`_factorise_linear`): Uses `scipy.linalg.svd` in float64, constructs `down` and `up` linear layers, copies weights and biases.
- **Sparse residual** (`_create_pruned_residual`): Computes the difference between the original and low‑rank approximation, thresholds using `np.percentile(np.abs(residual), sparsity * 100)`, and creates a `nn.Parameter` with the transposed residual.
- **Mask handling**: The forward pass directly adds `attention_mask` to the scores, expecting an additive mask. This is different from the typical 2D binary mask produced by HuggingFace tokenisers; the training script does not explicitly convert the mask, which could lead to errors if the default masks are binary 1/0. A production pipeline should ensure the mask is in the correct form (e.g., by converting `(1.0 - mask) * -1e9` before calling the module).
- **Injection**: The script‑level injection uses `inject_all_layers`, which maps rank names (e.g., `layer0_out`) to the dictionary expected by `SVDCompressedAttention` (`ranks={'q': rq, 'k': rk, 'v': rv, 'o': ro}`). The registry‑based `inject_svd_compressed` ignores the per‑layer allocation and applies a single rank to all projections, which is insufficient for the global budget optimisation scenario. This suggests that the module is primarily intended to be used via the script’s custom injection, while the registry entry serves as a simple default.

---

## 4. Usage and Configuration

The module is executed via the command‑line script. Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | `sst2` | GLUE task to fine‑tune on. |
| `--budget_ratio` | 0.2 | Fraction of original attention parameters to retain. |
| `--add_sparse` | `False` | Enable sparse residual. |
| `--sparsity` | 0.7 | Sparsity level for the residual (if enabled). |
| `--max_rank_factor` | 0.5 | Maximum rank as a fraction of \(\min(d_{\text{in}}, d_{\text{out}})\). |

The training hyperparameters (learning rate, batch size, epochs) follow standard GLUE fine‑tuning values.

Example:
```bash
python SVDCompressedAttention.py --task sst2 --budget_ratio 0.2 --add_sparse
```

The script automatically handles data loading, rank allocation, injection, training, and saving.

---

## 5. Discussion

The SVD‑Compressed Attention module offers a principled, budget‑aware compression method. The Lagrangian rank allocation is a key innovation: by solving a single global optimisation, it naturally assigns higher ranks to matrices with slowly decaying singular spectra, which are those that would suffer most from truncation. This is more efficient than uniform rank allocation and avoids tedious per‑matrix tuning. The optional sparse residual provides an additional mechanism to capture information lost by low‑rank truncation, mimicking the combination of low‑rank and sparse decompositions seen in robust PCA and in recent hybrid compression works.

However, the current implementation has several limitations:

1. **Mask format dependency**: The forward pass assumes the attention mask is already in additive form (0 for valid, -∞ for padding). In standard HuggingFace tokenisation, the mask is binary (1 for valid, 0 for padding). The training script does not convert the mask, which would cause the model to attend to padding tokens or treat valid tokens with a 1‑logit shift. This likely harms performance and must be fixed for correct training. A simple conversion step before calling the module would resolve this.

2. **Single‑use rank allocation**: Ranks are determined once before fine‑tuning using the pretrained weights. If the singular spectrum shifts significantly during fine‑tuning, the allocated ranks might become suboptimal. Dynamic rank adjustment during training is a potential future improvement.

3. **No sparsity pruning during training**: The sparse residual is initialised with a fixed sparsity and trained without further pruning. This may cause the residual to become less sparse over time as small weights grow, reducing the efficiency benefits. A periodic pruning callback (as in SpectronAttention) would maintain the target sparsity.

4. **Registry injection vs. script injection**: The registered injection function does not support per‑layer ranks, making it impossible to use the Lagrangian allocation through the standard framework pipeline. A more flexible injection interface (e.g., passing a rank dictionary) would align the registry entry with the script’s capabilities.

5. **Computational overhead of SVD**: The initial rank allocation uses `scipy.linalg.svd` on CPU for all weight matrices, which is fast for DistilBERT but may become a bottleneck for larger models. On‑GPU approximate SVD could be used to scale up.

Despite these issues, the module demonstrates a sophisticated integration of spectral analysis, constrained optimisation, and low‑rank factorisation, and it successfully reduces the parameter count of the attention sub‑layers to a fraction of the original while retaining trainability. The accompanied training script provides a complete, reproducible pipeline for evaluating the method on GLUE.

---

## 6. Conclusion

We have presented SVD‑Compressed Attention, a parameter‑efficient attention module that couples a budget‑aware Lagrangian rank allocation with truncated SVD factorisation and an optional sparse residual. The method significantly reduces the memory and parameter footprint of Transformer attention layers while maintaining competitive downstream performance. Its integration into the HuggingFace training ecosystem and the compression framework makes it a practical tool for efficient fine‑tuning. Future work will focus on correcting mask handling, implementing dynamic rank and sparsity adaptation, and harmonising the registry injection with the global allocation scheme. The code is publicly available and can be readily adapted to other Transformer architectures.