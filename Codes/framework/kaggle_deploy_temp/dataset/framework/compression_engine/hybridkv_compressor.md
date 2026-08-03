# HybridKV: Hybrid KV Cache Compression for Multimodal LLMs (Adapted for Weight Pruning)

**Registered as `hybridkv`**  
*Classifies attention projection heads into static/dynamic types and applies distinct structured pruning strategies under a hierarchical budget, inspired by hybrid KV cache compression.*

---

## Overview

HybridKV is a post‑training compression method that targets the linear projections of multi‑head attention layers (queries, keys, values). It first **classifies each attention head** (i.e., each output channel of the projection matrix) as either **static** or **dynamic** based on the variance of its weight vector. This classification is motivated by observations in KV cache compression: static heads (low variance) are responsible for encoding stable, often text‑related patterns, while dynamic heads (high variance) handle more variable, multimodal signals. Under a global parameter **budget** (fraction of output neurons to retain), the method allocates a larger share to static heads (default 70% of the budget) and a smaller share to dynamic heads. It then applies **different pruning strategies** to each type:

- **Static heads** are pruned by keeping the rows with the highest L2 norm (global importance).
- **Dynamic heads** are pruned by a **chunk‑wise top‑k** scheme, preserving local diversity across groups of channels.

This mixed strategy aims to retain the most critical components of both head types, achieving higher accuracy under a given parameter budget than a uniform pruning approach.

The module is designed for the DistilBERT compression framework and is registered under `"hybridkv"`. It requires no calibration data, operating purely on the weight matrices of attention projections.

---

## Methodology

### 1. Head Classification via Weight Variance

For a linear projection matrix **W** ∈ ℝ<sup>d_out × d_in</sup> (where `d_out` typically equals `num_heads * head_dim` or similar), each row corresponds to one output neuron (part of an attention head). The variance of each row’s weight values is computed:

```
var_i = Var(W[i, :])
```

The row variances are normalized by the maximum variance (plus a small epsilon):

```
norm_var_i = var_i / (max(var) + 1e-8)
```

A row is considered “dynamic” if its normalized variance exceeds a threshold (default 0.5). The fraction of dynamic rows in the layer is then:

```
dynamic_ratio = (norm_var > 0.5).mean()
```

If `dynamic_ratio > 0.3`, the entire head is classified as **dynamic**; otherwise, it is **static**. This heuristic captures the intuition that layers with many high‑variance rows tend to exhibit varied, input‑dependent behaviour.

### 2. Budget Allocation

The user specifies an overall `budget` (fraction of output neurons to retain, e.g., 0.3) and a `static_ratio` (fraction of the budget allocated to static heads, default 0.7). For a layer with `d_out` neurons:

- Number of neurons to keep overall: `keep_total = d_out * budget`
- For a **static** head:
  ```
  keep_static = keep_total * static_ratio   (capped at d_out)
  ```
- For a **dynamic** head:
  ```
  keep_dynamic = keep_total * (1 - static_ratio)
  ```

The actual number kept is at least 1.

### 3. Pruning Strategies

#### Static Heads: Global Top‑k by L2 Norm

Static heads are assumed to have a few dominant directions. We compute the L2 norm of each row (over input features):

```
r_i = ‖W[i, :]‖₂
```

The `keep_static` rows with the largest norms are selected, and all other rows are zeroed. This is a structured pruning: entire output neurons are removed.

#### Dynamic Heads: Chunk‑wise Top‑k

Dynamic heads are thought to encode more diverse information across different channel groups. To preserve this diversity, we partition the `d_out` neurons into **chunks of size 8** (hard‑coded). Within each chunk, we keep a proportional number of rows:

```
n_keep_chunk = max(1, floor( (chunk_size) * keep_dynamic / d_out ))
```

The kept rows within the chunk are those with the highest L2 norm. This ensures that even with a low overall budget, every chunk retains at least one representative neuron, preventing complete deletion of a region.

### 4. Application

Only linear layers whose names contain attention‑related keywords (`q_lin`, `k_lin`, `v_lin`, `query`, etc.) are compressed. For each such layer:

1. Classify as static or dynamic.
2. Compute the retention mask according to the type‑specific strategy.
3. Multiply the weight matrix by the mask (broadcast over input dimension), zeroing out pruned rows.
4. Bias is left unchanged.

After compression, the model has the same architecture, but many output channels of attention projections are zero, which can be exploited for structured sparsity acceleration.

---

## Implementation Details

### HybridKVCompressor Class

```python
class HybridKVCompressor(BaseCompressor):
    def __init__(self, model, budget=0.3, static_ratio=0.7, **kwargs):
        super().__init__(model, name="hybridkv", **kwargs)
        self.budget = budget
        self.static_ratio = static_ratio
```

- **`budget`**: Fraction of output neurons to retain overall (default 0.3, i.e., prune 70%).
- **`static_ratio`**: Fraction of the budget devoted to static heads (default 0.7).

#### `_classify_head(weight, threshold=0.5)`
- Computes row variance, normalizes, counts dynamic rows, returns `'dynamic'` if `dynamic_ratio > 0.3`, else `'static'`.

#### `compress()`
1. Iterates over all `nn.Linear` modules.
2. Filters those whose names pass `_is_attention_proj` (case‑insensitive match with attention keywords).
3. For each:
   - Determines `out_f` (number of rows).
   - Classifies head type.
   - Builds mask:
     - **Static**: `keep = max(1, int(out_f * budget * static_ratio))`, topk by row norm.
     - **Dynamic**: `keep = max(1, int(out_f * budget * (1 - static_ratio)))`, chunk‑size 8 loop, local topk.
   - Applies mask: `W_pruned = W * mask.unsqueeze(1)`.
   - Writes back to `module.weight.data` preserving dtype.
4. Updates `_compressed_params` and returns model with stats.

#### `_is_attention_proj(name)`
Returns `True` if any of the following substrings are in the lowercased name: `q_lin`, `k_lin`, `v_lin`, `q_proj`, `k_proj`, `v_proj`, `query`, `key`, `value`, `attention`.

### Factory Function

```python
@COMPRESSOR_REGISTRY.register("hybridkv")
def hybridkv_factory(model, comp_config):
    budget = getattr(comp_config, 'hybridkv_budget', 0.3)
    static_ratio = getattr(comp_config, 'hybridkv_static_ratio', 0.7)
    compressor = HybridKVCompressor(model, budget=budget, static_ratio=static_ratio)
    compressed_model, _ = compressor.compress()
    return compressed_model
```

The factory reads `hybridkv_budget` and `hybridkv_static_ratio` from `comp_config`.

---

## Usage and Configuration

Set the compression method to `"hybridkv"` and optionally adjust the budget and static ratio.

### Example

```python
comp_config.comp_method = "hybridkv"
comp_config.hybridkv_budget = 0.4          # keep 40% of neurons overall
comp_config.hybridkv_static_ratio = 0.8   # give 80% of the budget to static heads
```

Then run the compression pipeline. Attention projection layers (queries, keys, values) will be pruned according to the hybrid strategy.

### Parameters

| Parameter                  | Default | Description                                                                                                      |
|----------------------------|---------|------------------------------------------------------------------------------------------------------------------|
| `hybridkv_budget`          | `0.3`   | Overall fraction of output neurons to retain. 1.0 = no pruning, 0.1 = very aggressive.                           |
| `hybridkv_static_ratio`    | `0.7`   | Fraction of the budget allocated to static heads (the rest goes to dynamic heads). Must be between 0 and 1.      |

**Note**: The classification uses hard‑coded thresholds (0.5 for normalized variance, 0.3 for dynamic ratio). These are not configurable without modifying the code.

---

## Discussion

### Advantages

- **Head‑type awareness**: By distinguishing static vs. dynamic heads, the compression adapts to functional differences between attention projections.
- **Structured pruning**: Zeroing entire output neurons can lead to real speed‑ups on hardware that supports structured sparsity, unlike unstructured pruning.
- **No calibration data**: Works purely on weight statistics; no forward passes needed.
- **Hierarchical budget**: The two‑level budget allocation allows prioritizing the more “stable” components.
- **Easy integration**: Few configuration parameters, automatic detection of attention layers.

### Limitations

- **Heuristic classification**: The variance‑based static/dynamic split is a proxy; it may not perfectly align with actual head behaviour. A more data‑driven classification (e.g., using activations) could be more accurate but requires calibration.
- **Fixed budget allocation**: The `static_ratio` is a single global parameter; the optimal ratio may vary per layer. An adaptive per‑layer split could improve results.
- **Chunk size hard‑coded**: The chunk size of 8 for dynamic heads is fixed; different models or tasks might benefit from other sizes.
- **Only attention projections**: Other linear layers (FFN, output) are not compressed, missing potential additional savings.
- **Weight overwrite, not structural reduction**: The weight matrix remains dense with zeros; actual memory savings require storing the mask or using sparse linear algebra. Parameter count via `_count_params` may still count zeros, so the reported reduction might be optimistic unless the counting logic is aware of the mask.
- **Potential accuracy drop**: As with any pruning, fine‑tuning is often necessary to recover performance, though the hybrid strategy mitigates this.

### Comparison with Other Structured Pruners

| Method            | Granularity          | Classification | Budget Type              |
|-------------------|----------------------|----------------|--------------------------|
| HybridKV (this)   | Output neurons (rows)| Static/dynamic by weight variance | Hierarchical (global→static/dynamic) |
| CacheNotes        | Output neurons       | Task‑type heuristic | Fixed keep‑ratio        |
| FLAP              | Input features       | Activation variance | Per‑layer sequential     |

HybridKV is unique in its attempt to classify heads into static/dynamic and apply different pruning rules, drawing inspiration from multimodal KV cache compression.

---

## Conclusion

HybridKV (registered as `"hybridkv"`) brings a novel perspective to attention weight pruning by separating heads into static and dynamic categories and tailoring the pruning budget and strategy accordingly. This hierarchical, variance‑based approach aims to preserve more essential information under tight parameter budgets, especially in models where attention heads have diverse functional roles. While the current implementation relies on simple weight variance heuristics, it provides a solid foundation for future extensions that could incorporate activation‑based classification or learned budget allocation. Its seamless integration into the compression framework makes it an attractive choice for experimenting with structured attention pruning in DistilBERT.