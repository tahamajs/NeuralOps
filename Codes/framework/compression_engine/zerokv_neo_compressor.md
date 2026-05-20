# ZeroKV‑Neo: L2‑Norm Based KV Cache Eviction with Attention Sink Preservation – Adapted for Attention Weight Pruning

**Registered as `zerokv_neo`**  
*A structured pruning method for attention projection layers, inspired by token‑level KV cache compression: it retains the most important output neurons (rows) based on their L2 norm, explicitly preserving a few “attention sink” neurons and a fraction of “heavy hitters”.*

---

## Overview

ZeroKV‑Neo is a post‑training compression technique that selectively prunes entire output features of attention‑related linear layers (queries, keys, values). The pruning criterion is borrowed from **KV cache eviction strategies**: in efficient KV cache management, tokens with high L2 norm of their key/value vectors are deemed important, and a small set of initial tokens (called *attention sinks*) are always kept because they absorb disproportionate attention. Translating this idea to weight compression, ZeroKV‑Neo treats each **output neuron** (row of the weight matrix) as analogous to a token. The method:

1. Computes the L2 norm of each row of the weight matrix as its “importance”.
2. Keeps a fixed number of **attention sink** neurons – those with the very highest L2 norms (default 4).
3. Additionally keeps a fraction of **heavy hitter** neurons – those with high L2 norms beyond the sinks (default 10% of total output features).
4. Zeros out the remaining rows, effectively pruning the corresponding output dimensions.

The result is a sparse weight matrix where only a subset of output neurons are active, reducing the effective capacity of the layer. The method requires no calibration data, is purely weight‑based, and can be applied to any model containing attention projection layers (names matching `q_lin`, `k_lin`, `v_lin`, etc.). It is registered under `"zerokv_neo"` and controlled by three hyper‑parameters: the number of attention sinks, the heavy‑hitter ratio, and a `keep_ratio` (currently unused in the actual selection logic, see discussion).

---

## Methodology

### 1. Analogy Between KV Cache Tokens and Weight Neurons

In autoregressive generation, the KV cache stores key and value vectors from previous tokens. To limit memory, various strategies evict “unimportant” tokens, retaining those with high L2 norm (simple yet effective) and always preserving a few initial tokens (attention sinks) that were found to be crucial for maintaining model quality (H2O, StreamLLM). ZeroKV‑Neo mirrors this: each row of the weight matrix **W** (output neuron) corresponds to a “token” in parameter space. The L2 norm of a row measures the magnitude of that neuron’s connection to all input features, serving as a simple proxy for its overall contribution.

### 2. Importance Scoring via Row L2 Norm

For a weight matrix **W** ∈ ℝ<sup>d_out × d_in</sup>, the importance of the `i`‑th output neuron is:

```
s_i = ‖W[i, :]‖₂
```

These scores are computed over the full‑precision weight (float32).

### 3. Selection of Neurons to Keep

Three groups of neurons are retained:

- **Attention sinks** – the top `num_attention_sinks` (default 4) neurons with the highest L2 norm. They are analogous to the initial tokens always kept in a KV cache; they serve as “anchor” outputs that are always active.
- **Heavy hitters** – a further `n_heavy = max(1, floor(d_out * heavy_hitter_ratio))` neurons are kept, chosen as the next highest L2 norm values after the sinks. Together with the sinks, the total number of kept neurons is `n_keep = num_sinks + n_heavy`, capped at `d_out`.

The remaining `d_out - n_keep` neurons are pruned (their rows are zeroed).

*Note:* The parameter `keep_ratio` is provided in the constructor but is **not used** in the actual computation; the number of neurons kept depends solely on `num_attention_sinks` and `heavy_hitter_ratio`. This appears to be an inconsistency; the intended design might have been `keep_ratio` to determine total keep count, but it is not implemented.

### 4. Soft Pruning (Row Zeroing)

The selected neurons’ rows are kept unchanged, while all other rows are multiplied by zero using a binary mask:

```
mask = zeros(d_out)
mask[top_indices] = 1
W_pruned = W * mask.unsqueeze(1)   # broadcast over input dimension
```

The pruned weight matrix is written back in its original data type. Biases are not modified; pruned neurons will still have non‑zero biases, which could cause a constant offset. A more precise implementation would also prune the biases, but this is not done.

### 5. Scope of Compression

Only linear layers whose names contain attention‑projection keywords (e.g., `q_lin`, `k_lin`, `v_lin`, `query`, `key`, `value`, `attention`) are compressed. This targets the query, key, and value projections in multi‑head attention, leaving other linear layers (e.g., feed‑forward networks, output projections) untouched. This aligns with the KV cache analogy: the pruned rows correspond to whole attention head dimensions.

---

## Complexity Analysis

### A. Compression Time Complexity

For each targeted linear layer of size `d_out × d_in`:

- **Row norm computation**: O(d_out · d_in) operations.
- **Top‑k selection**: O(d_out log d_out) via sorting or O(d_out + k log d_out) via `torch.topk`.
- **Mask and multiplication**: O(d_out · d_in).
Overall, compression time per layer is linear in the number of parameters. For a typical model, it completes in a fraction of a second.

### B. Inference Complexity

After compression, the weight matrices are still dense (same shape), but many rows contain zeros. Standard matrix multiplication `y = x·W^T` still performs the full `d_out · d_in` operations, **no FLOP reduction** occurs. The method yields structured sparsity that can be exploited only if the inference engine skips zero rows or the layer is replaced by a smaller one with reduced output dimensions.

### C. Memory Usage

- **During compression**: negligible extra memory (temporary arrays for norms and indices).
- **After compression**: the weight tensor size is unchanged (`d_out × d_in`). No memory reduction. To actually reduce model size, the zero rows would need to be removed, which would require adjusting subsequent layers to expect fewer input features.

Thus, ZeroKV‑Neo primarily produces a **structured sparse pattern** that can guide further compression steps (e.g., dimension reduction) or be used for analysis.

---

## Implementation Details

### ZeroKVNeoCompressor Class

```python
class ZeroKVNeoCompressor(BaseCompressor):
    def __init__(self, model, keep_ratio=0.3, num_attention_sinks=4,
                 heavy_hitter_ratio=0.1, **kwargs):
        ...
```

- **`keep_ratio`**: (Unused) Intended fraction of total neurons to keep; currently overridden by sinks+heavy.
- **`num_attention_sinks`**: Number of top‑norm neurons always preserved (default 4).
- **`heavy_hitter_ratio`**: Fraction of output features to retain as heavy hitters (default 0.1, i.e., 10%).

**`compress()` method**:
1. Iterates over all `nn.Linear` modules.
2. Filters those whose names contain attention‑related substrings (case‑insensitive).
3. For each:
   - Computes row norms `torch.norm(W.float(), dim=1)`.
   - Determines `n_sinks = min(num_attention_sinks, out_f)`.
   - Determines `n_heavy = max(1, int(out_f * heavy_hitter_ratio))`.
   - Total keep = `min(n_sinks + n_heavy, out_f)`.
   - Finds the top‑`total_keep` indices by norm.
   - Creates a binary mask and zeroes out other rows.
4. Updates `_compressed_params` (which will still count the full weight size) and returns stats.

**`_is_attention_proj(name)`** static method checks for keywords `q_lin, k_lin, v_lin, q_proj, k_proj, v_proj, query, key, value, attention` in the layer name.

### Factory Function

```python
@COMPRESSOR_REGISTRY.register("zerokv_neo")
def zerokv_neo_factory(model, comp_config):
    keep_ratio = getattr(comp_config, 'zerokv_keep_ratio', 0.3)
    num_sinks = getattr(comp_config, 'zerokv_num_sinks', 4)
    heavy_ratio = getattr(comp_config, 'zerokv_heavy_ratio', 0.1)
    compressor = ZeroKVNeoCompressor(model, keep_ratio=keep_ratio,
                                     num_attention_sinks=num_sinks,
                                     heavy_hitter_ratio=heavy_ratio)
    compressed_model, _ = compressor.compress()
    return compressed_model
```

Parameters `zerokv_keep_ratio`, `zerokv_num_sinks`, `zerokv_heavy_ratio` can be set in the compression configuration. Note that `zerokv_keep_ratio` is accepted but ignored in the current logic.

---

## Usage and Configuration

Set compression method to `"zerokv_neo"` and optionally tune the sinks and heavy‑hitter ratio.

### Example

```python
comp_config.comp_method = "zerokv_neo"
comp_config.zerokv_num_sinks = 8          # keep 8 attention sink neurons
comp_config.zerokv_heavy_ratio = 0.05     # keep 5% of heavy hitters
```

Then run the compression pipeline. Attention projection layers will be pruned accordingly.

### Parameters

| Parameter                 | Default | Description                                                                                                 |
|---------------------------|---------|-------------------------------------------------------------------------------------------------------------|
| `zerokv_keep_ratio`       | `0.3`   | **Currently unused** in the pruning logic. May be intended for future use.                                  |
| `zerokv_num_sinks`        | `4`     | Number of attention sink neurons (top L2 norm rows) to always keep.                                         |
| `zerokv_heavy_ratio`      | `0.1`   | Fraction of output features to keep as heavy hitters (in addition to sinks). Must be between 0 and 1.        |

**Note**: The total fraction of kept neurons is `(num_sinks + heavy_ratio * d_out) / d_out`. For a 768‑dimensional projection, with `num_sinks=4` and `heavy_ratio=0.1`, about 80 neurons are kept (≈10.4%). The `keep_ratio` value is not used.

---

## Discussion

### Advantages

- **Simple and fast**: L2 norm is extremely cheap to compute; no calibration or training needed.
- **Interpretable**: The retained neurons correspond to large‑magnitude weights, which are intuitively more important.
- **Controlled sparsity**: Explicitly preserves a few top neurons (sinks) and a fraction of heavy hitters, aligning with proven KV cache eviction strategies.
- **No data required**: Works purely on weight statistics.

### Limitations

- **No memory or FLOP reduction**: The weight remains dense; the sparsity does not reduce computation unless zero rows are physically removed.
- **`keep_ratio` ignored**: The parameter is misleading; the actual keep count is determined by sinks and heavy ratio.
- **Bias not pruned**: The corresponding bias entries remain non‑zero, potentially causing a shift in the output distribution.
- **Global sinks count**: The same number of sinks is used for all layers, which may not be optimal for layers of different sizes.
- **Neglects interactions**: L2 norm is a rough importance measure; neurons with moderate norm but crucial directions might be pruned.
- **Only attention projections**: Other linear layers (e.g., output projections, FFN) are not compressed, missing additional savings.

### Comparison with Other Structured Pruners

| Method         | Importance Criterion | Keeps                  | Parameter Reduction | Data‑Free |
|----------------|----------------------|------------------------|---------------------|-----------|
| ZeroKV‑Neo     | L2 norm of rows      | Top sinks + heavy hitters | No (weight unchanged)| Yes       |
| HybridKV       | Weight variance      | Top‑k by norm (static/dynamic) | No (weight unchanged) | Yes       |
| CacheNotes     | Task heuristic + norm| Keep ratio of rows     | No                  | Task type |
| FLAP           | Weighted input variance | Top fraction of columns | Yes (if alignment correct) | Yes (calibration) |

ZeroKV‑Neo is distinguished by its direct inspiration from KV cache eviction policies, treating neurons like tokens. It is a lightweight, interpretable baseline for structured attention pruning.

---

## Conclusion

ZeroKV‑Neo (registered as `"zerokv_neo"`) translates successful KV cache compression heuristics—L2 norm importance and attention sink preservation—into a weight pruning strategy for attention projections. By retaining only the most “token‑like” output neurons, it induces structured sparsity with minimal computational overhead. While the current implementation does not directly reduce model size or inference cost, it provides a sparse weight pattern that can guide more aggressive dimension reduction and serves as a unique, theory‑driven pruning baseline. Future improvements could incorporate bias pruning, adaptive sink counts per layer, and follow‑up steps to physically remove the zeroed rows for actual memory and computational savings.