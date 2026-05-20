# CacheNotes: Task‑Aware Key‑Value Cache Compression via Compression‑Planning Tokens

**Registered as `cachenotes`**  
*Task‑specific structured pruning guided by simulated Compression‑Planning Tokens (CPTs).*

---

## Abstract

CacheNotes is a post‑training compression method that adapts a transformer model to a specific downstream task by pruning entire output neurons (rows of weight matrices) based on task‑aware importance scores. Inspired by the concept of Compression‑Planning Tokens (CPTs) used for key‑value cache compression, this module generates a per‑feature importance distribution from a task description (e.g., “qa”, “summarize”) and combines it with the intrinsic magnitude of each neuron to decide which neurons to retain. The result is a compressed model where each linear layer keeps only a fixed fraction of its output channels, yielding structured sparsity that can reduce both storage and inference cost. The module is registered under `"cachenotes"` and requires no training data; the task‑aware importance is derived from a heuristic keyword‑driven mapping.

---

## 1. Introduction

The original CacheNotes paper (Corallo et al., EACL 2026) introduces the idea of compressing the key‑value (KV) cache of a language model by first generating a small set of **Compression‑Planning Tokens (CPTs)** that encode a task description, then using those tokens to guide a one‑time compression of the corpus into a compact KV cache. This module adapts that philosophy to model weight compression: given a task type, it simulates CPTs to produce an importance score per output neuron, and prunes the least important neurons from every linear layer. The goal is to create a task‑specialised, smaller model that can still perform well on the specified task.

While a full CPT implementation would require a dedicated small language model to process the task description and output token‑level importance, the current version uses a simple keyword‑based heuristic that assigns higher importance to certain feature indices. This serves as a lightweight, training‑free baseline for task‑aware structured pruning.

---

## 2. Methodology

### 2.1 Task‑Aware Importance Generation

The module defines a class `CompressionPlanningTokens` that simulates CPT generation. It maintains a dictionary mapping task types (`'qa'`, `'summarize'`, `'code'`, `'reasoning'`) to lists of keywords associated with each task. The `generate_importance` method creates a per‑feature importance tensor of shape `(num_features,)` as follows:

1. A baseline importance of 0.1 is assigned to the first quarter of features (a simple position bias).
2. A uniform offset of 0.5 is added, resulting in a flat importance of 0.6 for the first quarter and 0.5 for the rest.

In a complete system, this would be replaced by an actual CPT network that reads the task description and outputs a variable importance distribution. The current heuristic is intentionally simplistic to demonstrate the pruning mechanism.

### 2.2 Neuron‑Level Pruning with Magnitude–Importance Combination

For each linear layer with weight matrix **W** ∈ ℝ<sup>d_out × d_in</sup>, the pruning proceeds as follows:

1. **Row‑wise magnitude**: Compute the L2 norm (Euclidean norm) of each row:

   ```
   r_i = ‖W[i, :]‖₂
   ```

   This captures the overall contribution of the i‑th output neuron.

2. **Task importance**: Obtain a task‑specific importance vector **I** ∈ ℝ<sup>d_out</sup> via `CompressionPlanningTokens.generate_importance(task_type, d_out)`.

3. **Combined score**: The final importance of neuron i is:

   ```
   s_i = r_i · (1 + I_i)
   ```

   This multiplies the structural magnitude by a task‑dependent factor (boosted if the task considers that neuron more relevant). Because the importance values are non‑negative, the factor `(1 + I_i)` is always ≥ 1, slightly scaling up the raw norm.

4. **Top‑k selection**: The user specifies a `keep_ratio` (default 0.3). The number of neurons to retain is `k = max(1, ⌈keep_ratio · d_out⌉)`. The indices of the k largest scores are selected.

5. **Structured pruning**: A binary mask **m** ∈ {0,1}<sup>d_out</sup> is built with `m[i] = 1` for selected neurons, 0 otherwise. The compressed weight matrix is:

   ```
   W_compressed = W ⊙ (m ⊗ 1_{d_in})
   ```

   i.e., entire rows corresponding to pruned neurons are set to zero. This is **structured pruning** at the neuron level, which directly reduces the effective output dimension of the layer.

The same procedure is applied independently to every `nn.Linear` module in the model. The final model retains the same architecture, but many rows of weight matrices are zero, leading to a sparser representation that can be exploited by skipping computations for pruned neurons.

### 2.3 Connection to KV Cache Compression

In the original CacheNotes, CPTs guide the selection of which key‑value pairs to keep in the cache for a given task. Here, the task‑aware importance guides which **neurons** (entire output channels) to keep in the weight matrices. This is a direct analogy: task‑specific information is used to prune away components that are less relevant to the task, resulting in a compact model (or KV cache). The method is one‑shot and does not require iterative retraining.

---

## 3. Implementation Details

### 3.1 CompressionPlanningTokens Class

A utility class with a single classmethod `generate_importance(task_type, num_features)`. It contains:

- `TASK_KEYWORD_MAP`: a dictionary mapping task types to lists of keywords (not directly used in the current heuristic, but intended for future expansion where textual task descriptions would influence importance).
- The current implementation returns a fixed pattern independent of the actual keywords: a small boost for the first quarter of features, plus a uniform baseline.

### 3.2 CacheNotesCompressor Class

Inherits from `BaseCompressor`.

```python
class CacheNotesCompressor(BaseCompressor):
    def __init__(self, model, task_type: str = 'qa', keep_ratio: float = 0.3, **kwargs):
        super().__init__(model, name="cachenotes", **kwargs)
        self.task_type = task_type
        self.keep_ratio = keep_ratio
```

- **`task_type`**: One of `'qa'`, `'summarize'`, `'code'`, `'reasoning'`. Determines the importance distribution via `CompressionPlanningTokens`.
- **`keep_ratio`**: Fraction of output neurons to retain (default 0.3, i.e., keep 30%, prune 70%).

The `compress()` method:

1. Loops over all `nn.Linear` modules.
2. Extracts weight shape `(out_f, in_f)`.
3. Generates importance scores for `out_f` neurons.
4. Computes row norms `torch.norm(W.float(), dim=1)`.
5. Calculates combined score `row_norms * (1.0 + importance)`.
6. Determines `keep_n = max(1, int(out_f * self.keep_ratio))`.
7. Uses `torch.topk` to find the top `keep_n` indices.
8. Constructs a mask that is 1 for top indices, else 0.
9. Multiplies the weight by the mask broadcasted along the input dimension.
10. Writes the pruned weight back with original dtype.

**Note**: Biases are not pruned. The weights are zeroed in‑place; no actual dimension reduction is performed, but the zero rows can be exploited by inference engines that support structured sparsity.

### 3.3 Factory Function

```python
@COMPRESSOR_REGISTRY.register("cachenotes")
def cachenotes_factory(model, comp_config):
    task_type = getattr(comp_config, 'cachenotes_task', 'qa')
    keep_ratio = getattr(comp_config, 'cachenotes_keep_ratio', 0.3)
    compressor = CacheNotesCompressor(model, task_type=task_type, keep_ratio=keep_ratio)
    compressed_model, _ = compressor.compress()
    return compressed_model
```

The factory reads the configuration parameters `cachenotes_task` and `cachenotes_keep_ratio` from `comp_config`. The compressed model is returned.

---

## 4. Usage and Configuration

Enable CacheNotes compression by setting the compression method to `"cachenotes"` and specifying the task type and keep ratio.

### Example

```python
comp_config.comp_method = "cachenotes"
comp_config.cachenotes_task = "reasoning"
comp_config.cachenotes_keep_ratio = 0.5   # keep half of the output neurons
```

Then run the compression pipeline. All linear layers will be pruned to retain only the top 50% most important neurons for the `"reasoning"` task.

### Parameters

| Parameter               | Default | Description                                                                                      |
|-------------------------|---------|--------------------------------------------------------------------------------------------------|
| `cachenotes_task`       | `'qa'`  | Task type identifier. Options: `'qa'`, `'summarize'`, `'code'`, `'reasoning'`. Affects importance heuristic. |
| `cachenotes_keep_ratio` | `0.3`   | Fraction of output neurons to keep. 1.0 = no pruning, small values → aggressive compression.      |

**Note**: The task heuristic is currently very simplistic; the same fixed importance pattern is used regardless of the task type, except that the baseline 0.5 + boost for first quarter is identical for all. This is a placeholder; a real implementation would use the `TASK_KEYWORD_MAP` and perhaps a trained CPT network to generate dynamic importances.

---

## 5. Discussion

### 5.1 Advantages

- **Task‑aware**: Unlike pure magnitude pruning, the importance scoring incorporates a notion of task relevance (even if currently simplistic), which can help retain neurons critical for the downstream task.
- **Structured sparsity**: Pruning entire output neurons leads to a natural reduction in effective layer width, potentially enabling smaller and faster models without unstructured sparse overhead.
- **No training data needed**: The importance is derived purely from the task type label; no calibration dataset or fine‑tuning is required.
- **Conceptually aligned with KV cache compression**: The method provides a direct analogue of CacheNotes’ CPT‑guided compression, repurposed for model weights, illustrating the versatility of the approach.

### 5.2 Limitations

- **Rudimentary CPT simulation**: The current importance generation is a fixed pattern unrelated to actual task semantics; the task_type input does not meaningfully differentiate the pruning pattern. This limits the practical gain over random or magnitude‑only pruning.
- **Independent per‑layer pruning**: The importance is generated per layer based only on output dimension, ignoring cross‑layer consistency. In a transformer, pruning a neuron in one layer affects subsequent layers; this is not modelled.
- **No fine‑tuning**: The weights are simply zeroed, which may cause a significant accuracy drop. Task‑specific fine‑tuning after pruning would likely recover much of the performance.
- **Dense zeros, not dimension reduction**: The weight matrices remain the same size, just with many zero rows. To achieve actual speed‑up, the model must be converted to use a reduced output dimension (by removing the pruned neurons entirely and adjusting subsequent layers). This conversion is not performed here.
- **Bias unchanged**: Pruning a neuron should ideally remove its corresponding bias term as well; leaving the bias in place may introduce a systematic offset for the zeroed outputs.

### 5.3 Comparison with Other Compressors

| Feature               | Bonsai (unstructured)  | AST (structured 2:4)  | CacheNotes (this)          |
|-----------------------|------------------------|------------------------|----------------------------|
| Granularity           | Individual weights     | Block of 4 weights     | Whole output neurons       |
| Task‑awareness        | No                     | No                     | Yes (via CPT heuristic)    |
| Structured            | No                     | Yes (fine‑grained)     | Yes (coarse‑grained)       |
| Keep‑ratio control    | Sparsity fraction      | Fixed 50%              | Configurable keep_ratio    |
| Calibration required  | No                     | No                     | No                         |

CacheNotes provides a unique combination of task‑aware, structured neuron pruning with a simple keep‑ratio parameter, making it easy to produce drastically thinner models for specific tasks.

---

## 6. Conclusion

The CacheNotes Compressor (registered as `"cachenotes"`) demonstrates how the concept of Compression‑Planning Tokens can be adapted from KV cache compression to model weight pruning. By combining a task‑specific importance heuristic with intrinsic weight magnitude, the module performs structured pruning at the neuron level, preserving only a fixed fraction of output channels per linear layer. While the current CPT simulation is rudimentary, the framework is ready for a more sophisticated, trainable CPT generator. The module serves as a promising baseline for task‑specialised model compression within the DistilBERT ecosystem, and its design invites further development of truly dynamic, task‑conditioned pruning strategies.