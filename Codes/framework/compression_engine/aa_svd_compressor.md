# AASVD: Anchored Adaptive SVD Compression

**Registered as `aasvd`**  
*Data‑aware low‑rank compression guided by input activation directions.*

---

## Overview

AASVD (Anchored Adaptive Singular Value Decomposition) is a post‑training compression method that reduces the parameter count of linear layers in a transformer model by replacing each weight matrix with a low‑rank approximation. Unlike standard SVD, which treats all directions equally, AASVD first identifies the principal directions of the **input activations** (collected from a calibration dataset) and restricts the low‑rank approximation to the subspace spanned by those directions. This “anchoring” ensures that the compressed weight matrix retains the components most relevant to the actual data distribution, often preserving accuracy better than plain SVD at the same rank.

The module is designed as a drop‑in compressor for the DistilBERT compression framework. It uses a small set of calibration texts (e.g., a single sentence) to capture activation statistics, then performs a two‑stage SVD that is both **anchored** (guided by input directions) and **adaptive** (tailored to each layer’s activations).

---

## Methodology

### 1. Activation Capture

During a single forward pass on the calibration texts, hooks attached to every `nn.Linear` module record the input tensors (the pre‑activation). For a linear layer with weight matrix **W** ∈ ℝ<sup>out×in</sup>, the captured input **X** has shape `(batch × seq_len, in)`. This matrix represents the empirical distribution of input vectors.

### 2. Anchor Directions

The input matrix **X** is decomposed via SVD:

```
X = U_x Σ_x V_x^T
```

The first `r` left singular vectors **U_x[:, :r]** (size `in × r`) capture the principal directions of the input variation. These `r` vectors form the **anchor subspace**.

### 3. Anchored Weight Projection

The original weight matrix **W** is projected into the anchor subspace and back:

```
W_anchored = W · A · A^T
```

where **A** = **U_x[:, :r]**. This projection keeps only the components of **W** that act within the anchor subspace; the remaining components (orthogonal to **A**) are discarded. If the input activations never explore certain directions, those weight components are less critical.

### 4. Low‑Rank Approximation

The anchored weight matrix **W_anchored** (now of size `out × in`, but with rank at most `r`) is factorized via SVD:

```
W_anchored = U S V^T
```

A truncated rank `r' = min(rank, out, in, original_rank)` is applied:

```
W_approx = U[:, :r'] · diag(S[:r']) · V[:r', :]
```

This low‑rank matrix replaces the original `module.weight.data`. Because the anchoring already filtered irrelevant directions, the subsequent truncation discards even smaller singular values in a data‑aware manner.

### Why This Works

- **Data awareness**: Plain SVD treats all input dimensions equally; if some input directions never occur (or are noise), their corresponding weight columns can be pruned. AASVD uses a calibration set to identify the “active” subspace.
- **Preservation of relevant signal**: By projecting **W** onto the anchor subspace before SVD, we force the low‑rank approximation to remain faithful to the parts of **W** that actually interact with real inputs. This tends to retain more task‑critical information than generic SVD compression.
- **Lightweight calibration**: A single short sentence can be enough to capture the activation statistics for many language tasks, because the principal directions often correspond to general syntactic/semantic patterns.

---

## Implementation Details

### AASVDCompressor Class

Inherits from `BaseCompressor` and requires the model, a tokenizer, a target rank, and optional calibration texts.

```python
class AASVDCompressor(BaseCompressor):
    def __init__(self, model, tokenizer, rank=64, calibration_texts=None, **kwargs):
        ...
```

- **calibration_texts**: list of strings used to collect activations. Default: `["The quick brown fox jumps over the lazy dog."]`.
- **rank**: target rank for the anchored SVD (default 64).

The `compress()` method:

1. Sets model to eval mode.
2. Registers forward hooks on all `nn.Linear` modules to capture their inputs.
3. Tokenizes the calibration texts and runs a forward pass (no gradients).
4. Removes hooks.
5. For each linear layer that received an input:
   - Flattens the captured activation to shape `(N, in_features)`.
   - Computes SVD of the input matrix, taking the first `rank` left singular vectors as the anchor `A`.
   - Projects the original weight: `W_anchored = W @ A @ A.T` (using NumPy for the SVD part after moving to CPU).
   - Computes SVD of `W_anchored`, truncates to `r = min(rank, min(W.shape), len(S))`.
   - Constructs `W_approx` and writes it back to `module.weight.data` (converting to original dtype and device).
6. Updates internal parameter count and returns the compressed model with stats.

**Important**: The code uses NumPy for the weight SVD (`np.linalg.svd`). This requires moving the weight to CPU, computing the SVD, and moving back, which is acceptable for post‑training compression but may be slow for very large models. All other operations (hooks, input SVD) stay in PyTorch.

### Factory Function

```python
@COMPRESSOR_REGISTRY.register("aasvd")
def aasvd_factory(model, comp_config):
    rank = getattr(comp_config, 'aasvd_rank', 64)
    tokenizer = getattr(model, 'tokenizer', None)
    if tokenizer is None:
        raise ValueError("aasvd requires tokenizer attached as model.tokenizer")
    compressor = AASVDCompressor(model, tokenizer, rank=rank)
    compressed_model, _ = compressor.compress()
    return compressed_model
```

The factory retrieves the rank from `comp_config.aasvd_rank` and the tokenizer from `model.tokenizer`. It is mandatory to attach the tokenizer to the model before compression (e.g., `model.tokenizer = tokenizer`).

### Registration

The class and the factory are both registered under the name `"aasvd"` – the registry will use the factory function when called via the standard compression pipeline (since the factory takes precedence). The class registration is retained for potential direct instantiation.

---

## Usage and Configuration

Enable AASVD compression by setting the compression method to `"aasvd"` and ensuring the model has a tokenizer attribute.

### Example

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model.tokenizer = tokenizer  # attach tokenizer

# In compression config:
comp_config.comp_method = "aasvd"
comp_config.aasvd_rank = 32
comp_config.calibration_texts = [  # optional
    "This is a sample sentence.",
    "Another example for calibration."
]

# Then invoke the compression framework
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `aasvd_rank` | 64 | Target rank for the anchored SVD. |
| `calibration_texts` | `["The quick brown fox..."]` | List of strings used to capture activations. |
| (via model) `tokenizer` | required | Tokenizer attached to the model (used for encoding calibration texts). |

**Note**: If `model.tokenizer` is missing, the factory will raise a `ValueError`. The calibration texts should be representative of the target domain for best results, though even a single generic sentence often works well.

---

## Discussion

### Advantages

- **Higher accuracy retention**: By focusing on the input subspace, AASVD often outperforms standard SVD compression at the same rank, especially when the input data has a low‑rank structure.
- **Minimal calibration**: One or two sentences suffice, making the method fast and easy to apply.
- **No fine‑tuning**: Purely algebraic; the model remains in its pretrained form, just with compressed weights.
- **Transparent**: The model structure does not change—still standard `nn.Linear` layers.

### Limitations

- **Tokenizer dependency**: Requires the tokenizer to be attached to the model, which may not be the default in some pipelines.
- **CPU–GPU transfer**: The weight SVD is performed on CPU via NumPy, causing data movement. For huge matrices, this may be memory‑intensive and slow.
- **Single‑sample statistics**: The default calibration text is minimal; if it does not cover the input variety (e.g., for multilingual models), the anchor may be suboptimal. Using a more diverse calibration set is recommended for such cases.
- **No inference speedup**: The weight matrix remains a dense low‑rank approximation, not factorized into two smaller layers. Actual FLOPs are unchanged unless the low‑rank factors are extracted and used explicitly.
- **Orthogonal projection**: The anchor is an orthogonal matrix, so `A @ A.T` is a projection onto the subspace. This operation discards information orthogonal to the input’s principal directions, which could harm performance if those directions are actually used (e.g., for out‑of‑domain data).

### Recommendations

- For best results, provide a few dozen calibration sentences that cover the typical inputs.
- If the model will be fine‑tuned after compression, AASVD can serve as a strong initialization; the fine‑tuning will recover any lost orthogonal components.
- Combine with factorization (replacing the linear layer with two smaller layers) to actually reduce inference latency.

---

## Conclusion

AASVD introduces a simple yet effective twist to weight matrix compression: using input activation directions to anchor the low‑rank approximation. This data‑aware method bridges the gap between generic SVD and fully trained factorization, offering a practical trade‑off between compression ratio and accuracy retention. Its integration into the compression framework via the `"aasvd"` entry point makes it a convenient tool for shrinking transformer models with minimal performance degradation.