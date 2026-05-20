# HashedNets: Weight Sharing via Hash Functions

**Registered as standalone module**  
*Dramatically reduces parameter count by sharing weights across positions using a hash table, enabling compact linear layers with minimal accuracy loss when fine‑tuned.*

---

## Overview

HashedNets is a compression technique that replaces the weight matrix of a linear layer with a **hash table** of learnable parameters. Instead of storing a full `out_features × in_features` matrix, only a small parameter vector of size `n_buckets` is kept. Each entry of the conceptual weight matrix is mapped to one of these buckets via a **fixed hash function**, so the actual weight used at position `(i, j)` is the value stored in that bucket. This approach can achieve massive parameter reduction (e.g., from millions to thousands) while retaining representational power through weight sharing. The module also provides a method to initialize the shared table from a pretrained dense weight, making it suitable for post‑training compression followed by fine‑tuning.

This implementation provides a `HashedLinear` class that directly replaces a standard `nn.Linear` layer. The hash mapping is precomputed and stored as a buffer for fast forward computation. The class is not registered via a compressor registry but can be used standalone or integrated into a compression pipeline.

---

## Methodology

### 1. Weight Sharing via Hash Mapping

Consider a linear layer with weight matrix **W** ∈ ℝ<sup>d_out × d_in</sup>. Instead of allocating `d_out * d_in` independent parameters, HashedNets allocates a small parameter vector **θ** ∈ ℝ<sup>B</sup>, where `B = n_buckets` is a user‑chosen number of buckets (e.g., 1024). The weight at position `(i, j)` is given by:

```
W_ij = θ[ h(i, j) ]
```

where `h : [0..d_out-1] × [0..d_in-1] → [0..B-1]` is a hash function. The function is deterministic, fast, and fixed. In this implementation, the hash uses two large primes and modulo:

```
h(i, j) = (i * P1 + j * P2) mod B
```

with `P1 = 2654435761`, `P2 = 2246822519`. This provides a good pseudo‑random distribution across buckets, minimizing collisions and ensuring that different weight entries are mapped uniformly.

### 2. Parameter Reduction

The original layer has `d_out * d_in` parameters (ignoring bias). The compressed layer has `B + d_out` parameters (table + bias), where `B ≪ d_out * d_in`. The compression ratio is approximately `(d_out * d_in) / B`. For a typical 768×768 layer (589k params), using B=1024 yields a reduction of ∼575×. The hash indices themselves are stored as a buffer (or can be recomputed on‑the‑fly), but they are not parameters.

### 3. Forward Computation

During the forward pass, the full weight matrix is reconstructed from the table:

```python
W = self.weight_table[self.hash_idx]   # shape (out_features, in_features)
y = x @ W.T + bias
```

The reconstruction is a simple indexing operation, which is very fast on GPUs. The gradient can flow back through the indexing to update the shared table values. Thus, the layer can be trained or fine‑tuned normally.

### 4. Initialization from a Pretrained Dense Layer

The class method `from_linear(cls, linear, n_buckets)` creates a `HashedLinear` that approximates an existing `nn.Linear`. It initializes the weight table by averaging the pretrained weights that fall into each bucket:

```
for each bucket b:
    θ[b] = mean( W_ij for all (i,j) where h(i,j) == b )
```

This provides a rough copy of the original layer’s weights into the compressed form, minimizing the initial accuracy drop before fine‑tuning. The bias is copied directly.

### 5. Fixed Hash Mapping

The hash indices are precomputed at construction and stored as a buffer `self.hash_idx` of shape `(out_features, in_features)`. They are not updated during training. This avoids recomputing the hash at each forward pass and ensures the mapping remains consistent.

---

## Implementation Details

### `fast_hash` function

```python
def fast_hash(indices_i, indices_j, n_buckets):
    return ((indices_i * _PRIME1 + indices_j * _PRIME2) % n_buckets).long()
```

Used to generate the mapping tensor. Large primes reduce regularities and collisions.

### `HashedLinear` class

```python
class HashedLinear(nn.Module):
    def __init__(self, in_features, out_features, n_buckets=1024, bias=True):
        ...
```

- **`n_buckets`**: Number of buckets in the hash table (size of shared parameter vector). Must be positive.
- **`weight_table`**: `nn.Parameter` of size `(n_buckets,)`.
- **`hash_idx`**: `torch.long` buffer of shape `(out_features, in_features)`, precomputed hash indices.
- **`bias`**: Optional parameter of size `(out_features,)`.

`forward(x)`:
- Computes `W = self.weight_table[self.hash_idx]` → shape `(out_features, in_features)`.
- Applies `F.linear(x, W, self.bias)`.

**`from_linear(cls, linear, n_buckets)`**:
1. Creates a `HashedLinear` instance with the same dimensions.
2. Copies bias if present.
3. For each bucket `b`, computes the mean of `linear.weight.data` over positions where `hash_idx == b` and sets `weight_table[b]` to that mean.

### Integration

The module is not registered as a compressor; it is meant to be used directly. However, one can easily write a simple compressor that replaces all `nn.Linear` with `HashedLinear` using `from_linear`.

---

## Usage and Configuration

### Replacing a Single Layer

```python
import torch.nn as nn
from hashednets import HashedLinear

# Original layer
original = nn.Linear(768, 768)

# Compressed version with 1024 buckets
compressed = HashedLinear.from_linear(original, n_buckets=1024)

# Use normally
x = torch.randn(32, 768)
y = compressed(x)
```

### Replacing All Linear Layers in a Model

```python
def compress_model(model, n_buckets=1024):
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            setattr(model, name, HashedLinear.from_linear(module, n_buckets))
    return model

compressed_model = compress_model(pretrained_model, n_buckets=1024)
```

After compression, fine‑tuning is strongly recommended to recover accuracy.

### Parameters

| Parameter    | Default | Description                                                                        |
|--------------|---------|------------------------------------------------------------------------------------|
| `n_buckets`  | 1024    | Number of hash buckets. Smaller → more compression, larger → better approximation. |

**Note**: The hash indices are fixed; changing `n_buckets` requires rebuilding the layer.

---

## Discussion

### Advantages

- **Extreme compression**: Reduces parameters by orders of magnitude (e.g., from 589k to ~1k).
- **Fast reconstruction**: The weight matrix is materialized by a simple indexing operation; no multiplication overhead.
- **Gradient friendly**: Standard backpropagation works; the table is learnable.
- **Pretrained weight initialization**: The `from_linear` method provides a good starting point for fine‑tuning.
- **Simplicity**: Very few lines of code; easy to integrate.

### Limitations

- **Accuracy drop**: Weight sharing inevitably discards information. Fine‑tuning is necessary to regain performance; zero‑shot use after `from_linear` will likely show significant degradation.
- **No per‑layer granularity**: All layers get the same number of buckets; important layers may need more capacity.
- **Collisions**: Some weight positions share the exact same value. If a bucket contains both important positive and negative weights, the averaged initial value may be near zero, hurting learning.
- **Hash index storage**: The hash indices buffer consumes `d_out * d_in * 4` bytes (if int32) which, for large layers, can exceed the size of the weight table itself, partially offsetting the parameter savings. However, this buffer could be regenerated on‑the‑fly to trade compute for memory.
- **No theoretical guarantee**: The hash function’s pseudo‑randomness is heuristic; performance depends on the data and architecture.

### Comparison with Other Low‑Parameter Techniques

| Method             | Approach                           | Parameter Count           | Trainability          |
|--------------------|------------------------------------|---------------------------|-----------------------|
| HashedNets (this)  | Hash table weight sharing          | `n_buckets + out`         | Yes (fine‑tuning)     |
| Low‑rank (SVD)     | Two smaller matrices               | `r * (out + in)`          | Yes (fine‑tuning)     |
| Random projection   | Fixed random matrix                | `out * in` (dense)        | No (fixed)            |

HashedNets is unique in its ability to arbitrarily compress a layer to a fixed‑size table independent of the original dimensions, whereas low‑rank methods need a rank proportional to dimensions for large matrices.

---

## Conclusion

HashedNets offers a radical compression strategy by replacing dense weight matrices with a small shared parameter table addressed via a hash function. The provided `HashedLinear` class makes it easy to integrate into existing models, and the `from_linear` initialization provides a sensible starting point for fine‑tuning. While it requires retraining to achieve good performance, the massive reduction in parameters makes it attractive for deployment on extremely memory‑constrained devices. This implementation’s vectorized design and precomputed hash indices ensure efficient execution, making it a practical tool for neural network compression. Future improvements could include learning the hash function or using multiple tables per layer to increase capacity.