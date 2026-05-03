
# Weight Sharing via Hashing

These modules implement parameter‑efficient linear layers using **weight sharing** through hash functions or tile‑based grouping, dramatically reducing the number of unique trainable parameters.

## Modules

| Class | Registry Key | Description |
|-------|--------------|-------------|
| `HashedLinear` | – | Drop‑in replacement for `nn.Linear` with a fixed hash‑based weight mapping. |
| `HashedNetCompressor` | `hashed_net` | Replaces all Linear layers with `HashedLinear`. |
| `ROASTLinear` | – | Tile‑based hardware‑aware weight sharing. |
| `ROASTCompressor` | `roast` | Replaces all Linear layers with `ROASTLinear`. |

## Usage

### HashedLinear

```python
from framework.compression.hashed_linear import HashedLinear

# Create from scratch
layer = HashedLinear(768, 768, n_buckets=2048)

# Initialize from pretrained linear layer
pretrained = nn.Linear(768, 768)
compressed = HashedLinear.from_linear(pretrained, n_buckets=2048)
```

### HashedNetCompressor

```python
from framework.compression import HashedNetCompressor

compressor = HashedNetCompressor(model, n_buckets=2048)
model, stats = compressor.compress()
```

### ROASTLinear / ROASTCompressor

```python
from framework.compression.roast_hashed_linear import ROASTLinear, ROASTCompressor

# Direct layer
layer = ROASTLinear.from_linear(original_linear, n_buckets=1024, tile_size=32)

# Full model compression
compressor = ROASTCompressor(model, n_buckets=1024, tile_size=32)
model, stats = compressor.compress()
```

## How It Works

- **HashedLinear**: The weight matrix `W(i,j)` is determined by a hash function `hash(i,j)`, mapping each entry to a small weight table. The hash function is fixed (precomputed as a buffer). This reduces parameter count from `(out × in)` to `n_buckets`.
- **ROASTLinear**: Divides the weight matrix into 2D tiles (e.g., 32×32). All entries within a tile share the same value, further reducing parameters to `tiles_out × tiles_in` buckets.

## Performance Notes

- Both layers reconstruct the full weight matrix in each forward pass, so they trade memory for parameters. The computation overhead is minimal compared to the original dense layer.
- Use larger `n_buckets` for higher capacity; smaller values for extreme compression.
- ROAST is particularly efficient on hardware that prefers block‑aligned memory access.

## Parameters

| Parameter | Typical | Description |
|-----------|---------|-------------|
| `n_buckets` | 1024–4096 | Number of unique weight values (size of the weight table) |
| `tile_size` (ROAST only) | 32 | Size of square tiles for weight sharing |

---

*Hashing your way to tiny models – with almost no accuracy loss.*
