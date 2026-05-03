
## 📘 README – LIFTR (Linearized Function Trick)

```markdown
# LIFTR – Linearized Function Trick Compressor

A compression method that **linearizes the entire model** by replacing:
- **ReLU**, **GELU**, **SiLU** → linear approximations (identity or scaled identity)
- **Softmax attention** → simple average pooling over values

The result is a model that behaves essentially like a large linear function, providing **maximum inference speed** at the cost of accuracy. Useful for:
- Fast baselines for latency‑critical applications.
- Edge devices where accuracy can be partially sacrificed for speed.
- Debugging and understanding the non‑linear contributions in a model.

## Usage

```python
from framework.compression import LIFTRCompressor

compressor = LIFTRCompressor(model, replace_attention=True)
linearized_model, stats = compressor.compress()
print(stats)
```

Or via CLI:

```bash
python cli.py compress --method liftr
```

## How It Works

1. Recurse through all submodules.
2. Replace any instance of `nn.ReLU` with `nn.Identity`.
3. Replace `nn.GELU` / `nn.SiLU` with a simple linear scaling `0.8 * x` (best linear fit).
4. If `replace_attention=True`, the `forward` method of every `nn.MultiheadAttention` is overwritten to simply average the value tensor – effectively ignoring queries and keys entirely.

## Parameters

- `replace_attention` (bool, default `True`): if False, only activations are linearized; the attention remains quadratic (but still loses its non‑linearity because activations are gone).

## Limitations

- This compressor is **lossy** – expect a significant drop in accuracy.
- It does not modify convolutional or other specialised layers (only common activations and attention).
- The linearized model is **not trainable** in a useful way (gradients will not reflect real non‑linear behaviour).

## When to Use

- **Extreme latency reduction**: If you need the model to run in microseconds and can afford lower quality.
- **Sanity check**: Does your task actually need non‑linearities? Linearizing can show whether the model relies on them.
- **Edge deployment**: Combine with other compressors (quantisation) for the smallest, fastest model possible.

---

*Linearize to the limit – when speed matters more than accuracy.*
