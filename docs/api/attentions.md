# Attentions API Reference

NeuralOps registers **131 canonical attention mechanisms** unified under `neuralops.core.registry.ATTENTION_REGISTRY`.

## Core Categories
- **Linear Attentions**: `svd_linear_attention`, `cosformer`, `linear_attention`, `performer`, `fast_attention`, `rfa`.
- **State-Space & Recurrent**: `mamba_block`, `rwkv6`, `retention`, `gla`, `gated_linear_attention`.
- **Sparse & Segmented**: `longformer`, `bigbird`, `reformer`, `local_attention`, `sliding_window`.
- **Hybrid & Gated**: `mixattention_moska`, `hstu`, `differential_attention`.

## Usage

```python
import neuralops

print(f"Total registered mechanisms: {len(neuralops.list_attentions())}")
# Inspect specific mechanism
info = neuralops.get_attention_info("svd_linear_attention")
```
