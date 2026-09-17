# Attention Analysis

## Purpose

Quantify how compression changes attention behavior.

## Per (backbone, rank, layer, head)

- Attention entropy: `-Σ p log p` over each query row
- Attention sink mass: fraction of mass on token 0
- Local vs global ratio: mass within window / total
- Diagonal dominance: mean attention on self
- Head redundancy: cosine similarity between head outputs
- KL divergence vs. baseline attention

## Implementation

```python
def register_attention_hooks(model, save_dir):
    import os, torch
    os.makedirs(save_dir, exist_ok=True)
    collected = {}
    def make_hook(name):
        def hook(module, inp, outp):
            if isinstance(outp, tuple) and len(outp) >= 2 and outp[1] is not None:
                collected[name] = outp[1].detach().cpu()
        return hook
    handles = []
    for name, module in model.named_modules():
        if any(k in name.lower() for k in ("attn", "attention")):
            handles.append(module.register_forward_hook(make_hook(name)))
    return handles, collected
```

## Output

- `papers/modules/<module>/metrics/attention_entropy.csv`
- `papers/modules/<module>/metrics/sink_mass.csv`
- `papers/modules/<module>/metrics/local_global_ratio.csv`
- `papers/modules/<module>/metrics/head_importance.csv`
- `papers/modules/<module>/metrics/head_redundancy.npy`
- `papers/modules/<module>/metrics/attn_kl_divergence.csv`
- `papers/modules/<module>/metrics/diag_dominance.csv`
- `papers/modules/<module>/figures/attn_maps/<sample_id>.png` for 5 samples
