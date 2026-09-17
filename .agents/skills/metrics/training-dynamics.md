# Training Dynamics

## Purpose

Record every training detail for reproducibility.

## Per step

- Loss value
- Gradient norm
- Learning rate
- Weight update norm
- NaN/Inf events

## Implementation

Add this callback to the LoRA training loop:

```python
def training_callback(step, loss, model, optimizer, lr):
    grad_norm = sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    update_norm = 0.0
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is not None:
                update_norm += p.grad.norm() ** 2
    return {
        "step": step,
        "loss": loss.item(),
        "grad_norm": grad_norm.item(),
        "update_norm": update_norm ** 0.5,
        "lr": lr,
        "nan": bool(torch.isnan(loss).item()),
    }
```

## Output

- `papers/modules/<module>/metrics/training_loss.jsonl`
- `papers/modules/<module>/metrics/grad_norm.jsonl`
- `papers/modules/<module>/metrics/update_norm.jsonl`
- `papers/modules/<module>/metrics/convergence.json`
- `papers/modules/<module>/logs/nan_inf.log`
- `papers/modules/<module>/logs/oom.log`
- `papers/modules/<module>/figures/training_curves_ci.png`
