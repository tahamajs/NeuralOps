# Training Curves with CI

## Purpose

Show loss and accuracy over recovery steps, with confidence bands.

## X-axis

Recovery step (0 to N)

## Y-axis

Loss (left), Accuracy (right)

## Implementation

```python
def plot_training_curves(loss_per_seed, acc_per_seed, out_png):
    import numpy as np, matplotlib.pyplot as plt
    loss = np.array(loss_per_seed)  # shape (n_seeds, n_steps)
    acc = np.array(acc_per_seed)
    steps = np.arange(loss.shape[1])
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(steps, loss.mean(0), color="C0", label="Loss (mean)")
    ax1.fill_between(steps, loss.mean(0) - loss.std(0),
                     loss.mean(0) + loss.std(0), alpha=0.2, color="C0")
    ax1.set_xlabel("Recovery step")
    ax1.set_ylabel("Loss", color="C0")
    ax2 = ax1.twinx()
    ax2.plot(steps, acc.mean(0), color="C1", label="Accuracy")
    ax2.fill_between(steps, acc.mean(0) - acc.std(0),
                     acc.mean(0) + acc.std(0), alpha=0.2, color="C1")
    ax2.set_ylabel("Accuracy (%)", color="C1")
    fig.savefig(out_png)
    plt.close(fig)
```

## Output

- `papers/modules/<module>/figures/training_curves_ci.png`
