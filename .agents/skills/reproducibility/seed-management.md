# Seed Management

## Seeds used

| Seed | Purpose |
|---|---|
| 42 | Primary runs |
| 123 | Replication |
| 456 | Replication |

## Global seed setting

```python
import random, numpy as np, torch
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

## Output

- `papers/modules/<module>/env/seeds.json`
- `papers/modules/<module>/env/deterministic.txt`
