<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 `copy_weights` Raises on Failure

Matches the pattern in AEGIS V4/V5, EAT, RALA, NHA. **Robustness 
regression** compared to modules that return None.

## 10.2 `rff_proj` is Not Persistent

```python
persistent=False
```

The RFF projection is re-initialized randomly on reload. **Train/test 
mismatch** on checkpoint restoration.

## 10.3 Softpick Collapses on Masked Rows

Without a fallback, fully masked rows produce zero attention. Contrast 
with NASC which has a batch-level fallback.

## 10.4 Slower than Baseline

~2.3 s× slowdown from more kernel launches.

## 10.5 Head Split is Heuristic

The 2/3 : 1/3 split is chosen by `H // 3`. No principled justification.

## 10.6 Rank Selection is Global

The same `svd_max_rank` is used for all projections. Per-layer rank 
selection could be better.

## 10.7 No Multi-Seed

Single seed.

## 10.8 Task Coverage

Only SST-2. Behavior on other tasks is untested.

## 10.9 Model Coverage

Only three models. Qwen3.5 untested.

## 10.10 RFF Memory

`rff_proj` shape is $(H_k, d_h, F)$. For $H_k = 8$, $d_h = 64$, $F = 256$: 
131K floats = ~0.5 MB per layer. Small but not negligible.

## 10.11 Python Loop in Decoder Kernel

For $N > 64$, the causal kernel path uses a Python loop. At $N = 128$: 2 
iterations. At $N = 512$: 8 iterations.

## 10.12 The Full Matrix Materialization

```python
@property
def weight(self):
    return torch.matmul(self.up.weight, self.down.weight)
```

Materializes the full $d_\text{out} \times d_\text{in}$ matrix. For large 
FFN projections, this is a memory spike.

## 10.13 Code Quality

- Two bugs (documented)
- No logging of head split
- No logging of rank selection

## 10.14 What This Paper Is Not

- Not a perfectly robust module (two bugs)
- Not the fastest method (slower than baseline)
- Not validated beyond SST-2

---