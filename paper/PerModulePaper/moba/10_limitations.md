# 10. Limitations

## 10.1 Quadratic Complexity

Despite the SSM inspiration, our implementation is **$\mathcal{O}(L^2 d)$** in 
time and memory, not $\mathcal{O}(L)$. The docstring's claim of "strictly 
O(L)" is incorrect.

## 10.2 Not True Mamba

We do not perform selective scan, input-dependent state transitions, or 
conv1d preprocessing. This is **Mamba-inspired**, not Mamba.

## 10.3 Weight Reuse is Approximate

The $p_1$ and $p_2$ branches are populated with Q/V or Q/K weights by 
heuristic. There is no theoretical justification for this specific mapping.

## 10.4 Task Coverage

Only SST-2 evaluated. Behavior on generation, multi-class, and long-context 
tasks unknown.

## 10.5 Model Coverage

| Size | Tested |
|---|---|
| ≤124M | Yes |
| 1B–10B | No |
| >10B | No |

## 10.6 Single-Seed

SE ~1.7%. Differences below 3.4% may be noise.

## 10.7 Memory Scaling

At $L = 1024$, the $[L, L, d]$ weight tensor is ~750 MB per layer at FP16. 
This limits scalability.

## 10.8 copy_weights Raises

`copy_weights` raises exceptions on failure, which may abort training if 
not caught by the injector.

## 10.9 A_log Initialization

Init $A_{\log, i} = \log(i)$ may not be optimal for all tasks. Different 
initializations could help.

## 10.10 Preservation Gate Interaction

The preservation gate $\alpha$ is shared across all heads. A per-head gate 
might be more expressive.

## 10.11 No Comparison with Actual Mamba

We do not compare with true Mamba (different parameter counts, different 
architecture).

## 10.12 Code Quality

Research-grade:
- Docstring complexity claim is incorrect
- `raise` in copy_weights
- No fused kernel

## 10.13 Ethical Considerations

See Appendix E.

---