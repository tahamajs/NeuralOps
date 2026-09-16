# 10. Limitations

## 10.1 Cumsum Memory Footprint

The causal linear stream materializes a 
$(\text{B}, \text{H}, L, 2F, d_h)$ intermediate via `torch.cumsum`. For 
$B = 8, H = 12, L = 128, 2F = 256, d_h = 64$, this is roughly **8 GB at 
FP32**. In practice, we reduce memory by operating at lower batch size, but 
the fundamental scaling is $\mathcal{O}(L \cdot 2F \cdot d_h)$.

**This is the primary limitation.**

## 10.2 Short-Sequence Slowdown

At $L = 128$, CHELA-SLAY is **{{SLOWDOWN}}× slower** than baseline because 
both streams are computed and the cumsum has overhead.

## 10.3 No Linear Speedup on SST-2

The theoretical $\mathcal{O}(L)$ benefit only appears at $L \geq 
{{BREAKEVEN}}$. On SST-2 ($L = 128$), the quadratic dense term is fast.

## 10.4 Gate Remains Dense-Dominant

After training, the gate stays near ${{D_GATE_MEAN}}$ (close to 1). The 
model effectively uses only the dense stream — the linear stream 
contributes little at short L.

## 10.5 Task Coverage

Only SST-2 is evaluated. Behavior on:
- Generation tasks
- Long-context tasks
- Multi-class classification

is untested.

## 10.6 Model Coverage

| Model size | Tested |
|---|---|
| ≤124M | Yes |
| 1B+ | No |

## 10.7 Hardware

All experiments on RTX 4070. Other hardware untested.

## 10.8 No Theoretical Bound

We provide no formal approximation bound for the linear stream relative to 
softmax.

## 10.9 Feature Dim Choice

We use $F = {{FEATURE_DIM}}$ as default. Optimal $F$ is model-dependent.

## 10.10 Gate Bias

$b_g = {{GATE_INIT_BIAS}}$ is a fixed default. Other values might work 
better.

## 10.11 Single-Seed Results

Multi-seed would strengthen conclusions.

## 10.12 No Comparison with Some Methods

We do not compare with Mamba, GLA, or other modern linear methods.

## 10.13 Code Quality

Research-grade:
- Broad exception handling
- No chunked cumsum optimization
- No unit tests for edge cases

## 10.14 Ethical Considerations

Hybrid attention is dual-use. See Appendix E.

---