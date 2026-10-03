# 10. Limitations

## 10.1 Task Coverage

Only SST-2 evaluated. Behavior on:
- Generation: unknown
- Multi-class: unknown
- Very long sequences ($L > 512$): partially tested

## 10.2 Model Coverage

| Constraint | Tested |
|---|---|
| ≤124M params | Yes |
| >1B params | No |
| >24 layers | Not fully |
| MoE | No |
| Vision | No |

## 10.3 Parameter Increase

LEKAv3 adds ~7.5% parameters per layer. This is a real cost:
- LoRA: +48.5\%%
- Kernel projections: +48.5\%%
- Landmark scorer: +48.5\%%
- W_corr: +48.5\%%

## 10.4 Inference Overhead

At short $L$ (< 64), landmark scoring may dominate:
- Scorer cost: $\mathcal{O}(L d_\text{feat} \cdot 64)$
- For small $L$, this is comparable to attention itself

## 10.5 Approximation Error

RFF-based approximation has inherent error. On peaked attention 
distributions (typical for transformers), the error may be larger than 
the RFF bound suggests.

## 10.6 Single-Seed Results

SE ~1.7%. Differences below 3.4% may be noise.

## 10.7 No Formal Downstream Bound

We provide no bound on downstream accuracy loss.

## 10.8 Causality for Cross-Attention

LEKAv3 is designed for self-attention. Cross-attention is not supported 
(explicitly raises NotImplementedError).

## 10.9 The `softpick` Function

Our sparsifying softpick differs from standard softmax. It introduces 
additional hyperparameters (temperature) that we did not tune.

## 10.10 W_corr Memory

$W_\text{corr} \in \mathbb{R}^{H \times d_h \times d_h}$ costs 
$H d_h^2$ parameters per layer. For $H = 12, d_h = 64$: 49,152 params.

## 10.11 No Comparison with All Baselines

We do not compare with:
- Performer (would need their exact implementation)
- Nyströmformer
- Reformer
- Longformer with global tokens

## 10.12 Code Quality

Research-grade:
- Broad exception handling
- No fused Triton kernels
- Some numerical stability issues at extreme feature dims

## 10.13 Ethical Considerations

Dual-use. See Appendix E.

---