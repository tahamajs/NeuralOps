<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 – 1 page (300 – 600 words)
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 Task Coverage

Only **SST-2** is evaluated. Adaptive SVD may behave differently on:
- Generation tasks (summarization, translation)
- Multi-class classification (MNLI, AG News)
- Token-level tasks (NER, POS)
- Long-context tasks (>512 tokens)

## 10.2 Model Coverage

We test on three models (66.9M–124.6M params). Not tested:
- Large models (>1B)
- Deep models (>24 layers)
- Mixture-of-Experts
- Vision transformers
- Multimodal transformers

## 10.3 Hardware

All experiments on **RTX 4070 Laptop (8 GB)**. Behavior on:
- A100/H100: SVD faster, method should scale
- CPU inference: Two matmuls may be slower
- Edge devices (Jetson): Untested

## 10.4 Inference Latency

Factorized layers are slower at inference:
- Two matmuls instead of one
- Intermediate $r$-dim tensor materialized
- Adapter adds one more matmul
- ~1.2× slower on GPU

**Mitigation.** The compressed model uses less memory, so larger batch sizes 
may fit, potentially offsetting latency.

## 10.5 Per-Projection Rank is Heuristic

The energy threshold is a **principled heuristic**, not an optimization. The 
optimal rank for downstream task may differ from the energy-thresholded rank.

## 10.6 No Downstream Error Bound

We provide:
- **Reconstruction error bound** (Prop 4.9.1)
- **No bound** on downstream task accuracy

The gap between reconstruction and task accuracy is not characterized.

## 10.7 Residual Adapter Overhead

The adapter adds $r_L (d_\text{in} + d_\text{out})$ parameters per projection. 
For $d = 768$ and $r_L = 16$:

$$
\text{Overhead per projection} = 16 \cdot 1536 = 24,576 \text{ params}
$$

vs. the compressed base of $r_P \cdot 1536$. This is ~12% 
of the base.

## 10.8 Single-Seed Results

All results from a single seed. Standard error $\approx 1.69\%$ at $N = 872$. 
Differences below 3.4% may be noise.

## 10.9 No Comparison with All Baselines

We do not compare with:
- DistilBERT / TinyBERT (distillation)
- Performer / Linformer (attention replacement)
- SparseGPT / Wanda (pruning)
- Direct head pruning

## 10.10 T5 and Qwen Verification

T5 support is not tested, Qwen support is partial. Full 
verification on these architectures is deferred to future work.

## 10.11 Theoretical Guarantee on Per-Layer Optimality

The energy threshold is not provably optimal. Alternatives (learned $\tau$, 
per-layer $\tau$, activation-aware $\tau$) are not tested.

## 10.12 Code Quality

Research-grade:
- Broad exception handling
- No unit tests for edge cases
- Numerical stability could be improved
- T5 bias path untested on decoder side

## 10.13 Ethical Considerations

Compression is dual-use. See Appendix E.
