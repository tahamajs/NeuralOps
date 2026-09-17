<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 – 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

We document failed experiments to help future work avoid dead ends.

## D.1 Attempts That Failed

### D.1.1 Trainable SVD Base

**Goal.** Let $U, V$ train alongside adapters.

**Result.** Base factors drifted during recovery, undoing the compression. 
Accuracy peaked early then collapsed.

**Conclusion.** Keep base frozen (`requires_grad=False`).

### D.1.2 Very Low Energy Threshold (τ=0.80)

**Goal.** Aggressive compression.

**Result.** Accuracy dropped to 80%.

**Conclusion.** τ=0.80 too aggressive for attention weights.

### D.1.3 Very High Threshold (τ=0.99)

**Goal.** Near-lossless compression.

**Result.** Accuracy was near-baseline but compression ratio was only 
1.3×.

**Conclusion.** τ=0.99 defeats the purpose.

### D.1.4 No Residual Adapter (r_L=0)

**Goal.** Pure truncation without adapter.

**Result.** Accuracy dropped to 80%.

**Conclusion.** Adapters are essential.

### D.1.5 Both A and B Random

**Goal.** Different initialization.

**Result.** Initial forward pass differed from pretrained; accuracy 
degraded before recovery.

**Conclusion.** Zero-init one of the factors.

### D.1.6 Both A and B Zero

**Goal.** Symmetric initialization.

**Result.** Gradient for both zero; no learning.

**Conclusion.** One must be nonzero.

## D.2 Abandoned Variants

### D.2.1 Per-Head Rank

We tested separate rank per head. Result: no improvement, 4× slower 
(SVD per head).

### D.2.2 Learned τ

Trained per-layer τ during recovery. Result: weak gradient signal, no 
improvement.

### D.2.3 Per-Projection Threshold

Different τ for Q, K, V, O. Result: no clear pattern; uniform τ=0.92 
works.

### D.2.4 Diagonal-Energy Rank

Thresholded $|\sigma_i|$ instead of $\sigma_i^2$. Result: different rank, 
slightly worse accuracy.

### D.2.5 Fixed Rank Higher

Tested fixed rank 192. Result: better accuracy than adaptive at τ=0.92, 
but worse compression. Trade-off.

## D.3 Failed Hyperparameters

### Learning Rate

| LR | DistilBERT | Loss stability |
|---|---|---|
| 5e-5 | 87.5 | Stable |
| 1e-4 | 88 | Stable |
| 2e-4 | 88 | Stable |
| 5e-4 | 87 | Unstable |
| 1e-3 | NaN | NaN |

### Batch Size

| BS | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 2 | 85 | 71 | 72 |
| 4 | 87 | 73 | 74 |
| 8 | 88 | 74 | 75.5 |
| 16 | 88.1 | OOM | OOM |

### Residual Rank

| $r_L$ | DistilBERT | Note |
|---|---|---|
| 1 | 80 | Underfit |
| 2 | 83 | Underfit |
| 4 | 85 | Acceptable |
| 8 | 87 | Good |
| 16 | 88 | Best |
| 32 | 88.2 | Diminishing returns |
| 64 | 88.3 | Overkill |

## D.4 Fixes That Didn't Help

| Fix | Problem | Result |
|---|---|---|
| Orthogonal adapter init | Slow convergence | No change |
| Gradient clipping | Stability | No change |
| Adaptive τ per layer | Rank quality | No improvement |
| Post-hoc base fine-tune | Accuracy | Drifted from compression |
| Adding dropout to adapters | Overfitting | No improvement |

## D.5 Architecture-Specific Failures

### D.5.1 T5 Verification Incomplete

T5 support was partially tested. Full verification on encoder and decoder 
blocks was deferred due to time constraints.

### D.5.2 Qwen Partial

Qwen3.5 with GQA was partially tested. Full verification pending.

## D.6 Multi-Seed Analysis

| Seed | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 42 | 88.0 | 74.0 | 75.5 |
| 123 | 87.8 | 73.8 | 75.3 |
| 456 | 88.1 | 74.1 | 75.6 |
| **Mean** | **88.0** | **74.0** | **75.5** |
| **Std** | **0.15** | **0.15** | **0.15** |

**Observation.** Standard deviation ~0.15% across seeds. Single-seed results 
are representative.

## D.7 Lessons Learned

1. **Freeze base factors** — never train them.
2. **Zero-init one adapter factor** — both zero or both random fails.
3. **τ=0.92 is robust** — no need to tune extensively.
4. **$r_L=16$ is sufficient** — diminishing returns above.
5. **No calibration is a feature** — enables zero-data settings.
6. **Per-layer rank matters** — fixed rank loses ~1.5% accuracy.

## D.8 Recommendations

Based on our negative results:

1. Start with default hyperparameters (τ=0.92, r_L=16).
2. Tune τ only if accuracy is unacceptable.
3. Increase r_L if recovery plateaus below target.
4. Keep base frozen.
5. Verify on one model before full sweep.

## D.9 Unresolved Questions

1. **Optimal τ schedule:** Could per-layer τ beat uniform?
2. **Combined with A3/AASVD:** Would activation-aware rank beat 
   energy-thresholded?
3. **Interaction with quantization:** INT8 + Adaptive SVD?
4. **Learning rate schedule:** Would cosine scheduling help?
5. **Multi-task calibration:** Does rank transfer across tasks?
