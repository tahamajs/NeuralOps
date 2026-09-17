<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 No Speedup

ASA-Span **does not** reduce inference time because all scores are computed. 
The soft ramp only changes the **values**, not the **structure**. To 
realize speedup, hard truncation + sparse kernels are needed (not implemented).

## 10.2 No Parameter Reduction

ASA-Span adds $H$ parameters (span logits). Total parameter count is 
**unchanged**.

## 10.3 Task Coverage

Only SST-2 evaluated. Long-context tasks ($L > 512$) not tested.

## 10.4 Model Coverage

Tested on models with $H = 12$. Not tested on:
- Very wide models ($H = 32+$)
- Mixture-of-Experts
- Vision transformers

## 10.5 Fixed Ramp Size

The ramp size $R$ is a hyperparameter, not learned. Future work could learn 
it per-head.

## 10.6 Distance-Only Masking

ASA-Span uses only distance, not content. This is complementary to 
content-based methods but limits expressiveness.

## 10.7 Single Seed

All results from seed 42.

## 10.8 No Comparison with Some Baselines

We do not compare with:
- Longformer
- BigBird
- Mistral sliding window

## 10.9 T5 / Qwen Not Verified

The T5 relative position bias and Qwen RoPE + GQA paths are implemented but 
not tested in this paper.

## 10.10 Failure Modes

Long-context evaluation, hard truncation speedup

## 10.11 Code Quality

Research-grade implementation.

## 10.12 Ethical Considerations

Dual-use. See Appendix E.

---