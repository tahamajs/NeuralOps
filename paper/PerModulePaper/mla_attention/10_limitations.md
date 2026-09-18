<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 This is a Reproduction, Not a Novel Method

MLA is from DeepSeek-AI (2024). We do not claim to improve it. Our 
contributions are the multi-architecture port and the portability study.

## 10.2 Task Coverage

Only SST-2 is evaluated. MLA's KV-cache savings are most relevant for 
long-context generation, which is **not** the task we evaluate. Our results 
show it preserves classification accuracy, but we do not measure 
long-context generation quality.

## 10.3 Model Coverage

| Constraint | Status |
|---|---|
| ≤124M params | Tested |
| >1B params | Not tested |
| DeepSeek-V2 | Not tested |
| MoE | Not tested |

The original MLA was designed for models with $d \approx 5120$ and 
$H = 128$. Our models have $d = 768$ and $H = 12$. Portability to 
larger models is untested.

## 10.4 RoPE Handling Simplified

The original DeepSeek-V2 applies RoPE on a **decoupled** head dimension. We 
apply it uniformly. This may lose accuracy on Qwen-style models.

## 10.5 Latent Dimension Not Auto-Selected

We use a fixed $c = 32$. The original paper does not specify how to 
choose $c$; ours is empirical.

## 10.6 Output Scale Not in Original

Our `output_scale` trainable parameter is not in the original DeepSeek-V2. 
It absorbs distributional shift but slightly deviates from the paper.

## 10.7 Cross-Attention Not Supported

Encoder-decoder models (T5, BART) are not supported.

## 10.8 Single-Seed Results

All main results from a single seed. SE ~1.69%.

## 10.9 No Long-Context Evaluation

We do not test sequences beyond 128 tokens. The KV-cache savings are 
largest for long contexts.

## 10.10 Inference Latency

MLA adds a down and up projection. For small batch and short sequence, this 
can be slower than standard attention.

## 10.11 No Comparison with Original DeepSeek-V2

We do not run the original DeepSeek-V2 model due to compute constraints.

## 10.12 Code Quality

Research-grade. Known issues:
- Broad exception handling in `copy_weights`
- `raise RuntimeError` in UniversalAdapter fallback (should log)
- No unit tests for edge cases

## 10.13 Ethical Considerations

See Appendix E.

---