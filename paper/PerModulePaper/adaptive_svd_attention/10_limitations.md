# 10. Limitations

## 10.1 Attention-Only Compression

We compress only the attention projections. The FFN remains uncompressed. 
Total model reduction is therefore modest.

## 10.2 Attention Remains Quadratic

The attention computation (Q·K^T) is not compressed. Long-context inference 
is not improved.

## 10.3 Task Coverage

Only SST-2 is evaluated. Generalization to:
- Generation tasks
- Long-context
- Multi-class classification
- Token-level tasks
- **T5-style seq2seq** (though we support the architecture, we don't test)

is untested.

## 10.4 Model Coverage

| Model size | Tested |
|---|---|
| ≤124M | Yes |
| 1B+ | No |
| T5-base/large | Not tested |

## 10.5 Hardware

All experiments on RTX 4070. Other hardware untested.

## 10.6 Energy Threshold Choice

We use τ = 0.92 as a fixed default. A per-layer adaptive τ might improve 
compression-accuracy tradeoffs, but we do not explore this.

## 10.7 Residual Rank

We fix $r_a = 16$. Optimal $r_a$ may differ per model or per projection.

## 10.8 No Theoretical Bound on Task Accuracy

We provide a bound on **reconstruction** error but not on **downstream task** 
accuracy. The gap is not characterized.

## 10.9 Single-Seed Results

Multi-seed would strengthen conclusions.

## 10.10 Nested Encoder Handling

For RoBERTa, we set `is_nested_encoder = True`, but we still factorize the 
output projection. The interaction is not fully characterized.

## 10.11 No Comparison with Some Methods

We do not compare with:
- DistilBERT (distillation)
- SparseGPT / Wanda (pruning)
- Performer / Linformer (attention replacement)

## 10.12 Code Quality

Research-grade:
- Broad exception handling
- No fused kernels for factorized forward
- No unit tests for edge cases

## 10.13 Ethical Considerations

Compression is dual-use. See Appendix E.

---
