# 10. Limitations

## 10.1 Only Attention Projections

SOA Fused Attention compresses **only** Q, K, V. The FFN and $W_O$ are 
untouched. Total model reduction is therefore modest (~{{D_TOTAL_DELTA}}% 
on DistilBERT).

## 10.2 Attention Remains Quadratic

The attention computation itself is not compressed. At long L, the quadratic 
cost dominates.

## 10.3 Task Coverage

Only SST-2 is evaluated. Generalization to:
- Generation tasks
- Long-context tasks
- Multi-class classification
- Token-level tasks

is untested.

## 10.4 Model Coverage

| Model size | Tested |
|---|---|
| ≤124M | Yes |
| 1B+ | No |

## 10.5 Hardware

All experiments on RTX 4070. Other hardware untested.

## 10.6 Rank Selection

We use a fixed default rank of 128. A per-layer adaptive rank could give 
better compression-accuracy tradeoffs, but we do not explore this.

## 10.7 No Theoretical Bound

We provide no formal bound on downstream accuracy loss. The reconstruction 
error bound is on the weight matrix, not on the task output.

## 10.8 Single-Seed Results

Multi-seed would strengthen conclusions.

## 10.9 Nested Encoder Handling

For RoBERTa, we set `is_nested_encoder = True` and rely on the outer 
`output.dense` for the output projection. If misconfigured, this could 
cause silent degradation.

## 10.10 No Comparison with Some Methods

We do not compare with:
- DistilBERT (distillation)
- SparseGPT / Wanda (pruning)
- Performer / Linformer (attention replacement)

## 10.11 Code Quality

Research-grade:
- Broad exception handling in some paths
- No fused kernels for factorized forward
- No unit tests for edge cases

## 10.12 Ethical Considerations

Parameter compression is dual-use. See Appendix E.
```

---