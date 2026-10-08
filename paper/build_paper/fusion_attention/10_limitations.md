# 10. Limitations

## 10.1 Short-Sequence Silencing

At $L = 128$ (our test length), FUSIONATTN uses the **short path** and 
provides no sparsity benefit. All our reported results reflect this — they 
are essentially LoRA-only results.

## 10.2 Long Path Untested

We did not test at $L > 128$ due to dataset and model constraints. Long-path 
behavior is extrapolated.

## 10.3 Not a Compression Method

Parameter count increases by ~66.4M%.

## 10.4 Gate MLP Overhead

The gate MLP adds ~66.4MM parameters per model but is unused at 
L ≤ 128.

## 10.5 Task Coverage

Only SST-2 at $L = 128$. Generation tasks, QA, and long-context tasks 
untested.

## 10.6 Model Coverage

Three models (66.9M–124.6M).

## 10.7 Hardware

Only RTX 4070 Laptop.

## 10.8 Causal Decoder Block Attention

Block attention for decoders requires within-block causal masking. This is 
implemented but not extensively tested.

## 10.9 No Formal Bound

No bound on approximation error of top-K block selection.

## 10.10 Single Seed

Standard error ~1.69%.

## 10.11 Critical Bug: `raise RuntimeError` in inject

The `inject_fusion_attn` function has:

```python
except Exception:
    raise RuntimeError("copy_weights failed")
```

This kills injection on any failure. Recommend changing to `return None`.

## 10.12 Code Quality

- Research-grade
- No unit tests for edge cases
- Broad exception handling

## 10.13 Ethical Considerations

Dual-use. See Appendix E.

---