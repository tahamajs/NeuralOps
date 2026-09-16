<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 `inject_performer` Raises on Failure

Contrary to the docstring, `inject_performer` raises `RuntimeError` on 
construction failure. This is a **robustness regression** compared to 
modules that return `None`.

## 10.2 `output_scale` is a New Parameter

`LoRALinear.output_scale` is not in baseline checkpoints. It defaults to 
1.0 but causes `missing_keys` warnings.

## 10.3 Cross-Attention Not Supported

`encoder_hidden_states` → returns input unchanged.

## 10.4 Feature Approximation Error

FAVOR+ approximates softmax. The approximation error is $\mathcal{O}(1/\sqrt{F})$. 
Small $F$ → larger error.

## 10.5 Slower than FlashAttention-2

At short sequences, the extra operations (feature map, gather, etc.) make 
Performer slower than FlashAttention-2.

## 10.6 Task Coverage

Only SST-2. Long-context tasks not tested.

## 10.7 Model Coverage

Only three models. Qwen untested.

## 10.8 Single Seed

No multi-seed runs.

## 10.9 Feature Map Epsilon

The $+\epsilon$ in the feature map deviates slightly from theory.

## 10.10 `kernel_type` Options

Three options (`favor_plus`, `elu`, `relu`), but the paper only advocates 
`favor_plus`.

## 10.11 Python Loop in Causal Path

For $N = 128$, chunk=64: 2 iterations. Acceptable.

## 10.12 What This Paper Is Not

- Not a new algorithm (Performer is published).
- Not the fastest method.
- Not a compression method.

---