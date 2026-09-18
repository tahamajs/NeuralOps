<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 Encoder Default is Dense

The `use_dense` flag defaults to standard attention for bidirectional 
encoders. NASC's cascade path is not exercised on DistilBERT/RoBERTa 
unless explicitly enabled.

## 10.2 Batch-Level Gate

The gate is computed from the **batch-mean hidden state**, so all tokens 
in a batch share the same 3-way mixture. This eliminates per-token 
adaptivity.

## 10.3 `softpick` Batch Fallback

The fallback to softmax triggers at the **batch level** (any row triggers 
fallback for the entire batch via `torch.where`). This is a coarse fix.

## 10.4 Causal RFF Memory

The `KV_cumsum` tensor is $(B, H, L, 2F, d)$. For $L > 512$, this exceeds 
GPU memory.

## 10.5 Python Loop in Sparse Block

The `for i in range(nb)` loop adds Python overhead. Vectorization would 
reduce latency.

## 10.6 `inject_nasc` Raises on Failure

```python
except Exception as e:
    logger.error("inject_nasc failed: %s", e)
    raise e
```

This **violates the safe-injection contract** used by other modules.

## 10.7 Parameter Overhead

RFF projection ($dF$), gate ($3d$), and LoRA all add parameters.

## 10.8 Inference Overhead

NASC is slower than baseline (~2.3 s×) at L=128.

## 10.9 Task Coverage

Only SST-2. Long-context (where NASC's RFF should help) is not tested.

## 10.10 Model Coverage

Only three models. Qwen3.5 untested.

## 10.11 Single Seed

No multi-seed runs.

## 10.12 `_logged_once` Workaround

The path logging uses a per-instance flag rather than a proper logging 
level. Minor.

## 10.13 Code Quality

- Sparse block Python loop
- `softpick` fallback is coarse
- `inject_nasc` raises on failure
- Some `try/except` blocks may mask errors

## 10.14 What This Paper Is Not

- Not a causal-decoder method for all cases
- Not a compression method
- Not a speedup method
- Not a per-token adaptive gate method

---