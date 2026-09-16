<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 Not a Novel Method

Cosformer is a published method (Qin et al., 2022). Our contribution is 
the **port** to pretrained transformers, not the algorithm.

## 10.2 RoBERTa Instability

Our implementation shows significant drop on RoBERTa. Possible causes:
- Nested encoder + cos/sin mismatch
- Hyperparameter mismatch
- Training instability

We did not fully resolve this.

## 10.3 Decoder Overhead

The chunked prefix-sum adds ~5× overhead compared to baseline SDPA. This 
limits the practical advantage of linear attention.

## 10.4 Parameter Count Increases

Cosformer adds LoRA adapters, increasing parameter count by ~1-4%. It is 
not a compression method.

## 10.5 Task Coverage

Only SST-2. Behavior on:
- Long-context tasks (where Cosformer should shine)
- Generation tasks
- Retrieval tasks
is untested.

## 10.6 Model Coverage

Only three models. Qwen3.5 was not tested. No 7B+ models.

## 10.7 Single Seed

No multi-seed runs.

## 10.8 No Long-Sequence Testing

We test only $L = 128$. Cosformer's main advantage is at long sequences. 
Our results do not demonstrate this.

## 10.9 Fragile RoPE Extraction

The triple-nested try/except for RoPE call signatures is fragile.

## 10.10 Dead Code: `self.tau`

The `tau` parameter is declared but never used.

## 10.11 The `chunk_size = 32` Choice

We chose 32 arbitrarily. Smaller chunks would reduce memory but increase 
Python loop iterations; larger chunks would do the opposite. Optimal value 
untested.

## 10.12 Code Quality

- Some `try/except` blocks may mask errors
- No unit tests for the attention path
- Limited logging of the chunked scan state

## 10.13 What This Paper Is Not

- Not a new algorithm
- Not a compression method
- Not a long-context benchmark
- Not a scaling study

---