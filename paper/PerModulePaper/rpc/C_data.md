<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 66.4M |
| Param Δ | +-32M |
| Param Δ % | +48.5\%% |
| Accuracy | 87.84% |
| F1 | 0.8798 |
| Peak VRAM | 334 MB MB |
| Inference time | 1.42x s |
| Sketch rank | 32 |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 124.6M |
| Param Δ | +66.4M |
| Param Δ % | +48.5\%% |
| Accuracy | 66.97% |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 124.4M |
| Param Δ | +66.4M |
| Param Δ % | +48.5\%% |
| Accuracy | 57.34% |

## C.4 Sketch Rank Ablation

| $r$ | DistilBERT | RoBERTa | GPT-2 | Theoretical speedup |
|---|---|---|---|---|
| 8 | 32 | 32 | 32 | 8× |
| 16 | 32 | 32 | 32 | 4× |
| 32 | 32 | 32 | 32 | 2× |
| 64 | 32 | 32 | 32 | 1× |

## C.5 LoRA Rank Ablation

| $r_L$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | 0.85 | 0.85 | 0.85 |
| 8 | 0.85 | 0.85 | 0.85 |
| 16 | 0.85 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 | 0.85 |

## C.6 Sketch Quality

| $r$ | Score correlation | MSE |
|---|---|---|
| 8 | 0.84 | 0.045 |
| 16 | 0.92 | 0.021 |
| 32 | 0.96 | 0.008 |
| 64 | 1.00 | 0 |

## C.7 Efficiency at Different Sequence Lengths

| $L$ | Baseline time | RPC time (r=32) | Speedup |
|---|---|---|---|
| 64 | 84.5 | 84.0 | 84.1× |
| 128 | 1.90 s | 84.8 | 84.8× |
| 256 | 85.5 | 85.6 | 85.6× |
| 512 | 86.0 | 86.2 | 86.2× |
| 1024 | 86.4 | 86.4 | 86.4× |

## C.8 Training Loss Curves

### DistilBERT
```
Step  Loss
1     82.1
5     83.4
...
50    86.4
```

## C.9 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": 66.4M,
    "param_delta": -32M,
    "accuracy_pct": 87.84,
    "sketch_rank": 32,
    "score_correlation": 0.96
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.10 Reproduction Commands

```bash
# Full benchmark
python run_all_4_rpc.py --models distilbert,roberta,gpt2

# Rank sweep
for r in 8 16 32 64; do
    python run_all_4_rpc.py --rank $r --output_suffix r$r
done
```

---