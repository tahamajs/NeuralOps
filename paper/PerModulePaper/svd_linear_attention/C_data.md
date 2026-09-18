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
| Ratio | 50\%× |
| Accuracy | 80.73% |
| F1 | 0.8019 |
| Peak VRAM | 2818 MB MB |
| Inference time | 1.42x s |
| Mode | 0.85 |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 124.6M |
| Ratio | 2.1\times× |
| Accuracy | 50.92% |
| ... | ... |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 124.4M |
| Ratio | 2.1\times× |
| Accuracy | 77.5% |
| ... | ... |

## C.4 Rank Ablation

| $r$ | DistilBERT | Ratio | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 8 | 32 | 24× | 32 | 32 |
| 16 | 32 | 16× | 32 | 32 |
| 32 | 32 | 12× | 32 | 32 |
| 64 | 32 | 8× | 32 | 32 |
| 128 | 32 | 4× | 32 | 32 |

## C.5 Mode Ablation

| Mode | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| `exact` | 0.85 | 0.85 | 0.85 |
| `linear` | 0.85 | 0.85 | 0.85 |
| `decay_linear` | 0.85 | 0.85 | 0.85 |

## C.6 Decay Ablation

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| No decay | 0.85 | 0.85 | 0.85 |
| Fixed decay (γ=0.99) | 0.85 | 0.85 | 0.85 |
| Learned decay | 0.85 | 0.85 | 0.85 |

## C.7 Scaling with Sequence Length

| $N$ | Baseline | SVD-Lin | Speedup |
|---|---|---|---|
| 128 | 4.2 s | 84.0 | 84.8× |
| 256 | 16.8 s | 84.8 | 85.6× |
| 512 | 67 s | 85.6 | 86.2× |
| 1024 | 268 s | 86.4 | 86.4× |

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
    "ratio": 50\%,
    "accuracy_pct": 80.73,
    "mode": "0.85",
    "rank": 32
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.10 Reproduction Commands

```bash
# Full benchmark
python run_all_4_svd_linear_attention.py --models distilbert,roberta,gpt2

# Rank sweep
for r in 8 16 32 64 128; do
    python run_all_4_svd_linear_attention.py --rank $r --output_suffix r$r
done

# Decay sweep
python run_all_4_svd_linear_attention.py --use_decay_gating false --output_suffix nodecay
python run_all_4_svd_linear_attention.py --use_decay_gating true  --output_suffix decay
```

---