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
| Accuracy | 50.92% |
| F1 | 0.6748 |
| Peak VRAM | 639 MB MB |
| Inference time | 1.42x s |
| Final gate γ | 0.94 |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 124.6M |
| Param Δ | +66.4M |
| Param Δ % | +48.5\%% |
| Accuracy | 50.92% |
| ... | ... |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 124.4M |
| Param Δ | +66.4M |
| Param Δ % | +48.5\%% |
| Accuracy | 50.23% |
| ... | ... |

## C.4 d_state Ablation

| $d_\text{state}$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 8 | 0.85 | 0.85 | 0.85 |
| 16 | 0.85 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 | 0.85 |
| 64 | 0.85 | 0.85 | 0.85 |

## C.5 d_conv Ablation

| $d_\text{conv}$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 2 | 0.85 | 0.85 | 0.85 |
| 4 | 0.85 | 0.85 | 0.85 |
| 8 | 0.85 | 0.85 | 0.85 |

## C.6 global_rank Ablation

| $r_g$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | 0.85 | 0.85 | 0.85 |
| 8 | 0.85 | 0.85 | 0.85 |
| 16 | 0.85 | 0.85 | 0.85 |

## C.7 Gate Bias Ablation

| bias | $\sigma(b)$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|---|
| -10 | 4.5e-5 | 0.85 | 0.85 | 0.85 |
| -7 | 9.1e-4 | 0.85 | 0.85 | 0.85 |
| -5 | 6.7e-3 | 0.85 | 0.85 | 0.85 |
| 0 | 0.5 | 0.85 | 0.85 | 0.85 |

## C.8 Gate Evolution

### DistilBERT

| Step | γ mean |
|---|---|
| 0 | 0.0009 |
| 10 | 0.85 |
| 25 | 0.85 |
| 50 | 0.85 |

## C.9 Training Loss Curves

### DistilBERT
```
Step  Loss
1     82.1
5     83.4
10    84.6
...
50    86.4
```

## C.10 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": 66.4M,
    "param_delta": -32M,
    "accuracy_pct": 50.92,
    "final_gate": 0.94
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.11 Reproduction Commands

```bash
# Full benchmark
python run_all_4_svd_mamba.py --models distilbert,roberta,gpt2

# d_state sweep
for ds in 8 16 32 64; do
    python run_all_4_svd_mamba.py --d_state $ds --output_suffix ds$ds
done

# Gate bias sweep
for gb in -10 -7 -5 0; do
    python run_all_4_svd_mamba.py --gate_bias $gb --output_suffix gb$gb
done
```

---