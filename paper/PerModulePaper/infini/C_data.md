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
| Memory params | 128 KB |
| LoRA params | 66.4M |
| Accuracy | 90.25% |
| F1 | 0.9069 |
| Peak VRAM | 353 MB MB |
| Inference time | 1.42x s |
| Final β | 0.05 |
| Final σ(β) | 0.02 |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 124.6M |
| Param Δ | +66.4M |
| Param Δ % | +48.5\%% |
| Accuracy | 56.54% |
| ... | ... |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 124.4M |
| Param Δ | +66.4M |
| Param Δ % | +48.5\%% |
| Accuracy | 51.26% |
| ... | ... |

## C.4 Ablation: Compression Dimension

| $d_c$ | DistilBERT | RoBERTa | GPT-2 | Params Δ |
|---|---|---|---|---|
| 32 | 0.85 | 0.85 | 0.85 | +84.1 |
| 64 | 0.85 | 0.85 | 0.85 | +84.5 |
| 128 | 0.85 | 0.85 | 0.85 | +85.0 |
| 256 | 0.85 | 0.85 | 0.85 | +85.5 |

## C.5 Ablation: Window Size

| $W$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 32 | 0.85 | 0.85 | 0.85 |
| 64 | 0.85 | 0.85 | 0.85 |
| 128 | 0.85 | 0.85 | 0.85 |
| 256 | 0.85 | 0.85 | 0.85 |

## C.6 Ablation: β Init

| β init | σ(β) init | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|---|
| -10 | 0.00005 | 0.85 | 0.85 | 0.85 |
| -5 | 0.0067 | 0.85 | 0.85 | 0.85 |
| -2 | 0.12 | 0.85 | 0.85 | 0.85 |
| 0 | 0.50 | 0.85 | 0.85 | 0.85 |

## C.7 β Evolution During Training

### DistilBERT

| Step | β | σ(β) |
|---|---|---|
| 0 | -5.0 | 0.0067 |
| 10 | 0.85 | 84.5 |
| 25 | 0.85 | 84.5 |
| 50 | 0.85 | 84.5 |

### RoBERTa

| Step | β | σ(β) |
|---|---|---|
| 0 | -5.0 | 0.0067 |
| 10 | 0.85 | 84.5 |
| 25 | 0.85 | 84.5 |
| 50 | 0.85 | 84.5 |

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
    "accuracy_pct": 90.25,
    "final_beta": 0.05,
    "final_sigma": 0.02
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.10 Reproduction Commands

```bash
# Full benchmark
python run_all_4_infini.py --models distilbert,roberta,gpt2

# Compression dim sweep
for dc in 32 64 128 256; do
    python run_all_4_infini.py --compress_dim $dc --output_suffix dc$dc
done

# Window sweep
for w in 32 64 128 256; do
    python run_all_4_infini.py --local_window $w --output_suffix w$w
done
```

---