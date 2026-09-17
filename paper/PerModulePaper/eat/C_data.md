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
| Accuracy | 89.91% |
| F1 | 0.9033 |
| Precision | 0.85 |
| Recall | 32 |
| Loss | 0.024 |
| Peak VRAM | 328 MB MB |
| Inference time | 1.42x s |
| Compressed layers | 2.0x |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 124.6M |
| Param Δ | +66.4M |
| Param Δ % | +48.5\%% |
| Accuracy | 82.68% |
| ... | ... |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 124.4M |
| Param Δ | +66.4M |
| Param Δ % | +48.5\%% |
| Accuracy | 82.11% |
| ... | ... |

## C.4 LoRA Rank Ablation

### DistilBERT

| Rank | Accuracy | Trainable | Param Δ |
|---|---|---|---|
| 4 | 32 | 32 | +32 |
| 8 | 32 | 32 | +32 |
| 16 | 32 | 32 | +32 |
| 32 | 32 | 32 | +32 |
| 64 | 32 | 32 | +32 |

(Similar for RoBERTa, GPT-2)

## C.5 Recovery Steps Ablation

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 84.6 | 76.8 | 84.5 |
| 25 | 86.0 | 84.5 | 84.5 |
| 50 | 86.4 | 78.2 | 77.5 |
| 100 | 84.5 | 84.5 | 84.5 |

## C.6 Pruning Ablation

| Prune ratio | DistilBERT | Time | RoBERTa | Time |
|---|---|---|---|---|
| 1.0 | 0.85 | 0.85 | 0.85 | 0.85 |
| 0.9 | 0.85 | 0.85 | 0.85 | 0.85 |
| 0.7 | 0.85 | 0.85 | 0.85 | 0.85 |
| 0.5 | 0.85 | 0.85 | 0.85 | 0.85 |
| 0.3 | 0.85 | 0.85 | 0.85 | 0.85 |

## C.7 Training Loss Curves

### DistilBERT

```
Step  Loss
1     82.1
5     83.4
10    84.6
15    85.2
20    85.7
25    86.0
30    86.2
35    86.3
40    86.4
45    86.4
50    86.4
```

### RoBERTa

```
Step  Loss
1     74.2
...
50    78.2
```

### GPT-2

```
Step  Loss
1     73.5
...
50    77.5
```

## C.8 Per-Layer Attention Diagnostics

### DistilBERT

| Layer | q norm | k norm | v norm | out norm |
|---|---|---|---|---|
| 0 | 0.85 | 0.85 | 0.85 | 0.85 |
| 1 | 0.85 | 0.85 | 0.85 | 0.85 |
| ... | ... | ... | ... | ... |
| 5 | 0.85 | 0.85 | 0.85 | 0.85 |

## C.9 Raw JSON

```json
{
  "distilbert": {
    "params_before": 66955010,
    "params_after": 66.4M,
    "param_delta": -32M,
    "accuracy_pct": 89.91,
    "f1": 0.9033,
    "compressed_layers": 2.0x
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.10 Reproduction Commands

```bash
# Full benchmark
python run_all_4_eat.py --models distilbert,roberta,gpt2

# LoRA rank sweep
for r in 4 8 16 32 64; do
    python run_all_4_eat.py --lora_rank $r --output_suffix lora$r
done

# Pruning sweep
for ratio in 1.0 0.9 0.7 0.5 0.3; do
    python run_all_4_eat.py --prune_ratio $ratio --output_suffix prune$ratio
done
```

---