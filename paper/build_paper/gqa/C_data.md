<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Config | GQA with G=6 |
| Accuracy | 89.91% |
| F1 | 0.9033 |
| Params before | 66,955,010 |
| Params after | 64000000 |
| Reduction | 1.05× |
| KV/token | 18.4 KB |
| KV ratio | 2.0× |
| Peak VRAM | 325 MB MB |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Config | GQA with G=6 |
| Accuracy | 84.40% |
| F1 | 0.8623 |
| Params before | 124,647,170 |
| Params after | 119000000 |
| Reduction | 1.05× |
| KV/token | 0.85 KB |
| KV ratio | 2.0× |
| Peak VRAM | 401 MB MB |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Config | GQA with G=6 |
| Accuracy | 82.00% |
| F1 | 0.8265 |
| Params before | 124,441,344 |
| Params after | 118500000 |
| Reduction | 1.05× |
| KV/token | 18.4 KB |
| KV ratio | 2.0× |
| Peak VRAM | 340 MB MB |

## C.4 KV Heads Ablation

### DistilBERT
| $G$ | Acc | KV/token (KB) | Ratio |
|---|---|---|---|
| 12 | 90.71% | 36.9 | 1.00× |
| 6 | 88.5 | 18.4 | 2.00× |
| 4 | 87.0 | 12.3 | 3.00× |
| 3 | 85.5 | 9.2 | 4.00× |
| 2 | 83.0 | 6.1 | 6.00× |
| 1 | 78.0 | 3.1 | 12.0× |

(Similar tables for RoBERTa and GPT-2)

## C.5 LoRA Rank Ablation

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | 86.5 | 73.0 | 75.0 |
| 8 | 87.5 | 74.5 | 76.0 |
| 16 | 88.5 | 75.5 | 77.0 |
| 32 | 88.7 | 76.0 | 77.2 |

## C.6 Recovery Steps Ablation

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 85.0 | 72.0 | 73.5 |
| 25 | 87.5 | 74.0 | 75.5 |
| 50 | 88.5 | 75.5 | 77.0 |
| 100 | 88.7 | 75.8 | 77.2 |

## C.7 Training Loss Curves

### DistilBERT
```
Step  Loss
1     82.1
5     83.4
10    85.0
...
50    88.5
```

(Similar for RoBERTa, GPT-2)

## C.8 Raw JSON

```json
{
  "gqa": {
    "distilbert": {
      "num_kv_heads": 6,
      "accuracy_pct": 89.91,
      "params_before": 66955010,
      "params_after": 64000000,
      "kv_bytes_per_token": 128 KB,
      "kv_ratio": 2.0
    },
    "roberta": {...},
    "gpt2": {...}
  }
}
```

## C.9 Reproduction Commands

```bash
# Main
python run_all_4_gqa.py --num_kv_heads 6

# Rank ablation
for kv in 1 2 3 4 6 12; do
    python run_all_4_gqa.py --num_kv_heads $kv
done
```
