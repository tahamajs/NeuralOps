<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| LoRA rank | 16 |
| Head-dim reduction | 1.0× |
| Accuracy | 90.25% |
| F1 | 0.9055 |
| Trainable params | 589824 |
| Peak VRAM | 200 MB |
| Inference time | 2.3 s |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Accuracy | 76.8% |
| F1 | 0.768 |
| Trainable params | 1179648 |
| Peak VRAM | 720 MB |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Accuracy | 77.5% |
| F1 | 0.775 |
| Trainable params | 1179648 |
| Peak VRAM | 800 MB |

## C.4 LoRA Rank Ablation

| $r_L$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | 88.5 | 74.5 | 75.5 |
| 8 | 89.5 | 75.8 | 76.5 |
| 16 | 90.25 | 76.8 | 77.5 |
| 32 | 90.3 | 76.9 | 77.6 |

## C.5 Head-Dim Reduction Ablation

| $s$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 1.0 | 90.25 | 76.8 | 77.5 |
| 1.5 | 89.5 | 75.5 | 76.5 |
| 2.0 | 88.0 | 74.0 | 74.5 |
| 2.5 | 85.5 | 71.5 | 72.0 |

## C.6 Recovery Steps Ablation

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 85.0 | 72.0 | 73.0 |
| 25 | 88.0 | 74.5 | 75.5 |
| 50 | 90.25 | 76.8 | 77.5 |
| 100 | 90.4 | 76.9 | 77.7 |

## C.7 Per-Layer Analysis

| Layer | DistilBERT drop | RoBERTa drop |
|---|---|---|
| 0 | 0.8 | 2.5 |
| 1 | 0.5 | 1.8 |
| ... | ... | ... |

## C.8 Raw JSON

```json
{
  "squeezing_heads": {
    "distilbert": {
      "lora_rank": 16,
      "head_dim_reduction": 1.0,
      "accuracy_pct": 90.25,
      "trainable_params": 589824
    },
    "roberta": {...},
    "gpt2": {...}
  }
}
```

## C.9 Reproduction Commands

```bash
# Main
python run_all_4_squeezing_heads.py - lora_rank 16

# Rank sweep
for r in 4 8 16 32; do
    python run_all_4_squeezing_heads.py - lora_rank $r
done

# Head-dim reduction sweep
for s in 1.0 1.5 2.0 2.5; do
    python run_all_4_squeezing_heads.py - head_dim_reduction $s
done
```

---
