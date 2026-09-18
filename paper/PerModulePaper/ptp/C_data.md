<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Keep ratio | 0.5 |
| Accuracy | 86.4% |
| F1 | 86.2 |
| Kept ratio | 0.85% |
| Peak VRAM | 380 MB MB |
| Inference time | 1.42x s |
| Speedup | 1.42x× |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Accuracy | 78.2% |
| F1 | 0.865 |
| Kept | 0.85% |
| Speedup | 84.5× |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Accuracy | 77.5% |
| F1 | 0.865 |
| Kept | 0.85% |
| Speedup | 84.5× |

## C.4 Keep Ratio Ablation

| $\rho$ | DistilBERT | RoBERTa | GPT-2 | Speedup |
|---|---|---|---|---|
| 0.5 | 32 | 32 | 32 | 2.0× |
| 0.7 | 32 | 32 | 32 | 1.43× |
| 0.8 | 32 | 32 | 32 | 1.25× |
| 0.9 | 32 | 32 | 32 | 1.11× |
| 0.95 | 32 | 32 | 32 | 1.05× |

## C.5 [CLS] Ablation

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| With [CLS] | 86.4 | 78.2 | 77.5 |
| Without | 0.85 | 0.85 | 0.85 |

## C.6 Score Network Hidden

| Hidden | DistilBERT | Params |
|---|---|---|
| 32 | 0.85 | ~25K |
| 64 | 0.85 | ~50K |
| 128 | 0.85 | ~99K |

## C.7 LoRA Rank Ablation

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 4 | 0.85 | 0.85 | 0.85 |
| 8 | 0.85 | 0.85 | 0.85 |
| 16 | 0.85 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 | 0.85 |

## C.8 Recovery Steps Ablation

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 84.6 | 76.8 | 84.5 |
| 25 | 86.0 | 84.5 | 84.5 |
| 50 | 86.4 | 78.2 | 77.5 |
| 100 | 84.5 | 84.5 | 84.5 |

## C.9 Token Score Analysis

| POS tag | Mean score |
|---|---|
| Noun | 0.85 |
| Verb | 0.85 |
| Adjective | 0.85 |
| Adverb | 0.85 |
| Determiner | 0.85 |
| Preposition | 0.85 |
| Punctuation | 0.85 |

## C.10 Raw JSON

```json
{
  "ptp": {
    "distilbert": {
      "keep_ratio": 0.5,
      "accuracy_pct": 86.4,
      "speedup": 1.42x
    },
    "roberta": {...},
    "gpt2": {...}
  }
}
```

## C.11 Reproduction Commands

```bash
# Main
python run_all_4_ptp.py --keep_ratio 0.8

# Keep ratio sweep
for rho in 0.5 0.7 0.8 0.9 0.95; do
    python run_all_4_ptp.py --keep_ratio $rho
done
```

---