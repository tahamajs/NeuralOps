<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Accuracy | 50.92% |
| Δ | -39.79% |
| F1 | 0.6748 |
| Params (after) | 66.4M |
| Peak VRAM | 1320 MB MB |
| Inference time | 1.42x s |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Accuracy | 50.92% |
| Δ | -27.64% |
| F1 | 0.6748 |
| Params (after) | 124.6M |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Accuracy | 77.5% |
| Δ | 0.85% |
| F1 | 0.865 |
| Params (after) | 124.4M |

## C.4 Feature Ablation

| $F$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 32 | 0.85 | 0.85 | 0.85 |
| 64 | 0.85 | 0.85 | 0.85 |
| 128 | 0.865 | 0.865 | 0.865 |
| 256 | 0.85 | 0.85 | 0.85 |

## C.5 Kernel Ablation

| Kernel | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| `favor_plus` | 0.85 | 0.85 | 0.85 |
| `elu` | 0.85 | 0.85 | 0.85 |
| `relu` | 32 | 32 | 32 |

## C.6 Raw JSON

```json
{
  "distilbert": {"acc": 50.92, "params": 66.4M},
  "roberta": {"acc": 50.92, "params": 124.6M},
  "gpt2": {"acc": 77.5, "params": 124.4M}
}
```

## C.7 Reproduction Commands

```bash
python run_all_4_performer.py --models distilbert,roberta,gpt2

for F in 32 64 128 256; do
    python run_all_4_performer.py --num_features $F --output_suffix f$F
done
```

---