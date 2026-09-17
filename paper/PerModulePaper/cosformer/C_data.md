<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full data
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Accuracy | 77.75% |
| Δ | -12.96% |
| F1 | 0.7651 |
| Precision | 0.876 |
| Recall | 0.874 |
| Loss | 0.42 |
| Params (before) | 66,955,010 |
| Params (after) | 70791394 |
| Peak VRAM | 335 MB MB |
| Inference time | 3.5 s |
| Path | Encoder |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Accuracy | 58.72% |
| Δ | -19.84% |
| F1 | 0.6629 |
| Params (before) | 124,647,170 |
| Params (after) | 131923074 |
| Peak VRAM | 659 MB MB |
| Inference time | 7.5 s |
| Path | Encoder (nested) |
| Status | PASS_LOW |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Accuracy | 70.18% |
| Δ | -9.18% |
| F1 | 0.7510 |
| Params (before) | 124,441,344 |
| Params (after) | 126630020 |
| Peak VRAM | 1283 MB MB |
| Inference time | 20.5 s |
| Path | Decoder |

## C.4 Chunk Size Ablation

| chunk_size | GPT-2 time | GPT-2 acc |
|---|---|---|
| 16 | 25.0 | 76.5 |
| 32 | 20.5 | 76.5 |
| 64 | 18.0 | 76.3 |
| 128 | 16.5 | 76.0 |

## C.5 Activation Ablation

| Act | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| `elu+1` | 87.5 | 68.0 | 76.5 |
| `softplus` | 87.3 | 67.5 | 76.2 |

## C.6 Training Loss

### DistilBERT

```
Step  Loss
1     0.68
5     0.55
10    0.48
...
50    87.5
```

## C.7 Raw JSON

```json
{
  "distilbert": {"acc": 77.75, "path": "encoder"},
  "roberta": {"acc": 58.72, "path": "encoder_nested"},
  "gpt2": {"acc": 70.18, "path": "decoder"}
}
```

## C.8 Reproduction Commands

```bash
python run_all_4_cosformer.py --models distilbert,roberta,gpt2

for cs in 16 32 64 128; do
    python run_all_4_cosformer.py --chunk_size $cs --output_suffix cs$cs
done
```

---