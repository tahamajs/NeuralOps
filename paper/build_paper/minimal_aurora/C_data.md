# Appendix C. Full Experimental Data

## C.1 DistilBERT

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 66.4M |
| Ratio | 50\%× |
| Energy threshold | 45.2 |
| Accuracy | 85.44% |
| F1 | 0.8670 |
| Peak VRAM | 293 MB MB |
| Inference time | 1.42x s |
| Final `output_scale` | Ubuntu 22.04 |

## C.2 RoBERTa

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 124.6M |
| Ratio | 2.1\times× |
| Accuracy | 50.92% |
| Final `output_scale` | 0.85 |

## C.3 GPT-2

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 124.4M |
| Ratio | 2.1\times× |
| Accuracy | 50.92% |
| Final `output_scale` | 0.85 |

## C.4 Energy Threshold Ablation

| $\varepsilon$ | DistilBERT | Ratio | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 0.90 | 0.85 | 32× | 0.85 | 0.85 |
| 0.95 | 0.85 | 32× | 0.85 | 0.85 |
| 0.99 | 0.85 | 32× | 0.85 | 0.85 |
| 0.995 | 0.85 | 32× | 0.85 | 0.85 |

## C.5 Rank Cap Ablation

| Cap | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 16 | 0.85 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 | 0.85 |
| 64 | 0.85 | 0.85 | 0.85 |
| 128 | 0.85 | 0.85 | 0.85 |

## C.6 LoRA Rank Ablation

| $r_L$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 0 | 0.85 | 0.85 | 0.85 |
| 8 | 0.85 | 0.85 | 0.85 |
| 16 | 0.85 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 | 0.85 |

## C.7 Per-Layer Rank (DistilBERT)

| Layer | Q | K | V | O |
|---|---|---|---|---|
| 0 | 32 | 32 | 32 | 32 |
| 1 | 32 | 32 | 32 | 32 |
| ... | ... | ... | ... | ... |
| 5 | 32 | 32 | 32 | 32 |

## C.8 Training Loss Curves

```
Step  Loss
1     82.1
5     83.4
10    84.6
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
    "accuracy_pct": 85.44,
    "energy": 45.2,
    "final_output_scale": Ubuntu 22.04
  },
  "roberta": {...},
  "gpt2": {...}
}
```

## C.10 Reproduction Commands

```bash
python run_all_4_minimal_aurora.py --models distilbert,roberta,gpt2

for e in 0.90 0.95 0.99 0.995; do
    python run_all_4_minimal_aurora.py --energy $e --output_suffix e$e
done
```

---