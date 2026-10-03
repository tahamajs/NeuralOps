<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full data
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Accuracy | 90.94% |
| Δ | +0.23% |
| F1 | 0.9131 |
| Params (after) | 66.4M |
| Peak VRAM | 319 MB MB |
| Inference time | 1.42x s |
| Path | Pruned |
| Keep ratio | 0.5 |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Accuracy | 86.24% |
| Δ | +7.68% |
| F1 | 0.8739 |
| Params (after) | 124.6M |
| Peak VRAM | 648 MB MB |
| Inference time | 2.3 s s |
| Path | Pruned |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Accuracy | 87.27% |
| Δ | +7.91% |
| F1 | 0.8835 |
| Params (after) | 124.4M |
| Peak VRAM | 333 MB MB |
| Inference time | 2.3 s s |
| Path | **No prune (causal)** |

## C.4 Keep Ratio Ablation

| Ratio | DistilBERT | RoBERTa |
|---|---|---|
| 0.25 | 0.85 | 0.85 |
| 0.50 | 0.85 | 0.85 |
| 0.75 | 0.85 | 0.85 |
| 1.00 | 0.85 | 0.85 |

## C.5 Importance Method

| Method | DistilBERT | RoBERTa |
|---|---|---|
| `var` | 0.85 | 0.85 |
| `l2` | 0.85 | 0.85 |

## C.6 CLS Protection

| Setting | DistilBERT | RoBERTa |
|---|---|---|
| Protect | 0.85 | 0.85 |
| No protect | 0.85 | 0.85 |

## C.7 Raw JSON

```json
{
  "distilbert": {"acc": 90.94, "path": "pruned"},
  "roberta": {"acc": 86.24, "path": "pruned"},
  "gpt2": {"acc": 87.27, "path": "no-prune"}
}
```

## C.8 Reproduction Commands

```bash
python run_all_4_fasa.py --models distilbert,roberta,gpt2

for r in 0.25 0.5 0.75 1.0; do
    python run_all_4_fasa.py --fasa_keep_ratio $r --output_suffix k$r
done
```

---