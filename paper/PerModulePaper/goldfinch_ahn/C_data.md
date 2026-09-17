<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Accuracy | 50.92% |
| Δ | -39.79% |
| F1 | 0.6748 |
| Params (before) | 66,955,010 |
| Params (after) | 66.4M |
| Peak VRAM | 353 MB MB |
| Inference time | 1.42x s |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Accuracy | 50.92% |
| Δ | -27.64% |
| F1 | 0.6748 |
| Params (after) | 124.6M |
| Peak VRAM | 698 MB MB |
| Inference time | 2.3 s s |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Accuracy | 50.92% |
| Δ | -28.44% |
| F1 | 0.6748 |
| Params (after) | 124.4M |
| Peak VRAM | 596 MB MB |
| Inference time | 2.3 s s |

## C.4 Window Size Ablation

| Window | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 16 | 0.85 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 | 0.85 |
| 64 | 0.85 | 0.85 | 0.85 |
| 128 | 0.85 | 0.85 | 0.85 |
| ∞ (full) | 0.85 | 0.85 | 0.85 |

## C.5 Gate Bias Ablation

| Bias | Initial g | DistilBERT |
|---|---|---|
| 0.0 | 0.50 | 0.85 |
| 1.0 | 0.73 | 0.85 |
| 1.8 | 0.86 | 0.85 |
| 3.0 | 0.95 | 0.85 |

## C.6 Preservation Gate Ablation

| Init | Initial α | DistilBERT |
|---|---|---|
| 0.0 | 0.50 | 0.85 |
| 1.0 | 0.73 | 0.85 |
| 2.0 | 0.88 | 0.85 |
| 5.0 | 0.99 | 0.85 |

## C.7 Remove Long-Memory Branch

| Config | DistilBERT |
|---|---|
| Full | 50.92% |
| Without long-memory | 0.85% |

## C.8 Raw JSON

```json
{
  "distilbert": {"acc": 50.92, "params": 66.4M},
  "roberta": {"acc": 50.92, "params": 124.6M},
  "gpt2": {"acc": 50.92, "params": 124.4M}
}
```

## C.9 Reproduction Commands

```bash
python run_all_4_goldfinch_ahn.py --models distilbert,roberta,gpt2

for W in 16 32 64 128; do
    python run_all_4_goldfinch_ahn.py --window_size $W --output_suffix w$W
done
```

---