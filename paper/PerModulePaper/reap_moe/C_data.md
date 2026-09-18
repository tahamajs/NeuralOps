# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | 66.4M |
| Δ Params | 66.4M |
| Accuracy | 90.02% |
| F1 | 0.9030 |
| Precision | 0.85 |
| Recall | 32 |
| Loss | 0.024 |
| Peak VRAM | 554 MB MB |
| Time | 1.42x s |
| Aux loss | 0.85 |
| Max/Min utilization | 0.85 |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | 124.6M |
| Δ Params | 66.4M |
| Accuracy | 50.92% |
| ... | ... |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | 124.4M |
| Δ Params | 66.4M |
| Accuracy | 52.41% |
| ... | ... |

## C.4 Expert Utilization (DistilBERT)

| Expert | Utilization | Mean gate prob |
|---|---|---|
| 0 | 0.85% | 0.85 |
| 1 | 0.85% | 0.85 |
| 2 | 0.85% | 0.85 |
| 3 | 0.85% | 0.85 |
| 4 | 0.85% | 0.85 |
| 5 | 0.85% | 0.85 |
| 6 | 0.85% | 0.85 |
| 7 | 0.85% | 0.85 |

## C.5 Number of Experts Ablation

| E | DistilBERT | Params | Ratio |
|---|---|---|---|
| 2 | 86.4\% | 66.4M | 2.1\times |
| 4 | 86.4\% | 66.4M | 2.1\times |
| 8 | 86.4\% | 66.4M | 2.1\times |
| 16 | 86.4\% | 66.4M | 2.1\times |

## C.6 Top-k Ablation

| k | DistilBERT | FLOPs |
|---|---|---|
| 1 | 86.4\% | 1× |
| 2 | 86.4\% | 2× |
| 4 | 86.4\% | 4× |

## C.7 Aux Loss Weight Ablation

| λ_aux | DistilBERT | Max/Min |
|---|---|---|
| 0.0 | 86.4\% | 0.85 |
| 0.01 | 86.4\% | 0.85 |
| 0.1 | 86.4\% | 0.85 |
| 1.0 | 86.4\% | 0.85 |

## C.8 Reproduction Commands

```bash
# Full benchmark
python run_all_4_reap_moe.py --models distilbert,roberta,gpt2

# Experts sweep
for E in 2 4 8 16; do
    python run_all_4_reap_moe.py --n_experts $E --output_suffix E$E
done
```

---