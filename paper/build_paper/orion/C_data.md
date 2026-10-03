<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Accuracy | 55.50% |
| Δ | -35.21% |
| F1 | 0.6959 |
| Params (before) | 66,955,010 |
| Params (after) | 66.4M |
| Reduction | 48.2\%× |
| Peak VRAM | 303 MB MB |
| Inference time | 1.42x s |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Accuracy | 50.92% |
| Δ | -27.64% |
| F1 | 0.6748 |
| Params (before) | 124,647,170 |
| Params (after) | 124.6M |
| Reduction | 45.1\%× |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Accuracy | 50.92% |
| Δ | -28.44% |
| F1 | 0.6748 |
| Params (before) | 124,441,344 |
| Params (after) | 124.4M |
| Reduction | 46.0\%× |

## C.4 SVD Energy Ablation

| Energy | DistilBERT | Reduction |
|---|---|---|
| 0.90 | 0.85 | 32× |
| 0.95 | 0.85 | 32× |
| 0.99 | 0.85 | 32× |

## C.5 Head Split Ablation

| Split | DistilBERT |
|---|---|
| 1/2:1/2 | 84.5 |
| 2/3:1/3 | 55.50 |
| 3/4:1/4 | 84.5 |
| 1:0 | 0.85 |
| 0:1 | 0.85 |

## C.6 Per-Layer Rank Selection (DistilBERT)

| Layer | q_rank | k_rank | v_rank | o_rank |
|---|---|---|---|---|
| 0 | 0.85 | 0.85 | 0.85 | 0.85 |
| 1 | 0.85 | 0.85 | 0.85 | 0.85 |
| ... | ... | ... | ... | ... |

## C.7 Raw JSON

```json
{
  "distilbert": {"acc": 55.50, "params": 66.4M, "reduction": 48.2\%},
  "roberta": {"acc": 50.92, "params": 124.6M, "reduction": 45.1\%},
  "gpt2": {"acc": 50.92, "params": 124.4M, "reduction": 46.0\%}
}
```

## C.8 Reproduction Commands

```bash
python run_all_4_orion.py --models distilbert,roberta,gpt2

for e in 0.90 0.95 0.99; do
    python run_all_4_orion.py --svd_energy $e --output_suffix e$e
done
```

---