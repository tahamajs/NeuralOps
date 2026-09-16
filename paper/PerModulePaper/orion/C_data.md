<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full data
     DEPENDENCIES: 07_results.md
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Accuracy | {{D_ACC}}% |
| Δ | {{D_DELTA}}% |
| F1 | {{D_F1}} |
| Params (before) | 66,955,010 |
| Params (after) | {{D_PARAMS}} |
| Reduction | {{D_RED}}× |
| Peak VRAM | {{D_VRAM}} MB |
| Inference time | {{D_TIME}} s |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Accuracy | {{R_ACC}}% |
| Δ | {{R_DELTA}}% |
| F1 | {{R_F1}} |
| Params (before) | 124,647,170 |
| Params (after) | {{R_PARAMS}} |
| Reduction | {{R_RED}}× |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Accuracy | {{G_ACC}}% |
| Δ | {{G_DELTA}}% |
| F1 | {{G_F1}} |
| Params (before) | 124,441,344 |
| Params (after) | {{G_PARAMS}} |
| Reduction | {{G_RED}}× |

## C.4 SVD Energy Ablation

| Energy | DistilBERT | Reduction |
|---|---|---|
| 0.90 | {{D_E90}} | {{D_E90_R}}× |
| 0.95 | {{D_E95}} | {{D_E95_R}}× |
| 0.99 | {{D_E99}} | {{D_E99_R}}× |

## C.5 Head Split Ablation

| Split | DistilBERT |
|---|---|
| 1/2:1/2 | {{D_SPLIT_HALF}} |
| 2/3:1/3 | {{D_ACC}} |
| 3/4:1/4 | {{D_SPLIT_34}} |
| 1:0 | {{D_ALL_KERNEL}} |
| 0:1 | {{D_ALL_FULL}} |

## C.6 Per-Layer Rank Selection (DistilBERT)

| Layer | q_rank | k_rank | v_rank | o_rank |
|---|---|---|---|---|
| 0 | {{D_L0_Q}} | {{D_L0_K}} | {{D_L0_V}} | {{D_L0_O}} |
| 1 | {{D_L1_Q}} | {{D_L1_K}} | {{D_L1_V}} | {{D_L1_O}} |
| ... | ... | ... | ... | ... |

## C.7 Raw JSON

```json
{
  "distilbert": {"acc": {{D_ACC}}, "params": {{D_PARAMS}}, "reduction": {{D_RED}}},
  "roberta": {"acc": {{R_ACC}}, "params": {{R_PARAMS}}, "reduction": {{R_RED}}},
  "gpt2": {"acc": {{G_ACC}}, "params": {{G_PARAMS}}, "reduction": {{G_RED}}}
}
```

## C.8 Reproduction Commands

```bash
python run_all_4_orion.py --models distilbert,roberta,gpt2

for e in 0.90 0.95 0.99; do
    python run_all_4_orion.py --svd_energy $e --output_suffix e$e
done
```
```

---