# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Params before | 66,955,010 |
| Params after | {{D_PARAMS}} |
| Δ Params | {{D_PARAM_CHANGE}} |
| Accuracy | {{D_ACC}}% |
| F1 | {{D_F1}} |
| Precision | {{D_PREC}} |
| Recall | {{D_REC}} |
| Loss | {{D_LOSS}} |
| Peak VRAM | {{D_VRAM}} MB |
| Time | {{D_TIME}} s |
| Aux loss | {{D_AUX}} |
| Max/Min utilization | {{D_MAXMIN}} |

## C.2 RoBERTa Metrics

| Metric | Value |
|---|---|
| Params before | 124,647,170 |
| Params after | {{R_PARAMS}} |
| Δ Params | {{R_PARAM_CHANGE}} |
| Accuracy | {{R_ACC}}% |
| ... | ... |

## C.3 GPT-2 Metrics

| Metric | Value |
|---|---|
| Params before | 124,441,344 |
| Params after | {{G_PARAMS}} |
| Δ Params | {{G_PARAM_CHANGE}} |
| Accuracy | {{G_ACC}}% |
| ... | ... |

## C.4 Expert Utilization (DistilBERT)

| Expert | Utilization | Mean gate prob |
|---|---|---|
| 0 | {{D_E0}}% | {{D_P0}} |
| 1 | {{D_E1}}% | {{D_P1}} |
| 2 | {{D_E2}}% | {{D_P2}} |
| 3 | {{D_E3}}% | {{D_P3}} |
| 4 | {{D_E4}}% | {{D_P4}} |
| 5 | {{D_E5}}% | {{D_P5}} |
| 6 | {{D_E6}}% | {{D_P6}} |
| 7 | {{D_E7}}% | {{D_P7}} |

## C.5 Number of Experts Ablation

| E | DistilBERT | Params | Ratio |
|---|---|---|---|
| 2 | {{D_E2_ACC}} | {{D_E2_PARAMS}} | {{D_E2_RATIO}} |
| 4 | {{D_E4_ACC}} | {{D_E4_PARAMS}} | {{D_E4_RATIO}} |
| 8 | {{D_E8_ACC}} | {{D_E8_PARAMS}} | {{D_E8_RATIO}} |
| 16 | {{D_E16_ACC}} | {{D_E16_PARAMS}} | {{D_E16_RATIO}} |

## C.6 Top-k Ablation

| k | DistilBERT | FLOPs |
|---|---|---|
| 1 | {{D_K1_ACC}} | 1× |
| 2 | {{D_K2_ACC}} | 2× |
| 4 | {{D_K4_ACC}} | 4× |

## C.7 Aux Loss Weight Ablation

| λ_aux | DistilBERT | Max/Min |
|---|---|---|
| 0.0 | {{D_L0_ACC}} | {{D_L0_MAXMIN}} |
| 0.01 | {{D_L001_ACC}} | {{D_L001_MAXMIN}} |
| 0.1 | {{D_L01_ACC}} | {{D_L01_MAXMIN}} |
| 1.0 | {{D_L1_ACC}} | {{D_L1_MAXMIN}} |

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