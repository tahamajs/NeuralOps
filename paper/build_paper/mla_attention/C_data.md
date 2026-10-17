<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 Model Evaluations on SST-2 Validation

| Model | Variant | Accuracy | Macro F1 | Active Params | KV Cache / Token |
|:---|:---|:---:|:---:|:---:|:---:|
| DistilBERT | Baseline MHA | 90.71% | 0.9082 | 66,955,010 | 18.4 KB |
| DistilBERT | **MLA ($d_c=128$)** | **89.91%** | **0.8998** | **59,142,400** | **4.6 KB (4.0×)** |
| RoBERTa-base | Baseline MHA | 94.84% | 0.9492 | 124,647,170 | 36.9 KB |
| RoBERTa-base | **MLA ($d_c=128$)** | **93.58%** | **0.9362** | **110,845,440** | **9.2 KB (4.0×)** |
| GPT-2 Small | Baseline MHA | 83.20% | 0.8340 | 124,441,344 | 36.9 KB |
| GPT-2 Small | **MLA ($d_c=128$)** | **82.35%** | **0.8251** | **110,639,614** | **9.2 KB (4.0×)** |

## C.2 Full MNLI Benchmark Verification

Evaluation over $N=9,815$ validation pairs confirms robust hypothesis-premise inference:
- Accuracy: **81.42%** (Wilson 95% CI: $[80.64\%, 82.18\%]$)
- Baseline Accuracy: 81.97% ($\Delta = -0.55\%$, within equivalence margin)
- Compression Gate: **PASS**
