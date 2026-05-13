
| Attention Type              | Eval Loss ↓ | Matthews Corr. | Duration (s) | Parameters  | Notes                      |
|-----------------------------|-------------|----------------|--------------|-------------|----------------------------|
| gated_deltanet_block        | 0.6708      | 0.0            | 1663.6       | 63,420,674  | Lowest loss among all      |
| hgrn2_layer                 | 0.6713      | 0.0            | 1919.9       | 63,411,458  | Slightly more parameters   |
| flash3_fp8                  | 0.6790      | 0.0            | 228–272      | 66,955,010  | Faster, but still no learning |
| soa_attn                    | 0.6838      | 0.0            | 194–311      | 56,914,178  | Smaller model size         |
| goldfinch_ahn               | 0.6910      | 0.0            | 227–255      | 70,489,352  | Largest parameter count    |
| infini                      | 0.6914      | 0.0            | 324          | 67,544,834  |                            |
| aegis_v4                    | 0.6924      | 0.0            | 319–333      | 55,355,210  | Smallest parameter count   |

**Failed runs (no output / immediate error):**  
`svd_proj_attn`, `linformer`, `compressed_mha`, `moba`, `kda`, `routing`, `factorised_matrix_state`, `aegis_hybrid`

---

### Key Observations
1. **Zero Matthews correlation across the board** – None of the efficient attention mechanisms managed to capture the linguistic patterns needed for CoLA. The models essentially default to predicting the majority label, indicating a complete breakdown of learning on this binary acceptability task.
2. **Loss values cluster around 0.67–0.69** – While lower loss (closer to log(0.5) ≈ 0.693) suggests the model is slightly better than random, the final predictions are still trivially constant. The marginally lower losses for `gated_deltanet_block` and `hgrn2_layer` may reflect slightly less overconfident guesses, not genuine task learning.
3. **Many efficient attention modules are incompatible with the CoLA setup** – Several methods (`svd_proj_attn`, `linformer`, `compressed_mha`, etc.) crashed immediately or produced no results, likely due to architectural or implementation constraints when plugged into a classification‑only DistilBERT.
4. **Training duration varies widely** – From ~200 s (`soa_attn`) to ~1900 s (`hgrn2_layer`), partly due to hardware (Apple MPS vs. other backends) and partly due to the inherent computational cost of each attention mechanism.

### Teacher Summary
> *On the CoLA benchmark, substituting standard DistilBERT self‑attention with any of the tested efficient attention modules results in catastrophic failure – all models achieve zero Matthews correlation. Even the best‑performing variant (`gated_deltanet_block`) simply predicts a constant label. This suggests that these lightweight attention replacements cannot capture the fine‑grained syntactic and semantic information required for linguistic acceptability, at least with only two epochs and default hyperparameters. Further investigation with longer training, learning rate tuning, or incorporation of task‑specific inductive biases would be necessary to assess their true potential on GLUE’s CoLA.*