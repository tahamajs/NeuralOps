# Parameter Delta and Real Compression Audit

Every number cited below is extracted directly from the execution logs and `full_metrics.json` outputs on the remote GPU cluster.

## 1. Summary of Categories
- **COMPRESSOR** ($\Delta < 0$ and $\text{Comp. Layers} > 0$): Truly reduces parameter count via low-rank factorisation.
- **INFLATED** ($\Delta > 0$ and $\text{Comp. Layers} > 0$): Replaces attention layers but introduces net-positive parameters (e.g., LoRA residual adapters or expanded projections without dimension reduction).
- **ADAPTER_ONLY** ($\text{Comp. Layers} = 0$): Custom architectural block that bypasses standard SVD layer compression entirely.
- **BROKEN** ($\Delta > 50\text{M}$ or catastrophic accuracy drop): Architectural failure causing pathological parameter growth or model corruption.

---

## 2. Complete Verification Table
<!-- source: results/all_modules/*/*full_metrics.json and results/fixed_aegis/*/*full_metrics.json -->
| Module | Model / Dataset | Params Before | Params After | Delta | Comp. Layers | Category | Real Compression? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`adaptive_svd_attention`** | DistilBERT / SST-2 | 66,955,010 | 58,107,650 | **-8,847,360** | 6 | **COMPRESSOR** | **YES** |
| **`decay_gated_linear_attention`** | DistilBERT / SST-2 | 66,955,010 | 55,803,722 | **-11,151,288** | 6 | **COMPRESSOR** | **YES** |
| `adaptive_span_attention` | DistilBERT / SST-2 | 66,955,010 | 67,544,906 | +589,896 | 6 | INFLATED | NO |
| `asvd_attention` | DistilBERT / SST-2 | 66,955,010 | 68,940,424 | +1,985,414 | 6 | INFLATED | NO |
| `chela_slay_attention` | DistilBERT / SST-2 | 66,955,010 | 67,600,202 | +645,192 | 6 | INFLATED | NO |
| `cosformer_attention` | DistilBERT / SST-2 | 66,955,010 | 67,544,840 | +589,830 | 6 | INFLATED | NO |
| `decay_linear_attn` | DistilBERT / SST-2 | 66,955,010 | 67,600,274 | +645,264 | 6 | INFLATED | NO |
| `fasa_attention` | DistilBERT / SST-2 | 66,955,010 | 67,544,834 | +589,824 | 6 | INFLATED | NO |
| `gated_linear_attn` | DistilBERT / SST-2 | 66,955,010 | 71,273,624 | +4,318,614 | 6 | INFLATED | NO |
| `gqa_attention` | DistilBERT / SST-2 | 66,955,010 | 67,544,834 | +589,824 | 6 | INFLATED | NO |
| `griffin_ssd_block` | DistilBERT / SST-2 | 66,955,010 | 71,217,482 | +4,262,472 | 6 | INFLATED | NO |
| `halo_regla_attention` | DistilBERT / SST-2 | 66,955,010 | 67,600,202 | +645,192 | 6 | INFLATED | NO |
| `hla_attention` | DistilBERT / SST-2 | 66,955,010 | 67,531,088 | +576,078 | 6 | INFLATED | NO |
| `aegis_hybrid` | DistilBERT / SST-2 | 66,955,010 | 56,703,506 | -10,251,504 | 0 | ADAPTER_ONLY | NO |
| `aegis_v1` | DistilBERT / SST-2 | 66,955,010 | 67,741,514 | +786,504 | 0 | ADAPTER_ONLY | NO |
| `aegis_v2` | DistilBERT / SST-2 | 66,955,010 | 68,499,986 | +1,544,976 | 0 | ADAPTER_ONLY | NO |
| `aegis_v2` | RoBERTa / MNLI | 124,647,939 | 134,824,995 | +10,177,056 | 0 | ADAPTER_ONLY | NO |
| `aegis_v3` | GPT-2 / AG News | 124,442,880 | 102,422,928 | -22,019,952 | 0 | ADAPTER_ONLY | NO |
| `aegis_v4` | GPT-2 / AG News | 124,442,880 | 102,422,928 | -22,019,952 | 0 | ADAPTER_ONLY | NO |
| `aegis_v5` | DistilBERT / SST-2 | 66,955,010 | 55,945,034 | -11,009,976 | 0 | ADAPTER_ONLY | NO |
| `adaptive_sparse_attention` | DistilBERT / SST-2 | 66,955,010 | 81,719,120 | +14,764,110 | 0 | ADAPTER_ONLY | NO |
| `adasplash_attention` | DistilBERT / SST-2 | 66,955,010 | 67,544,834 | +589,824 | 0 | ADAPTER_ONLY | NO |
| `cluster_attention` | DistilBERT / SST-2 | 66,955,010 | 67,544,846 | +589,836 | 0 | ADAPTER_ONLY | NO |
| `factorised_matrix_state` | DistilBERT / SST-2 | 66,955,010 | 68,134,730 | +1,179,720 | 0 | ADAPTER_ONLY | NO |
| `flash_attention3` | DistilBERT / SST-2 | 66,955,010 | 67,544,834 | +589,824 | 0 | ADAPTER_ONLY | NO |
| `full_attention` | DistilBERT / SST-2 | 66,955,010 | 67,544,834 | +589,824 | 0 | ADAPTER_ONLY | NO |
| `galerkin_attention` | DistilBERT / SST-2 | 66,955,010 | 67,544,834 | +589,824 | 0 | ADAPTER_ONLY | NO |
| `hamming_attention` | DistilBERT / SST-2 | 66,955,010 | 67,544,984 | +589,974 | 0 | ADAPTER_ONLY | NO |
| **`expert_choice_moe_attention`** | DistilBERT / SST-2 | 66,955,010 | 180,947,720 | **+113,992,710** | 6 | **BROKEN** | **NO** |

---

## 3. Investigation of "Comp. Layers = 6"
Inspection of the underlying framework implementation in `neuralops/run_quality_sweep.py` confirms that for injected module classes, `num_compressed_layers` counts the number of transformer attention blocks that were successfully replaced with the custom attention class (e.g., 6 layers for DistilBERT). However, unless the replacement module itself actively reduces rank (as in `adaptive_svd_attention` and `decay_gated_linear_attention`), it preserves full base projections and adds auxiliary weights (such as LoRA recovery parameters: $4 \times 768 \times 16 \times 2 = 98,304$ params per layer $\times 6 = 589,824$ total), resulting in a net-positive parameter delta.
