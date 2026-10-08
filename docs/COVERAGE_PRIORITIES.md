# Coverage Priorities

Benchmark prioritization framework to guide future evaluation runs and close remaining coverage gaps across `neuralops`.

## Priority Criteria

- **Critical**: Missing modules evaluated on primary anchor architectures (`distilbert-base-uncased`, `gpt2`) and core GLUE tasks (`sst2`, `mnli`, `wikitext2`).
- **High**: Partially covered modules (<= 2 combinations covered) missing evaluations on standard downstream classification tasks (`mrpc`, `ag_news`).
- **Medium**: Exploration of moderate-depth combinations (e.g. `qnli`, `c4`, multi-rank bottleneck sweeps).
- **Low**: Rare or specialized evaluations (e.g. edge-case datasets `cola`, `listops`, `piqa`).

---

## 1. Critical Priority Gaps

These 15 modules have **0 benchmark runs** in the master dataset and are essential to achieving complete coverage across standard benchmarks:

| Priority | Module | Target Models | Target Datasets | Requirements |
|:---|:---|:---|:---|:---|
| **Critical** | `grouped_query_attention` | `distilbert-base-uncased`, `gpt2` | `sst2`, `wikitext2` | Standard PyTorch |
| **Critical** | `adaptive_sparse_attn` | `distilbert-base-uncased`, `gpt2` | `sst2`, `wikitext2` | Standard PyTorch |
| **Critical** | `x_attention` | `distilbert-base-uncased`, `gpt2` | `sst2`, `wikitext2` | Fast kernel runtime |
| **Critical** | `svd_linear_attn` | `distilbert-base-uncased`, `gpt2` | `sst2`, `wikitext2` | SVD attention runtime |
| **Critical** | `simplessm` | `distilbert-base-uncased`, `gpt2` | `sst2`, `wikitext2` | Sequence recurrent runtime |
| **Critical** | `selective_mamba` | `distilbert-base-uncased`, `gpt2` | `sst2`, `wikitext2` | GPU / Mamba CUDA kernel |
| **Critical** | `hgrn2_gated_outer` | `distilbert-base-uncased`, `gpt2` | `sst2`, `wikitext2` | GPU / Recurrent kernel |
| **Critical** | `hymba_transx_attn` | `distilbert-base-uncased`, `gpt2` | `sst2`, `wikitext2` | Hybrid SSM kernel |
| **Critical** | `fused_svd_kernel_attention` | `distilbert-base-uncased`, `gpt2` | `sst2`, `wikitext2` | Triton / GPU |
| **Critical** | `inject_triton_cosformer` | `distilbert-base-uncased`, `gpt2` | `sst2`, `wikitext2` | Triton / GPU |
| **Critical** | `inject_triton_svd_linear_attention` | `distilbert-base-uncased`, `gpt2` | `sst2`, `wikitext2` | Triton / GPU |
| **Critical** | `triton_cosformer_inject` | `distilbert-base-uncased`, `gpt2` | `sst2`, `wikitext2` | Triton / GPU |
| **Critical** | `triton_svd_linear_attention_inject` | `distilbert-base-uncased`, `gpt2` | `sst2`, `wikitext2` | Triton / GPU |
| **Critical** | `inject_xattention` | `distilbert-base-uncased`, `gpt2` | `sst2`, `wikitext2` | Fast kernel runtime |
| **Critical** | `adaptive_svd_attn` | `distilbert-base-uncased`, `gpt2` | `sst2`, `wikitext2` | Standard SVD runtime |

---

## 2. High Priority Gaps

Modules with single-combination coverage (only 1 evaluated configuration) that require expansion to cross-architecture pairs:

| Priority | Module | Current Coverage | Recommended Expansion |
|:---|:---|:---:|:---|
| **High** | `chela_slay_attention` | 1 | Expand to `gpt2` + `wikitext2`, `roberta` + `mnli` |
| **High** | `eat_attention` | 1 | Expand to `gpt2` + `wikitext2`, `distilbert` + `sst2` |
| **High** | `expert_choice_moe_attention` | 1 | Expand to `gpt2` + `wikitext2` |
| **High** | `fusion_attention` | 1 | Expand to `distilbert` + `sst2`, `mrpc` |
| **High** | `selective_ssm` | 1 | Expand to `gpt2` + `wikitext2` |
| **High** | `triton_cosformer` | 1 | Expand to `distilbert` + `sst2` (CUDA) |
| **High** | `triton_svd_linear_attention` | 1 | Expand to `distilbert` + `sst2` (CUDA) |

---

## 3. Medium Priority Gaps

Modules with exactly 2 combinations evaluated:
- `aatc_xkv_compressor`, `compressed_mha_attention`, `flash_attention3_fp8`, `gfws`, `lsc_compressor`, `mgie_budget`, `naive_svd`, `rala_compressor`, `residual_svd_compressor`, `roast_hashed_linear`, `subspace_projection`, `svd_linear_decay`, `svd_seq`, `tomoe_compressor`, `upi_mamba`, `vig_attention`, `vspec_attention`, `wgqa_attention`, `zeros`, `zeros_attention`.

*Action*: Evaluate across multi-rank sweeps (r = 16, 32, 64, 128) and additional GLUE subsets (`mrpc`, `ag_news`).

---

## 4. Low Priority Gaps

- Long-sequence scaling evaluations (`listops`, `ruler`, `piqa`) on specialized benchmark configurations.
- Niche legacy modules that have been superseded by canonical replacements.
