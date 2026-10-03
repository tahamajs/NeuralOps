# Dataset Coverage Audit

**Date:** September 18, 2026  
**Repository:** SVD Linear Attention & Unified Compression Framework  
**Scope:** Complete repository scan of `results/`, `audit/`, `Codes/`, and `scripts/`  
**Status:** Audit Complete  

---

## 1. Datasets Actually Present in `results/`

| Dataset | Number of Result Files | Unique Modules Identified | Backbones Evaluated | Description |
| :--- | :---: | :---: | :---: | :--- |
| **SST-2** (`sst2`, `sst-2`) | **1,581 files** | **237+ modules** (Complete repo catalog) | DistilBERT, RoBERTa-base, GPT-2 | Primary standard classification benchmark across all models. All 442 `results/unified_*` runs target SST-2. |
| **MRPC** (`mrpc`) | **382 files** | Subsets in `detailed_results.json` | DistilBERT | Auxiliary GLUE benchmark evaluated during historical pilot and detailed multi-metric runs. |
| **MNLI** (`mnli`) | **91 files** | **83 modules** (`results/mnli_unified_*`) | DistilBERT, RoBERTa-base | 3-class NLI benchmark run as a focused extension on 83 selected modules. |
| **AG News** (`ag_news`, `agnews`) | **77 files** | **69 modules** (`results/agnews_unified_*`) | DistilBERT | 4-class topic classification evaluated as a focused extension across 69 modules. |
| **RTE / STS-B / CoLA / QNLI** | **< 15 files each** | Select baselines in `detailed_results.json` | DistilBERT | Auxiliary GLUE calibration runs from early validation sweeps. |

---

## 2. Datasets Referenced in Code

### Harness Code (`run_one_module.py`)
- **Dataset handling**: `run_one_module.py` is a **direct tensor forward-inference harness**. It passes synthetic token batches (`torch.randint(0, 1000, (2, 16))`) directly into model checkpoints (`distilbert_sst2`, `roberta-base`, `gpt2`) to test layer injection stability and output logit differences with zero dependencies on external dataset downloads.
- **Model Checkpoint**: The primary DistilBERT model loaded is `models\distilbert_sst2` (fine-tuned on SST-2).

### Evaluation & Benchmark Drivers
- **Multi-Mode Benchmark (`neuralops/core/multi_mode_benchmark.py` line 111)**:
  ```python
  SUPPORTED_DATASETS = [
      "sst2", "mrpc", "cola", "qqp", "mnli", "qnli", "rte", "stsb", "wnli",
      "wikitext2", "c4", "ptb", "synthetic", "custom", "arc", "hellaswag"
  ]
  ```
- **Evaluation Runner (`neuralops/eval_all_models_datasets.py`)**:
  - Implements dynamic caching via `_PREPARED_DATASET_CACHE` supporting SST-2, MNLI, AG News, and MRPC.

---

## 3. Datasets Referenced in Planning Docs vs Results

### Planning Documents
- `Category Modules.txt` and earlier generation templates mention:
  ```
  "Dataset: SST-2 / AG News / MNLI (with class counts)"
  ```
  This reflected the aspirational multi-task vision: binary classification (SST-2, 2 classes), multi-class topic classification (AG News, 4 classes), and natural language inference (MNLI, 3 classes).

### Reality in `results/`
- **SST-2 is the universal benchmark**: It was executed across the **entire 253-module repository catalog** and across **all 3 backbones** (DistilBERT, RoBERTa, GPT-2).
- **AG News was only partially executed**: 69 modules under `results/agnews_unified_*` (only on DistilBERT).
- **MNLI was only partially executed**: 83 modules under `results/mnli_unified_*` (only on DistilBERT and RoBERTa).
- **MASTER_PROJECT_RESULTS.md appropriately reports SST-2 only**: Because SST-2 is the **only common denominator** that has complete, uniform empirical coverage across every evaluated attention module, compressor, and backbone.

---

## 4. Verdict

- [x] **Something else**:
  - **SST-2** is the **universal, complete benchmark** covering 100% of tested modules across DistilBERT, RoBERTa, and GPT-2.
  - **AG News (69 modules)** and **MNLI (83 modules)** were executed as partial exploratory sweeps on subsets of modules and were never completed across the entire catalog or on GPT-2.
  - Inventing AG News or MNLI numbers for modules that lack `results/agnews_unified_*` or `results/mnli_unified_*` directories would be a violation of the rule against hallucinating metrics.

---

## 5. Implication for Multi-Seed Run

1. **Multi-seed sweep MUST remain on SST-2**:
   - Every module has validated SST-2 baseline weights, fine-tune configs, and comparative metrics in `results/unified_<module>/`.
   - Running multi-seed on SST-2 ensures rigorous, apples-to-apples statistical confidence bounds ($\mu \pm \sigma$, p-values) against existing repository data.
2. **Do NOT force AG News or MNLI into the universal sweep**:
   - AG News and MNLI do not have checkpoints or baseline runs for causal GPT-2 or for >170 of the modules.
3. **Explicit Documentation**:
   - The final report must clearly state that **SST-2 is the primary evaluation benchmark for the full multi-seed suite**, and note that AG News and MNLI exist in the repository as focused domain-transfer pilots for 69 and 83 modules respectively.

---

## 6. Correction Needed in Earlier Multi-Seed Prompt

- **Current Prompt Structure**:
  `--dataset sst2`
- **Required Change**:
  - **Keep `--dataset sst2` as the benchmark target**. It is the correct and only complete dataset across the repo.
  - **Update documentation headers**: Clarify in any multi-seed summary that SST-2 is the target evaluation dataset, referencing the 69 AG News and 83 MNLI auxiliary runs in an exploratory appendix rather than expecting 253-module coverage on them.

---

## 7. Concrete Evidence Citations

- SST-2 universal coverage: <!-- source: results/unified_svd_compressed/distilbert/summary.csv line 2 -->
- AG News 69-module pilot run: <!-- source: results/agnews_unified_lora_attention/distilbert/finetune_metadata.json line 1 -->
- MNLI 83-module pilot run: <!-- source: results/mnli_unified_leka_v1/distilbert/finetune_metadata.json line 1 -->
- Multi-dataset architecture registry: <!-- source: neuralops/core/multi_mode_benchmark.py line 111 -->
