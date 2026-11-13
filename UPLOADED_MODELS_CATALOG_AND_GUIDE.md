# NeuralOps Hugging Face Hub Catalog & Model Card Guide

**Namespace:** `Hooshaai` ([Hugging Face Hub](https://huggingface.co/Hooshaai))
**Repository:** NeuralOps (`v2.0.6`)  
**Authority:** Aligned with `AGENTS.md` (§21 Cached Distillation & Dual Zoo Verification Protocol, §0 Non-negotiable principles, §8 Three-level kernel contract, §12 Results & Artifacts).

---

## 1. Executive Summary & Architecture Overview

The **NeuralOps Hugging Face Hub Ecosystem** is an open, reproducible catalog of pretrained transformer architectures that have undergone surgical attention transplant and post-surgery calibration healing across six foundational model zoos.

All published artifacts adhere strictly to the **Zero-Fake Serialization** and **Parameter-Efficient Delta** standard: instead of re-uploading hundreds of redundant gigabytes containing identical frozen backbone weights, NeuralOps serializes modular surgical deltas (`attention_weights.safetensors` + `surgical_manifest.json`) that can be loaded in one line directly from Hugging Face Hub via `neuralops.hub.load_surgeried_model`.

```
                      Hugging Face Hub (tahamajs/*)
                                    │
    ┌───────────────────────────────┴───────────────────────────────┐
    │                                                               │
┌───▼──────────────────────────┐                ┌───────────────────▼──────────────────┐
│   BERT Subquadratic Zoo      │                │    BERT Compressor & LoRA Zoo        │
│   (115 Attention Modules)    │                │    (82 Compression Engines)          │
│   Target: GLUE SST-2         │                │    Target: GLUE SST-2 / WikiText-2   │
└───┬──────────────────────────┘                └───────────────────┬──────────────────┘
    │                                                               │
    ├───────────────────────────────┬───────────────────────────────┤
    │                               │                               │
┌───▼──────────────────────────┐ ┌──▼──────────────────────────┐ ┌──▼──────────────────┐
│ Qwen2.5-0.5B Attention Zoo   │ │ Qwen2.5-0.5B Compressor Zoo │ │ LLaVA Multimodal Zoo │
│ (Hybrid 3:1 Subquadratic)    │ │ (Multi-Rank Pareto KD)      │ │ (ScienceQA Reasoning)│
│ Target: WikiText-2 PPL       │ │ Target: WikiText-2 PPL      │ │ Target: ScienceQA Acc│
└───┬──────────────────────────┘ └──┬──────────────────────────┘ └──┬───────────────────┘
    │                               │                               │
    └───────────────────────────────┼───────────────────────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │ TinyLlama-1.1B Attention Zoo  │
                    │ (Subquadratic GQA Decoders)   │
                    │ Target: WikiText-2 PPL        │
                    └───────────────────────────────┘
```

---

## 2. The Six NeuralOps Model Zoos

### Track 1: BERT Subquadratic Attention Zoo
- **Base Architecture:** `google-bert/bert-base-uncased` (109.5M parameters, 12 layers, hidden=768, heads=12, head_dim=64)
- **Scope:** 115 registered attention mechanisms transplanted across all 12 encoder self-attention blocks (`bert.encoder.layer.{i}.attention`).
- **Benchmark Suite:** GLUE SST-2 (Validation split, 872 binary sequence classification samples).
- **Teacher Baseline:** Fine-tuned BERT-Base teacher (`91.63%` accuracy).
- **Hub Repository Pattern:** `https://huggingface.co/Hooshaai/bert-{slug}-sst2`

### Track 2: BERT Compressor & LoRA-Healed Zoo
- **Base Architecture:** `google-bert/bert-base-uncased` (109.5M parameters)
- **Scope:** 82 registered compression algorithms (SVD, HNC-SVD, ASVD, ADMM, Sparse-GPT, Fisher-weighted factorization) evaluated across 4 physical ranks ($k \in \{64, 48, 32, 16\}$) with LoRA recovery healing.
- **Benchmark Suite:** GLUE SST-2 and WikiText-2.
- **Teacher Baseline:** `91.63%` SST-2 Accuracy | `10.48` WikiText-2 PPL.
- **Hub Repository Pattern:** `https://huggingface.co/Hooshaai/bert-comp-{slug}-sst2`

### Track 3: Qwen2.5-0.5B Subquadratic Attention Surgery Zoo
- **Base Architecture:** `Qwen/Qwen2.5-0.5B` (494M parameters, 24 layers, hidden=896, 14Q:2KV 7:1 GQA)
- **Scope:** Hybrid 3:1 subquadratic architecture (every 4th layer retains quadratic full-rank attention anchor, remaining 18 layers transplanted with subquadratic kernels).
- **Benchmark Suite:** WikiText-2 raw validation (causal language modeling perplexity).
- **Hub Repository Pattern:** `https://huggingface.co/Hooshaai/qwen-2.5-0.5b-{slug}-wikitext2`

### Track 4: Qwen2.5-0.5B Compressor Zoo
- **Base Architecture:** `Qwen/Qwen2.5-0.5B` (494M parameters)
- **Scope:** Multi-rank linear layer compression with 2-stage sparse knowledge distillation.
- **Benchmark Suite:** WikiText-2 raw validation (causal language modeling perplexity).
- **Hub Repository Pattern:** `https://huggingface.co/Hooshaai/qwen-2.5-0.5b-comp-{slug}-wikitext2`

### Track 5: LLaVA-Qwen-0.5B Multimodal Vision-Language Zoo
- **Base Architecture:** `llava-hf/llava-interleave-qwen-0.5b-hf` (892M parameters)
- **Scope:** Frozen SigLIP vision encoder + 2-layer MLP projector + 24-layer Qwen2 decoder with subquadratic attention surgery.
- **Benchmark Suite:** ScienceQA Image-Bearing STEM subset (1,024 test questions).
- **Teacher Baseline:** `68.20%` ScienceQA multimodal visual reasoning accuracy.
- **Hub Repository Pattern:** `https://huggingface.co/Hooshaai/llava-qwen-0.5b-{slug}-scienceqa`

### Track 6: TinyLlama-1.1B Attention Zoo
- **Base Architecture:** `TinyLlama/TinyLlama_v1.1` (1.10B parameters, 22 layers, hidden=2048, 32Q:4KV 8:1 GQA, bias-free)
- **Scope:** Subquadratic attention replacement with hybrid anchor retention.
- **Benchmark Suite:** WikiText-2 raw validation (causal language modeling perplexity).
- **Hub Repository Pattern:** `https://huggingface.co/Hooshaai/tinyllama-1.1b-{slug}-wikitext2`

---

## 3. Parameter-Efficient Delta Serialization (`Zero-Fake` Standard)

In accordance with **AGENTS.md §0.2** and **§21.2**, NeuralOps strictly enforces cryptographically authentic, reproducible serialization:

| File Name | Typical Size | Cryptographic & Architectural Role |
|---|---:|---|
| `attention_weights.safetensors` | 2.4 MB – 12.0 MB | Modular FP16 surgical attention delta or compressed factor tensors. |
| `surgical_manifest.json` | 1.8 KB – 4.5 KB | SHA-256 topology manifest specifying layer indices, donor mechanisms, tensor counts, and tensor shapes. |
| `config.json` | 1.2 KB – 2.0 KB | Complete Hugging Face architecture specification with `neuralops_surgery` metadata. |
| `training_metrics.json` | 400 B – 1.5 KB | Verifiable empirical metrics: exact wall-clock seconds, steps, throughput, peak VRAM, pre/post accuracy, and loss. |
| `compression_summary.json` | 800 B – 3.0 KB | Pareto evaluation ledger across all tested compression ranks with zero-shot and healed metrics. |
| `README.md` | 3.5 KB – 8.0 KB | Verified Model Card containing universal loader, telemetry, and empirical audit tables. |

### Footprint Reduction Evidence
For standard BERT-Base:
- Full uncompressed Hugging Face checkpoint (`pytorch_model.bin` / `model.safetensors`): **441.6 MB**
- NeuralOps modular surgical delta (`attention_weights.safetensors`): **2.6 MB**
- **Effective Storage & Transfer Reduction: 99.4% (170x smaller artifact)**.

---

## 4. AGENTS.md §21.1 Compliance: Offline Teacher Representation Caching

Every distillation training run conforms to the **NeuralOps Cached Distillation Protocol (§21.1)**:

1. **No Forward Passes through Teacher Models:** Distillation training strictly **MUST NOT** execute forward passes through heavy teacher models during student optimization.
2. **Pre-computed Offline Representations:** Teacher hidden states $\mathbf{H}_t \in \mathbb{R}^{B \times L \times D}$ and task logits $\mathbf{z}_t \in \mathbb{R}^{B \times C}$ are computed once and stored in high-speed RAM/shared memory (`/dev/shm/cached_teacher_representations.pt` or `data/cached_teacher_representations.pt`).
3. **Guarantees:**
   - **0 MB Teacher GPU VRAM Allocation** on the student GPU.
   - **2x Training Throughput Acceleration** by eliminating teacher computational graphs.
   - **Zero Teacher CUDA Out-Of-Memory (OOM) Risk**.

---

## 5. Hardware Telemetry Specifications

All empirical training, calibration, and benchmarking are conducted in a controlled environment:

- **Cluster Host:** Sharif University HPC Cluster (Rocks 6.1, Torque PBS scheduler)
- **Physical GPU:** NVIDIA GeForce RTX 3090 (24GB GDDR6X, Ampere GA102, 10,496 CUDA Cores, 328 Tensor Cores)
- **CUDA Runtime:** CUDA 11.3 (NVIDIA Linux x86_64 Driver 470.x / 525.x)
- **Machine Learning Framework:** PyTorch 2.x with native Tensor Core Mixed Precision (FP16 / BF16) and Triton JIT 2.x/3.x.
- **Teacher GPU VRAM Overhead:** **0 MB** (Offline Teacher Representation Caching).

---

## 6. Benchmark Evaluation Metrics Across Supported Suites

Every published model card provides empirical validation across three primary benchmark evaluation suites:

| Benchmark Suite | Modality / Task | Target Evaluation Metric | Teacher / Baseline Value | Distilled Student Parity Status |
|---|---|---|:---:|:---:|
| **GLUE SST-2** | Text Classification (BERT) | **SST-2 Accuracy (%)** | `91.63%` | Evaluated & Verified |
| **WikiText-2** | Language Modeling (Qwen / LLaMA) | **WikiText-2 PPL** | `10.48` | Evaluated & Verified |
| **ScienceQA** | Multimodal Reasoning (LLaVA VLM) | **ScienceQA Accuracy (%)** | `68.20%` | Evaluated & Verified |

---

## 7. Universal One-Line Loading & Inference Guide

NeuralOps provides the canonical high-level loader `load_surgeried_model` in `neuralops.hub`. It dynamically downloads the base model and transparently patches the surgical delta weights in memory.

### Universal Quick Start
```python
import torch
from neuralops.hub import load_surgeried_model

# Load model and tokenizer in a single call directly from Hugging Face Hub
model, tokenizer = load_surgeried_model("Hooshaai/bert-gated-delta-net-sst2")
model.eval()

# Sequence classification inference
inputs = tokenizer(
    "NeuralOps achieves subquadratic scaling while maintaining benchmark parity.",
    return_tensors="pt",
)
with torch.no_grad():
    logits = model(**inputs).logits
    predicted_class = logits.argmax(dim=-1).item()

print(f"Predicted Sentiment (0 = Negative, 1 = Positive): {predicted_class}")
```

### Loading Causal Language Models (Qwen / TinyLlama)
```python
import torch
from neuralops.hub import load_surgeried_model

model, tokenizer = load_surgeried_model("Hooshaai/qwen-2.5-0.5b-mamba-wikitext2")
model.eval()

prompt = "Subquadratic attention mechanisms replace standard softmax attention by"
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    tokens = model.generate(**inputs, max_new_tokens=32)

print(tokenizer.decode(tokens[0], skip_special_tokens=True))
```

---

## 8. Model Card Template & Tagging Compliance Standard

Every model card on the Hugging Face Hub must strictly contain:

1. **YAML Frontmatter:**
   ```yaml
   ---
   language:
     - en
   license: mit
   license_link: https://github.com/tahamajs/NeuralOps/blob/main/LICENSE
   library_name: neuralops
   tags:
     - transformers
     - pytorch
     - distillation
     - neuralops
     - sub-quadratic-attention
   ---
   ```
2. **Universal One-Line Loader Snippet:** Must import `load_surgeried_model` from `neuralops.hub`.
3. **Explicit 0 MB Teacher GPU VRAM Disclosure:** Must disclose AGENTS.md §21.1 offline teacher caching.
4. **Hardware Telemetry Specifications:** Must specify `NVIDIA RTX 3090, 24GB GDDR6X, CUDA 11.3`.
5. **Benchmark Evaluation Metrics:** Must specify `SST-2 Accuracy`, `WikiText-2 PPL`, and `ScienceQA Accuracy`.
6. **Zero-Fake Artifact Audit Table:** Details the actual files, byte sizes, and verification roles.

---

## 9. Operational Tooling & Network Resilience

NeuralOps includes automated tooling to audit and upgrade model cards across all zoos:

### 1. Model Card Upgrader: `scripts/benchmarks/upgrade_hf_model_cards.py`
This utility upgrades local and remote model card READMEs with the canonical template, loader snippets, and telemetry disclosures.

#### Resilience to Rate Limits and Network Timeouts:
- **HTTP 429 Rate Limiting:** Catches 429 errors from Hugging Face Hub, extracts `Retry-After` response headers, and applies exponential backoff with randomized jitter ($t_{\text{wait}} = \text{base} \cdot 2^{\text{attempt}} + \mathcal{U}(0.5, 2.0)$).
- **Socket and Connection Timeouts:** Automatically retries transient connection errors and 5xx gateway timeouts (up to `--max_retries 5`).
- **Offline Audit Mode:** Run `--audit_only 1` to inspect catalog completeness without network calls or authentication tokens.

```bash
# 1. Audit catalog completeness without modifying files or network access
python3 scripts/benchmarks/upgrade_hf_model_cards.py --audit_only 1

# 2. Upgrade local README.md files across results/
python3 scripts/benchmarks/upgrade_hf_model_cards.py --push_hub 0

# 3. Synchronize upgraded README.md files to Hugging Face Hub
python3 scripts/benchmarks/upgrade_hf_model_cards.py --push_hub 1 --max_retries 5
```

### 2. High-Speed Upload Daemon: `scripts/hpc/bert_zoo_upload_hf.py`
Monitors cluster checkpoints, purges redundant 440 MB base models, and streams lightweight safetensors to Hugging Face Hub in parallel threads.

---

## 10. Showcase Repository Catalog

Below is the verified master index of prominent NeuralOps repositories hosted on the Hugging Face Hub:

| Repository ID | Zoo Family | Mechanism / Engine | Base Architecture | Target Benchmark | Format |
|---|---|---|---|---|---|
| [`Hooshaai/bert-gated-delta-net-sst2`](https://huggingface.co/Hooshaai/bert-gated-delta-net-sst2) | BERT Attention | `gated_delta_net` | `bert-base-uncased` | GLUE SST-2 (Acc) | `attention_only_fp16` |
| [`Hooshaai/bert-linear-attention-sst2`](https://huggingface.co/Hooshaai/bert-linear-attention-sst2) | BERT Attention | `linear_attention` | `bert-base-uncased` | GLUE SST-2 (Acc) | `attention_only_fp16` |
| [`Hooshaai/bert-flash-attention-sst2`](https://huggingface.co/Hooshaai/bert-flash-attention-sst2) | BERT Attention | `flash_attention` | `bert-base-uncased` | GLUE SST-2 (Acc) | `attention_only_fp16` |
| [`Hooshaai/bert-mamba-sst2`](https://huggingface.co/Hooshaai/bert-mamba-sst2) | BERT Attention | `mamba` | `bert-base-uncased` | GLUE SST-2 (Acc) | `attention_only_fp16` |
| [`Hooshaai/bert-hyena-sst2`](https://huggingface.co/Hooshaai/bert-hyena-sst2) | BERT Attention | `hyena` | `bert-base-uncased` | GLUE SST-2 (Acc) | `attention_only_fp16` |
| [`Hooshaai/bert-retention-sst2`](https://huggingface.co/Hooshaai/bert-retention-sst2) | BERT Attention | `retention` | `bert-base-uncased` | GLUE SST-2 (Acc) | `attention_only_fp16` |
| [`Hooshaai/bert-rwkv-sst2`](https://huggingface.co/Hooshaai/bert-rwkv-sst2) | BERT Attention | `rwkv` | `bert-base-uncased` | GLUE SST-2 (Acc) | `attention_only_fp16` |
| [`Hooshaai/bert-cosformer-sst2`](https://huggingface.co/Hooshaai/bert-cosformer-sst2) | BERT Attention | `cosformer` | `bert-base-uncased` | GLUE SST-2 (Acc) | `attention_only_fp16` |
| [`Hooshaai/bert-performer-sst2`](https://huggingface.co/Hooshaai/bert-performer-sst2) | BERT Attention | `performer` | `bert-base-uncased` | GLUE SST-2 (Acc) | `attention_only_fp16` |
| [`Hooshaai/bert-nystrom-sst2`](https://huggingface.co/Hooshaai/bert-nystrom-sst2) | BERT Attention | `nystrom` | `bert-base-uncased` | GLUE SST-2 (Acc) | `attention_only_fp16` |
| [`Hooshaai/bert-comp-hnc-svd-sst2`](https://huggingface.co/Hooshaai/bert-comp-hnc-svd-sst2) | BERT Compressor | `hnc_svd` | `bert-base-uncased` | GLUE SST-2 (Acc) | `safetensors_lora` |
| [`Hooshaai/bert-comp-svd-linear-sst2`](https://huggingface.co/Hooshaai/bert-comp-svd-linear-sst2) | BERT Compressor | `svd_linear` | `bert-base-uncased` | GLUE SST-2 (Acc) | `safetensors_lora` |
| [`Hooshaai/bert-comp-asvd-sst2`](https://huggingface.co/Hooshaai/bert-comp-asvd-sst2) | BERT Compressor | `asvd` | `bert-base-uncased` | GLUE SST-2 (Acc) | `safetensors_lora` |
| [`Hooshaai/bert-comp-admm-sst2`](https://huggingface.co/Hooshaai/bert-comp-admm-sst2) | BERT Compressor | `admm` | `bert-base-uncased` | GLUE SST-2 (Acc) | `safetensors_lora` |
| [`Hooshaai/qwen-2.5-0.5b-gated-delta-net-wikitext2`](https://huggingface.co/Hooshaai/qwen-2.5-0.5b-gated-delta-net-wikitext2) | Qwen Attention | `gated_delta_net` | `Qwen2.5-0.5B` | WikiText-2 (PPL) | `safetensors` |
| [`Hooshaai/qwen-2.5-0.5b-mamba-wikitext2`](https://huggingface.co/Hooshaai/qwen-2.5-0.5b-mamba-wikitext2) | Qwen Attention | `mamba` | `Qwen2.5-0.5B` | WikiText-2 (PPL) | `safetensors` |
| [`Hooshaai/qwen-2.5-0.5b-comp-hnc-svd-wikitext2`](https://huggingface.co/Hooshaai/qwen-2.5-0.5b-comp-hnc-svd-wikitext2) | Qwen Compressor | `hnc_svd` | `Qwen2.5-0.5B` | WikiText-2 (PPL) | `safetensors` |
| [`Hooshaai/llava-qwen-0.5b-gated-delta-net-scienceqa`](https://huggingface.co/Hooshaai/llava-qwen-0.5b-gated-delta-net-scienceqa) | LLaVA VLM | `gated_delta_net` | `llava-interleave-qwen-0.5b` | ScienceQA (Acc) | `safetensors` |
| [`Hooshaai/tinyllama-1.1b-gated-delta-net-wikitext2`](https://huggingface.co/Hooshaai/tinyllama-1.1b-gated-delta-net-wikitext2) | TinyLlama | `gated_delta_net` | `TinyLlama_v1.1` | WikiText-2 (PPL) | `safetensors` |

---

## 11. Verification & Quality Assurance

To ensure adherence to the specification across all releases:

```bash
# 1. Verify model card upgrader unit tests
pytest tests/benchmarks/test_upgrade_hf_model_cards.py -v

# 2. Verify universal hub loader and surgical delta IO
pytest tests/test_surgical_delta_and_hub_loader.py -v

# 3. Verify BERT zoo export pipeline
pytest tests/test_bert_hf_hub_zoo.py -v
```
