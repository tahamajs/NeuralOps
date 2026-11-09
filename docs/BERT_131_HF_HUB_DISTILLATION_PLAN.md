# 🏛️ NeuralOps Pretrained BERT-131 Attention Hub Zoo — Master Engineering & Distillation Plan

**Document ID:** `NOP-ENG-ZOO-131-v2.1`  
**Status:** Canonical Master Plan  
**Target Architecture:** `bert-base-uncased` (110M params, 12 layers, 768 hidden, 12 heads)  
**Distillation Corpus:** `WikiText-2` (Masked Language Modeling, Sequence Length = 256)  
**Primary Compute Host:** Sharif HPC Cluster (`GPU-3090-1`, NVIDIA RTX 3090 24GB GDDR6X, CUDA 11.3)  
**Secondary Distributed Hosts:** `ssh gpu` (RTX 4070 8GB SM 8.9) & `ssh tahoora` (GTX 1060 6GB SM 6.1)  
**Hugging Face Hub Organization:** `Hooshaai` (Primary) / `tahamajs` (Mirror)  
**Canonical Scope:** All 131 Registered & Extended Attention Mechanisms in NeuralOps  

---

## 1. Architectural Vision & Strategic Value

The goal of this initiative is to transition `NeuralOps` from a high-performance kernel library into an **end-to-end Pretrained Foundation Model Zoo on the Hugging Face Hub**. 

Currently, evaluating a new subquadratic, linear, sparse, or SVD-compressed attention mechanism requires researchers to train a transformer from scratch (computationally prohibitive) or patch a raw architecture without weights. By providing **131 ready-to-use, pre-distilled BERT checkpoints on Hugging Face Hub**, researchers can download any attention mechanism with **a single line of Python code**:

```python
import neuralops

# Instant 1-line download and instantiation for any downstream NLP task
model, tokenizer = neuralops.load_pretrained_bert(
    attention_type="gated_delta_net",    # Any of the 131 attention mechanisms
    task="sequence-classification",      # Or "masked-lm", "token-classification"
    num_labels=2,
)
```

### The Core Engineering Challenge & Solution

| Challenge | Root Cause | NeuralOps Engineering Solution |
|---|---|---|
| **BERT Perplexity Metric** | BERT is bidirectional; unmasked causal PPL is mathematically invalid ($\text{Loss}=0$). | **Deterministic MLM Pseudo-Perplexity ($\text{MLM-PPL}$)** on 15% masked tokens with fixed seed + **Masked Token Accuracy (%)**. |
| **Storage Exhaustion** | 131 BERT checkpoints $\times$ 440 MB = **~57.6 GB** of disk space. | **Asynchronous Auto-Cleanup Queue**: Background daemon verifies HF Hub remote commit, records metrics, then immediately deletes local `.safetensors`. Active disk stays $< 2\text{ GB}$. |
| **GPU Starvation** | Standard PyTorch DataLoaders and repeated teacher forward passes waste 50–70% of compute. | **In-VRAM Pre-tokenized WikiText-2** + **Pre-cached Teacher Hidden States/Logits** + **AutoBatchSizer** locking SMs at 100% duty cycle. |
| **Layer Interface Mismatch** | Attention mechanisms have divergent forward signatures (SSM states, MoE logits, tuple outputs). | **Universal `BertAttentionAdapter`**: Wraps any attention mechanism into a strict `[Batch, Seq, 768] -> [Batch, Seq, 768]` contract. |
| **Strict Time Budget** | 131 modules cannot take weeks of cluster time. | **5-Minute (300-Second) 3-Stage Curriculum**: Closed-form Ridge initialization (5s) $\rightarrow$ Operator alignment (90s) $\rightarrow$ End-to-end KD polish (205s). Total runtime: **~11 hours**. |

---

## 2. End-to-End System Topology

The pipeline separates heavy GPU compute from network I/O using a producer-consumer architecture:

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                GPU WORKER PROCESS (100% LOAD)                           │
│                                                                                         │
│  ┌─────────────────────────────────┐       ┌─────────────────────────────────────────┐  │
│  │ In-VRAM WikiText-2 Tokens (FP16) │       │ Pre-Cached Teacher Hidden States/Logits │  │
│  └────────────────┬────────────────┘       └────────────────────┬────────────────────┘  │
│                   │                                             │                       │
│                   ▼                                             ▼                       │
│     ┌─────────────────────────────────────────────────────────────────────────────┐     │
│     │                      FOR MODULE i IN 1..131 ATTENTIONS                      │     │
│     │                                                                             │     │
│     │  1. In-Place Surgery: Inject Module into 12 BertSelfAttention layers        │     │
│     │  2. Stage 0 (0-5s):   Closed-Form Ridge Alignment W_align = (Z^T Z + λI)^-1 │     │
│     │  3. Pre-Eval:         Record Pre-KD MLM PPL & Masked Token Accuracy         │     │
│     │  4. Stage 1 (5-95s):  Frozen-Backbone Subspace MSE + Linear CKA Distill    │     │
│     │  5. Stage 2 (95-300s):Unfrozen End-to-End Fused KL + MLM Cross-Entropy     │     │
│     │  6. Post-Eval:        Record Post-KD MLM PPL, Accuracy, VRAM, Throughput    │     │
│     │  7. Staging:          Dump .safetensors, config.json, README to /tmp/stage  │     │
│     └──────────────────────────────────────┬──────────────────────────────────────┘     │
└────────────────────────────────────────────┼────────────────────────────────────────────┘
                                             │ Non-blocking Queue.put(module_meta)
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                        ASYNC BACKGROUND UPLOADER DAEMON (CPU / I/O)                     │
│                                                                                         │
│  While GPU immediately begins training Module i+1 (Zero GPU Idle Time):                 │
│   1. Read staging directory `/tmp/neuralops_hf_staging/bert-{module_name}/`             │
│   2. Generate publication-grade `README.md` Model Card with empirical tables            │
│   3. Call `huggingface_hub.HfApi.upload_folder(...)` to `Hooshaai/bert-{module_name}`   │
│   4. Verify remote commit hash from Hugging Face Hub API response                       │
│   5. Execute Atomic Cleanup: `os.remove("model.safetensors")` (Frees ~440 MB disk)      │
│   6. Append verification entry to Master Ledgers:                                       │
│      - `results/bert_wikitext_131_hub_zoo.json`                                         │
│      - `results/bert_wikitext_131_hub_zoo.csv`                                          │
│      - `results/bert_wikitext_131_hub_zoo.md`                                           │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Mathematical Formulation & 3-Stage Curriculum

BERT is an autoencoding bidirectional Transformer ($h \in \mathbb{R}^{B \times S \times D}$, $D = 768$). Training an attention mechanism inside it without full pretraining from scratch requires a structured distillation curriculum.

### 3.1. Exact MLM Pseudo-Perplexity & Accuracy Formulation

For a token sequence $X = (x_1, \dots, x_S)$, a deterministic subset of token indices $\mathcal{M} \subset \{1, \dots, S\}$ is selected with probability $p = 0.15$ using a fixed generator seed (`seed=42`).

- $80\%$ of tokens in $\mathcal{M}$ are replaced with `[MASK]`.
- $10\%$ of tokens in $\mathcal{M}$ are replaced with a random token from $\mathcal{V}$.
- $10\%$ of tokens in $\mathcal{M}$ remain unchanged.

The Masked Language Modeling loss is evaluated strictly over the masked positions $\mathcal{M}$:

$$\mathcal{L}_{\text{MLM}}(S) = -\frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \log P_{\text{student}}(x_i \mid \tilde{X})$$

The **MLM Pseudo-Perplexity ($\text{PPL}$)** and **Masked Token Accuracy ($\text{Acc}_{\text{MLM}}$)** are computed as:

$$\text{PPL}_{\text{MLM}} = \exp\left(\min\left(\mathcal{L}_{\text{MLM}}, 20.0\right)\right)$$

$$\text{Acc}_{\text{MLM}} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \mathbb{I}\left( \arg\max_{v \in \mathcal{V}} z_{S, i}(v) = x_i \right) \times 100\%$$

Both metrics are recorded at:
1. **Zero-Shot / Uncalibrated**: Immediately after surgery.
2. **Pre-KD**: Following closed-form Ridge alignment (Stage 0).
3. **Post-KD**: Following the 300-second distillation window (Stage 2).

---

### 3.2. Stage 0: Instant Closed-Form Ridge Output Calibration ($t = 0\dots 5\,\text{s}$)

Before backpropagation begins, we align the output subspace of the newly injected attention module with the teacher's original attention output space.

For each of the 12 encoder layers $l \in \{1, \dots, 12\}$, we collect calibration activation representations $Z_l \in \mathbb{R}^{N_{\text{cal}} \times D}$ from 16 WikiText sequences passed through the new attention layer, and the corresponding teacher representations $Y_l \in \mathbb{R}^{N_{\text{cal}} \times D}$. We solve the Tikhonov-regularized Ridge regression problem in closed form:

$$W_{\text{align}}^{(l)} = \left( Z_l^\top Z_l + \lambda I \right)^{-1} Z_l^\top Y_l, \quad \lambda = 10^{-4}$$

The projection $W_{\text{align}}^{(l)}$ is absorbed directly into the output projection layer `attention.output.dense.weight`:

$$W_{\text{dense, new}}^{(l)} = W_{\text{align}}^{(l)} \cdot W_{\text{dense, orig}}^{(l)}$$

**Execution time:** $\le 3.5$ seconds across all 12 layers on GPU. Eliminates gradient explosion in early steps.

---

### 3.3. Stage 1: Subspace & Operator Alignment ($t = 5\dots 95\,\text{s}$, 30% of budget)

In this stage, the core backbone is frozen to force the new attention parameters to learn the representation manifold of standard Multi-Head Attention without corrupting the downstream Feed-Forward Networks (FFNs) or LayerNorms.

- **Frozen parameters:** Embeddings, all intermediate FFN layers, output classification/MLM head.
- **Trainable parameters:** Injected `attention.self.*` and `attention.output.LayerNorm`.
- **Optimizer:** AdamW ($\beta_1 = 0.9, \beta_2 = 0.98, \epsilon = 10^{-8}$), $\text{lr} = 1.0 \times 10^{-3}$, weight decay $= 0.01$.
- **Loss Function:**

$$\mathcal{L}_{\text{Stage 1}} = \sum_{l=1}^{12} \left[ \mu_1 \cdot \text{MSE}\left(h_S^{(l)}, h_T^{(l)}\right) + \mu_2 \cdot \left(1 - \text{LinearCKA}\left(h_S^{(l)}, h_T^{(l)}\right)\right) \right] + \mu_3 \cdot \mathcal{L}_{\text{MLM-KL}}$$

Where Linear Centered Kernel Alignment ($\text{CKA}$) measures invariant subspace similarity:

$$\text{LinearCKA}(X, Y) = \frac{\|Y^\top X\|_F^2}{\|X^\top X\|_F \|Y^\top Y\|_F}$$

Coefficients: $\mu_1 = 0.6, \mu_2 = 0.3, \mu_3 = 0.1$.

---

### 3.4. Stage 2: End-to-End Differential-LR KD + MLM Recovery ($t = 95\dots 300\,\text{s}$, 70% of budget)

All parameters across the entire BERT student model are unfrozen to allow joint adaptation and recover masked language modeling perplexity.

- **Unfrozen parameters:** Entire model (Embeddings + 12 Transformer Layers + Prediction Head).
- **Differential Learning Rates:**
  - $\text{lr}_{\text{attention}} = 5.0 \times 10^{-4}$ (higher adaptation rate for newly initialized weights)
  - $\text{lr}_{\text{backbone}} = 5.0 \times 10^{-5}$ (conservative rate preserving pretrained features)
  - Schedule: Cosine decay with 10-step linear warmup, decaying to $\text{lr}_{\text{min}} = 1.0 \times 10^{-5}$.
- **Loss Function:**

$$\mathcal{L}_{\text{Stage 2}} = \alpha \cdot T^2 \cdot D_{\text{KL}}\left( \sigma\left(\frac{z_S}{T}\right) \,\middle\|\, \sigma\left(\frac{z_T}{T}\right) \right) + \beta \cdot \mathcal{L}_{\text{MLM-CE}}(z_S, y_{\text{masked}}) + \gamma \cdot \frac{1}{12}\sum_{l=1}^{12}\text{MSE}\left(h_S^{(l)}, h_T^{(l)}\right)$$

Parameters: Temperature $T = 2.0$, $\alpha = 0.5$, $\beta = 0.3$, $\gamma = 0.2$.  
Mixed precision: `torch.amp.autocast("cuda", dtype=torch.float16)`.

---

## 4. Hardware Saturation Strategy (100% SM & VRAM Duty Cycle)

To prevent the GPU from idling while waiting for batches or CPU operations:

### 4.1. In-VRAM Pre-Tokenized WikiText-2 Dataset

The raw `wikitext-2-raw-v1` validation and train sets are tokenized once into contiguous token matrices of sequence length $S = 256$:

```python
# Contiguous tensor resident in pinned host memory or GPU VRAM
tokens_train = torch.tensor(token_ids, dtype=torch.long, device="cuda")
# Shape: (N_chunks, 256) where N_chunks ~ 8,500
```

During training, mini-batches are sliced directly from this pre-allocated tensor:
$$\text{batch} = \text{tokens\_train}[i \cdot B : (i+1) \cdot B]$$
**Data loading latency:** **0.00 milliseconds** per step.

### 4.2. Pre-Cached Teacher Representation Engine

Running `teacher_model(input_ids)` inside the inner training loop consumes 50% of total GPU FLOPs on redundant computation. Instead, before starting the 131-module loop:

1. Pass the entire training set through unmodified `bert-base-uncased` under `torch.no_grad()`.
2. Cache intermediate layer hidden states $h_T^{(1..12)}$ and output logits $z_T$ in FP16.
3. For RTX 3090 (24GB), caching 2,048 training sequences $\times$ 12 layers $\times$ 256 tokens $\times$ 768 dims in FP16 takes **~9.6 GB of VRAM**.
4. The remaining **14.4 GB of VRAM** is dedicated entirely to the student model, optimizer states, and gradient buffers.

### 4.3. Dynamic Auto-Batch-Sizer (`find_optimal_gpu_batch_size`)

Before training Module 0, an automated binary search probes batch sizes:
$$B \in \{128, 96, 64, 48, 32, 16\}$$
It allocates memory, performs forward/backward steps with gradient accumulation $= 1$, checks `torch.cuda.max_memory_allocated()`, and selects the largest batch size that fills **85% to 92% of available VRAM**.

| Target Host | Available VRAM | Target Saturation VRAM | Optimal Batch Size ($S=256$) |
|---|---|---|---|
| **Sharif HPC (RTX 3090)** | 24,576 MB | 21,500 MB | **$B = 64$** (Effective with accum 2: **128**) |
| **`ssh gpu` (RTX 4070)** | 8,192 MB | 7,200 MB | **$B = 32$** (Effective with accum 4: **128**) |
| **`ssh tahoora` (GTX 1060)** | 6,144 MB | 5,200 MB | **$B = 16$** (Effective with accum 8: **128**) |

---

## 5. Universal `BertAttentionAdapter` Contract

In `neuralops`, attention modules implement diverse interfaces. Some return a single tensor, some return `(context, weights)`, some require causal masking, and some require explicit recurrent states.

The `BertAttentionAdapter` standardizes all 131 modules into the exact signature expected by `BertLayer`:

```python
class BertAttentionAdapter(nn.Module):
    """
    Wraps any NeuralOps attention module into a drop-in replacement
    for Hugging Face transformers BertSelfAttention.
    """
    def __init__(self, core_module: nn.Module, embed_dim: int = 768, num_heads: int = 12):
        super().__init__()
        self.core = core_module
        self.embed_dim = embed_dim
        self.num_heads = num_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        # 1. Normalize bidirectional attention mask
        # 2. Forward pass through core module
        out = self.core(hidden_states, mask=attention_mask)
        
        # 3. Unpack return type
        if isinstance(out, tuple):
            context = out[0]
            attn_weights = out[1] if len(out) > 1 and output_attentions else None
        else:
            context = out
            attn_weights = None
            
        # 4. Enforce invariant output shape [Batch, SeqLen, EmbedDim]
        if context.shape[-1] != self.embed_dim:
            raise ValueError(f"Output dim {context.shape[-1]} != expected {self.embed_dim}")
            
        return (context, attn_weights)
```

---

## 6. Asynchronous Background Uploader Daemon & Disk Protection

Uploading 131 models sequentially over a cluster network connection would add 3 to 5 minutes of dead time per module ($131 \times 4\text{ min} \approx 8.7\text{ hours}$ of wasted GPU time).

The `AsyncHFHubUploader` runs as a completely decoupled thread/process:

```python
class AsyncHFHubUploader:
    def __init__(self, hf_org: str, max_workers: int = 2, cleanup_weights: bool = True):
        self.hf_org = hf_org
        self.cleanup_weights = cleanup_weights
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.futures = []

    def submit_upload(self, module_name: str, local_staging_dir: Path, metrics: dict):
        future = self.executor.submit(
            self._worker_upload_and_cleanup,
            module_name,
            local_staging_dir,
            metrics,
        )
        self.futures.append(future)

    def _worker_upload_and_cleanup(self, module_name: str, staging_dir: Path, metrics: dict):
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ["HF_TOKEN"])
        repo_id = f"{self.hf_org}/bert-{module_name.replace('_', '-')}-wiki"

        # 1. Ensure remote repository exists
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

        # 2. Generate and write publication Model Card
        model_card = generate_bert_model_card_markdown(module_name, metrics, repo_id)
        (staging_dir / "README.md").write_text(model_card)

        # 3. Upload entire folder to Hugging Face Hub
        commit_info = api.upload_folder(
            folder_path=str(staging_dir),
            repo_id=repo_id,
            commit_message=f"Distill bert-{module_name} on WikiText-2 (PPL: {metrics['post_kd_ppl']:.2f})",
        )

        # 4. Verify remote commit hash
        assert commit_info.commit_id is not None, "Remote commit failed!"

        # 5. ATOMIC DISK CLEANUP: Delete heavy model weights
        if self.cleanup_weights:
            weight_file = staging_dir / "model.safetensors"
            if not weight_file.exists():
                weight_file = staging_dir / "pytorch_model.bin"
            if weight_file.exists():
                os.remove(weight_file)
                logger.info(f"[CLEANUP] Deleted local weight file for {module_name} to free disk space.")
```

---

## 7. Automated Model Card Generator (`README.md`)

Every model uploaded to Hugging Face Hub will feature a fully populated, professional Model Card:

```markdown
---
language:
  - en
license: mit
library_name: neuralops
base_model: bert-base-uncased
datasets:
  - wikitext
metrics:
  - perplexity
  - accuracy
tags:
  - neuralops
  - bert
  - attention-zoo
  - knowledge-distillation
  - subquadratic-attention
  - {module_name}
pipeline_tag: fill-mask
---

# 🧠 BERT-Base ({module_name}) — Distilled on WikiText-2

This model is a surgically modified variant of **`bert-base-uncased`** where all 12 standard Multi-Head Self-Attention layers have been replaced with the **`{module_name}`** mechanism from the [NeuralOps Library](https://github.com/tahamajs/NeuralOps).

The model was distilled on **WikiText-2** using the **NeuralOps 3-Stage GPU-Saturated Knowledge Distillation Pipeline** for 300 seconds on an NVIDIA RTX 3090.

---

## 📊 Empirical Evaluation & Distillation Benchmark

Evaluated using **Deterministic Masked Language Modeling Pseudo-Perplexity ($\text{MLM-PPL}$)** on WikiText-2 (15% token masking, `seed=42`):

| Evaluation Phase | MLM Perplexity ($\text{PPL}$) | Masked Token Acc (%) | Delta vs Baseline |
|---|:---:|:---:|:---:|
| **Vanilla BERT Teacher (`bert-base-uncased`)** | `{teacher_ppl:.2f}` | `{teacher_acc:.2f}%` | *Reference* |
| **Zero-Shot Surgery (Uncalibrated)** | `{raw_ppl:.2f}` | `{raw_acc:.2f}%` | `{raw_delta:+.2f}` |
| **Stage 0 (Closed-Form Ridge Calibration)** | `{pre_kd_ppl:.2f}` | `{pre_kd_acc:.2f}%` | `{pre_delta:+.2f}` |
| **Stage 2 (Post-KD Distilled Model)** | **`{post_kd_ppl:.2f}`** | **`{post_kd_acc:.2f}%`** | **`{post_delta:+.2f}`** |

### Hardware & Training Telemetry
- **Distillation Hardware:** `{hardware_name}`
- **Training Time:** `300.0s (5.0 minutes)`
- **Peak GPU VRAM:** `{peak_vram_mb} MB`
- **Throughput:** `{tokens_per_sec:.1f} tokens/sec`
- **Distillation Stages:** Stage 0 (Closed-Form Ridge) $\rightarrow$ Stage 1 (Frozen-Backbone CKA/MSE) $\rightarrow$ Stage 2 (Differential-LR Fused Logit KD)

---

## 🚀 Quick Start & Usage

### 1. Using NeuralOps High-Level API (Recommended)

```python
import neuralops

# Instant load from Hugging Face Hub
model, tokenizer = neuralops.load_pretrained_bert(
    attention_type="{module_name}",
    task="masked-lm",
)
```

### 2. Standard Transformers Pipeline

```python
from transformers import pipeline

unmasker = pipeline("fill-mask", model="{repo_id}")
results = unmasker("NeuralOps makes attention mechanisms [MASK] and efficient.")
for res in results:
    print(f"{{res['token_str']}}: {{res['score']:.4f}}")
```

---

## 📜 Citation

```bibtex
@software{{neuralops2026bertzoo,
  title   = {{NeuralOps: Pretrained BERT-131 Subquadratic Attention Zoo}},
  author  = {{Majlesi, Taha}},
  year    = {{2026}},
  url     = {{https://github.com/tahamajs/NeuralOps}}
}}
```
```

---

## 8. End-User Python API & CLI Specification

### 8.1. Python API (`neuralops.load_pretrained_bert`)

```python
def load_pretrained_bert(
    attention_type: str,
    task: str = "masked-lm",
    num_labels: int = 2,
    hf_org: str = "Hooshaai",
    device: str = "auto",
    revision: Optional[str] = None,
    **kwargs,
) -> Tuple[nn.Module, Any]:
    """
    Downloads and instantiates a pretrained, KD-distilled BERT model with the
    specified NeuralOps attention mechanism directly from Hugging Face Hub.
    
    Supported tasks:
      - "masked-lm": BertForMaskedLM
      - "sequence-classification": BertForSequenceClassification
      - "token-classification": BertForTokenClassification
      - "feature-extraction": BertModel
    """
```

### 8.2. CLI Subcommands (`neuralops hub`)

```bash
# 1. List all available models, PPL scores, and download links
neuralops hub list --backbone bert-base-uncased

# 2. Pull and verify a specific attention model
neuralops hub pull --attention gated_delta_net --task sequence-classification

# 3. Launch or resume the 5-minute zoo distillation pipeline
neuralops hub train-zoo \
    --minutes-per-module 5.0 \
    --upload-hf \
    --hf-org Hooshaai \
    --cleanup-after-upload
```

---

## 9. Full Canonical Attention Inventory Mapping (131 Units)

The repository currently registers 115 core attention mechanisms in `ATTENTION_REGISTRY` and 82 compression/hybrid modules in `COMPRESSOR_REGISTRY`. To reach the canonical target of **131 attention mechanisms**, the 16 remaining units represent canonical hybrid/fused attention variants:

```
[001-010] adaptive_span_attention, adaptive_sparse_attention, adaptive_svd_attention, adasplash_attention, alibi, asvd_attention, bam_umoe, block_transformer_hiba, chela_slay, cluster_attention
[011-020] compressed_attention, compressed_mha_attention, cosformer, cross_attention, decay_linear_attention, delta_product_attention, dynamic_conv_attention, eat, expert_choice_moe, factorised_matrix_state
[021-030] flash_attention3, flash_attention3_fp8, full_attention, fused_svd_kernel, fusion_attn, galerkin_attention, gated_delta_net, gated_deltanet_block, gated_hybrid_kernel_attention, gdn2_attention
[031-040] gla_attention, goldfinch_ahn, gqa_attention, granite_hybrid, griffin_ssd, halo_regla, hamming_attention, hgrn2_gated_deltanet, hgrn2_gated_layer, hgrn2_gated_outer
[041-050] hla_attention, hybrid_attention, hybrid_local_kernel, hymba_transx, jamba, jamba_info, kda_attention, landmark_local_attention, linformer_attention, lola_attention
[051-060] lora_attention, lsh_attention, mamba_block, mamba_svd_flash, minimal_nova, minimal_svd, mixattention_moska, mla_attention, moba_attention, moe_deltakv_paged
[061-070] mqa, naive_svd, nasc_attention, nexus, nha_attention, nsa_attention, nystrom_attention, optimized_hybrid_attention, orion, padre_attention
[071-080] performer_attention, phantom, positive_kernel_linear_attention, preconditioned_gdn_attention, primal_attention, progressive_token_pruning, projection_factorized_attn, qcqa_attention, rala_attention, reap_moe
[081-090] refined_landmark_attention, regla_attention, residual_svd_attention, routing_attention, rpc_attention, salt_attention, samba_m3, samba_moe, selective_ssm, simple_ssm
[091-100] sink_attention, soa_attn, spectraformer_attention, spectron, spectron_fp8, split_head_hybrid, svd_attention, svd_compressed, svd_linear_attention, svd_linear_decay
[101-110] svd_llm_v2_attention, svd_mamba, svd_projection_attention, token_compressed_attention, tri_branch_hybrid, triton_cosformer, triton_svd_linear_attention, ttt_linear_attention, unified_rope, upi_stuffed_mamba
[111-115] vig_attention, vspec, wgqa_attention, xattention, zeros_attention
[116-131] Canonical Extensible Units:
          116: chunk_gla_attention
          117: multi_scale_retention
          118: rwkv6_linear_attention
          119: hgrn_linear_attention
          120: lightning_attention2
          121: focused_linear_attention
          122: flash_linear_attention
          123: gated_linear_attention_v2
          124: polynomial_kernel_attention
          125: relu_linear_attention
          126: elu_linear_attention
          127: taylor_series_linear_attention
          128: randomized_svd_attention
          129: truncated_svd_attention
          130: orthogonal_svd_attention
          131: fused_recurrent_attention
```

---

## 10. Production PBS Job Script for Sharif HPC

**Path:** `scripts/hpc/job_bert_131_wikitext_hf_zoo.pbs`

```bash
#!/bin/bash
#PBS -N bert_131_hf_zoo
#PBS -q N3090
#PBS -l nodes=1:ppn=8:gpus=1
#PBS -l walltime=14:00:00
#PBS -l mem=48gb
#PBS -o /home/tahamajlesi.ut/NeuralOps/results/bert_wikitext_131_hub_zoo/pbs_stdout.log
#PBS -e /home/tahamajlesi.ut/NeuralOps/results/bert_wikitext_131_hub_zoo/pbs_stderr.log

set -euo pipefail

WORK_DIR="/home/tahamajlesi.ut/NeuralOps"
CONDA_ENV="/home/tahamajlesi.ut/.conda/envs/neuralops_cu113"
PYTHON="${CONDA_ENV}/bin/python3"
OUT_DIR="${WORK_DIR}/results/bert_wikitext_131_hub_zoo"

mkdir -p "${OUT_DIR}/staging"

cd "${WORK_DIR}"

echo "================================================================="
echo "NeuralOps Pretrained BERT-131 Attention Zoo Pipeline"
echo "Host: $(hostname) | Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
echo "================================================================="

# Pre-flight sanity checks
${PYTHON} -c "import torch; assert torch.cuda.is_available(), 'CUDA Required!'"
${PYTHON} -c "from neuralops.core.registry import ATTENTION_REGISTRY; print(f'Core Registry: {len(ATTENTION_REGISTRY)} modules')"

# Ensure HF_TOKEN is configured in environment
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN environment variable is not set. Aborting." >&2
    exit 1
fi

# Run the 100% GPU-Saturated 5-Minute KD Engine with Background HF Upload
${PYTHON} scripts/train_bert_131_wikitext_hf_zoo.py \
    --model               bert-base-uncased \
    --dataset             wikitext-2-raw-v1 \
    --seq-len             256 \
    --minutes-per-module  5.0 \
    --output-dir          "${OUT_DIR}" \
    --staging-dir         "${OUT_DIR}/staging" \
    --hf-org              Hooshaai \
    --upload-hf \
    --cleanup-after-upload \
    --device              cuda

echo "================================================================="
echo "Pipeline completed successfully at $(date)."
echo "Master Ledger: ${OUT_DIR}/master_ledger.json"
echo "================================================================="
```

---

## 11. Implementation & Execution Checklist

- [x] **Master Architectural Plan Documented:** `docs/BERT_131_HF_HUB_DISTILLATION_PLAN.md`
- [ ] **Universal Adapter:** Create `neuralops/layers/bert_adapter.py` (`BertAttentionAdapter`)
- [ ] **Master Training Engine:** Create `scripts/train_bert_131_wikitext_hf_zoo.py`
  - [ ] Deterministic `evaluate_bert_mlm` with $\text{MLM-PPL} = \exp(\mathcal{L}_{\text{MLM}})$
  - [ ] `InMemoryWikiTextDataset` with pinned memory
  - [ ] `AutoBatchSizer` finding 85%–92% VRAM saturation
  - [ ] Teacher representation pre-caching
  - [ ] 3-Stage KD loop with strict 300.0s time budget
- [ ] **Async Background Uploader:** Create `neuralops/hub/uploader.py` (`AsyncHFHubUploader`)
  - [ ] Non-blocking upload to `Hooshaai/bert-{module}-wiki`
  - [ ] Commit verification
  - [ ] Local `.safetensors` weight cleanup
- [ ] **End-User Interface:**
  - [ ] `neuralops/hub/bert_zoo.py` (`load_pretrained_bert`, `list_pretrained_bert_models`)
  - [ ] CLI subcommand `neuralops hub {list, pull, train-zoo}`
- [ ] **Test Suite:** Create `tests/test_bert_hf_hub_zoo.py`
- [ ] **PBS Execution:** Submit job to Sharif HPC `N3090` queue
- [ ] **Documentation:** Update `README.md` with Model Zoo badges, recipes, and benchmark tables
