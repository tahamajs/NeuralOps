# Efficient Transformer Toolkit

A **model‑agnostic, production‑ready** framework for compressing, adapting, and fine‑tuning **any HuggingFace Transformer**.  
Swap in **108 state‑of‑the‑art attention mechanisms**, compress linear layers with **58 compression engines**, and train with a full‑featured pipeline—all with automatic architecture detection and zero-shot pretrained weight transfer.

---

## 🚀 Quick Start

```python
# 1. Load a pretrained model with a custom attention variant
from framework.models import load_model

model, tokenizer = load_model(
    "distilbert-base-uncased", attention_type="aegis_hybrid", attention_rank=64
)

# 2. Prepare data (GLUE / custom / synthetic)
from framework.data import DataConfig, TaskType, create_dataloader, load_dataset

dataset = load_dataset(ExperimentConfig(dataset_name="sst2"))
config = DataConfig(
    task_type=TaskType.SINGLE_CLASSIFICATION,
    text_column_1="sentence",
    label_column="label",
)
train_loader = create_dataloader(dataset["train"], tokenizer, config)

# 3. Train with a full experiment tracker and automatic results saving
from framework.results_saver import ResultsSaver, SaveConfig
from framework.tracking import ExperimentTracker
from framework.training import HuggingFaceTrainer, ResultsManager

results = ResultsManager("my_experiment")
tracker = ExperimentTracker(backend="tensorboard", log_dir="./runs")
trainer = HuggingFaceTrainer(
    model=model,
    train_dataset=train_loader.dataset,
    eval_dataset=val_loader.dataset,
    config={"learning_rate": 2e-5, "epochs": 3, "batch_size": 16},
    results_manager=results,
)
trainer.train()

# 4. Save and share your compressed model
saver = ResultsSaver(
    run_id=results.run_dir.name,
    base_dir=results.base_dir,
    config=SaveConfig(storage_backends=["hf_hub"]),
)
saver.save_all(model, config, results.history, metrics, tokenizer)
saver.push_all()
```

---

## ✨ Key Features

- **Universal Model Loading** – DistilBERT, BERT, RoBERTa, GPT‑2, Llama, Mistral, Gemma, … all supported out‑of‑the‑box.
- **108 Attention Mechanisms** – Complete suite across 7 paradigm families (Dense MHA/GQA, Spectral & SVD, Kernel, SSM & Recurrent, Sparse & MoE, Aegis Hybrid, and Fused Hardware-Accelerated).
- **58 Compression Engines** – SVD factorizers, Tensor Train, Kronecker product, SparseGPT, SliceGPT, ADMM pruners, KV-Cache delta compressors, and Astro INT4/INT2 quantizers.
- **Zero‑Shot Weight Transfer** – Automatically decomposes and copies pretrained projection weights into new attention modules via SVD and leverage scores.
- **Complete Training Stack** – HuggingFace Trainer wrapper, custom training loop, hyperparameter optimization (Optuna), learning rate schedulers, and callbacks.
- **Security & Integrity** – Ed25519-based cryptographic model signing and verification.

---

## 📚 Complete Attention Modules Reference (108 Mechanisms)

Every attention mechanism is registered in `ATTENTION_REGISTRY` and has a dedicated technical manual (`.md`) and PyTorch module (`.py`) in [`attentions/`](./attentions/).

### 1. Dense MHA, GQA & Windowed Attention

| Registry Key | Module File | Documentation | Description | Time Complexity |
|--------------|-------------|---------------|-------------|-----------------|
| `full` | `full_attention.py` | [full_attention.md](./attentions/full_attention.md) | Standard Multi-Head Scaled Dot-Product Attention | $\mathcal{O}(B \cdot H \cdot L^2 \cdot d_h)$ |
| `gqa` | `gqa_attention.py` | [gqa_attention.md](./attentions/gqa_attention.md) | Grouped-Query Attention (GQA) for fast KV decoding | $\mathcal{O}(B \cdot H \cdot L^2 \cdot d_h)$ |
| `wgqa` | `wgqa_attention.py` | [wgqa_attention.md](./attentions/wgqa_attention.md) | Weighted Grouped-Query Attention with dynamic head scaling | $\mathcal{O}(B \cdot H \cdot L^2 \cdot d_h)$ |
| `qca` / `qcqa` | `qcqa_attention.py` | [qcqa_attention.md](./attentions/qcqa_attention.md) | Quality-Capacity-Aware GQA with adaptive head pruning | $\mathcal{O}(B \cdot H \cdot L^2 \cdot d_h)$ |
| `mla` | `mla_attention.py` | [mla_attention.md](./attentions/mla_attention.md) | Multi-Head Latent Attention with compressed KV projections | $\mathcal{O}(B \cdot L \cdot (d_c + d_k))$ |
| `compressed_mha` | `compressed_mha_attention.py` | [compressed_mha_attention.md](./attentions/compressed_mha_attention.md) | Latent compressed Multi-Head Attention | $\mathcal{O}(B \cdot L \cdot r + B \cdot H \cdot L^2 \cdot d_h)$ |
| `adaptive_span` | `adaptive_span_attention.py` | [adaptive_span_attention.md](./attentions/adaptive_span_attention.md) | Dynamic learnable context window span | $\mathcal{O}(B \cdot H \cdot L \cdot S_{\text{span}})$ |
| `span_expanded` | `span_expanded_attention.py` | [span_expanded_attention.md](./attentions/span_expanded_attention.md) | Multi-scale span expansion attention | $\mathcal{O}(B \cdot H \cdot L \cdot S_{\text{max}})$ |
| `tpa` | `tpa_attention.py` | [tpa_attention.md](./attentions/tpa_attention.md) | Temporal Pattern Attention | $\mathcal{O}(B \cdot H \cdot L^2 \cdot d_h)$ |
| `unified_rope` | `unified_rope.py` | [unified_rope.md](./attentions/unified_rope.md) | Unified Rotary Position Embedding Attention | $\mathcal{O}(B \cdot H \cdot L^2 \cdot d_h)$ |

---

### 2. Spectral & SVD Factorized Attention

| Registry Key | Module File | Documentation | Description | Time Complexity |
|--------------|-------------|---------------|-------------|-----------------|
| `minimal_svd` | `minimal_svd.py` | [minimal_svd.md](./attentions/minimal_svd.md) | Minimal truncated SVD low-rank projections | $\mathcal{O}(B \cdot L \cdot r \cdot D)$ |
| `minimal_svd_v2` | `minimal_svd_v2.py` | [minimal_svd_v2.md](./attentions/minimal_svd_v2.md) | Optimized Minimal SVD with residual compensation | $\mathcal{O}(B \cdot L \cdot r \cdot D)$ |
| `svd_linear_attn` | `svd_linear_attention.py` | [svd_linear_attention.md](./attentions/svd_linear_attention.md) | Low-rank factorized linear Q/K/V/O projections | $\mathcal{O}(B \cdot L \cdot r \cdot (d_{\text{in}} + d_{\text{out}}))$ |
| `svd` | `svd_attention.py` | [svd_attention.md](./attentions/svd_attention.md) | SVD attention with spectral entropy truncation | $\mathcal{O}(B \cdot L \cdot r \cdot D + B \cdot H \cdot L^2 \cdot d_h)$ |
| `svd_compressed` | `svd_compressed.py` | [svd_compressed.md](./attentions/svd_compressed.md) | Lagrangian energy-budget SVD rank allocator | $\mathcal{O}(B \cdot L \cdot r_{\text{opt}} \cdot D)$ |
| `svd_projection` | `svd_projection_attention.py` | [svd_projection_attention.md](./attentions/svd_projection_attention.md) | Orthogonal singular subspace projection | $\mathcal{O}(B \cdot L \cdot r \cdot D)$ |
| `svd_llm_v2` | `svd_llm_v2_attention.py` | [svd_llm_v2_attention.md](./attentions/svd_llm_v2_attention.md) | Layer-wise SVD-LLM v2 spectral truncation | $\mathcal{O}(B \cdot L \cdot r \cdot D)$ |
| `adaptive_svd` | `adaptive_svd_attention.py` | [adaptive_svd_attention.md](./attentions/adaptive_svd_attention.md) | Dynamic runtime rank selection per layer | $\mathcal{O}(B \cdot L \cdot r_t \cdot D)$ |
| `nha` | `nha_attention.py` | [nha_attention.md](./attentions/nha_attention.md) | Normalized Head Attention with head whitening | $\mathcal{O}(B \cdot H \cdot L^2 \cdot d_h)$ |
| `rala` | `rala_attention.py` | [rala_attention.md](./attentions/rala_attention.md) | Rank-Augmented Linear Attention | $\mathcal{O}(B \cdot L \cdot r \cdot D + B \cdot L \cdot d_h^2)$ |
| `factorised_matrix` | `factorised_matrix_state.py` | [factorised_matrix_state.md](./attentions/factorised_matrix_state.md) | Low-rank matrix state state-space tracking | $\mathcal{O}(B \cdot L \cdot r \cdot d_h)$ |
| `spectra_v2` | `spectra_v2.py` | [spectra_v2.md](./attentions/spectra_v2.md) | Discrete Fourier / Cosine Transform spectral attention | $\mathcal{O}(B \cdot H \cdot L \log L \cdot d_h)$ |
| `spectron` | `spectron.py` | [spectron.md](./attentions/spectron.md) | Spectrally-decomposed hybrid attention | $\mathcal{O}(B \cdot L \cdot D \log D)$ |

---

### 3. Kernel, Landmark & Linear Attention

| Registry Key | Module File | Documentation | Description | Time Complexity |
|--------------|-------------|---------------|-------------|-----------------|
| `galerkin` | `galerkin_attention.py` | [galerkin_attention.md](./attentions/galerkin_attention.md) | Galerkin operator linear attention | $\mathcal{O}(B \cdot L \cdot d_h^2)$ |
| `linformer` | `linformer_attention.py` | [linformer_attention.md](./attentions/linformer_attention.md) | Linear projections to fixed sequence length $E, F$ | $\mathcal{O}(B \cdot H \cdot L \cdot E)$ |
| `performer` | `performer_attention.py` | [performer_attention.md](./attentions/performer_attention.md) | FAVOR+ random Fourier feature linear attention | $\mathcal{O}(B \cdot L \cdot M_{\text{rff}} \cdot d_h)$ |
| `nystrom` | `nystrom_attention.md` | [nystrom_attention.md](./attentions/nystrom_attention.md) | Nyström landmark token approximation | $\mathcal{O}(B \cdot H \cdot L \cdot M_{\text{landmarks}})$ |
| `kda` | `kda_attention.py` | [kda_attention.md](./attentions/kda_attention.md) | Kernel Dynamic Attention | $\mathcal{O}(B \cdot L \cdot d_h^2)$ |
| `adasplash` | `adasplash_attention.py` | [adasplash_attention.md](./attentions/adasplash_attention.md) | Adaptive Splash sparse kernel attention | $\mathcal{O}(B \cdot H \cdot L \cdot B_{\text{splash}})$ |
| `vig` | `vig_attention.py` | [vig_attention.md](./attentions/vig_attention.md) | Vision Graph Grid Attention | $\mathcal{O}(B \cdot K \cdot L \cdot d_h)$ |
| `rpc` | `rpc_attention.py` | [rpc_attention.md](./attentions/rpc_attention.md) | Random Projection Compressor attention | $\mathcal{O}(B \cdot L \cdot r \cdot d_h)$ |
| `lola` | `lola_attention.py` | [lola_attention.md](./attentions/lola_attention.md) | Low-Overhead Linear Attention | $\mathcal{O}(B \cdot L \cdot d_h^2)$ |
| `primal` | `primal_attention.py` | [primal_attention.md](./attentions/primal_attention.md) | Primal dual-form linear attention | $\mathcal{O}(B \cdot L \cdot d_h^2)$ |
| `chela_slay` | `chela_slay.py` | [chela_slay.md](./attentions/chela_slay.md) | Chebyshev polynomial kernel expansion | $\mathcal{O}(B \cdot K \cdot L \cdot d_h)$ |

---

### 4. State-Space Models (SSM) & Recurrent Attention

| Registry Key | Module File | Documentation | Description | Time Complexity |
|--------------|-------------|---------------|-------------|-----------------|
| `mamba` | `mamba.py` | [mamba.md](./attentions/mamba.md) | Selective State-Space Model (Mamba) linear layer | $\mathcal{O}(B \cdot L \cdot D^2)$ |
| `gated_delta_net` | `gated_delta_net.py` | [gated_delta_net.md](./attentions/gated_delta_net.md) | Hierarchical Gated Recurrent Network (HGRN2) DeltaNet | $\mathcal{O}(B \cdot L \cdot D^2)$ |
| `gated_deltanet_block` | `gated_deltanet_block.py` | [gated_deltanet_block.md](./attentions/gated_deltanet_block.md) | Gated DeltaNet with optional MHA bypass | $\mathcal{O}(B \cdot L \cdot D^2)$ |
| `hgrn2_layer` | `hgrn2_layer.py` | [hgrn2_layer.md](./attentions/hgrn2_layer.md) | HGRN2 recurrent state tracking layer | $\mathcal{O}(B \cdot L \cdot D^2)$ |
| `hgrn2_outer` | `hgrn2_outer.py` | [hgrn2_outer.md](./attentions/hgrn2_outer.md) | Outer product HGRN2 state update | $\mathcal{O}(B \cdot L \cdot d_h^2)$ |
| `hgrn2_gated_deltanet` | `hgrn2_gated_deltanet.py` | [hgrn2_gated_deltanet.md](./attentions/hgrn2_gated_deltanet.md) | Hybrid HGRN2 + Gated DeltaNet module | $\mathcal{O}(B \cdot L \cdot D^2)$ |
| `griffin_ssd` | `griffin_ssd.py` | [griffin_ssd.md](./attentions/griffin_ssd.md) | Selective State-Space Duality (SSD Griffin) | $\mathcal{O}(B \cdot L \cdot D \cdot d_{\text{state}})$ |
| `jamba2` | `jamba2.py` | [jamba2.md](./attentions/jamba2.md) | Hybrid Jamba2 Transformer-Mamba block | $\mathcal{O}(B \cdot L \cdot D^2)$ |
| `jamba_info` | `jamba_info.py` | [jamba_info.md](./attentions/jamba_info.md) | Jamba Info state tracking block | $\mathcal{O}(B \cdot L \cdot D^2)$ |
| `infini` | `infini_attention.py` | [infini_attention.md](./attentions/infini_attention.md) | Infini-Attention bounded long-context memory | $\mathcal{O}(B \cdot L \cdot d_h^2)$ |
| `hymba_transx` | `hymba_transx.py` | [hymba_transx.md](./attentions/hymba_transx.md) | Hymba TransX recurrent hybrid attention | $\mathcal{O}(B \cdot L \cdot D^2)$ |
| `simple_ssm` | `simple_ssm.py` | [simple_ssm.md](./attentions/simple_ssm.md) | Lightweight linear State-Space Model | $\mathcal{O}(B \cdot L \cdot D \cdot d_{\text{state}})$ |
| `upi_stuffed_mamba` | `upi_stuffed_mamba.py` | [upi_stuffed_mamba.md](./attentions/upi_stuffed_mamba.md) | Universal Prompt Injection Stuffed Mamba | $\mathcal{O}(B \cdot L \cdot D^2)$ |

---

### 5. Sparse Routing & Mixture-of-Experts (MoE) Attention

| Registry Key | Module File | Documentation | Description | Time Complexity |
|--------------|-------------|---------------|-------------|-----------------|
| `moba` | `moba_attention.py` | [moba_attention.md](./attentions/moba_attention.md) | Mixture of Block Attention (MoBA) | $\mathcal{O}\left(B \cdot H \cdot \frac{L^2}{E} \cdot d_h\right)$ |
| `bam_umoe` | `bam_umoe.py` | [bam_umoe.md](./attentions/bam_umoe.md) | Unified Mixture-of-Experts Attention | $\mathcal{O}(B \cdot L \cdot E_{\text{topk}} \cdot d_h)$ |
| `samba_moe` | `samba_moe.py` | [samba_moe.md](./attentions/samba_moe.md) | Samba MoE expert-routed attention | $\mathcal{O}(B \cdot L \cdot E_{\text{topk}} \cdot d_h)$ |
| `samba_m3` | `samba_m3.py` | [samba_m3.md](./attentions/samba_m3.md) | Samba M3 multi-scale expert block | $\mathcal{O}(B \cdot L \cdot E_{\text{topk}} \cdot d_h)$ |
| `cluster` | `cluster_attention.py` | [cluster_attention.md](./attentions/cluster_attention.md) | ClusterFormer token $K$-means clustering attention | $\mathcal{O}\left(B \cdot H \cdot \frac{L^2}{K} \cdot d_h\right)$ |
| `routing` | `routing_attention.py` | [routing_attention.md](./attentions/routing_attention.md) | Vector-quantized Routing Attention | $\mathcal{O}(B \cdot H \cdot L \cdot K \cdot d_h)$ |
| `lsh` | `lsh_attention.py` | [lsh_attention.md](./attentions/lsh_attention.md) | Reformer Locality-Sensitive Hashing (LSH) | $\mathcal{O}(B \cdot H \cdot L \log L \cdot d_h)$ |
| `xattention` | `xattention.py` | [xattention.md](./attentions/xattention.md) | Block-sparse XAttention | $\mathcal{O}(B \cdot H \cdot L \cdot B_{\text{block}} \cdot d_h)$ |
| `block_transformer` | `block_transformer_hiba.py` | [block_transformer_hiba.md](./attentions/block_transformer_hiba.md) | Hierarchical Block Transformer attention | $\mathcal{O}(B \cdot H \cdot L \cdot B_{\text{block}})$ |
| `nsa` | `nsa_attention.py` | [nsa_attention.md](./attentions/nsa_attention.md) | Native Sparse Attention | $\mathcal{O}(B \cdot H \cdot L \cdot S_{\text{sparse}})$ |
| `eat` | `eat.py` | [eat.md](./attentions/eat.md) | Expert-Aware Token Routing Attention | $\mathcal{O}(B \cdot L \cdot E_{\text{topk}} \cdot d_h)$ |
| `squeezing_heads_distill` | `squeezing_heads_distill.py` | [squeezing_heads_distill.md](./attentions/squeezing_heads_distill.md) | Head distillation & squeezing attention | $\mathcal{O}(B \cdot H_{\text{distill}} \cdot L^2 \cdot d_h)$ |

---

### 6. Aegis Architecture Suite & Hybrid Attention

| Registry Key | Module File | Documentation | Description | Time Complexity |
|--------------|-------------|---------------|-------------|-----------------|
| `aegis_hybrid` | `aegis_hybrid.py` | [aegis_hybrid.md](./attentions/aegis_hybrid.md) | Aegis v1 Hybrid: Low-Rank + Local Window + Global Linear | $\mathcal{O}(B \cdot L \cdot r \cdot D + B \cdot H \cdot L W)$ |
| `aegis_v2` | `aegis_v2.py` | [aegis_v2.md](./attentions/aegis_v2.md) | Aegis v2 with adaptive gating branches | $\mathcal{O}(B \cdot L \cdot r \cdot D + B \cdot H \cdot L W)$ |
| `aegis_v3` | `aegis_v3.py` | [aegis_v3.md](./attentions/aegis_v3.md) | Aegis v3 with spectral residual routing | $\mathcal{O}(B \cdot L \cdot r \cdot D + B \cdot H \cdot L W)$ |
| `aegis_v4` | `aegis_v4.py` | [aegis_v4.md](./attentions/aegis_v4.md) | Aegis v4 multi-scale local band fusion | $\mathcal{O}(B \cdot L \cdot r \cdot D + B \cdot H \cdot L W)$ |
| `aegis_v5` | `aegis_v5.py` | [aegis_v5.md](./attentions/aegis_v5.md) | Aegis v5 with depthwise convolution branches | $\mathcal{O}(B \cdot L \cdot r \cdot D + B \cdot H \cdot L W)$ |
| `aegis_v6` | `aegis_v6.py` | [aegis_v6.md](./attentions/aegis_v6.md) | Aegis v6 full state-space hybrid architecture | $\mathcal{O}(B \cdot L \cdot r \cdot D + B \cdot H \cdot L W)$ |
| `granite_hybrid` | `granite_hybrid.py` | [granite_hybrid.md](./attentions/granite_hybrid.md) | Granite Hybrid block attention | $\mathcal{O}(B \cdot H \cdot L^2 \cdot d_h)$ |
| `orion` | `orion.py` | [orion.md](./attentions/orion.md) | Orion SVD + head-split kernel + Softpick | $\mathcal{O}(B \cdot L \cdot r \cdot D + B \cdot H \cdot L W)$ |
| `phantom` | `phantom.py` | [phantom.md](./attentions/phantom.md) | Phantom CUR leverage-score factorized LSH | $\mathcal{O}(B \cdot L \cdot r \cdot D + B \cdot H \cdot L \log L)$ |
| `nexus` | `nexus.py` | [nexus.md](./attentions/nexus.md) | Nexus multi-paradigm fused hybrid module | $\mathcal{O}(B \cdot L \cdot r \cdot D + B \cdot H \cdot L W)$ |
| `leka_v1` / `v2` / `v3` | `leka_v1.py` / `v2` / `v3` | [leka_v1.md](./attentions/leka_v1.md) | Leka multi-stage kernelized attention suite | $\mathcal{O}(B \cdot L \cdot d_h^2)$ |
| `salt` | `salt.py` | [salt.md](./attentions/salt.md) | Sparse Adaptive Linear Transformer | $\mathcal{O}(B \cdot L \cdot d_h^2)$ |
| `vspec` | `vspec.py` | [vspec.md](./attentions/vspec.md) | Vectorized Spectral Attention | $\mathcal{O}(B \cdot H \cdot L \log L \cdot d_h)$ |
| `minimal_aurora` | `minimal_aurora.py` | [minimal_aurora.md](./attentions/minimal_aurora.md) | Minimal Aurora hybrid module | $\mathcal{O}(B \cdot L \cdot r \cdot D)$ |
| `minimal_nova` | `minimal_nova.py` | [minimal_nova.md](./attentions/minimal_nova.md) | Minimal Nova hybrid module | $\mathcal{O}(B \cdot L \cdot r \cdot D)$ |
| `minimal_prism` | `minimal_prism.py` | [minimal_prism.md](./attentions/minimal_prism.md) | Minimal Prism hybrid module | $\mathcal{O}(B \cdot L \cdot r \cdot D)$ |
| `optimized_hybrid` | `optimized_hybrid_attention.py` | [optimized_hybrid_attention.md](./attentions/optimized_hybrid_attention.md) | Optimized Multi-Branch Hybrid Attention | $\mathcal{O}(B \cdot L \cdot r \cdot D + B \cdot H \cdot L W)$ |

---

### 7. Hardware-Accelerated & Fused Attention

| Registry Key | Module File | Documentation | Description | Time Complexity |
|--------------|-------------|---------------|-------------|-----------------|
| `flash3` | `flash_attention3.py` | [flash_attention3.md](./attentions/flash_attention3.md) | Hopper TMA asynchronous FlashAttention-3 | $\mathcal{O}(B \cdot H \cdot L^2 \cdot d_h)$ (IO-optimal) |
| `flash3_fp8` | `flash_attention3_fp8.py` | [flash_attention3_fp8.md](./attentions/flash_attention3_fp8.md) | FlashAttention-3 with FP8 ($E4M3$) Tensor Core math | $\mathcal{O}(B \cdot H \cdot L^2 \cdot d_h)$ ($2\times$ throughput) |
| `soa` / `soa_fused` | `soa_attn.py` / `SOAFusedAttention.py` | [soa_attn.md](./attentions/soa_attn.md) | Structure-of-Arrays (SoA) Fused Attention | $\mathcal{O}(B \cdot H \cdot L^2 \cdot d_h)$ |
| `fusion_attn` | `fusion_attn.py` | [fusion_attn.md](./attentions/fusion_attn.md) | Fused PyTorch CUDA kernel attention | $\mathcal{O}(B \cdot H \cdot L^2 \cdot d_h)$ |
| `goldfinch_ahn` | `goldfinch_ahn.py` | [goldfinch_ahn.md](./attentions/goldfinch_ahn.md) | Goldfinch Adaptive Head Network attention | $\mathcal{O}(B \cdot H_{\text{active}} \cdot L^2 \cdot d_h)$ |
| `halo_regla` | `halo_regla.py` | [halo_regla.md](./attentions/halo_regla.md) | Halo Regularized Linear Attention | $\mathcal{O}(B \cdot L \cdot d_h^2)$ |
| `hamming` | `hamming_attention.py` | [hamming_attention.md](./attentions/hamming_attention.md) | Bitwise Hamming distance binary attention | $\mathcal{O}(B \cdot H \cdot L^2 \cdot d_h / 64)$ |
| `hla` | `hla_attention.py` | [hla_attention.md](./attentions/hla_attention.md) | Hierarchical Linear Attention | $\mathcal{O}(B \cdot L \cdot d_h^2)$ |
| `lora` | `lora_attention.py` | [lora_attention.md](./attentions/lora_attention.md) | Low-Rank Adapted (LoRA) Multi-Head Attention | $\mathcal{O}(B \cdot L \cdot (D d_h + r D))$ |
| `mixattention_moska` | `mixattention_moska.py` | [mixattention_moska.md](./attentions/mixattention_moska.md) | Multi-Scale Gated Mixture Attention | $\mathcal{O}(B \cdot H \cdot L^2 \cdot d_h)$ |
| `regla` | `regla_attention.py` | [regla_attention.md](./attentions/regla_attention.md) | Regularized Linear Attention | $\mathcal{O}(B \cdot L \cdot d_h^2)$ |
| `weighted_ensemble` | `weighted_ensemble.py` | [weighted_ensemble.md](./attentions/weighted_ensemble.md) | Weighted Ensemble Multi-Module Attention | $\mathcal{O}(B \cdot M \cdot H \cdot L^2 \cdot d_h)$ |
| `zeros` | `zeros_attention.py` | [zeros_attention.md](./attentions/zeros_attention.md) | Zero-memory state attention testing stub | $\mathcal{O}(B \cdot L \cdot D)$ |

---

## 🛠️ Customization & Registration

### Adding a New Attention Module
1. Inherit from `BaseAttention` or `nn.Module`.
2. Implement the `forward` method with standard signature `(hidden_states, attention_mask=None, output_attentions=False, **kwargs)`.
3. Register with `@ATTENTION_REGISTRY.register("my_attn_key")`.

```python
import torch
import torch.nn as nn
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("custom_linear_attn")
class CustomLinearAttention(BaseAttention):

  def __init__(self, config, rank=None, **kwargs):
    super().__init__(config, rank)
    self.proj = nn.Linear(config.hidden_size, config.hidden_size)

  def forward(self, hidden_states, attention_mask=None, **kwargs):
    return self.proj(hidden_states), None
```

---

## 🔒 Dependencies

- `torch ≥ 2.0`
- `transformers`
- `datasets`
- `numpy`
- `scipy`

---

## 📄 License

This project is licensed under the MIT License.

*Built with ❤️ for the efficient Transformer community.*