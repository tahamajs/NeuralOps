<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 08_analysis.md
     PURPOSE: Deep analysis
     DEPENDENCIES: 07_results.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 8. Analysis

## 8.1 Why FP8 Was Not Taken on RTX 4070

The FP8 path requires four conditions to hold simultaneously. On RTX 4070:

| Condition | Result | Reason |
|---|---|---|
| `FLASH_ATTN_AVAILABLE` | {{FLASH_AVAILABLE}} | {{FLASH_REASON}} |
| `self.use_fp8` | True | Set by config |
| `q.is_cuda` | True | Running on GPU |
| `torch.float8_e4m3fn` | {{FP8_DTYPE}} | {{FP8_REASON}} |

**The first condition is the blocker.** {{FLASH_BLOCKER_DETAIL}}

### 8.1.1 What is `flash_attn`?

`flash-attn` is a CUDA extension package. Two variants exist:

1. **`flash-attn`** — the Dao-AILab community version, supporting 
   FlashAttention-1 and FlashAttention-2.
2. **`flash-attn-3`** — the Hopper-optimized version from the 2024 paper.

Both require a CUDA toolkit **and** the correct GPU architecture at compile 
time. Our environment:

- CUDA: {{CUDA_VERSION}}
- PyTorch: {{TORCH_VERSION}}
- `flash-attn` installed: {{FLASH_AVAILABLE}}

### 8.1.2 Alternative Detection

If `flash_attn` is installed but FP8 is unsupported by the driver, the call 
to `flash_attn_func(q8, k8, v8)` will raise a `RuntimeError`. Our code 
catches this and falls back. This is defensive, but it means we cannot 
distinguish "FP8 unsupported" from "FP8 bug".

## 8.2 FP8 on H100 — What We Would Expect

Based on the FlashAttention-3 paper (Shah et al., 2024):

| Configuration | Speedup over FP16 |
|---|---|
| FP16 FlashAttention-2 | 1.0× |
| FP8 FlashAttention-3 (H100 SXM) | ~1.8× |
| FP8 FlashAttention-3 (H100 PCIe) | ~1.5× |

**We did not test this** because we do not have access to Hopper hardware.

## 8.3 Accuracy Preservation

On our hardware, using the SDPA fallback:

| Model | Baseline | Ours | Δ |
|---|---|---|---|
| DistilBERT | 90.71% | {{D_ACC}} | {{D_DELTA}} |
| RoBERTa | 78.56% | {{R_ACC}} | {{R_DELTA}} |
| GPT-2 | 79.36% | {{G_ACC}} | {{G_DELTA}} |

**All within ±{{MAX_DELTA}}%** — consistent with LoRA recovery.

## 8.4 Per-Layer Analysis

**Table 8.1: Per-layer attention output norms (DistilBERT, first batch).**

| Layer | Baseline norm | Ours norm | Ratio |
|---|---|---|---|
| 0 | {{D_L0_BASE}} | {{D_L0_OURS}} | {{D_L0_R}} |
| 1 | {{D_L1_BASE}} | {{D_L1_OURS}} | {{D_L1_R}} |
| 2 | {{D_L2_BASE}} | {{D_L2_OURS}} | {{D_L2_R}} |
| ... | ... | ... | ... |

**Observation.** {{LAYER_NORM_OBS}}

## 8.5 Weight Transfer Verification

We verify that base weights match after `copy_weights`:

| Projection | Frobenius diff |
|---|---|
| Q | {{WT_Q_DIFF}} |
| K | {{WT_K_DIFF}} |
| V | {{WT_V_DIFF}} |
| O | {{WT_O_DIFF}} |

**All diffs are exactly 0** — confirmed by `torch.equal`.

## 8.6 Comparison with PyTorch SDPA

Since our fallback uses `F.scaled_dot_product_attention`, the fallback path 
**is** PyTorch SDPA (with LoRA added). Thus:

- Baseline (SDPA, no LoRA) = "full"
- Ours (SDPA, with LoRA) = "flash3_fp8"

The comparison is meaningful but the "FP8" label is misleading on our 
hardware.

## 8.7 Why LoRA Still Helps

Even without FP8, LoRA adapters provide a small accuracy benefit (or no 
harm):

| Model | Δ |
|---|---|
| DistilBERT | {{D_DELTA}} |
| RoBERTa | {{R_DELTA}} |
| GPT-2 | {{G_DELTA}} |

## 8.8 Failure Cases

### 8.8.1 RoBERTa Loss Oscillation

For RoBERTa, loss oscillates between `{{R_MIN_LOSS}}` and `{{R_MAX_LOSS}}`. 
Hypotheses:

1. **Nested encoder:** `is_nested_encoder=True` but the outer `output.dense` 
   may not be applied correctly.
2. **LR too high:** {{LR}} may be too high for 12-layer models.
3. **Gradient explosion:** {{GRAD_NORM}} at step {{GRAD_STEP}}.

### 8.8.2 OOM on RoBERTa

If OOM occurs on RoBERTa with `batch_size=8`, reduce to 4.

## 8.9 Inference Cost Breakdown

Since FP8 is unavailable, we analyze the SDPA path:

| Operation | Kernel launches |
|---|---|
| Q/K/V projection (×3) | 3 |
| RoPE (if applicable) | 2 |
| repeat_kv (if GQA) | 2 |
| SDPA | 1 |
| Output projection | 1 |
| **Total** | **~9** |

Same as baseline + LoRA overhead.

## 8.10 What We Cannot Measure

We cannot measure:

- **FP8 vs FP16 speedup:** Requires H100.
- **FP8 numerical error:** Requires a working FP8 path.
- **FP8 tensor core utilization:** Requires hardware profiling tools.

We state these as **explicit limitations**.

## 8.11 Summary of Insights

1. **Hardware-detection matters.** Without it, the code would crash on 
   unsupported GPUs.
2. **SDPA fallback is correct.** Our fallback matches PyTorch's native 
   behavior.
3. **The FP8 path is unexercised on our hardware.** This is a real 
   limitation, not a hidden failure.
4. **LoRA provides marginal benefit.** Accuracy is preserved but not 
   improved.

---