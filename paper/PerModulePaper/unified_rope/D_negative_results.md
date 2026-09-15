<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 No Nested Encoder Detection

For RoBERTa: without `is_nested_encoder = True`, projection applied twice. 
Collapse.

### D.1.2 Conv1D Not Transposed

For GPT-2: without transposition, Q/K/V swapped. Collapse.

### D.1.3 GQA Not Expanded

For Qwen: without `repeat_kv`, shape mismatch.

### D.1.4 No LoRA Recovery

For DistilBERT, RoBERTa, GPT-2: without LoRA, adding RoPE causes 
significant accuracy loss.

### D.1.5 Wrong RoPE Base

For DistilBERT: base=1000 → accuracy drop.

## D.2 Abandoned Variants

### D.2.1 Per-Head RoPE Base

Different base per head. Result: no gain, more parameters.

### D.2.2 Learned RoPE Base

Learnable base. Result: unstable, no gain.

### D.2.3 Partial RoPE

Rotary dim < head_dim. Result: similar accuracy.

## D.3 Failed Hyperparameters

### RoPE Base

| Base | DistilBERT |
|---|---|
| 1000 | {{D_BASE1K}} |
| 5000 | {{D_BASE5K}} |
| 10000 | {{D_ACC}} |
| 100000 | {{D_BASE100K}} |

### Rotary Dim

| Rot dim | DistilBERT |
|---|---|
| 16 | {{D_ROT16}} |
| 32 | {{D_ROT32}} |
| 64 | {{D_ROT64}} |
| Full | {{D_ACC}} |

## D.4 Fixes That Didn't Help

| Fix | Problem | Result |
|---|---|---|
| Larger LoRA rank | Slow recovery | No gain |
| Higher LR | Convergence | Unstable |
| Longer warmup | Stability | No gain |
| Different dropout | Overfitting | No gain |

## D.5 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | {{SEED42}} |
| 123 | {{SEED123}} |
| 456 | {{SEED456}} |
| **Mean** | **{{MEAN}}** |
| **Std** | **{{STD}}** |

## D.6 Companion Method Comparison

| Method | DistilBERT | Reduces |
|---|---|---|
| MinimalSVD | {{MINSVD_D}} | Params |
| SVD-LLM V2 | {{SVDLLM_D}} | Params |
| Dobi-SVD | {{DOBI_D}} | Params |
| MLA | {{MLA_D}} | KV |
| Nyströmformer | {{NYS_D}} | Compute |
| **UnifiedRoPE** | **{{D_ACC}}** | **Nothing** |

## D.7 Lessons Learned

1. **Nested encoder bypass is essential.**
2. **Conv1D transposition is essential.**
3. **GQA expansion is essential.**
4. **LoRA recovery needed for adding RoPE.**
5. **RoPE base 10000 works best.**
6. **Full rotary dim is best.**

---
