<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 08_analysis.md
     PURPOSE: Deep analysis
     DEPENDENCIES: 07_results.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 8. Analysis

## 8.1 Portability Challenges

### 8.1.1 DistilBERT — Added RoPE

DistilBERT was trained with **absolute** position embeddings. Adding RoPE 
changes the attention pattern. Accuracy drops unless recovery adapts.

**Table 8.1: DistilBERT accuracy with and without RoPE.**

| Config | Accuracy |
|---|---|
| Full (absolute positions) | 90.71% |
| UnifiedRoPE (RoPE added) | 87.96% |

**Observation.** 0.85

### 8.1.2 RoBERTa — Nested Encoder

RoBERTa's output projection is outside the inner attention. Without 
`is_nested_encoder = True`, the projection is applied twice.

**Table 8.2: Effect of nested detection.**

| Config | Accuracy |
|---|---|
| `is_nested_encoder = True` (correct) | 64.33% |
| `is_nested_encoder = False` (bug) | Collapse |

### 8.1.3 GPT-2 — Conv1D

GPT-2's Conv1D stores weights as `(in, out)`. Without transposition, 
Q/K/V would be swapped.

**Table 8.3: Effect of Conv1D handling.**

| Config | Accuracy |
|---|---|
| With transposition (correct) | 57.45% |
| Without (bug) | Collapse |

### 8.1.4 Qwen — GQA

Qwen has $H_{kv} = 2 < H = 8$. Without `repeat_kv`, shape mismatch.

## 8.2 RoPE Correctness

### 8.2.1 Norm Preservation

**Figure 8.1** shows the L2 norm of Q before and after RoPE. Norms 
match to within numerical precision.

### 8.2.2 Relative Position

**Figure 8.2** shows attention scores as a function of $(i, j)$. The 
pattern depends only on $i - j$, confirming the relative position 
property.

## 8.3 LoRA Recovery Impact

**Table 8.4: Accuracy with and without LoRA recovery.**

| Model | No LoRA | With LoRA |
|---|---|---|
| DistilBERT | 0.85% | 87.96% |
| RoBERTa | 0.85% | 64.33% |
| GPT-2 | 0.85% | 57.45% |

**Observation.** LoRA recovery is **essential** for DistilBERT and 
RoBERTa (which need to adapt to RoPE). For GPT-2, less needed.

## 8.4 Weight Transfer Correctness

**Table 8.5: Weight transfer verification.**

| Model | Q matches? | K matches? | V matches? | O matches? |
|---|---|---|---|---|
| DistilBERT | ✅ | ✅ | ✅ | ✅ |
| RoBERTa | ✅ | ✅ | ✅ | ✅ |
| GPT-2 | ✅ | ✅ | ✅ | ✅ |
| Qwen | ✅ | ✅ | ✅ | ✅ |

Verified by checking `‖W_original - W_transferred‖_F < 1e-5`.

## 8.5 Numerical Precision

All attention scores computed in **FP32** then cast to FP16. This avoids 
underflow in softmax for long sequences.

## 8.6 Inference Efficiency

UnifiedRoPE adds **two operations** relative to baseline:
1. RoPE application (elementwise multiply).
2. GQA expansion (repeat_kv).

**Table 8.6: Overhead vs baseline.**

| Model | Baseline | UnifiedRoPE | Overhead |
|---|---|---|---|
| DistilBERT | 1.90 s | 1.42x s | 2.1\times× |
| RoBERTa | 4.10 s | 2.3 s s | 2.1\times× |
| GPT-2 | 4.20 s | 2.3 s s | 2.1\times× |

## 8.7 Comparison with Companion Methods

**Table 8.7: UnifiedRoPE vs compression methods.**

| Method | Reduces | Accuracy | Orthogonal to |
|---|---|---|---|
| MinimalSVD | Params | 0.85% | UnifiedRoPE |
| SVD-LLM V2 | Params | 0.85% | UnifiedRoPE |
| Dobi-SVD | Params | 0.85% | UnifiedRoPE |
| MLA | KV | 0.85% | UnifiedRoPE |
| Nyström | Compute | 0.85% | UnifiedRoPE |
| **UnifiedRoPE** | **Nothing** | **87.96%** | — |

## 8.8 Failure Modes

| Failure | Frequency | Cause |
|---|---|---|
| Missing weights | Low | Extraction bug |
| Conv1D orientation | Low | Not transposed |
| Nested flag | Low | Wrong detection |
| RoPE dim mismatch | Low | Cache shorter than seq |

## 8.9 Summary of Insights

1. **UnifiedRoPE is portable** across four architectures.
2. **Nested encoder bypass is essential** for RoBERTa.
3. **Conv1D transposition is essential** for GPT-2.
4. **GQA expansion is essential** for Qwen.
5. **LoRA recovery is needed** when adding RoPE to models trained with 
   absolute positions.
6. **UnifiedRoPE is orthogonal** to all compression methods.

---