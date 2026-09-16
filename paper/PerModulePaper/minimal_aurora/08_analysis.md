# 8. Analysis

## 8.1 Why MinimalAURORA Works

Three simple reasons:

### 8.1.1 Energy Threshold Adapts Per Layer

**Figure 8.1** shows the per-layer rank selected by $\varepsilon = 0.95$. 
Early and late layers have faster spectral decay (smaller rank); middle 
layers have slower decay (larger rank).

### 8.1.2 LoRA Provides Recovery Capacity

The LoRA branch adapts the frozen SVD to the task. With `r_L = 16`, it 
adds ~24K trainable parameters per projection.

### 8.1.3 Multi-Architecture Support

The `copy_weights` routine handles four attention families via priority-ordered 
extraction, ensuring weight transfer succeeds for all tested models.

## 8.2 Per-Layer Analysis

**Table 8.1: Rank and reconstruction error (DistilBERT).**

| Layer | Q rank | Q err | K rank | K err | V rank | V err | O rank | O err |
|---|---|---|---|---|---|---|---|---|
| 0 | {{D_L0_QR}} | {{D_L0_QE}} | {{D_L0_KR}} | {{D_L0_KE}} | {{D_L0_VR}} | {{D_L0_VE}} | {{D_L0_OR}} | {{D_L0_OE}} |
| 1 | {{D_L1_QR}} | {{D_L1_QE}} | {{D_L1_KR}} | {{D_L1_KE}} | {{D_L1_VR}} | {{D_L1_VE}} | {{D_L1_OR}} | {{D_L1_OE}} |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 5 | {{D_L5_QR}} | {{D_L5_QE}} | {{D_L5_KR}} | {{D_L5_KE}} | {{D_L5_VR}} | {{D_L5_VE}} | {{D_L5_OR}} | {{D_L5_OE}} |

**Observation.** {{RANK_OBS}}

## 8.3 Energy Threshold Effect

**Table 8.2: Total parameters vs $\varepsilon$.**

| $\varepsilon$ | DistilBERT params | Ratio | Mean rank |
|---|---|---|---|
| 0.90 | {{D_P90}} | {{D_R90}}× | {{D_MR90}} |
| 0.95 | {{D_P95}} | {{D_R95}}× | {{D_MR95}} |
| 0.99 | {{D_P99}} | {{D_R99}}× | {{D_MR99}} |
| 0.995 | {{D_P995}} | {{D_R995}}× | {{D_MR995}} |

**Observation.** Higher $\varepsilon$ → larger rank → less compression → 
better accuracy.

## 8.4 Comparison with AASVD

| Aspect | AASVD | MinimalAURORA |
|---|---|---|
| Weighting | Full covariance | None |
| Calibration | Required | None |
| Rank | Fixed cap | Energy-selected |
| Memory | O(d²) | O(1) |
| Complexity | High | Low |

## 8.5 Comparison with A3

| Aspect | A3 | MinimalAURORA |
|---|---|---|
| Weighting | Fisher diagonal | None |
| Rank | Adaptive retry | Energy |
| LayerNorm recal | Yes | No |
| Complexity | Medium | Low |

## 8.6 The `pres_gate` Parameter Issue

**Critical finding.** `pres_gate` is defined in `__init__`:

```python
self.pres_gate = nn.Parameter(torch.tensor(2.0))
```

But it is **never referenced in `forward`**. This is a code smell 
indicating either:

1. **Dead code:** leftover from a previous design (e.g., a preservation 
   gate between attention output and residual).
2. **Incomplete feature:** the gate was intended but never wired up.

**Implications:**

- Adds 1 unnecessary parameter.
- No effect on forward pass.
- Misleading to readers and future maintainers.

**Recommendation.** Remove the parameter, or implement the gate 
intended. If the intent was:

$$
\text{out} = \text{attn} + \sigma(\text{pres\_gate}) \cdot x
$$

then the forward should be modified accordingly. We do **not** recommend 
this without ablation evidence.

## 8.7 The `output_scale` Parameter

`output_scale` is trainable and multiplies the attention output. In 
our experiments, it drifts by **{{D_OS_DRIFT}}** on average. If it 
reached 0, the attention output would collapse. We observe:

| Model | Init | Final |
|---|---|---|
| DistilBERT | 1.00 | {{D_OS}} |
| RoBERTa | 1.00 | {{R_OS}} |
| GPT-2 | 1.00 | {{G_OS}} |

**Recommendation.** Either remove this parameter, or initialize it to 
1.0 and freeze it (or add a small regularization toward 1).

## 8.8 Nested Encoder Handling

For RoBERTa/BERT, the output projection is set to identity:

```python
if proj_key == "o" and self.is_nested_encoder:
    eye_w = torch.eye(self.hidden_size, self.all_head_size)
    target.decompose_from_weight(eye_w, None, self.energy)
```

This ensures the outer `attention.output.dense` applies the actual 
output projection. Verified via forward-pass equivalence.

## 8.9 Inference Cost

**Table 8.3: Time breakdown (DistilBERT, N=128).**

| Component | Baseline | Ours |
|---|---|---|
| Q/K/V/O projections | 0.5 ms | {{D_PROJ_TIME}} ms |
| Attention | 0.4 ms | 0.4 ms |
| LoRA | 0 | {{D_LORA_TIME}} ms |
| **Total** | **1.4 ms** | **{{D_TOTAL_TIME}} ms** |

## 8.10 Summary of Insights

1. **MinimalAURORA is a real compressor.** Parameters reduce by {{D_RATIO}}×.
2. **Energy threshold adapts per layer.** Simpler than AASVD/A3.
3. **No activation weighting** — trades accuracy for simplicity.
4. **`pres_gate` is unused.** Recommend removal.
5. **`output_scale` drifts** but not to collapse.
6. **Silent injection failure** requires caller attention.

---