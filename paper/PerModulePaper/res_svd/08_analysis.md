# 8. Analysis

## 8.1 Why Res-SVD Works

Two mechanisms:

### 8.1.1 Spectral Initialization Preserves More Information

At initialization, Res-SVD is equivalent to rank-$(r_b + r_r)$ SVD. This 
is optimal — no other rank-$(r_b + r_r)$ decomposition has lower Frobenius 
error.

**Figure 8.1** compares the initial loss of:
- Standard SVD + zero LoRA: $\|W - W_b\|_F = \sqrt{\sum_{i > r_b} \sigma_i^2}$
- Res-SVD: $\|W - W_b - R_r\|_F = \sqrt{\sum_{i > r_b + r_r} \sigma_i^2}$

The difference is the energy in directions $[r_b, r_b + r_r]$.

### 8.1.2 Trainable Residual Adapts to Task

The residual $B, A$ is trainable, allowing fine-tuning. The base 
$U_b, V_b$ stays frozen, so the compressed representation is preserved.

## 8.2 Per-Layer Analysis

**Table 8.1: Reconstruction error per layer.**

| Layer | Q | K | V | O |
|---|---|---|---|---|
| 0 | {{D_L0_Q}} | {{D_L0_K}} | {{D_L0_V}} | {{D_L0_O}} |
| 1 | {{D_L1_Q}} | {{D_L1_K}} | {{D_L1_V}} | {{D_L1_O}} |
| ... | ... | ... | ... | ... |
| 5 | {{D_L5_Q}} | {{D_L5_K}} | {{D_L5_V}} | {{D_L5_O}} |

**Observation.** {{REC_ERR_OBS}}

## 8.3 Comparison with Pure SVD at Equal Rank

As proven in §4.7.1, Res-SVD at $(r_b, r_r)$ is equivalent to SVD of 
rank $r_b + r_r$ at initialization. Differences during training arise 
because:

1. Only $B, A$ receive gradient, not $U_b, V_b$.
2. The optimizer may update $B, A$ in ways that diverge from optimal 
   rank-$(r_b + r_r)$ directions.

**Table 8.2: Accuracy comparison.**

| Method | DistilBERT | RoBERTa |
|---|---|---|
| SVD $r=20$ + zero LoRA | {{D_SVD20}} | {{R_SVD20}} |
| **Res-SVD $(16, 4)$** | **{{D_ACC}}** | **{{R_ACC}}** |
| Difference | {{D_DIFF}} | {{R_DIFF}} |

## 8.4 Comparison with PiSSA

PiSSA (Meng et al., 2024) uses the **complementary** approach: base takes 
the residual, LoRA takes the principal directions. 

**Table 8.3: Res-SVD vs PiSSA (math comparison).**

| Aspect | PiSSA | Res-SVD |
|---|---|---|
| Base | $W - BA$ (frozen) | $U_b V_b^T$ (frozen) |
| LoRA | $U_b \Sigma_b^{1/2}$ (principal) | $U_r \Sigma_r^{1/2}$ (residual) |
| Base rank | Full $-$ $r$ | $r_b$ |
| Total rank | $r$ (LoRA) + dense base | $r_b + r_r$ |

## 8.5 Inference Cost

**Table 8.4: Inference time (DistilBERT, N=128).**

| Component | Baseline | Res-SVD |
|---|---|---|
| Q/K/V/O projections | 0.5 ms | {{D_PROJ_TIME}} ms |
| Attention | 0.4 ms | 0.4 ms |
| Output proj | 0.2 ms | {{D_OUT_TIME}} ms |
| LoRA | 0 | {{D_LORA_TIME}} ms |
| **Total** | **1.4 ms** | **{{D_TOTAL_TIME}} ms** |

## 8.6 The Silent Injection Fallback

`inject_res_svd` catches exceptions and returns `None`:

```python
except Exception as e:
    logger.error("inject_res_svd failed at layer %d: %s", layer_idx, e)
    return None
```

This is a **silent failure** — the model is left unchanged but no error 
is raised. If the caller ignores the return value, the model runs as 
baseline.

## 8.7 Summary of Insights

1. **Res-SVD is a real compressor.** Parameters reduce by {{D_RATIO}}×.
2. **Spectral initialization is equivalent to rank-$(r_b+r_r)$ SVD** at start.
3. **Only the residual is trained**, making the base frozen.
4. **Silent injection failure** is a risk.
5. **Smaller total rank = more compression but more accuracy drop.**

---