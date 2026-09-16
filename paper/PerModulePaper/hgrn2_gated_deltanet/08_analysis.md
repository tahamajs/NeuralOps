# 8. Analysis

## 8.1 Why Linear Recurrence?

Standard attention is $\mathcal{O}(L^2)$ because every query attends to every 
key. Linear recurrence is $\mathcal{O}(L)$ because the state is fixed-size.

**Figure 8.1** plots inference time vs L for baseline and ours. The 
crossover point is $L^* = {{BREAKEVEN}}$. Below $L^*$, standard attention 
wins because of its highly optimized CUDA kernels. Above $L^*$, our 
recurrence wins because the quadratic term dominates baseline.

## 8.2 Why Two Branches?

The two branches are **complementary**:

- **HGRN-2:** element-wise state $s_t \in \mathbb{R}^d$. Fast, but cannot 
  represent cross-feature interactions.
- **DeltaNet:** matrix state $S_t \in \mathbb{R}^{d \times d}$. Rich, but 
  slower.

**Figure 8.2** shows the learned gate distribution. For DistilBERT, the gate 
is roughly uniform ({{D_GATE_MEAN}}); for RoBERTa, it leans toward HGRN 
({{R_GATE_MEAN}}); for GPT-2, it leans toward DeltaNet ({{G_GATE_MEAN}}).

## 8.3 Where Does the Short-Sequence Slowdown Come From?

Two factors:

### 8.3.1 Python Loop

The DeltaNet recurrence is implemented as a Python-level loop over timesteps:

```python
for t in range(L):
    S = alpha * S + beta * (v - S @ k).outer(k)
    y = S @ q
```

For L = 128, this is 128 Python iterations per layer. Across 12 layers and 
109 batches, this is ~170K Python iterations, each with GPU kernel launch 
overhead.

### 8.3.2 Activation Memory

Each timestep materializes intermediate tensors. Without a fused kernel, 
these accumulate in GPU memory.

### 8.3.3 Speedup at Long Sequences

At L = 1024, baseline is 45s. Our method is {{D_T1024}}s. The crossover 
happens because baseline's $\mathcal{O}(L^2)$ term dominates.

## 8.4 Per-Layer Analysis

**Table 8.1: Gate statistics per layer.**

| Layer | Mean γ | Std γ |
|---|---|---|
| 0 | {{D_L0_GATE_MEAN}} | {{D_L0_GATE_STD}} |
| 1 | {{D_L1_GATE_MEAN}} | {{D_L1_GATE_STD}} |
| ... | ... | ... |
| 5 | {{D_L5_GATE_MEAN}} | {{D_L5_GATE_STD}} |

## 8.5 Comparison to Prior Linear Attention

**Table 8.2: Accuracy comparison.**

| Method | DistilBERT | RoBERTa | GPT-2 | Speed @ L=128 |
|---|---|---|---|---|
| full | 90.71% | 78.56% | 79.36% | 1× |
| Linear Transformer | {{LT_D}} | {{LT_R}} | {{LT_G}} | {{LT_SP}}× |
| Performer | {{PERF_D}} | {{PERF_R}} | {{PERF_G}} | {{PERF_SP}}× |
| Mamba (est.) | {{MAMBA_D}} | {{MAMBA_R}} | {{MAMBA_G}} | {{MAMBA_SP}}× |
| **HGRN2-GDN** | **{{D_ACC}}** | **{{R_ACC}}** | **{{G_ACC}}** | **{{SP}}×** |

## 8.6 Training Dynamics

**Figure 8.3** shows training loss curves. The loss decreases steadily on 
DistilBERT (from 0.30 to 0.02), but **oscillates** on RoBERTa 
({{R_LOSS_MIN}} to {{R_LOSS_MAX}}). We hypothesize this is due to 
{{ROBERTA_HYPOTHESIS}}.

## 8.7 Failure Analysis

### 8.7.1 Cases Where It Fails

| Case | Reason | Fix |
|---|---|---|
| Cross-attention | Not implemented | Return identity + warning |
| Qwen hybrid layers | Skipped by design | — |
| Very large `lora_rank` | OOM | Reduce rank |

### 8.7.2 RoBERTa Degradation

RoBERTa shows {{R_DELTA}}% drop, larger than DistilBERT's {{D_DELTA}}%. 
Possible causes:

1. RoBERTa's nested attention requires the outer output projection, which 
   our module bypasses (`is_nested_encoder`).
2. RoBERTa's 12 layers amplify error.
3. The gate may not have converged.

**Test.** {{ROBERTA_TEST}}

## 8.8 Summary of Insights

1. **Linear recurrence works** at long L, but **Python loop hurts** at short L.
2. **Two-branch fusion helps** slightly over single-branch ({{BRANCH_GAIN}}%).
3. **Gate learns** sensible per-token preferences.
4. **RoBERTa degrades more** than DistilBERT, hypothesis: nested structure.
5. **Fused CUDA kernel** would eliminate the short-sequence slowdown.

---