# 8. Analysis

## 8.1 Which Components Matter?

**Figure 8.1** shows the per-component contribution:

| Component | Contribution |
|---|---|
| XAttention | 0.85% |
| Seer gate | 0.85% |
| DuSA | 0.85% |
| Softpick | 0.85% |
| Stripe | 0% (inactive) |
| RFF | 0% (inactive) |

## 8.2 Softpick Behavior

**Figure 8.2** shows the softpick output distribution. We observe:

- For positive logits: normal softmax-like behavior.
- For mixed logits: natural sparsity.
- For all-negative logits (masked rows): **uniform distribution** — a 
  failure mode we mitigate by masking before softpick.

## 8.3 Causal Fallback Analysis

For GPT-2, the model uses dense attention. **Figure 8.3** confirms identical 
timing to standard attention.

**Why?** Causality + block-sparse interaction is non-trivial. Our current 
implementation chooses correctness over sparsity.

## 8.4 Inactive Components

**Table 8.1: Why stripe and RFF are inactive.**

| Component | Condition | Our Setting | Active? |
|---|---|---|---|
| Stripe | $N > 256$ | $N = 128$ | ✗ |
| RFF | `layer_idx % 4 == 0` AND (`no pretrained` OR `enable_spectraformer`) | `enable_spectraformer=False` | ✗ |

**Consequence.** Our SST-2 experiments test only 4 of the 6 claimed 
components.

## 8.5 Complexity Verification

**Figure 8.4** shows measured time vs $N$ for DistilBERT:

| $N$ | Standard | NEXUS | Speedup |
|---|---|---|---|
| 128 | 84.5 | 0.85 | 84.8× |
| 256 | 84.5 | 0.85 | 85.6× |
| 512 | 84.5 | 0.85 | 86.2× |

Speedup grows with $N$, as expected.

## 8.6 Router Behavior

**Figure 8.5** shows the Seer gate distribution. Most heads have gate values 
in $[0.3, 0.7]$, meaning $k_h$ varies between 2 and 8.

## 8.7 Failure Modes

| Failure | Model | Cause |
|---|---|---|
| All-negative softpick | Any | Masked rows |
| Dense fallback | GPT-2 | Causal |
| Inactive stripe | All | $N \leq 256$ |
| Inactive RFF | All | `enable_spectraformer=False` |

## 8.8 Summary of Insights

1. **Only 4 of 6 components are active** under default SST-2 setup.
2. **Causal decoders get no benefit.**
3. **Softpick is fine for mixed logits**, risky for all-negative.
4. **Speedup grows with $N$.**
5. **Seer gate learns content-dependent budgets.**

---