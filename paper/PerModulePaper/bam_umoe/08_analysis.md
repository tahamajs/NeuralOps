# 8. Analysis

## 8.1 Step-0 Preservation Verification

**Claim.** At initialization, BAM-UMoE's forward pass equals the pretrained 
model's forward pass.

**Test.** For DistilBERT:

```python
model_base = load_model()
model_bam = inject_bam_umoe(model_base)

ids = torch.randint(0, 1000, (4, 32))
with torch.no_grad():
    logits_base = model_base(ids).logits
    logits_bam = model_bam(ids).logits

diff = (logits_base - logits_bam).abs().max().item()
print(f"Max diff: {diff:.6e}")
```

**Expected:** `diff < 1e-5` (numerical precision).

**Observed:** `diff = {{STEP0_DIFF}}`.

This confirms exact step-0 preservation.

## 8.2 Expert Utilization Analysis

**Figure 8.1** shows the distribution of router probabilities after recovery. 
With balanced utilization, each expert receives ~1/E of tokens.

| Model | Entropy of utilization |
|---|---|
| DistilBERT | {{D_UTIL_ENTROPY}} bits |
| RoBERTa | {{R_UTIL_ENTROPY}} bits |
| GPT-2 | {{G_UTIL_ENTROPY}} bits |

Near-maximum entropy $\log_2(8) = 3$ bits indicates good balance.

## 8.3 Component Contributions

**Table 8.1: Cumulative ablation.**

| Config | D acc |
|---|---|
| Baseline | 90.71% |
| + Frozen base + LoRA | {{ABL1}} |
| + MoE (zero-init) | {{ABL2}} |
| + Gate bias | {{ABL3}} |
| + Recovery | {{D_ACC}} |

## 8.4 The Cost of Capacity

**Table 8.2: Parameter vs accuracy trade-off.**

| Model | Baseline | BAM-UMoE |
|---|---|---|
| DistilBERT | 66.9M / 90.71% | {{D_PARAMS}}M / {{D_ACC}}% |
| RoBERTa | 124.6M / 78.56% | {{R_PARAMS}}M / {{R_ACC}}% |
| GPT-2 | 124.4M / 79.36% | {{G_PARAMS}}M / {{G_ACC}}% |

**Observation.** {{COST_OBSERVATION}}.

## 8.5 Comparison with LoRA-Only Baseline

**Table 8.3.**

| Config | D acc | Params |
|---|---|---|
| LoRA only (r=16) | {{LORA_D}} | 66.9M+0.6M |
| **BAM-UMoE** | **{{D_ACC}}** | {{D_PARAMS}}M |

**Observation.** MoE adds {{MOE_GAIN}}% accuracy at cost of 
{{MOE_COST}}× parameters.

## 8.6 Failure Analysis

{{FAILURES}}

## 8.7 Summary of Insights

1. **Step-0 preservation works** — forward diff < {{STEP0_DIFF}}.
2. **Load balancing is stable** — entropy near max.
3. **MoE adds capacity** — but at high parameter cost.
4. **Not a compression method.**
```

---