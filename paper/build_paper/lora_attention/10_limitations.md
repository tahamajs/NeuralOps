# 10. Limitations

## 10.1 No Parameter Reduction

LoRA Attention **adds** parameters (adapters). This is not a compression 
method. Post-merge, the model is baseline-sized.

## 10.2 Task Coverage

Only SST-2 evaluated. Behavior on generation tasks unknown.

## 10.3 Model Coverage

| Size | Tested |
|---|---|
| ≤124M | Yes |
| 1B–10B | No |
| >10B | No |

## 10.4 Single-Seed

SE ~1.7%. Differences below 3.4% may be noise.

## 10.5 Hard Failure on Missing Weights

Unlike some compression modules, LoRA Attention **raises exceptions** on 
missing weights. This is intentional (silent failure would corrupt base 
weights) but reduces robustness.

## 10.6 No Comparison with Full FT

We do not directly compare with full fine-tuning (compute budget). 

## 10.7 Rank Selection

Rank 16 is a default; task-specific tuning may help.

## 10.8 The `output_scale` Parameter

We add an `output_scale` scalar. This is not standard LoRA. Its effect is 
small but non-zero.

## 10.9 Cross-Attention

The implementation supports cross-attention via `key_value_states`, but 
we do not systematically evaluate this.

## 10.10 Merge Precision

Merging introduces small numerical error (FP16 rounding). We do not 
characterize this for very large models.

## 10.11 Bias Handling

Bias parameters are frozen but copied. LoRA does not adapt bias.

## 10.12 Initialization

$B = 0$ ensures exact recovery at start, but may slow initial learning 
compared to nonzero init.

## 10.13 Code Quality

Research-grade:
- Broad exception handling in `copy_weights`
- Duplicated code in `inject_lora_attention`
- Some numerical concerns at FP16

## 10.14 Ethical Considerations

Dual-use. See Appendix E.

---