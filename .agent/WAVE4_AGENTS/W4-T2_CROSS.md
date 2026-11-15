# Wave 4 Cross-Domain Findings: W4-T2

## 2026-11-15T03:32:00Z
File: neuralops/core/peft/lora.py
Line: 146
Symptom: In LoRALinear.__init__, lora_B is initialized with `normal_(mean=0.0, std=1e-4)` instead of `zeros_`, causing initial forward pass to deviate from base linear output. In wrapper mode, passing positional arguments `(linear, r, alpha, dropout)` causes `r` to bind to `out_features`, `alpha` to `bias`, and `dropout` to `lora_rank` (resulting in rank=0). Furthermore, `unmerge()` subtracts delta in float arithmetic, resulting in loss of bitwise equality with original weights.
Suspected bug ID: P1-CORE-LORA-INIT
Suggested fix: In neuralops/core/peft/lora.py, detect if `base_or_in_features` is nn.Module and map positional args (r, alpha, dropout); initialize lora_B with `zeros_`; and cache original weights on `merge()` to restore bitwise on `unmerge()`.

## 2026-11-15T03:32:30Z
File: neuralops/evaluation/metrics.py
Line: 25
Symptom: `compute_perplexity_from_loss` only accepts float type; passing PyTorch scalar tensors or NumPy float scalars raises TypeError.
Suspected bug ID: P2-EVAL-PERPLEXITY-TYPE
Suggested fix: Extract scalar value using `hasattr(loss, "item")` before computing `math.exp(val)`.
