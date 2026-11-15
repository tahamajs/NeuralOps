# Cross-Domain Bug Log: W4-L1

## 2026-11-15T02:30:00Z
File: neuralops/ops/triton/retnet_retention.py
Line: 164
Symptom: In `triton_multi_scale_retention_pytorch_ref`, `decay = torch.where(causal, g ** diff.float(), torch.zeros_like(g))` evaluates `g ** diff.float()` for all matrix entries including non-causal entries where `diff < 0`. Because PyTorch's `torch.where` eagerly evaluates both branch arguments, negative powers of `g` can produce division by zero or large magnitudes that can overflow in half-precision representations before being masked out.
Suspected bug ID: P1
Suggested fix: Clamp `diff` with `torch.clamp(diff, min=0)` before computing `g ** diff_clamped.float()`, ensuring all exponents remain non-negative during branch evaluation.

## 2026-11-15T02:30:30Z
File: neuralops/layers/ssm/rwkv_attention.py
Line: 157
Symptom: In `RWKVAttention.forward`, `w` is computed as `w = -torch.exp(torch.clamp(w_raw, max=5.0))`. It is then passed to `rwkv6_scan_pytorch(r, k, v, w, u)` which evaluates `decay = torch.exp(-torch.exp(wt))`. Since `w` is already strictly negative due to `-torch.exp()`, applying `-torch.exp(wt)` double-exponentiates the decay factor and restricts the decay to a narrow range near 1.0.
Suspected bug ID: P1
Suggested fix: Pass raw unconstrained decay logits `w_raw` to `rwkv6_scan_pytorch` so that `torch.exp(-torch.exp(wt))` operates as intended by the RWKV-6 parameterization.

## 2026-11-15T02:30:50Z
File: neuralops/layers/linear/gla_attention.py
Line: 133
Symptom: In `GLAttention.__init__`, `self.gate_W2.bias.fill_(16.0)` with `self.gate_tau = 16.0` sets initial gate logits `gate_logits / tau = 1.0`, yielding initial retention decay factor `sigmoid(1.0) ≈ 0.73` per token. For sequence lengths > 20 tokens, `0.73^20 ≈ 0.0018`, causing excessively aggressive memory forgetting upon initialization rather than preserving long-range context.
Suspected bug ID: P2
Suggested fix: Initialize `gate_W2.bias` to a higher positive value (e.g. 48.0 or 64.0) so initial retention decay is closer to 0.95-0.98.
