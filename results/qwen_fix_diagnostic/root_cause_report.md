# Qwen SST-2 compatibility forensic report

## Scope and controls

The investigation used only the cached Qwen checkpoint and cached SST-2. The
existing fine-tuned state was loaded from `qwen_sst2_compat/finetuned_model.pt`;
no fine-tuning or model download was performed during instrumentation. Qwen
configuration was verified as `hidden_size=896`, `num_attention_heads=14`,
`num_key_value_heads=2`, `head_dim=64`, `num_hidden_layers=24`, causal with
rotary position embeddings.

The original Qwen sweep had a 75.46% `full` baseline. The prior DistilBERT
SST-2 run had a 90.25% `full` baseline. The DistilBERT CSV contains multiple
runs for some module names; comparisons below use the matching rows and are
therefore directional, not a claim of a single deterministic DistilBERT run.

## Data comparison

| Module | Old Qwen accuracy | DistilBERT accuracy (matching result) | New diagnostic accuracy | New status | Path |
|---|---:|---:|---:|---|---|
| full | 75.46% | 90.25% | 75.46% | PASS | native_qwen |
| a3 | 50.11% | 83.14–83.72% | 49.54% | PASS | adapted_qwen |
| aasvd | 49.08% | 82.34–82.91% | 50.80% | PASS | adapted_qwen |
| adaptive_svd | 50.92% | 88.30–88.88% | 50.92% | PASS | adapted_qwen |
| astro | 68.23% | — | 68.23% | PASS | adapted_qwen |
| admm | 49.20% | 84.98% | 51.61% | PASS | adapted_qwen |
| faq | 75.23% | 89.79% | 75.23% | PASS | adapted_qwen |
| aegis_v1 | SKIP | 49.08% | SKIP | SKIP_NO_SUPPORT | strict_skip |

The new accuracy column is the full cached SST-2 validation evaluation with
`batch_size=16`, `seq_len=64`, and the existing seed-42 checkpoint. It is not
the one-batch probe accuracy.

## Instrumented forward evidence

The first transformer attention output was compared on one cached validation
batch before and after injection. Cosine similarity is against the native
Qwen output; values are measured after the attention block hook.

| Module | Cosine | Mean abs diff | Max abs diff | Finite | Collapsed | RoPE flag | Observation |
|---|---:|---:|---:|---|---|---|---|
| a3 | — | — | — | — | — | — | fp16 `geqrf_cuda` during orthogonal initialization |
| aasvd | 0.7183 | 0.00308 | 0.2639 | yes | no | false | attention projections factorized; native Qwen attention remains |
| adaptive_svd | 0.0000 | 0.00318 | 0.4778 | yes | yes | false | replacement output is collapsed; no RoPE and no Qwen GQA geometry |
| astro | 0.9929 | 0.00060 | 0.0154 | yes | no | false | near-native weight quantization path |
| admm | 0.6939 | 0.00438 | 0.2014 | yes | no | false | 2:4 pruning/4-bit projection perturbation |
| faq | 0.9944 | 0.00051 | 0.0166 | yes | no | false | preserves pretrained attention behavior closely |
| aegis_v1 | -0.0109 | 0.00767 | 0.4830 | yes | no | false | semantic replacement; no RoPE and bidirectional local/global attention |
| cosformer | — | — | — | — | — | — | skipped as unsupported Qwen semantic replacement |
| mamba | — | — | — | — | — | — | skipped as unsupported family/module |

All measured modified outputs were finite. Thus the near-chance results are
not NaN or zero-tensor failures in most modules; they are representation drift
large enough to break the randomly added/shortly trained classifier head.

## Exact root causes

1. **Qwen GQA is not ordinary MHA.** Qwen has 14 query heads but only 2 KV
   heads. Several replacement attentions construct K and V as 14×64 and then
   reshape them as if K/V had one projection per query head. This is a shape
   contract mismatch even when a wrapper prevents an immediate exception.

2. **Replacement attentions omit RoPE.** The native Qwen block receives
   `position_embeddings=(cos, sin)` and rotates Q/K before causal attention.
   `AdaptiveSVDAttention` and `AegisAttentionV1` do not consume those values;
   the compatibility wrapper therefore removes them. Their outputs are not
   in the native Qwen positional coordinate system.

3. **Causal-mask semantics differ.** Aegis V1's short-sequence local branch
   uses a bidirectional window mask. At `seq_len=64` and its default window,
   this branch is active and can attend to future tokens. Linear/random-feature
   replacements likewise implement their own attention semantics rather than
   Qwen's causal softmax/GQA path.

4. **Rank 16 is too destructive for Qwen projections.** A3/AASVD initially
   compressed every linear layer, including MLPs. The Qwen-only fix now limits
   these compressors to `q_proj/k_proj/v_proj/o_proj`, preserving MLP and head
   weights. A second run with a rank floor of 64 still remained near chance,
   proving that preserving MLPs alone is insufficient: the remaining attention
   projection approximation and missing RoPE/GQA path dominate.

5. **fp16 orthogonal initialization was unsafe.** A3's `nn.init.orthogonal_`
   called CUDA QR (`geqrf`) on half tensors. Its Qwen path now initializes a
   CPU fp32 temporary and casts the factors back. This converts a hard failure
   into a valid run, but does not by itself restore accuracy.

6. **The classifier head is fragile to attention drift.** The base head was
   trained for only 200 head-only steps. FAQ and ASTRO stay near baseline
   because their first-block perturbations are tiny; low-rank and semantic
   replacements move the representation outside the head's learned geometry.

## Changes applied and remaining status

- A3 fp16-safe orthogonal initialization: **applied**; no injection crash.
- A3/AASVD/ADMM Qwen-only attention projection targeting: **applied**; Qwen
  MLPs are no longer compressed. DistilBERT path is unchanged.
- Qwen rank floor set from the adapter head dimension (64): **applied and
  measured**; it did not recover A3/AASVD accuracy, so it is not sufficient.
- Adaptive-SVD: **not yet safe to mark compatible**. It needs a dedicated
  Qwen implementation with 2 KV heads, `repeat_kv`, native `position_embeddings`
  RoPE application, and causal additive-mask handling. Until then it should be
  classified `SKIP_QWEN_INCOMPATIBLE`, rather than reported as a successful
  semantic replacement at chance accuracy.
- Aegis V1, Cosformer, and Mamba: **incompatible in current scope** and should
  remain explicit Qwen skips. They require architectural semantic rewrites,
  not projection aliases.

## Diagnostic conclusion

For the seven-module rerun: 6 modules completed with PASS status and 1 was an
intentional `SKIP_NO_SUPPORT` (`aegis_v1`). Of the completed modules, FAQ is
effectively Qwen-compatible for this evaluation, ASTRO is partially compatible,
and A3/AASVD/ADMM are only projection-level adaptations whose attention drift
still breaks the head. Adaptive-SVD is functionally incompatible despite
completing without an exception. The evidence does not support labeling all
non-crashing modules as Qwen-compatible.

