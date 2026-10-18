# torch.compile Known Limitations (v1.4.0)

This document tracks known limitations and behavioral quirks when compiling NeuralOps attention modules with `torch.compile` (`mode="reduce-overhead"`, Inductor backend).

## Modules with Resolved Compiler Quirks

| Module | Issue | Workaround / Fix | Status |
|---|---|---|---|
| `gated_hybrid_kernel_attention` | `.item()` in conditional caused CPU-GPU sync and Dynamo graph break | Replaced `.item()` with `bool(tensor.all())` / scalar capture | **FULLY_COMPILED** |
| `infini_attention` | Integer counter mutation (`_aux_loss_count += 1`) triggered recompile limit | Guarded with `if torch.compiler.is_compiling(): return` & `allow_unspec_int_on_nn_module=True` | **FULLY_COMPILED** |
| `hgrn2_gated_layer` | Bidirectional mode (`is_decoder=False`) leaked future tokens on causal models | Added runtime safety check: raises `RuntimeError` if causal decoder uses `is_decoder=False` | **SAFE & VERIFIED** |

## HGRN2 Bidirectional Mode Policy (`is_decoder=False`)

HGRN-2 supports two execution paradigms:
- `is_decoder=False`: Bidirectional recurrent scan (forward + backward) — designed for encoder models (BERT, RoBERTa, DistilBERT).
- `is_decoder=True`: Causal recurrent scan (forward only) — strictly required for autoregressive causal decoders (GPT-2, Qwen, LLaMA).

**Critical Policy:**
Using `is_decoder=False` on causal decoders leaks future tokens into early hidden states (data leakage). The benchmark harness automatically infers `is_decoder=True` for all causal decoder architectures.

## Benchmark & Registry Canonical Policy

The registry maintains **125 canonical attention mechanisms** plus **6 legacy aliases** (`chela_slay_attention`, `cosformer_attention`, `decay_linear_attn`, `eat_attention`, `gated_linear_attn`, `hgrn2_outer`) for backwards compatibility.

1. The benchmark harness strictly iterates over the **125 canonical names**.
2. Any module exhibiting `RECOMPILE_LIMIT` or `PARTIAL_COMPILE` must be flagged as `UNRELIABLE` and excluded from speedup claims in camera-ready paper tables.
