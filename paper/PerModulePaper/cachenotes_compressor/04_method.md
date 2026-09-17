# 4. Methodology

## 4.1 Algorithmic Formulation
The `cachenotes_compressor` engine optimizes the layer parameterization via calibrated transformation routines.

```
CacheNotes: Task-Aware Key-Value Cache Compression and Static Weight Compression
with Learnable Continuous CPT (Compression-Planning-Token) Generator.

Given a task description, CacheNotes generates Compression-Planning-Tokens (CPTs)
that guide the compression of the corpus and KV cache into compact, reusable states.

Scientific Enhancement:
- Replaces static keyword heuristics with a continuous, differentiable CPTGenerator network.
- Learnable task representation using tokenized embedding bag and multi-frequency coordinate positional projection.
- Capable of evaluating task importance at ANY continuous output feature dimension d (64, 128, 768, 1024, 2048, 4096, etc.).
- Backprop-trainable end-to-end minimizing Frobenius reconstruction loss under CPT-guided low-rank SVD.
- Full fallback co
```

## 4.2 Guardrail Checks and Numerical Stability
Prior to layer substitution, the engine evaluates Frobenius reconstruction norms and parameter delta metrics. If numerical instability (NaN/Inf) or excessive reconstruction error is encountered, safety guardrails revert to uncompressed execution or initiate adaptive LoRA residual recovery.
