# NeuralOps Empirical Benchmark Executive Summary

- **Run ID**: `run_20261102_112616`
- **Timestamp**: 2026-11-02 11:26:16
- **Target Accelerator**: `cpu`
- **Modules Verified**: 10/10 (100.0%)

## Verified Module Telemetry

| Module | Category | Latency (ms) | p95 (ms) | VRAM (MB) | Throughput (tok/s) | SQNR (dB) | Status |
|---|---|---|---|---|---|---|---|
| `eat` | attention | 2.114 | 5.11 | 0.0 | 15138.9 | N/A | **PASS** |
| `block_transformer_hiba` | attention | 3.58 | 11.025 | 0.0 | 8937.5 | N/A | **PASS** |
| `adaptive_span_attention` | attention | 4.131 | 9.836 | 0.0 | 7745.7 | N/A | **PASS** |
| `progressive_token_pruning` | attention | 2.242 | 4.83 | 0.0 | 14273.6 | N/A | **PASS** |
| `expert_choice_moe` | attention | 1.732 | 2.047 | 0.0 | 18477.5 | N/A | **PASS** |
| `direct_svd` | compressor | N/A | N/A | 0.0 | N/A | 2.03 | **PASS** |
| `flat_llm` | compressor | N/A | N/A | 0.0 | N/A | 2.02 | **PASS** |
| `hashed_net_compressor` | compressor | N/A | N/A | 0.0 | N/A | 2.08 | **PASS** |
| `liftr` | compressor | N/A | N/A | 0.0 | N/A | 2.05 | **PASS** |
| `roast_hashed_linear` | compressor | N/A | N/A | 0.0 | N/A | 2.04 | **PASS** |
