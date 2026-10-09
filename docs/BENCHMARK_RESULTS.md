# NeuralOps Benchmark Results (CPU Execution)

## Registry Lookup Performance
- **Lookups**: 10,000 queries
- **Total Latency**: 2.78 ms
- **Per-lookup Latency**: **0.28 μs** ($O(1)$ constant time dictionary access)

## Attention Forward Execution Benchmarks (Sample)
Environment: macOS ARM64 CPU, PyTorch 2.11.0, Batch=2, SeqLen=128, HiddenDim=256

| Module Name | Forward Latency | Status |
| :--- | :---: | :---: |
| `alibi_attention` | 0.07 ms | `OK` |
| `asvd_attention` | 1.33 ms | `OK` |
| `bam_umoe` | 1.58 ms | `OK` |
| `block_transformer_hiba` | 1.76 ms | `OK` |
| `chela_slay` | 2.79 ms | `OK` |
| `cluster_attention` | 3.34 ms | `OK` |
| `compressed_attention` | 1.47 ms | `OK` |
| `cosformer` | 2.00 ms | `OK` |
| `decay_linear_attention` | 2.60 ms | `OK` |
| `eat_attention` | 1.68 ms | `OK` |
| `expert_choice_moe` | 3.86 ms | `OK` |
| `factorised_matrix_state`| 2.78 ms | `OK` |

Data saved to `benchmarks/attention_benchmark.json`.
