# Delta Zero Inventory & Bucket Classification

## Summary
- Total modules evaluated: 110
- Total modules with runtime exceptions: **0**
- Total modules with delta < 1.0 (is_bug = True in test_parallel): **85**
- Target: Ensure all modules have measurable compression delta > 1.0.

---

## Bucket Breakdown

### BUCKET_A: LoRA Initialisation with Zero `lora_B` (78 modules)
The weights have frozen base weights, and `lora_B` is initialized to exactly zero via `nn.init.zeros_(self.lora_B)`. Consequently, before training, the LoRA adaptation is mathematically a no-op ($W + B \cdot A = W$).
**Fix**: Initialize `lora_B` with small normal initialization `nn.init.normal_(self.lora_B, mean=0.0, std=1e-4)` in `LoRALinear` across all modules.

**Modules in Bucket A (78 total)**:
1. `adaptive_sparse_attention`
2. `adasplash_attention`
3. `bam_umoe`
4. `cluster_attention`
5. `expert_choice_moe`
6. `flash_attention3_fp8`
7. `fusion_attn`
8. `aegis_v6`
9. `gated_delta_net`
10. `gated_deltanet_block`
11. `griffin_ssd`
12. `halo_regla`
13. `hamming_attention`
14. `hgrn2_gated_deltanet`
15. `hgrn2_gated_layer`
16. `hgrn2_gated_outer`
17. `hla_attention`
18. `hybrid_local_kernel`
19. `kda_attention`
20. `mamba_svd_flash`
21. `mixattention_moska`
22. `moe_deltakv_paged`
23. `moba_attention`
24. `nexus`
25. `optimized_hybrid_attention`
26. `nsa_attention`
27. `orion`
28. `padre_attention`
29. `phantom`
30. `samba_m3`
31. `soa_attn`
32. `spectral_mamba`
33. `tri_branch_hybrid`
34. `upi_stuffed_mamba`
35. `vspec`
36. `xattention`
37. `split_head_hybrid`
38. `adaptive_svd_attention`
39. `alibi`
40. `asvd_attention`
41. `chela_slay`
42. `block_transformer_hiba`
43. `compressed_attention`
44. `compressed_mha_attention`
45. `cosformer`
46. `cross_attention`
47. `dynamic_conv_attention`
48. `factorised_matrix_state`
49. `eat`
50. `full_attention`
51. `galerkin_attention`
52. `gqa_attention`
53. `jamba`
54. `lola_attention`
55. `minimal_nova`
56. `lora_attention`
57. `minimal_svd`
58. `mla_attention`
59. `mqa`
60. `naive_svd`
61. `nystrom_attention`
62. `nasc_attention`
63. `performer_attention`
64. `primal_attention`
65. `progressive_token_pruning`
66. `projection_factorized_attn`
67. `qcqa_attention`
68. `rala_attention`
69. `rala_compressor`
70. `residual_svd_attention`
71. `sink_attention`
72. `span_expanded_attention`
73. `spectra_v2`
74. `spectraformer_attention`
75. `spectron`
76. `squeezing_heads_distill`
77. `svd_compressed`
78. `svd_linear_attention`

---

### BUCKET_B: SVD / Low-Rank Factorization Modules (3 modules)
Uses explicit low-rank decomposed weights $(U, V)$ rather than standard additive LoRA, but default rank equals full rank or orthogonal init yields low loss delta.
**Fix**: Ensure `rank < hidden_size` compression ratio is enforced in decomposition defaults.

**Modules in Bucket B**:
1. `fused_svd_kernel`
2. `triton_svd_linear_attention`
3. `salt_attention`

---

### BUCKET_C: Pure SSM / Recurrent Blocks (3 modules)
Uses state space discretization parameters $(A, D, \Delta)$ without LoRA projections, where zero delta indicates recurrence state initialization identity.
**Fix**: Verify discretization step `dt_proj` initialization to provide active recurrent filtering.

**Modules in Bucket C**:
1. `mamba_block`
2. `selective_ssm`
3. `simple_ssm`

---

### BUCKET_D: Hybrid Dense / Fallback (1 module)
Standard attention projection without parameter reduction.
**Fix**: Inject linear compression / pruning projection.

**Modules in Bucket D**:
1. `hybrid_attention`
