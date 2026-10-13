# Framework Hardening Progress

| Batch | Status | Files | Commit | Time | Notes |
|---|---|---|---|---|---|
| 1. family_aegis_spectron | ✅ | 11 | `b5a643c6b` | 15min | All 11 pass AST; output_scale eliminated; prefix is_decoder standardized |
| 2. family_moe | ✅ | 9 | `de3e7580f` | 5min | All 9 pass AST; output_scale eliminated; prefix is_decoder & RuntimeError fixed |
| 3. family_mamba | ✅ | 7 | `45de3d5b4` | 5min | All 7 pass AST; output_scale eliminated; prefix is_decoder & RuntimeError fixed |
| 4. family_hgrn | ✅ | 3 | `a6dbf5b18` | 3min | All 3 pass AST; output_scale eliminated; prefix is_decoder & RuntimeError fixed |
| 5. family_deltanet | ✅ | 2 | `bcf36210f` | 2min | All 2 pass AST; prefix is_decoder standardized |
| 6. family_compressed | ✅ | 2 | `98f3bc83f` | 2min | All 2 pass AST; output_scale eliminated; prefix is_decoder standardized |
| 7. family_svd_split_A | ✅ | 16 | `88f4129fe` | 6min | All 16 pass AST; output_scale eliminated; prefix is_decoder & RuntimeError fixed |
| 8. family_svd_split_B | ✅ | 17 | `83cdf4986` | 4min | All 17 pass AST; output_scale eliminated |
| 9. family_attn_general_1 | ✅ | 20 | `7d102240a` | 4min | All 20 pass AST; output_scale & RuntimeError & prefix is_decoder fixed |
| 10. family_attn_general_2 | ✅ | 20 | `bc3f34f7f` | 4min | All 20 pass AST; output_scale & RuntimeError & prefix is_decoder fixed |
| 11. family_attn_general_3 | ✅ | 20 | `560402ec8` | 4min | All 20 pass AST; output_scale & RuntimeError & prefix is_decoder fixed |
| 12. family_attn_general_4 | ✅ | 20 | `b6ce35e71` | 4min | All 20 pass AST; output_scale & RuntimeError & prefix is_decoder fixed |
| 13. family_attn_general_5 | ✅ | 23 | `28162beb1` | 4min | All 23 pass AST; output_scale & RuntimeError & prefix is_decoder fixed |
| 14. compression_engine | ✅ | 80 | `53e65cde8` | 5min | All 80 pass AST; output_scale eliminated across entire suite |
