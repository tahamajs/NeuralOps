# Benchmark Execution Timeline

| Timestamp | Module | Model | Dataset | Status | Accuracy |
|---|---|---|---|---|---|
| 2026-10-14T12:41:34Z | `gated_delta_net` | `roberta-base` | `sst2` | failed | - |
| 2026-10-14T07:31:05Z | `upi_stuffed_mamba` | `roberta-base` | `sst2` | failed | - |
| 2026-10-14T07:30:51Z | `ttt_linear_attention` | `roberta-base` | `sst2` | failed | - |
| 2026-10-14T07:29:56Z | `svd_mamba` | `roberta-base` | `sst2` | failed | - |
| 2026-10-14T07:29:34Z | `simple_ssm` | `roberta-base` | `sst2` | failed | - |
| 2026-10-14T07:29:19Z | `selective_ssm` | `roberta-base` | `sst2` | failed | - |
| 2026-10-14T07:29:02Z | `native_hybrid_attention` | `roberta-base` | `sst2` | success | 0.8067 |
| 2026-10-14T07:28:12Z | `moe_deltakv_paged` | `roberta-base` | `sst2` | success | 0.8233 |
| 2026-10-14T07:27:01Z | `mamba_svd_flash` | `roberta-base` | `sst2` | failed | - |
| 2026-10-14T07:26:47Z | `mamba_block` | `roberta-base` | `sst2` | failed | - |
| 2026-10-14T07:26:31Z | `lola_attention` | `roberta-base` | `sst2` | success | 0.6500 |
| 2026-10-14T07:25:52Z | `hgrn2_outer` | `roberta-base` | `sst2` | failed | - |
| 2026-10-14T07:25:36Z | `hgrn2_gated_outer` | `roberta-base` | `sst2` | failed | - |
| 2026-10-14T07:25:19Z | `hgrn2_gated_deltanet` | `roberta-base` | `sst2` | failed | - |
| 2026-10-14T07:25:01Z | `halo_regla` | `roberta-base` | `sst2` | success | 0.5167 |
| 2026-10-14T07:24:10Z | `granite_hybrid` | `roberta-base` | `sst2` | failed | - |
| 2026-10-14T07:23:56Z | `gated_deltanet_block` | `roberta-base` | `sst2` | failed | - |
| 2026-10-14T07:23:40Z | `gated_delta_net` | `roberta-base` | `sst2` | failed | - |
| 2026-10-14T07:23:23Z | `dynamic_conv_attention` | `roberta-base` | `sst2` | success | 0.4833 |