# Benchmark Execution Timeline

| Timestamp | Module | Model | Dataset | Status | Accuracy |
|---|---|---|---|---|---|
| 2026-10-14T16:31:18Z | `ttt_linear_attention` | `roberta-base` | `sst2` | success | 0.5200 |
| 2026-10-14T16:29:45Z | `svd_mamba` | `roberta-base` | `sst2` | success | 0.4800 |
| 2026-10-14T16:28:19Z | `simple_ssm` | `roberta-base` | `sst2` | success | 0.5200 |
| 2026-10-14T16:23:54Z | `selective_ssm` | `roberta-base` | `sst2` | success | 0.5200 |
| 2026-10-14T16:19:44Z | `mamba_svd_flash` | `roberta-base` | `sst2` | failed | - |
| 2026-10-14T16:19:29Z | `mamba_block` | `roberta-base` | `sst2` | success | 0.5200 |
| 2026-10-14T16:15:09Z | `hgrn2_outer` | `roberta-base` | `sst2` | success | 0.8100 |
| 2026-10-14T16:13:42Z | `hgrn2_gated_outer` | `roberta-base` | `sst2` | success | 0.8100 |
| 2026-10-14T16:12:17Z | `hgrn2_gated_deltanet` | `roberta-base` | `sst2` | failed | - |
| 2026-10-14T16:12:00Z | `granite_hybrid` | `roberta-base` | `sst2` | failed | - |
| 2026-10-14T16:11:40Z | `gated_deltanet_block` | `roberta-base` | `sst2` | failed | - |
| 2026-10-14T16:11:20Z | `gated_delta_net` | `roberta-base` | `sst2` | failed | - |
| 2026-10-14T16:11:04Z | `sink_attention` | `roberta-base` | `sst2` | success | 0.8300 |
| 2026-10-14T16:10:31Z | `hamming_attention` | `roberta-base` | `sst2` | success | 0.4800 |
| 2026-10-14T16:10:00Z | `dynamic_conv_attention` | `roberta-base` | `sst2` | success | 0.5200 |