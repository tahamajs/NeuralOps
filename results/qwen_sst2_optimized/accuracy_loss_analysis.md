# Qwen SST-2 Accuracy-Loss Analysis

## Scope

This report compares the live Qwen SST-2 sweep in `qwen_sst2_optimized/summary.csv` with the cached DistilBERT SST-2 results in `final_paper_run/summary.csv`. The Qwen CSV currently contains 48 completed rows; the sweep was not stopped. A result marked PASS means the module executed, not that its accuracy is good.

## Summary

The Qwen `full` baseline is **75.46%**, versus **90.25%** for DistilBERT, a 14.79-point gap before compression. Therefore compressed-module scores must be interpreted relative to the Qwen baseline, not only relative to the DistilBERT numbers. Most Qwen replacements cluster near 49%, which is close to the majority-class/unchanged-head behavior seen in this evaluation and indicates that the injected attention output is not preserving the representation used by the classification head.

The primary causes are architectural mismatch and insufficient task adaptation:

1. Qwen is a causal decoder-only transformer with rotary position embeddings and a prepared causal mask. Most modules were authored for BERT/DistilBERT-style bidirectional attention and different projection/output conventions. The compatibility layer supplies naming/configuration aliases and safely handles the mask, but it cannot make a non-causal or non-RoPE module semantically equivalent to native Qwen attention.
2. Only the classification head received 200 fine-tuning steps. The body was not adapted to SST-2, so even a structurally valid attention replacement can move hidden states away from the narrow representation learned by the head.
3. `max_steps=1` and `seq_len=64` make this a smoke evaluation, not a quality re-training or calibration pass. Modules that alter token mixing cannot recover through training in this run.
4. Several modules require fp32 QR/SVD or have mixed Half/Float paths. The runner’s fp32 retry prevents execution failure, but it does not remove numerical or representation mismatch.

## Categories (current 48 rows)

- **Inherently or semantically mismatched replacements: 15** — `adaptive_span`, `adaptive_sparse`, `aegis_hybrid`, `aegis_v1`, `aegis_v3`, `aegis_v4`, `aegis_v5`, `aegis_v6`, `bam_umoe`, `cosformer`, `cur`, `cur_moe`, `decay_linear_attn`, `dobi`, `dynamic_rank`. These include linear/recurrent/random-feature, routing, or alternate causal/update rules whose assumptions differ from Qwen’s RoPE causal block.
- **Dtype/shape-sensitive or numerically fragile: 6** — `a3`, `aegis_spectron_fp8`, `compot`, `decay_linear_attn`, `direct_svd`, `entropy_budget`. The CSV records fp32 retries for `a3`, `aegis_spectron_fp8`, `compot`, and `decay_linear_attn`; CPU QR/SVD and Half/Float transitions can change or damage the injected weights even when execution completes.
- **Partially compatible / moderate preservation: 10** — `armor`, `ast`, `astro`, `block_transformer_hiba`, `bonsai`, `cluster`, `coopq`, `delta_kv`, `eat`, `enlighten`. Their scores are higher than the cluster around 49%; `astro` is strongest at 68.23%, but remains 7.23 points below Qwen `full`.
- **SVD/projection/compression affected mainly by the head and representation shift: 17** — `adaptive_svd`, `adasvd`, `admm`, `admm_pruner`, `asvd`, `cachenotes`, `chela_slay`, `come`, `compressed_attention`, `compressed_attention_layer_energy`, `compressed_mha`, `cr`, `cur_moe_compressor`, `dynamic_rank`, `entropy_budget`, `entropy_rank_allocator`, `direct_svd`. This category overlaps with the first two where a module has multiple failure modes; compression preserves weights approximately, but approximate preservation is not sufficient when the head was trained on the uncompressed/native Qwen path.

The categories are diagnostic and intentionally overlap; they are not mutually exclusive taxonomic labels. `aasvd` is included because the current CSV records it as PASS, despite the earlier request to skip it.

## Module-by-module comparison

`DistilBERT SST-2` is the matched `mean_accuracy` when a same-named row exists. `prior FAIL` means a same-named DistilBERT row existed but did not produce a usable accuracy. `—` means no same-named SST-2 row was found.

| Module | Qwen % | DistilBERT % | Likely explanation |
|---|---:|---:|---|
| full | 75.46 | 90.25 | Native Qwen baseline; 200-step head-only fine-tune and causal decoder gap. |
| a3 | 50.11 | 83.14 | SVD/QR plus fp32 retry; representation damage after dtype fallback. |
| aasvd | 49.08 | 82.34 | Approximate SVD attention not calibrated to Qwen/RoPE. |
| adaptive_span | 47.36 | 89.22 | Span/mask behavior assumes encoder-style attention. |
| adaptive_sparse | 51.38 | 89.68 | Sparse pattern changes causal/RoPE token mixing. |
| adaptive_svd | 50.92 | 88.30 | SVD approximation plus head sensitivity. |
| adasplash | 51.95 | prior FAIL | Qwen execution succeeds, but prior DistilBERT result was unusable. |
| adasvd | 49.08 | 81.31 | SVD compression not preserving Qwen head features. |
| admm | 49.20 | 84.98 | ADMM projection changes native attention geometry. |
| admm_pruner | 52.18 | — | Pruning/shape behavior not directly comparable. |
| aegis_hybrid | 48.97 | 49.08 | Alternate attention path; near chance on both. |
| aegis_spectron_fp8 | 51.26 | prior FAIL | Mixed Half/Float retry; prior DistilBERT result unusable. |
| aegis_v1 | 48.97 | 49.08 | Alternate attention semantics. |
| aegis_v2 | 50.46 | 82.11 | Qwen projection/mask mismatch despite successful injection. |
| aegis_v3 | 51.26 | 83.94 | Alternate attention semantics and RoPE mismatch. |
| aegis_v4 | 47.71 | 49.08 | Alternate attention path; near chance. |
| aegis_v5 | 51.26 | 82.68 | Alternate attention semantics and head sensitivity. |
| aegis_v6 | 47.71 | 49.08 | Alternate attention path; near chance. |
| armor | 54.70 | 89.79 | Partial preservation; still not native Qwen attention. |
| ast | 56.08 | 89.45 | Partial preservation, but causal/RoPE assumptions differ. |
| astro | 68.23 | 90.14 | Best current partial match; still below native Qwen. |
| asvd | 50.92 | 85.67 | SVD projection shifts features expected by head. |
| bam_umoe | 49.08 | prior FAIL | MoE/routing path is not Qwen-native; prior row unusable. |
| block_transformer_hiba | 48.39 | 91.17 | Block structure incompatible with current Qwen injection. |
| bonsai | 52.98 | 87.84 | Approximation changes token mixing. |
| cachenotes | 49.43 | 89.79 | Cache/attention assumptions differ from Qwen KV/RoPE. |
| chela_slay | 50.23 | 89.11 | Compression/projection not calibrated to Qwen. |
| cluster | 54.13 | 90.71 | Partial preservation; altered attention geometry. |
| come | 50.46 | 88.88 | Compression changes representation seen by head. |
| compot | 49.08 | 81.42 | QR fp32 retry plus SVD projection loss. |
| compressed_attention | 49.08 | 84.40 | Bidirectional/encoder attention assumptions. |
| compressed_attention_layer_energy | 51.83 | 84.86 | Partial compatibility, but compressed feature basis differs. |
| compressed_mha | 50.57 | 87.61 | MHA replacement does not fully preserve Qwen RoPE/mask path. |
| coopq | 56.65 | 89.22 | Partial preservation with quantization/projection error. |
| cosformer | 50.34 | 49.08 | Random-feature/linear attention path; semantics differ. |
| cr | 49.08 | 81.88 | Compression path collapses features for Qwen head. |
| cur | 49.08 | 49.08 | Recurrent/compact update path; near chance. |
| cur_moe | 49.08 | 50.92 | MoE/recurrent path not aligned with Qwen block. |
| cur_moe_compressor | 49.08 | — | MoE compression and Qwen routing mismatch. |
| decay_linear_attn | 46.10 | prior FAIL | Linear-attention semantics and mixed dtype retry. |
| delta_kv | 53.78 | 89.68 | KV modification partially preserves signal, but not Qwen-native behavior. |
| direct_svd | 49.20 | 83.26 | SVD/QR approximation and fp16/fp32 boundary. |
| dobi | 48.97 | 83.60 | Dynamic/alternate attention path not RoPE-compatible. |
| dynamic_rank | 49.08 | 83.60 | Rank adaptation changes causal attention geometry. |
| eat | 51.72 | 90.02 | Partial compatibility; representation shift dominates. |
| enlighten | 57.34 | 90.25 | Moderate preservation; below native baseline. |
| entropy_budget | 49.20 | 82.91 | Allocation/projection plus fp32-sensitive decomposition. |
| entropy_rank_allocator | 49.08 | 83.60 | Rank allocation changes features expected by head. |

## Interpretation and limitations

The approximately 49% cluster should not be interpreted as proof that every algorithm is intrinsically poor. It is evidence that the current injection/evaluation contract is not a fair quality test for many Qwen modules. A meaningful Qwen comparison would require native Qwen attention signatures, explicit RoPE and causal-mask support in each module, a fixed deterministic fine-tuning protocol, and post-injection calibration or fine-tuning. The present `max_steps=1` evaluation measures immediate compatibility and degradation, not recoverable task accuracy.

The sweep remains active; this report reflects the 48 rows available when generated.
