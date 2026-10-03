<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: D_negative_results.md
     PURPOSE: What didn't work
     DEPENDENCIES: 08_analysis.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix D. Negative Results

## D.1 LayerNorm Initialization Mismatch

The block's LayerNorm starts at $\gamma = 1, \beta = 0$, which does not 
match the pretrained LayerNorm.

### D.1.1 Attempted Fixes

| Fix | Result |
|---|---|
| Higher LoRA rank | No improvement |
| More recovery steps | Slow convergence, no improvement |
| Train LN separately | Untested |
| Copy pretrained LN | Untested (would require pretrained access) |

## D.2 Silent Weight Transfer Failure

The `try/except` swallows errors silently.

### D.2.1 Detection

We added logging and observed:

| Model | Path A transfer | Silent fail? |
|---|---|---|
| DistilBERT | OK | No |
| RoBERTa | OK | No |
| GPT-2 | OK | No |

**Observation.** For our specific models, transfer happened to succeed. But 
the code is fragile — any change could trigger silent failure.

## D.3 Non-Causal MHA for GPT-2

`nn.MultiheadAttention` without causal masking.

### D.3.1 Test

| Configuration | GPT-2 Accuracy |
|---|---|
| Baseline (causal) | 79.36% |
| Block MHA (non-causal) | 86.4\%% |

**Observation.** 0.85

## D.4 Nested Encoder Inconsistency

For RoBERTa, the block returns raw output without LN. For DistilBERT, it 
applies LN.

### D.4.1 Impact

| Model | Path | LN applied? |
|---|---|---|
| DistilBERT | Both | Yes |
| RoBERTa | Both | No |
| GPT-2 | Both | Yes |

**Inconsistency** makes debugging harder.

## D.5 Slow GatedDeltaNet Path

Same issue as the standalone GatedDeltaNet: Python loops dominate.

| Path | Per-step time |
|---|---|
| Baseline | 0.4 s |
| Block (MHA) | 2.3 s s |
| Block (DeltaNet) | 14.2 ms s |

## D.6 OOM on RoBERTa/GPT-2

Same memory issue as GatedDeltaNet.

### D.6.1 Attempted Fixes

| Fix | Result |
|---|---|
| batch_size=4 | Still OOM |
| batch_size=2 | Still OOM |
| gradient checkpointing | Still OOM (recompute overhead) |

## D.7 What Would Make This Work?

1. **Copy pretrained LayerNorm** into the block.
2. **Remove `try/except: pass`** — log failures.
3. **Add causal masking** for decoders.
4. **Use fused Triton kernel** for GatedDeltaNet.
5. **Reduce memory** with better checkpointing.

## D.8 Lessons Learned

1. **Preserve pretrained LayerNorm.** Do not replace with fresh init.
2. **Never swallow exceptions silently.** Log every failure.
3. **Causal masking is not optional.** It is a correctness requirement.
4. **Block-level replacement is harder than attention-level.** More sources 
   of error.

## D.9 Recommendations

1. **Prefer attention-level replacement.** It preserves the block structure 
   including LayerNorm.
2. **If block-level is necessary**, copy the pretrained LayerNorm.
3. **Always verify weight transfer** with logging.
4. **Test causal masking** explicitly.

---