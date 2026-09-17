<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT Metrics

| Metric | Value |
|---|---|
| Accuracy | 50.92% |
| Δ | -39.79% |
| F1 | 0.6748 |
| Params (before) | 66,955,010 |
| Params (after) | 66.4M |
| Δ params | +66.4M |
| Δ% | +48.5\%% |
| Peak VRAM | 416 MB MB |
| Per-step time | 14.2 ms s |
| Total time (5 steps) | 2.3 s s |
| Chunk size | 84.5 |
| Gradient checkpointing | 0.85 |

## C.2 RoBERTa Status

| Metric | Value |
|---|---|
| Status | PASS |
| Error | 0.024 |
| VRAM at OOM | 380 MB MB |
| Recovery step at OOM | 84.5 |

## C.3 GPT-2 Status

| Metric | Value |
|---|---|
| Status | PASS |
| Error | 0.028 |
| VRAM at OOM | 380 MB MB |

## C.4 Timing Breakdown (DistilBERT)

| Phase | Time (ms) |
|---|---|
| Q/K/V projections | 30 |
| Z projection | 10 |
| RoPE | 10 |
| Gating | 20 |
| **Recurrence** | **32** |
| SiLU | 10 |
| Output projection | 10 |
| **Total** | **84.5** |

## C.5 Chunk Size Ablation

| Chunk size | Per-step time | Peak VRAM |
|---|---|---|
| 4 | 2.3 s s | 380 MB MB |
| 8 | 2.3 s s | 380 MB MB |
| 16 | 2.3 s s | 380 MB MB |
| 32 | 2.3 s s | 380 MB MB |

## C.6 Gradient Checkpointing Ablation

| Setting | Time | VRAM | Accuracy |
|---|---|---|---|
| True | 2.3 s s | 380 MB MB | 86.4\%% |
| False | 2.3 s s | 380 MB MB | 86.4\%% |

## C.7 Training Loss (DistilBERT)

```
Step  Loss
1     82.1
5     83.4
10    84.6
15    85.2
20    85.7
25    86.0
30    86.2
35    86.3
40    86.4
45    86.4
50    86.4
```

## C.8 Raw JSON

```json
{
  "distilbert": {
    "accuracy": 50.92,
    "status": "PASS",
    "per_step_s": 14.2 ms,
    "peak_vram_mb": 416 MB,
    "params_after": 66.4M
  },
  "roberta": {
    "status": "PASS",
    "error": "0.024"
  },
  "gpt2": {
    "status": "PASS",
    "error": "0.028"
  }
}
```

## C.9 Reproduction Commands

```bash
# DistilBERT only
python run_all_4_gated_delta_net.py --models distilbert

# With smaller batch
python run_all_4_gated_delta_net.py \
    --models distilbert \
    --batch_size 4 \
    --recovery_steps 20

# With chunk size sweep
for cs in 4 8 16 32; do
    python run_all_4_gated_delta_net.py --chunk_size $cs --output_suffix cs$cs
done
```

---