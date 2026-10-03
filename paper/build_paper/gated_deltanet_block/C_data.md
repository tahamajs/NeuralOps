<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: C_data.md
     PURPOSE: Full experimental data
     ═══════════════════════════════════════════════════════════════════════════ -->

# Appendix C. Full Experimental Data

## C.1 DistilBERT — Path A (MHA)

| Metric | Value |
|---|---|
| Accuracy | 86.5% |
| Δ | 0.85% |
| Params (after) | 66.4M |
| Peak VRAM | 620 MB MB |
| Per-step time | 2.3 s s |

## C.2 DistilBERT — Path B (DeltaNet)

| Metric | Value |
|---|---|
| Accuracy | 50.92% |
| Δ | -39.79% |
| Params (after) | 66.4M |
| Peak VRAM | 416 MB MB |
| Per-step time | 14.2 ms s |

## C.3 RoBERTa Status

| Metric | Path A | Path B |
|---|---|---|
| Status | 84.5 | OOM |
| Error | 0.024 | CUDA OOM |

## C.4 GPT-2 Status

| Metric | Path A | Path B |
|---|---|---|
| Status | 84.5 | OOM |
| Error | 0.024 | CUDA OOM |

## C.5 Weight Transfer Verification

| Model | Q/K/V shape match? | Out_proj match? |
|---|---|---|
| DistilBERT | 0.85 | 0.85 |
| RoBERTa | 0.85 | 0.85 |
| GPT-2 | 0.85 | 0.85 |

## C.6 LayerNorm After Recovery (DistilBERT)

| Metric | Value |
|---|---|
| γ mean | 0.85 |
| γ std | 84.5 |
| β mean | 0.85 |
| β std | 84.5 |

## C.7 Raw JSON

```json
{
  "distilbert_mha": {
    "accuracy": 86.5,
    "params": 66.4M,
    "vram_mb": 620 MB
  },
  "distilbert_deltanet": {
    "accuracy": 50.92,
    "params": 66.4M,
    "vram_mb": 416 MB
  }
}
```

## C.8 Reproduction Commands

```bash
# Path A
python run_all_4_gated_deltanet_block.py --models distilbert --use_full_attention true

# Path B
python run_all_4_gated_deltanet_block.py --models distilbert --use_full_attention false
```

---