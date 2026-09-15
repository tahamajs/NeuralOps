# 7. Results

## 7.1 Main Results

| Model | Baseline Acc. | Recovered Acc. | Delta | Peak VRAM | Latency |
|---|---:|---:|---:|---:|---:|
| DistilBERT | {{BASE_D}}% | {{ACC_D}}% | {{DELTA_D}} | {{VRAM_D}} | {{LAT_D}} |
| RoBERTa-base | {{BASE_R}}% | {{ACC_R}}% | {{DELTA_R}} | {{VRAM_R}} | {{LAT_R}} |
| GPT-2 | {{BASE_G}}% | {{ACC_G}}% | {{DELTA_G}} | {{VRAM_G}} | {{LAT_G}} |

## 7.2 Ablations

We vary rank, recovery steps, and gate initialization to understand stability and accuracy trade-offs.

## 7.3 Inference Cost Breakdown

We report memory reduction and runtime effects relative to the dense baseline under matched settings.
