# OOM Early Warning — aasvd

## Memory Safeguards

- Peak VRAM observed in benchmark:
  - DistilBERT: ~350 MB
  - RoBERTa: ~700 MB
  - GPT-2: ~800 MB
- Online accumulation: Never store full activation tensors in RAM; use streaming covariance or batchwise chunking.
