# Appendix C. Datasets
Evaluated on SST-2.



## 4. Complexity & Empirical Performance Profile

| Metric | Full Softmax Attention | AEGIS-V1 Baseline |
| :--- | :--- | :--- |
| **FLOP Complexity** | $\mathcal{O}(B \cdot H \cdot L^2 \cdot d_h)$ | $\mathcal{O}(B \cdot L \cdot D \cdot r + B \cdot H_l \cdot L \cdot W \cdot d_h + B \cdot H_g \cdot L \cdot F \cdot d_h)$ |
| **Activation Memory** | $\mathcal{O}(B \cdot H \cdot L^2)$ | $\mathcal{O}(B \cdot H_l \cdot L \cdot W + B \cdot H_g \cdot F \cdot d_h)$ |
| **Inference Step Memory** | $\mathcal{O}(L \cdot D)$ (Full KV-Cache) | $\mathcal{O}(W \cdot D + H_g \cdot F \cdot d_h)$ (Fixed Band + State) |
| **Empirical Benchmarks** | *Benchmarks Pending (Execute run_on_gpu_server.sh)* | *See [Empirical Validation Protocol](../empirical_validation_protocol.md)* |

---

## Benchmark Results

| Status | Accuracy | Val Loss | Peak VRAM | Inference Latency |
|--------|----------|----------|-----------|-------------------|
| Pending| Pending  | Pending  | Pending   | Pending           |

## Empirical Benchmark Results

| Model | Dataset | Status | Baseline Acc | Compressed Acc | Δ Acc | F1 Score | Comp Rate | Compressed Layers | Checkpoint | Date |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **distilbert-sst2** | sst2 | `PASS` | 90.94% | **63.65%** | -27.29% | 0.7167 | 0.99x | 24 | [svd-linear-attention-distilbert-sst2-aegis-v1](https://huggingface.co/Hooshaai/svd-linear-attention-distilbert-sst2-aegis-v1) | 2026-09-14 |
| **roberta-base** | sst2 | `PASS` | 75.46% | **80.62%** | +5.16% | 0.8284 | 0.94x | 48 | [svd-linear-attention-roberta-base-aegis-v1](https://huggingface.co/Hooshaai/svd-linear-attention-roberta-base-aegis-v1) | 2026-09-14 |