# GPU Smoke Test Report

## Configuration
- Model: distilbert-base-uncased
- Dataset: sst2
- Rank: 64
- Epochs: 1
- Batch Size: 32
- Sequence Length: 128
- Device: NVIDIA GeForce RTX 4070 Laptop GPU (CUDA 12.4, PyTorch 2.6.0+cu124)

## Results

### Baseline (No Compression)
- Accuracy: 91.40% (0.9140)
- F1 Score: 0.9156
- Loss: 0.2236
- Peak VRAM: 668.68 MB
- Throughput: 88,358 tokens/sec
- Status: SUCCESS

### SVD Linear Attention (`svd_linear_attention`)
- Accuracy: 89.11% (0.8911)
- F1 Score: 0.8936
- Loss: 0.2776
- Parameters: 55,803,722 (~55.8M vs 67.0M baseline, 1.20× compression)
- Peak VRAM: 578.14 MB
- Duration: 368.80s (~6.1 min)
- Status: SUCCESS

### Direct SVD (`direct_svd`)
- Accuracy: 86.01% (0.8601)
- F1 Score: 0.8638
- Loss: 0.3178
- Peak VRAM: 499.27 MB
- Duration: 292.88s (~4.9 min)
- Status: SUCCESS

## Criteria Evaluation
| Test | Target | Actual | Result |
|:---|:---:|:---:|:---:|
| Baseline Accuracy | > 0.80 | 0.9140 (91.40%) | ✓ PASS |
| SVD r=64 Accuracy | > 0.75 | 0.8911 (89.11%) | ✓ PASS |
| direct_svd Accuracy | > 0.75 | 0.8601 (86.01%) | ✓ PASS |
| Out Of Memory (OOM) | No OOM | 0 OOM (Peak < 670MB) | ✓ PASS |
| Execution Status | All Pass | 3/3 Succeeded | ✓ PASS |

## Conclusion
GPU server environment, data loaders, CUDA runtime, and module injection pass all criteria. Ready for Triton kernel checks and multi-module benchmarking.
