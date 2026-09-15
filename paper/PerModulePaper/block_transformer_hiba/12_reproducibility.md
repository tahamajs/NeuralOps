# 12. Reproducibility Statement

We believe in open science and rigorous verification. To reproduce the HIBA results presented in this paper:

## 12.1 Code and Framework
The full implementation, including the `BlockTransformerHIBA` class and the dynamic `copy_weights` injection factory, is open-sourced under the MIT License at:
`https://github.com/HooshaAI/SVD_Project/tree/main/framework/attentions/block_transformer_hiba.py`

## 12.2 Experimental Setup
All experiments were conducted using the following environment:
- **GPU:** Single NVIDIA RTX 4070 (8 GB VRAM)
- **Framework:** PyTorch 2.1+, CUDA 11.8+
- **Hyperparameters:** LoRA Rank = 16, LoRA Alpha = 32.0, Learning Rate = 2e-4, Batch Size = 8, Recovery Steps = 50.
- **Random Seed:** Set globally to `42` via `torch.manual_seed(42)`.

## 12.3 Execution Command
To reproduce the Table 7.1 main results, execute the unified sweep script:
```bash
python Codes/framework/run_quality_sweep.py \
    --model models/distilbert_sst2 \
    --modules full,block_transformer_hiba \
    --recovery_steps 50 \
    --lora_rank 16 \
    --batch_size 8 \
    --precision fp16

```

This script handles the automatic extraction of pretrained weights, the injection of the HIBA module, and the LoRA recovery fine-tuning loop. Model checkpoints (including LoRA adapters) will be saved to the `./results/unified_hiba/` directory for subsequent verification.


---
