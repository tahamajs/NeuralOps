# 🏆 Kaggle & Cloud GPU Deployment Guide

This guide outlines how to upload, configure, and execute the **Efficient Attention Research Framework** on Kaggle GPUs (P100 / T4 / L4 / A100), Google Colab, or any remote Linux GPU cluster.

---

## 📁 Repository Zip & Upload Options

### Option A: Direct Git Clone (Recommended if public repo)
In a Kaggle Notebook cell:
```bash
!git clone https://github.com/your-username/SVD_linear_Attention.git
%cd SVD_linear_Attention/Codes/framework
!python run_kaggle.py list
```

### Option B: Kaggle Dataset Upload
1. Zip the `Codes/framework` folder:
   ```bash
   cd Codes/framework
   zip -r framework_code.zip .
   ```
2. Upload `framework_code.zip` as a Kaggle Dataset named `framework-code`.
3. In your Kaggle Notebook:
   ```bash
   !cp -r /kaggle/input/framework-code /kaggle/working/framework
   %cd /kaggle/working/framework
   !python run_kaggle.py train --attn_type full --task sst2 --epochs 3
   ```

---

## ⚡ Execution Commands on Kaggle

All commands run via `python run_kaggle.py` (which auto-configures Kaggle paths and installs missing packages):

### 1. List All Components
```bash
!python run_kaggle.py list
```

### 2. Fine-Tune Attention Model
```bash
!python run_kaggle.py train \
    --attn_type full \
    --task sst2 \
    --epochs 3 \
    --batch_size 32 \
    --lr 2e-5 \
    --exp_name distilbert_sst2_baseline
```

### 3. Apply SVD Model Compression
```bash
!python run_kaggle.py compress \
    --compressor aasvd \
    --task sst2 \
    --compression_rank 64 \
    --exp_name distilbert_aasvd_compressed
```

### 4. Evaluate Checkpoint
```bash
!python run_kaggle.py eval \
    --attn_type full \
    --task sst2 \
    --model_path /kaggle/working/results/distilbert_aasvd_compressed/model \
    --exp_name eval_aasvd_compressed
```

### 5. Multi-Seed Grid Sweep
```bash
!python run_kaggle.py sweep \
    --attn_types full nha lsh \
    --seeds 42 123 456 \
    --task sst2
```

---

## 📊 Kaggle Output Artifacts

Results are saved to `/kaggle/working/results/<exp_name>/`:
- `config.json`: Hardware, hyperparameters, and model config
- `metrics/`: Final evaluation metrics (`eval_final.json`, `compression.json`)
- `history/`: Full training loss history (`log_history.json`, `log_history.csv`)
- `plots/`: `loss_curve.png` & `metric_curve.png`
- `model/`: HuggingFace saved model weights & tokenizer
- `SUMMARY.md`: Formatted Markdown report
