# 🌟 SVD-Ensemble: Lightweight Multi-Model Sentiment Analysis

A high-performance, resource-efficient implementation for combining multiple language models (Ensemble) using Matrix Factorization (SVD) without the need for extensive training. This project is specifically optimized for resource-constrained environments, such as Apple Silicon laptops, enabling fast inference and lightweight fine-tuning.

## 🚀 Key Features
- **Zero/Minimal Training:** Supports immediate inference or extremely short fine-tuning (only 1 epoch).
- **Smart Compression:** Compresses Attention layers in BERT/RoBERTa/ALBERT models using Singular Value Decomposition (SVD), significantly reducing memory footprint while preserving accuracy.
- **Flexible Fusion Strategies:** Implements three ensemble methods: Simple Average, Manual Weighted Sum, and Learnable Linear Weights (Quick Tune).
- **Robust Dimension Handling:** Automatically aligns output dimensions across different model architectures to prevent runtime errors.
- **Config-Driven:** Fully integrated with `config.py` for seamless management of paths, datasets, and hyperparameters.

---

## 📋 Prerequisites

Ensure you have the following Python packages installed:

```bash
pip install torch transformers pandas scikit-learn numpy
```

> **Note:** If using macOS with Apple Silicon (M1/M2/M3), ensure your PyTorch version supports MPS (`torch >= 2.0`).

---

## 🛠️ Project Structure

```text
Project/
├── src/
│   ├── smart_ensemble_svd.py    # Main execution script
│   ├── svd_compressor.py        # SVD compression logic & SVDLinearLayer
│   └── config.py                # Global configuration (paths, dataset settings)
├── data/
│   └── combined_master_dataset.csv  # Input dataset (columns: 'text', 'label')
├── final_ensemble_artifacts/    # Output directory for results and models
└── README.md
```

### Essential Files:
1.  **`config.py`**: Must define `batch_size` and point to the local dataset path.
2.  **`svd_compressor.py`**: Contains the `SVDLinearLayer` class for efficient matrix compression.
3.  **`combined_master_dataset.csv`**: A CSV file containing text samples and binary labels.

---

## ⚡ Quick Start

### 1. Setup Configuration
Verify that `config.py` correctly defines `batch_size` (e.g., 8 or 16) and points to your dataset.

### 2. Run the Script
Execute the following command in your terminal:

```bash
python src/smart_ensemble_svd.py
```

### 3. Execution Workflow
The script automatically performs the following steps:
1.  **Load & Compress:** Loads `distilbert-base`, `roberta-base`, and `albert-base-v2`. It compresses their attention layers using the specified SVD ranks.
2.  **Data Preparation:** Tokenizes the dataset and creates train/validation loaders.
3.  **Evaluate Methods:** Tests three fusion strategies:
    *   **Simple Average:** Equal weighting of all models.
    *   **Weighted Manual:** Pre-defined weights (e.g., 0.5, 0.3, 0.2).
    *   **Quick Tune:** Learns optimal weights in just 1 epoch.
4.  **Comparison:** Displays a final table comparing Accuracy and F1-Score for all methods and individual baselines.

---

## 🔧 Configuration & Tuning

You can modify parameters directly in `smart_ensemble_svd.py`:

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `NUM_LABELS` | Target number of classes | `2` |
| `MAX_LENGTH` | Maximum sequence length | `128` |
| `BATCH_SIZE` | Batch size for processing | Read from `config.py` |
| `MODELS_CONFIG` | List of models and their SVD ranks | `[distilbert(32), roberta(48), albert(64)]` |
| `lr` | Learning rate for Quick Tune | `0.5` |

### Adjusting SVD Rank
To balance speed vs. accuracy, change the `rank` values in `MODELS_CONFIG`:
*   **Lower Rank (e.g., 16):** Faster inference, slightly lower accuracy.
*   **Higher Rank (e.g., 64):** Higher accuracy (closer to original), slower inference.

---

## 📊 Outputs

Upon completion, a folder named `final_ensemble_artifacts` will be created containing:

1.  **`weights_quick_tune.pt`**: The learned weights for the best-performing method.
2.  **`tokenizer/`**: Saved tokenizer files for deployment.
3.  **`ensemble_results.csv`**: A comprehensive CSV file comparing all evaluation metrics.

### Sample Console Output:
```text
🏆 FINAL COMPARISON
================================================================================
Simple Average               | Acc: 0.8542 | F1: 0.8510
Weighted Manual              | Acc: 0.8615 | F1: 0.8590
Quick Tune (1 Epoch)         | Acc: 0.8720 | F1: 0.8705
Model 1                      | Acc: 0.8410 | F1: 0.8380
Model 2                      | Acc: 0.8550 | F1: 0.8530
Model 3                      | Acc: 0.8490 | F1: 0.8460
--------------------------------------------------------------------------------
🎯 Best Method: Quick Tune (1 Epoch) (Acc: 0.8720)
```

---

## 🛡️ Error Handling & Robustness

This implementation includes robust mechanisms to handle common issues:
*   **Dimension Alignment:** If a model outputs a different number of classes (e.g., 3 instead of 2), the code automatically slices or pads the tensor to match the target dimension before fusion.
*   **Memory Management:** On Apple Silicon (MPS), memory is cleared periodically to prevent OOM (Out Of Memory) errors.
*   **Graceful Degradation:** If one model fails to load, the script logs the error and continues with the remaining models.

---

## 📝 License

This project is released under the **MIT License**. You are free to use, modify, and distribute this code for academic and commercial purposes.

---

## 👨‍💻 Developer

**RAINO** - Smart Solutions Development Team (Apadana)
Developed for university projects and lightweight NLP applications.

---

## ❓ FAQ

**Q: Why use "Quick Tune" instead of full fine-tuning?**
A: Full fine-tuning is computationally expensive. "Quick Tune" only optimizes the linear combination weights, which takes seconds but often yields significant accuracy boosts by leveraging the diversity of the ensemble.

**Q: Can I add more models?**
A: Yes! Simply add new entries to the `MODELS_CONFIG` list in the script, specifying the model name and desired SVD rank.

**Q: What does "Placeholder storage has not been allocated on MPS" mean?**
A: This is a known issue with older PyTorch versions on macOS. Updating to the latest PyTorch usually resolves it. If it persists, the script automatically falls back to CPU mode.