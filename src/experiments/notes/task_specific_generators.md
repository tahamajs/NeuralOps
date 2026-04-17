

# 🧪 Specialized NLP Task Generator for Robust Model Evaluation

A comprehensive Python toolkit designed to generate **five distinct, high-difficulty synthetic tasks** specifically for stress-testing Transformer models (BERT, RoBERTa, etc.) under compression (SVD).

Unlike standard sentiment analysis datasets, this engine generates data that targets specific weaknesses in compressed models: **irony detection**, **long-context memory**, **domain generalization**, **semantic equivalence**, and **noise robustness**.

---

## 📋 Table of Contents
1. [Overview](#overview)
2. [The 5 Specialized Tasks](#the-5-specialized-tasks)
3. [Key Features](#key-features)
4. [Project Structure](#project-structure)
5. [Prerequisites & Installation](#prerequisites--installation)
6. [Usage Guide](#usage-guide)
7. [Integration with SVD Optimizer](#integration-with-svd-optimizer)
8. [Output Analysis](#output-analysis)
9. [Author & License](#author--license)

---

## ✨ Overview

This tool (`task_specific_generators.py`) addresses the "overfitting" problem in model compression. When a model is compressed using SVD, it often loses the ability to handle complex linguistic nuances. This generator creates **targeted benchmarks** to prove whether your optimization strategy preserves these critical capabilities.

---

## 🔑 The 5 Specialized Tasks

### 1. 🔍 Task A: Sentiment Reversal (Irony & Sarcasm)
*   **Goal:** Test if the model can detect negative sentiment despite positive keywords (e.g., *"What a **great** movie, I fell asleep"*).
*   **Why Critical?** Compressed models often rely on keyword matching. This task forces them to understand context.
*   **Label Logic:** Positive words $\rightarrow$ Negative Label (0).

### 2. 🧩 Task B: Long-Range Dependency Resolution
*   **Goal:** Evaluate performance on long paragraphs where the sentiment depends on the final clause (e.g., *"It started poorly... but finally succeeded!"*).
*   **Why Critical?** Compression reduces the model's effective "memory." This tests if information at the start of a sentence is lost by the time the end is processed.
*   **Metric:** Sentence length > 60 chars.

### 3. ⚖️ Task C: Domain Shift Robustness
*   **Goal:** Train/Test separation across different fields (e.g., Train on *Movie*, Test on *Finance*).
*   **Why Critical?** Ensures the model learns the *concept* of sentiment, not just domain-specific vocabulary.
*   **Columns:** `train_domain`, `test_domain`.

### 4. 🎨 Task D: Semantic Paraphrase Equivalence
*   **Goal:** Provide pairs of sentences with identical meaning but completely different wording (e.g., *"It broke"* vs. *"It doesn't work"*).
*   **Why Critical?** Tests if the compressed model retains semantic understanding rather than surface-level pattern recognition.
*   **Structure:** Pairs of sentences must share the same label.

### 5. 🌐 Task E: Low-Resource / Noise Sensitivity
*   **Goal:** Generate text with heavy noise (typos, random casing, missing punctuation) to simulate real-world user input.
*   **Why Critical?** Real-world data is messy. If a compressed model fails here, it's not ready for deployment.
*   **Noise Level:** ~30-50% character corruption per sentence.

---

## 🛠 Prerequisites & Installation

### Requirements
- Python 3.8+
- `pandas`
- `numpy`

### Install Dependencies
```bash
pip install pandas numpy
```

---

## 🚀 Usage Guide

### Step 1: Run the Generator
Execute the script to generate all 5 specialized CSV files.

```bash
python task_specific_generators.py
```

### Step 2: Verify Output
Check the `data/` directory. You will find:
- `task_a_irony.csv`
- `task_b_long_context.csv`
- `task_c_domain_shift.csv`
- `task_d_paraphrase.csv`
- `task_e_noise.csv`
- `all_specialized_tasks.csv` (Combined master file)

### Configuration
You can modify the `size` parameter inside `run_all_task_generations()` to increase sample counts (default: 1000 samples per task).

---

## 🔗 Integration with SVD Optimizer

To include these tasks in your main SVD evaluation pipeline (`complete_svd_optimizer.py`), update the `DATASETS_CONFIG` list:

```python
DATASETS_CONFIG = [
    {"name": "Master Dataset", "path": "./data/combined_master_dataset.csv"},
    
    # Add the 5 Specialized Tasks Here
    {"name": "Irony Test", "path": "./data/task_a_irony.csv"},
    {"name": "Long Context", "path": "./data/task_b_long_context.csv"},
    {"name": "Domain Shift", "path": "./data/task_c_domain_shift.csv"},
    {"name": "Paraphrase", "path": "./data/task_d_paraphrase.csv"},
    {"name": "Noise Robustness", "path": "./data/task_e_noise.csv"}
]
```

When you run `python complete_svd_optimizer.py`, the output report will now include a breakdown of accuracy for each of these 5 specific challenges.

---

## 📊 Output Analysis Example

After running the optimizer, your report might look like this:

| Model | Rank | Master Acc | Irony Acc | Long Context Acc | Noise Acc |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BERT-base** | Original | 92.0% | **88.5%** | 90.1% | 85.2% |
| **BERT-base** | Rank 32 | 91.8% | **82.0%** ❌ | 89.5% | 84.1% ❌ |
| **RoBERTa** | Rank 32 | 93.5% | **90.2%** ✅ | 92.0% | 89.5% ✅ |

**Interpretation:**
- If **Irony Acc** drops significantly after compression, the model lost its ability to handle sarcasm.
- If **Noise Acc** drops, the model became too sensitive to exact character matches.
- The goal is to find a Rank where **all columns remain stable**.

---

## 👤 Author & License

**Developed by**: Rayno  
**Organization**: Apadana Intelligent Solutions Development Company  
**Date**: April 2026  

**License**: Educational & Research Use Only.  
*All code and generated data are provided for academic research, benchmarking, and educational purposes.*

---

> **Pro Tip**: When presenting your results, highlight the **Irony** and **Noise** metrics. These are usually the first things to fail during aggressive compression, so maintaining high scores here is a strong indicator of a successful SVD implementation.

RESULTS:

(bert_svd_env) ➜  Project git:(main) ✗ python src/task_specific_generators.py 
🚀 Generating Specialized Tasks...
✅ Generated task_a_irony: 1000 samples
✅ Generated task_b_long_context: 1000 samples
✅ Generated task_c_domain_shift: 1000 samples
✅ Generated task_d_paraphrase: 1000 samples
✅ Generated task_e_noise: 1000 samples

✨ All specialized tasks saved to /Users/tahamajs/Documents/uni/Project/data
(bert_svd_env) ➜  Project git:(main) ✗ 