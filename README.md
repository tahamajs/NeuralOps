# 📘 Comprehensive Guide: BERT SVD Compression Pipeline

This guide provides a complete, step-by-step walkthrough for setting up, running, and analyzing the BERT model compression project using Singular Value Decomposition (SVD). All instructions are in English.

---

## 1. Prerequisites & Installation

Ensure you have Python installed (version 3.8 or higher recommended).

### Step 1: Activate Virtual Environment
If you haven't created one yet, create it first:
```bash
python -m venv bert_svd_env
source bert_svd_env/bin/activate  # macOS/Linux
# .\bert_svd_env\Scripts\activate  # Windows
```

### Step 2: Install Dependencies
Install all required libraries listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```
**Key Libraries:**
*   `torch`, `transformers`: Core AI framework and Hugging Face models.
*   `datasets`, `pandas`: Data handling and CSV processing.
*   `scikit-learn`: Metrics calculation (Accuracy, F1, Precision).
*   `numpy`: Numerical operations for matrix math.

---

## 2. Project Structure & File Roles

| File | Function | Technical Detail |
| :--- | :--- | :--- |
| **`config.py`** | Central Configuration | Defines dataset path, SVD rank (`svd_rank`), model name, and output directories. Changing this affects the entire run. |
| **`generate_samples.py`** | Synthetic Data Generator | Creates diverse, balanced synthetic datasets (CSV) if real data isn't available. |
| **`compress_bert_advanced.py`** | SVD Compression Engine | The core logic that decomposes weight matrices ($W = U \Sigma V^T$) and reconstructs them with lower rank to reduce parameters. |
| **`evaluator.py`** | Evaluation Module | Loads the model, runs inference on the test set, calculates metrics (Accuracy, F1, Latency), and handles batching. |
| **`run_pipeline.py`** | Main Orchestrator | Executes the full workflow: Load → Baseline Eval → Compress → Post-Compression Eval → Save Results. |

---

## 3. Execution Workflow

### Phase 1: Configure Settings
Open `src/config.py` and set your desired parameters:

```python
# In src/config.py

# 1. Select Dataset
# Options: "sst2", "imdb", "custom", "mrpc"
CURRENT_DATASET_NAME = "custom" 

# 2. Set SVD Rank
# Lower rank = Smaller/Faster model but potentially lower accuracy.
# Recommended range: 4 to 32 depending on your needs.
svd_rank = 8 
```

### Phase 2: Generate/Dataset Preparation
If you do not have a local CSV file for your chosen dataset, generate one now:
```bash
python src/generate_samples.py
```
*Expected Output:* `✅ Generated X samples for custom` (or sst2/imdb). This creates files like `data/data_custom.csv`.

### Phase 3: Run the Full Pipeline
Execute the main script to perform compression and evaluation:
```bash
python src/run_pipeline.py
```

**What happens during execution:**
1.  **Loading:** Downloads/loads `google-bert/bert-base-uncased`.
2.  **Baseline:** Evaluates the original model to establish a performance benchmark.
3.  **Compression:** Applies SVD to all linear layers, reducing their dimensionality to `svd_rank`.
4.  **Evaluation:** Tests the compressed model's performance.
5.  **Reporting:** Saves detailed JSON results comparing the two models.

---

## 4. Analyzing Results

After the script finishes, check the `experiments/results/` directory.

### A. `results_<timestamp>.json`
This file contains the specific metrics for your run.
```json
{
  "config": {
    "model_name": "google-bert/bert-base-uncased",
    "svd_rank": 8,
    "device": "mps"
  },
  "original_model": {
    "metrics": { "accuracy": 0.92, "f1_score": 0.91 },
    "stats": { "reduction_percent": 0.0 }
  },
  "compressed_model": {
    "metrics": { "accuracy": 0.89, "f1_score": 0.88 },
    "stats": { "reduction_mb": 145.5, "reduction_percent": 65.2 }
  },
  "comparison": {
    "accuracy_loss": 0.03,
    "size_reduction_pct": 65.2
  }
}
```
*   **`accuracy_loss`**: How much performance dropped after compression.
*   **`size_reduction_pct`**: How much smaller the model became.

### B. `history.json`
If you run the pipeline multiple times with different ranks, this file accumulates all results, allowing you to compare trade-offs over time.

---

## 5. Troubleshooting & Optimization

### Issue 1: Out of Memory (OOM) Error
*   **Symptom:** `CUDA out of memory` or `MemoryError`.
*   **Fix:** Open `config.py` and lower the `batch_size` (e.g., change from `8` to `4` or `2`).

### Issue 2: Significant Accuracy Drop
*   **Symptom:** Compressed model accuracy is much lower than baseline (>5% drop).
*   **Fix:** Increase `svd_rank` in `config.py` (e.g., try `16` or `32`). A higher rank preserves more information.

### Issue 3: "Local dataset not found"
*   **Symptom:** Script fails to load CSV.
*   **Fix:** Ensure `python src/generate_samples.py` was run successfully. Check that the file exists at the path defined in `config.py` (e.g., `./data/data_custom.csv`).

### Issue 4: Model Inference Errors after Compression
*   **Symptom:** The compressed model crashes during evaluation.
*   **Fix:** Ensure the `compress_bert_advanced.py` script correctly updates the `bias` terms alongside weights. Also, verify the model is moved to the correct device (`model.to(device)`) before evaluation.

---

## 6. Optimization Strategy (Recommended Workflow)

To find the best balance between speed and accuracy:

1.  **Baseline:** Run with default settings (no compression or very high rank) to get the reference score.
2.  **Light Compression:** Test with `svd_rank=16`. Expect minimal accuracy loss (<1%) and moderate size reduction.
3.  **Heavy Compression:** Test with `svd_rank=8` or `4`. Expect faster inference but potential accuracy drops.
4.  **Selection:** Choose the configuration where the accuracy loss is acceptable for your application while maximizing size reduction.

