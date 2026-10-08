# Benchmark Accuracy Diagnosis

## 1. Root Cause
- The initial verification test used `eval_only=True` to test the executor plumbing quickly without full training.
- Because `AutoModelForSequenceClassification` instantiates a brand new sequence classification head (`pre_classifier` and `classifier`) for `distilbert-base-uncased`, those classification head parameters are randomly initialized from scratch.
- Setting `eval_only=True` bypassed the fine-tuning training loop entirely (`epochs=0`).
- Evaluating a randomly initialized 2-class head on SST-2 binary classification resulted in expected random chance accuracy (~0.50):
  - `svd_linear_attention`: `accuracy = 0.4576`
  - `direct_svd`: `accuracy = 0.4346`

---

## 2. Evidence
- **Loss at Step 0**: `0.6825` (consistent with cross-entropy on random binary classification, $-\ln(0.5) \approx 0.693$).
- **Gradients Flow Correctly**:
  - Classifier gradient norm: `1.324`
  - Total model gradient norm: `2.291`
- **Parameters Active**: Both `pre_classifier` and `classifier` parameters have `requires_grad=True` and are fully trainable.
- **Evaluation Loop Correctness**:
  - Evaluates on the `validation` split (872 examples).
  - Explicit `model.eval()` mode called.
  - Wrapped inside `with torch.no_grad():`.
  - Predictions obtained via `torch.argmax(logits, dim=-1)` matching target label indices.
- **DistilBERT Compatibility**:
  - Batches unpack explicitly into `input_ids`, `attention_mask`, `labels`.
  - `token_type_ids` are omitted/ignored cleanly, avoiding tokenizer mismatches.

---

## 3. Fix
Default configuration across benchmark suite configs:
- `eval_only = False`
- `epochs = 3`
- `batch_size = 32`
- `lr = 2e-5`
- `max_length = 128`

Warnings:
- If `eval_only=True` is explicitly passed for a classification model with newly initialized heads, log a prominent warning that accuracy will reflect random chance unless pre-trained classifier weights are supplied.

---

## 4. Code Verification Summary
- **Training Loop**: Standard `AdamW` optimization with gradient clipping (`clip_grad_norm_ = 1.0`) and running loss tracking.
- **Gradient Flow**: Gradients flow end-to-end to attention modules and classification heads.
- **Evaluation Pipeline**: Metrics (`accuracy`, `f1`, `precision`, `recall`, `loss`) compute cleanly.
- **Hardware Profile**: Supports CPU, CUDA, and Apple Silicon MPS transparently.
