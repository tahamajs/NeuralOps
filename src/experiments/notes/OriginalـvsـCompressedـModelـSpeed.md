
(bert_svd_env) ➜  Project git:(main) ✗ python src/benchmark_speed.py

[CONFIG] MPS available. Using Apple Silicon GPU.
============================================================
BENCHMARKING: Original vs Compressed Model Speed
============================================================

🔄 Loading Original Model...
⚠️ Could not find trained models (Unrecognized model in /Users/tahamajs/Documents/huggingface_secure_cache/experiments_logs/custom/original_finetuned. Should have a `model_type` key in its config.json.). Loading base models for structure test only.
Loading weights: 100%|████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 3609.18it/s]
BertForSequenceClassification LOAD REPORT from: google-bert/bert-base-uncased
Key                                        | Status     | 
-------------------------------------------+------------+-
cls.predictions.transform.LayerNorm.weight | UNEXPECTED | 
cls.predictions.transform.LayerNorm.bias   | UNEXPECTED | 
cls.seq_relationship.weight                | UNEXPECTED | 
cls.predictions.transform.dense.bias       | UNEXPECTED | 
cls.seq_relationship.bias                  | UNEXPECTED | 
cls.predictions.transform.dense.weight     | UNEXPECTED | 
cls.predictions.bias                       | UNEXPECTED | 
classifier.bias                            | MISSING    | 
classifier.weight                          | MISSING    | 

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.

🔄 Starting TRUE SVD Compression (Rank=32)...
📊 Found 72 Linear layers to compress (Skipping Embeddings/Pooler/Classifier)
✅ Compressed: bert.encoder.layer.0.attention.self.query | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.0.attention.self.key | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.0.attention.self.value | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.0.attention.output.dense | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.0.intermediate.dense | Ratio: 18.75x
✅ Compressed: bert.encoder.layer.0.output.dense | Ratio: 19.08x
✅ Compressed: bert.encoder.layer.1.attention.self.query | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.1.attention.self.key | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.1.attention.self.value | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.1.attention.output.dense | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.1.intermediate.dense | Ratio: 18.75x
✅ Compressed: bert.encoder.layer.1.output.dense | Ratio: 19.08x
✅ Compressed: bert.encoder.layer.2.attention.self.query | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.2.attention.self.key | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.2.attention.self.value | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.2.attention.output.dense | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.2.intermediate.dense | Ratio: 18.75x
✅ Compressed: bert.encoder.layer.2.output.dense | Ratio: 19.08x
✅ Compressed: bert.encoder.layer.3.attention.self.query | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.3.attention.self.key | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.3.attention.self.value | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.3.attention.output.dense | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.3.intermediate.dense | Ratio: 18.75x
✅ Compressed: bert.encoder.layer.3.output.dense | Ratio: 19.08x
✅ Compressed: bert.encoder.layer.4.attention.self.query | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.4.attention.self.key | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.4.attention.self.value | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.4.attention.output.dense | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.4.intermediate.dense | Ratio: 18.75x
✅ Compressed: bert.encoder.layer.4.output.dense | Ratio: 19.08x
✅ Compressed: bert.encoder.layer.5.attention.self.query | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.5.attention.self.key | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.5.attention.self.value | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.5.attention.output.dense | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.5.intermediate.dense | Ratio: 18.75x
✅ Compressed: bert.encoder.layer.5.output.dense | Ratio: 19.08x
✅ Compressed: bert.encoder.layer.6.attention.self.query | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.6.attention.self.key | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.6.attention.self.value | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.6.attention.output.dense | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.6.intermediate.dense | Ratio: 18.75x
✅ Compressed: bert.encoder.layer.6.output.dense | Ratio: 19.08x
✅ Compressed: bert.encoder.layer.7.attention.self.query | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.7.attention.self.key | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.7.attention.self.value | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.7.attention.output.dense | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.7.intermediate.dense | Ratio: 18.75x
✅ Compressed: bert.encoder.layer.7.output.dense | Ratio: 19.08x
✅ Compressed: bert.encoder.layer.8.attention.self.query | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.8.attention.self.key | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.8.attention.self.value | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.8.attention.output.dense | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.8.intermediate.dense | Ratio: 18.75x
✅ Compressed: bert.encoder.layer.8.output.dense | Ratio: 19.08x
✅ Compressed: bert.encoder.layer.9.attention.self.query | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.9.attention.self.key | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.9.attention.self.value | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.9.attention.output.dense | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.9.intermediate.dense | Ratio: 18.75x
✅ Compressed: bert.encoder.layer.9.output.dense | Ratio: 19.08x
✅ Compressed: bert.encoder.layer.10.attention.self.query | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.10.attention.self.key | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.10.attention.self.value | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.10.attention.output.dense | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.10.intermediate.dense | Ratio: 18.75x
✅ Compressed: bert.encoder.layer.10.output.dense | Ratio: 19.08x
✅ Compressed: bert.encoder.layer.11.attention.self.query | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.11.attention.self.key | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.11.attention.self.value | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.11.attention.output.dense | Ratio: 11.82x
✅ Compressed: bert.encoder.layer.11.intermediate.dense | Ratio: 18.75x
✅ Compressed: bert.encoder.layer.11.output.dense | Ratio: 19.08x

📊 COMPRESSION SUMMARY
   Layers compressed: 72
   Layers skipped (kept full): 2
   Original params: 85,017,600
   Compressed params: 5,393,664
   Reduction: 93.66%
⚠️ Warning: Comparing UNTRAINED models. Accuracy will be random, but SPEED should show improvement.

🧪 Testing on 300 samples...

⏱️ Measuring Original Model Time...
   Average Time: 0.8181 seconds

⏱️ Measuring Compressed Model Time...
   Average Time: 0.5069 seconds

============================================================
📊 RESULTS SUMMARY
============================================================
Original Time:  0.8181s
Compressed Time: 0.5069s
Speed Up:       1.61x faster
Time Saved:     311.28 ms per batch
Reduction %:    38.05%
============================================================
✅ SUCCESS! Compression reduced inference time.
(bert_svd_env) ➜  Project git:(main) ✗ 