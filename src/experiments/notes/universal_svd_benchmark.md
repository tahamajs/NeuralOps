(bert_svd_env) ➜  Project git:(main) ✗ 
(bert_svd_env) ➜  Project git:(main) ✗ 
(bert_svd_env) ➜  Project git:(main) ✗ python src/universal_svd_benchmark.py
[CONFIG] MPS available. Using Apple Silicon GPU.
======================================================================
🌍 UNIVERSAL SVD BENCHMARK: MULTI-ARCH & MULTI-RANK TESTING
======================================================================

🔄 Processing Model: distilbert-base-uncased-finetuned-sst-2-english
Loading weights: 100%|████████████████████████████████████████████████████████████████| 104/104 [00:00<00:00, 3792.88it/s]
   ✅ Rank 8: Acc=0.8989, F1=0.8986
   ✅ Rank 16: Acc=0.9527, F1=0.9527
   ✅ Rank 32: Acc=0.9568, F1=0.9568
   ✅ Rank 64: Acc=0.9627, F1=0.9627

🔄 Processing Model: google-bert/bert-base-uncased
Loading weights: 100%|████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 2977.30it/s]
BertForSequenceClassification LOAD REPORT from: google-bert/bert-base-uncased
Key                                        | Status     | 
-------------------------------------------+------------+-
cls.seq_relationship.bias                  | UNEXPECTED | 
cls.predictions.transform.LayerNorm.weight | UNEXPECTED | 
cls.predictions.bias                       | UNEXPECTED | 
cls.seq_relationship.weight                | UNEXPECTED | 
cls.predictions.transform.dense.weight     | UNEXPECTED | 
cls.predictions.transform.LayerNorm.bias   | UNEXPECTED | 
cls.predictions.transform.dense.bias       | UNEXPECTED | 
classifier.bias                            | MISSING    | 
classifier.weight                          | MISSING    | 

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
   ✅ Rank 8: Acc=0.4994, F1=0.3378
   ✅ Rank 16: Acc=0.5232, F1=0.4133
   ✅ Rank 32: Acc=0.5087, F1=0.3740
   ✅ Rank 64: Acc=0.4814, F1=0.4540

======================================================================
📊 FINAL SUMMARY TABLE
======================================================================
                                          Model Architecture  Rank  Layers Compressed  Accuracy  F1 Score  Status
distilbert-base-uncased-finetuned-sst-2-english   distilbert     8                 24  0.898889  0.898584 Success
distilbert-base-uncased-finetuned-sst-2-english   distilbert    16                 24  0.952685  0.952672 Success
distilbert-base-uncased-finetuned-sst-2-english   distilbert    32                 24  0.956759  0.956758 Success
distilbert-base-uncased-finetuned-sst-2-english   distilbert    64                 24  0.962685  0.962685 Success
                  google-bert/bert-base-uncased         bert     8                 48  0.499444  0.337794 Success
                  google-bert/bert-base-uncased         bert    16                 48  0.523241  0.413308 Success
                  google-bert/bert-base-uncased         bert    32                 48  0.508704  0.373991 Success
                  google-bert/bert-base-uncased         bert    64                 48  0.481389  0.454045 Success

🏆 BEST CONFIGURATION FOUND:
   Model: distilbert-base-uncased-finetuned-sst-2-english
   Rank: 64
   Accuracy: 0.9627
   Layers Compressed: 24
======================================================================
(bert_svd_env) ➜  Project git:(main) ✗ 



