import os
from huggingface_hub import HfApi

hf_token = os.environ.get('HF_TOKEN')
repo_id = 'TahaaaaM/distilbert-agnews-a3'
model_dir = 'D:/svd_project/models/distilbert_agnews_full'

api = HfApi(token=hf_token)
api.create_repo(repo_id=repo_id, exist_ok=True, private=False)

card = """---
language:
- en
license: apache-2.0
tags:
- distilbert
- classification
- ag_news
- a3-compression
datasets:
- fancyzhx/ag_news
metrics:
- accuracy
- f1
model-index:
- name: distilbert-agnews-a3
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: AG News
      type: fancyzhx/ag_news
    metrics:
    - type: accuracy
      value: 0.9426
    - type: f1
      value: 0.9428
---

# DistilBERT-base Fine-Tuned on AG News (Full Dataset)

Fine-tuned `distilbert-base-uncased` on the full AG News dataset (120,000 training examples, 4 classes) for use as the evaluation checkpoint in the A3 compression sweep.

- **Base Model**: `distilbert-base-uncased`
- **Training Samples**: 120,000
- **Validation Samples**: 7,600
- **Epochs**: 2
- **Batch Size**: 128 (FP16)
- **Validation Accuracy**: **94.26%**
- **Macro F1**: **0.9428**
"""

with open(os.path.join(model_dir, 'README.md'), 'w', encoding='utf-8') as f:
    f.write(card)

api.upload_folder(
    folder_path=model_dir,
    repo_id=repo_id,
    commit_message='Upload DistilBERT-base fine-tuned on full AG News (94.26% accuracy)'
)
print('AG News Model successfully uploaded to Hugging Face!')
