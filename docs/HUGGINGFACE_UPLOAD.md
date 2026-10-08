# Uploading Models to Hugging Face Hub

This guide describes how to publish fine-tuned or compressed NeuralOps model checkpoints to the Hugging Face Hub.

---

## 1. Prerequisites

Install the official Hugging Face Hub library:
```bash
pip install huggingface_hub
```

Authenticate with your Hugging Face account token (write permissions required):
```bash
huggingface-cli login
```

---

## 2. Prepare Model Directory

Ensure your model directory contains standard Hugging Face assets:
```
models/distilbert_sst2/
├── config.json
├── model.safetensors
├── tokenizer_config.json
├── vocab.txt
└── README.md (Model card)
```

---

## 3. Upload via Python API

```python
from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="huggingface_models/distilbert-svd-attention",
    repo_id="tahamajs/distilbert-svd-attention",
    repo_type="model",
)
print("Upload complete!")
```

---

## 4. Verify on Hugging Face Hub

Visit your published model repository:
`https://huggingface.co/tahamajs/distilbert-svd-attention`
