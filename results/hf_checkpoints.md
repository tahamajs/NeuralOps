# HuggingFace Checkpoints

| Dataset | Repo | Accuracy | Labels |
|---|---|---|---|
| SST-2 | [TahaaaaM/distilbert-sst2-a3](https://huggingface.co/TahaaaaM/distilbert-sst2-a3) | 91.06% | 2 |
| MNLI | [TahaaaaM/distilbert-mnli-a3](https://huggingface.co/TahaaaaM/distilbert-mnli-a3) | 81.8% | 3 |
| AG News | [TahaaaaM/distilbert-agnews-a3](https://huggingface.co/TahaaaaM/distilbert-agnews-a3) | 94.26% | 4 |

These are the fully fine-tuned base checkpoints used as starting points for A3 SVD compression experiments. All three are public and can be loaded directly with `transformers.AutoModelForSequenceClassification.from_pretrained()`.
