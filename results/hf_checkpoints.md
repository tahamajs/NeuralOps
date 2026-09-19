# HuggingFace Base Checkpoints

These are DistilBERT fine-tuned on each dataset, used as starting points for compression experiments.

| Dataset | Repo | Accuracy | Labels |
|---|---|---|---|
| SST-2   | [Hooshaai/distilbert-sst2](https://huggingface.co/Hooshaai/distilbert-sst2)     | 91.06% | 2 |
| MNLI    | [Hooshaai/distilbert-mnli](https://huggingface.co/Hooshaai/distilbert-mnli)     | 81.80% | 3 |
| AG News | [Hooshaai/distilbert-agnews](https://huggingface.co/Hooshaai/distilbert-agnews) | 94.26% | 4 |

**Note:** These are base checkpoints, NOT compressed models.
Compression results (a3, kmeans, aasvd, etc.) are documented separately in the sweep summaries.
