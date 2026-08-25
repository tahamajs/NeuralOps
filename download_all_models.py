import os
import sys

# Configure Iranian high-speed Hugging Face mirror
os.environ["HF_ENDPOINT"] = "https://hf.devneeds.ir"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "300"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoModel
from datasets import load_dataset

models = [
    ('bert-base-uncased', 'seq_cls'),
    ('roberta-base', 'seq_cls'),
    ('distilbert-base-uncased', 'seq_cls'),
    ('t5-small', 'auto'),
    ('gpt2', 'causal_lm')
]

print('=== Downloading Models & Tokenizers via DevNeeds Mirror ===')
for m, mtype in models:
    try:
        print(f'Fetching {m}...')
        tok = AutoTokenizer.from_pretrained(m)
        if mtype == 'seq_cls':
            AutoModelForSequenceClassification.from_pretrained(m)
        elif mtype == 'causal_lm':
            AutoModelForCausalLM.from_pretrained(m)
        else:
            AutoModel.from_pretrained(m)
        print(f'SUCCESS: {m}')
    except Exception as e:
        print(f'ERROR {m}: {type(e).__name__}: {e}')

print('=== Pre-caching GLUE Datasets ===')
for task in ['sst2', 'mrpc', 'qqp', 'mnli', 'qnli', 'rte']:
    try:
        print(f'Fetching GLUE {task}...')
        ds = load_dataset('glue', task)
        print(f'SUCCESS: GLUE {task} cached ({len(ds["train"])} train samples)')
    except Exception as e:
        print(f'ERROR GLUE {task}: {type(e).__name__}: {e}')
