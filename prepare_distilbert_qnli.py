from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
cache = Path(r"D:/SVD_PROJECT/offline_hf/distilbert_qnli")
cache.mkdir(parents=True, exist_ok=True)
print("Downloading DistilBERT weights...", flush=True)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
tokenizer.save_pretrained(cache / "distilbert-base-uncased")
model.save_pretrained(cache / "distilbert-base-uncased")
print("Downloading GLUE QNLI...", flush=True)
ds = load_dataset("glue", "qnli")
ds.save_to_disk(str(cache / "glue_qnli"))
print("PREPARED", flush=True)
