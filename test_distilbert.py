import os
os.environ["HF_ENDPOINT"] = "https://hf.devneeds.ir/"
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
print("SUCCESS: DistilBERT loaded with shape", model.distilbert.embeddings.word_embeddings.weight.shape)
