import os
os.environ['HF_ENDPOINT'] = 'https://hf.devneeds.ir/'   # fast mirror

from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

print("Downloading config...")
config = BertConfig.from_pretrained("bert-base-uncased")

print("Downloading model...")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

print("✅ All files downloaded to Hugging Face cache.")