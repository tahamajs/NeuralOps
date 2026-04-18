import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification, RobertaConfig, RobertaForSequenceClassification

models = ["bert-base-uncased", "roberta-base", "distilbert-base-uncased"]
for model_name in models:
    print(f"Downloading {model_name}...")
    AutoTokenizer.from_pretrained(model_name)
    if "bert" in model_name:
        BertConfig.from_pretrained(model_name)
        BertForSequenceClassification.from_pretrained(model_name)
    elif "roberta" in model_name:
        RobertaConfig.from_pretrained(model_name)
        RobertaForSequenceClassification.from_pretrained(model_name)
    else:
        AutoConfig.from_pretrained(model_name)
        AutoModelForSequenceClassification.from_pretrained(model_name)
    print(f"✅ {model_name} downloaded.")