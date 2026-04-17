# ==================== debug_layers.py ====================
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import config

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(config.hf_cache_dir))
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2,
    cache_dir=str(config.hf_cache_dir)
)

print("🔍 Scanning Linear Layers in DistilBERT Model:\n")
for name, module in model.named_modules():
    if isinstance(module, __import__('torch').nn.Linear):
        # فقط لایه‌هایی که به Attention مربوط هستند یا وزن دارند را نشان بده
        if 'attention' in name or 'output' in name or 'dense' in name:
            print(f"Name: {name}")
            print(f"Shape: {module.weight.shape}")
            print("-" * 40)