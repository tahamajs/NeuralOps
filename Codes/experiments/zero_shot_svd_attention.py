import sys
import torch
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
sys.path.insert(0, str(Path(__file__).parent.parent))
from framework.compression.svd_linear import SVDLinearLayer
from framework.utils.device import get_device
from config import config

def compress_attention_layers(model, rank):
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if any(p in name.lower() for p in ['attention.q_lin', 'attention.k_lin', 'attention.v_lin', 'attention.out_lin',
                                                'attention.self.query', 'attention.self.key', 'attention.self.value']):
                new_layer = SVDLinearLayer(module, rank)
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model.get_submodule(parent_name) if parent_name else model
                setattr(parent, child_name, new_layer)
                count += 1
    return count

def main():
    device = get_device()
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    orig_params = sum(p.numel() for p in model.parameters())
    rank = 32
    compressed = compress_attention_layers(model, rank)
    comp_params = sum(p.numel() for p in model.parameters())
    print(f"Compressed {compressed} attention layers. Parameter reduction: {100*(1 - comp_params/orig_params):.2f}%")

    # Speed test
    test_texts = ["This is great.", "I hate this."] * 50
    inputs = tokenizer(test_texts, return_tensors='pt', truncation=True, padding=True).to(device)
    start = time.time()
    with torch.no_grad():
        model(**inputs)
    print(f"Inference time: {time.time()-start:.4f}s")

if __name__ == "__main__":
    main()