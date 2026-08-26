from datasets import load_dataset
from transformers import AutoTokenizer
import torch

def get_train_dataloader(dataset_name, tokenizer_name, batch_size, seq_len, device):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Check if dataset is stsb to handle differently?
    if dataset_name == 'stsb':
        k1, k2 = "sentence1", "sentence2"
    elif dataset_name == 'mrpc':
        k1, k2 = "sentence1", "sentence2"
    elif dataset_name == 'cola':
        k1, k2 = "sentence", None
    elif dataset_name == 'sst2':
        k1, k2 = "sentence", None
    
    val_data = load_dataset("glue", dataset_name, split="train")
    def tok_fn(ex):
        args = (ex[k1],) if k2 is None else (ex[k1], ex[k2])
        return tokenizer(*args, padding="max_length", truncation=True, max_length=seq_len)
    
    val_tokenized = val_data.map(tok_fn, batched=True)
    val_tokenized.set_format(type="torch", columns=[c for c in ["input_ids", "attention_mask", "label"] if c in val_tokenized.column_names])
    
    loader = torch.utils.data.DataLoader(val_tokenized, batch_size=batch_size, shuffle=True)
    return loader

