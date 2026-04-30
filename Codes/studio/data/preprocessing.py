# studio/data/preprocessing.py
from transformers import DataCollatorWithPadding

def preprocess_function(tokenizer, task_name, max_seq_length, is_regression=False):
    def fn(examples):
        if task_name in ["cola", "sst2"]:
            texts = examples["sentence"]
        elif task_name in ["mrpc", "qqp", "qnli", "rte"]:
            texts = [f"{a} [SEP] {b}" for a, b in zip(examples["sentence1"], examples["sentence2"])]
        else:
            texts = examples["text"]
        tokenized = tokenizer(texts, truncation=True, max_length=max_seq_length, padding=False)
        if is_regression:
            tokenized["labels"] = [float(l) for l in examples["label"]]
        else:
            tokenized["labels"] = examples["label"]
        return tokenized
    return fn

def tokenize_dataset(dataset, tokenizer, fn, num_proc=4):
    return dataset.map(fn, batched=True, num_proc=num_proc, remove_columns=["label"])

DataCollator = DataCollatorWithPadding