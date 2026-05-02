"""Tokenization helpers."""
def preprocess_function(tokenizer, task_name, max_seq_length, sent1_key, sent2_key, is_regression):
    def _preprocess(examples):
        if sent2_key is None:
            tokenized = tokenizer(
                examples[sent1_key],
                truncation=True, padding="max_length", max_length=max_seq_length
            )
        else:
            tokenized = tokenizer(
                examples[sent1_key], examples[sent2_key],
                truncation=True, padding="max_length", max_length=max_seq_length
            )
        if is_regression:
            tokenized["labels"] = [float(l) for l in examples["label"]]
        else:
            tokenized["labels"] = examples["label"]
        return tokenized
    return _preprocess
