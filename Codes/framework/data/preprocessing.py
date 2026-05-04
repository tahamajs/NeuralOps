# framework/data/preprocessing.py
def preprocess_function(tokenizer, task_name=None, max_seq_length=128,
                        sent1_key="text", sent2_key=None, is_regression=False,
                        multi_choice_config=None):
    """Return a preprocessing function for tokenization."""
    if multi_choice_config:
        # Multi‑choice is handled by MultiChoiceDataset, not here.
        return lambda examples: examples

    def _preprocess(examples):
        if sent2_key is None:
            texts = examples[sent1_key]
        else:
            texts = (examples[sent1_key], examples[sent2_key])
        tokenized = tokenizer(*texts,
                              truncation=True,
                              padding="max_length",
                              max_length=max_seq_length)
        tokenized["labels"] = (
            [float(l) for l in examples["label"]] if is_regression else
            [int(l) for l in examples["label"]]
        )
        return tokenized
    return _preprocess