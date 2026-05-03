"""Tokenization and preprocessing helpers for diverse task types."""

def preprocess_function(tokenizer, task_name=None, max_seq_length=128,
                        sent1_key="text", sent2_key=None, is_regression=False,
                        multi_choice_config=None):
    """
    Return a function that preprocesses a batch of examples.

    Args:
        tokenizer: HuggingFace tokenizer
        task_name: optional GLUE task name (used for column mapping)
        max_seq_length: maximum sequence length
        sent1_key: first sentence column name
        sent2_key: second sentence column name (if pair task)
        is_regression: whether labels are continuous
        multi_choice_config: dict with keys 'question_col', 'choices_col', 'label_col' if multi‑choice
    """
    if multi_choice_config:
        # Multi‑choice preprocessing is usually done in a custom Dataset class;
        # we provide a placeholder that returns the raw features unchanged.
        # In practice, we'll use MultiChoiceDataset directly.
        def _multi_preprocess(examples):
            return examples
        return _multi_preprocess

    def _preprocess(examples):
        texts = (examples[sent1_key],) if sent2_key is None else (examples[sent1_key], examples[sent2_key])
        tokenized = tokenizer(
            *texts,
            truncation=True,
            padding="max_length",
            max_length=max_seq_length
        )
        if is_regression:
            tokenized["labels"] = [float(l) for l in examples["label"]]
        else:
            tokenized["labels"] = [int(l) for l in examples["label"]]
        return tokenized
    return _preprocess