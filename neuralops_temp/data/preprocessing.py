from typing import Any


def preprocess_function(
    tokenizer, task_name, max_seq_length, sent1_key, sent2_key, is_regression
) -> Any:
    """
    Returns a preprocessing function for GLUE tasks that works with batched=True.
    """

    def _preprocess(examples):
        # Extract the text(s) - each is a list of strings
        """preprocess."""
        texts = (examples[sent1_key],)
        if sent2_key is not None:
            texts = texts + (examples[sent2_key],)

        # Tokenize (works with batched inputs)
        tokenized = tokenizer(
            *texts,
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
            return_tensors=None,
        )
        # Labels: examples["label"] is a list of ints/strings
        if is_regression:
            tokenized["labels"] = [float(x) for x in examples["label"]]
        else:
            tokenized["labels"] = [int(x) for x in examples["label"]]
        return tokenized

    return _preprocess
