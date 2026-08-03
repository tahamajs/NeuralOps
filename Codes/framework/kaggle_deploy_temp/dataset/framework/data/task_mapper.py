"""
Task metadata: metrics, number of labels, text columns.
"""

TASK_METRICS = {
    "cola": "matthews_correlation",
    "sst2": "accuracy",
    "mrpc": "accuracy",
    "qqp": "accuracy",
    "qnli": "accuracy",
    "rte": "accuracy",
    "mnli": "accuracy",
    "stsb": "pearson",
}

TASK_NUM_LABELS = {
    "cola": 2,
    "sst2": 2,
    "mrpc": 2,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "mnli": 3,
    "stsb": 1,
}

TASK_COLUMNS = {
    "cola": ["sentence"],
    "sst2": ["sentence"],
    "mrpc": ["sentence1", "sentence2"],
    "qqp": ["question1", "question2"],
    "qnli": ["question", "sentence"],
    "rte": ["sentence1", "sentence2"],
    "mnli": ["premise", "hypothesis"],
    "stsb": ["sentence1", "sentence2"],
}


def get_metric_name(task: str) -> str:
    """Return the metric name."""
    return TASK_METRICS.get(task, "accuracy")


def get_num_labels(task: str, dataset=None) -> int:
    """Return the num labels."""
    if task == "stsb":
        return 1
    if task in TASK_NUM_LABELS:
        return TASK_NUM_LABELS[task]
    if dataset and "label" in dataset["train"].column_names:
        return len(dataset["train"].unique("label"))
    return 2


def get_text_columns(task: str) -> list:
    """Return the text columns."""
    return TASK_COLUMNS.get(task, ["sentence"])
