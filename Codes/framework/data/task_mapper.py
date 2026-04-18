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
    "mnli": 3,      # entailment, neutral, contradiction
    "stsb": 1,      # regression
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

def get_metric_name(task):
    return TASK_METRICS.get(task, "accuracy")

def get_num_labels(task, dataset=None):
    if task == "stsb":
        return 1
    if task in TASK_NUM_LABELS:
        return TASK_NUM_LABELS[task]
    # fallback: detect from dataset
    if dataset and "label" in dataset["train"].column_names:
        return len(dataset["train"].unique("label"))
    return 2  # default binary

def get_text_columns(task):
    return TASK_COLUMNS.get(task, ["sentence"])