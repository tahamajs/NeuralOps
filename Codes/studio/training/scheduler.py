# studio/training/scheduler.py
from transformers import get_linear_schedule_with_warmup

def get_scheduler(optimizer, num_training_steps, warmup_ratio=0.1):
    num_warmup = int(warmup_ratio * num_training_steps)
    return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup, num_training_steps=num_training_steps)