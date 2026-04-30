import torch

class CacheNotesCompressor:
    def __init__(self, model, task_prefixes):
        self.model = model
        self.task_prefixes = task_prefixes
    def compress(self):
        # Use prefix trees to prune KV cache per task
        pass