from transformers import Trainer, TrainingArguments
import torch
import gc

class SVDAwareTrainer(Trainer):
    def __init__(self, visualizer=None, pre_train_callback=None, post_train_callback=None, **kwargs):
        super().__init__(**kwargs)
        self.visualizer = visualizer
        self.pre_train_callback = pre_train_callback
        self.post_train_callback = post_train_callback

    def train(self, *args, **kwargs):
        if self.pre_train_callback:
            self.pre_train_callback(self.model, self.train_dataset)
        result = super().train(*args, **kwargs)
        if self.post_train_callback:
            self.post_train_callback(self.model, self.train_dataset)
        # Cleanup
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result