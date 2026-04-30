# studio/training/callbacks.py
class TrainingCallback:
    def on_epoch_end(self, epoch, logs): pass
    def on_batch_end(self, batch, logs): pass