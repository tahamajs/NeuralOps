"""Track VRAM/RAM usage."""
import torch

class MemoryTracker:
    def __init__(self):
        self.peak = 0
        self.reset()

    def reset(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.peak = 0

    def update(self):
        if torch.cuda.is_available():
            current = torch.cuda.max_memory_allocated() / 1e9
            self.peak = max(self.peak, current)
            return current
        return 0.0

    def report(self):
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f'  Peak VRAM: {self.peak:.2f} GB / {total:.2f} GB ({self.peak/total*100:.1f}%)')
        else:
            print(f'  Peak VRAM: N/A (CPU mode)')

    def check_ok(self, threshold=0.9):
        if not torch.cuda.is_available():
            return True
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        if self.peak / total > threshold:
            raise MemoryError(f'VRAM usage > {threshold*100}%')
