"""Validate datasets before training."""
from collections import Counter
import logging
LOG = logging.getLogger(__name__)

class DataValidator:
    def __init__(self, dataset_name, config_name=None):
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.errors = []
        self.warnings = []

    def validate(self):
        from datasets import load_dataset
        print(f"\n=== Validating {self.dataset_name} ===")
        try:
            ds = load_dataset(self.dataset_name, self.config_name, trust_remote_code=True)
            print(f"  ✅ Loaded: {dict(ds.num_rows)}")
        except Exception as e:
            self.errors.append(f"Load failed: {e}")
            print(f"  ❌ Load failed: {e}")
            return False

        for split, count in ds.num_rows.items():
            if count == 0:
                self.errors.append(f"Empty split: {split}")
            elif count < 10:
                self.warnings.append(f"Tiny split: {split} ({count})")

        if 'label' in ds['train'].features:
            labels = ds['train']['label']
            dist = Counter(labels)
            print(f"  Label distribution: {dict(dist)}")
            if len(dist) < 2:
                self.errors.append(f"Only {len(dist)} class(es)!")

        print(f"  Errors: {len(self.errors)}, Warnings: {len(self.warnings)}")
        if self.warnings:
            for w in self.warnings:
                print(f"  ⚠️  {w}")
        return len(self.errors) == 0
