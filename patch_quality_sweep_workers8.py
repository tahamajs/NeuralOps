from pathlib import Path

p = Path("Codes/framework/run_quality_sweep.py")
content = p.read_text(encoding="utf-8")

# 1. Update train_loader workers to 8, prefetch_factor to 8
old_w = '''        workers = min(4, os.cpu_count() or 1)
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=current_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=workers,
            pin_memory=is_cuda,
            prefetch_factor=4 if workers > 0 else None,
            persistent_workers=(workers > 0),
        )'''

new_w = '''        workers = min(8, os.cpu_count() or 1)
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=current_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=workers,
            pin_memory=is_cuda,
            prefetch_factor=8 if workers > 0 else None,
            persistent_workers=(workers > 0),
        )'''

content = content.replace(old_w, new_w, 1)

# 2. Update val_loader workers to 8, prefetch_factor to 8
old_val_w = '''        val_workers = min(4, os.cpu_count() or 1)
        loader = torch.utils.data.DataLoader(
            val_tokenized,
            batch_size=current_batch_size,
            shuffle=False,
            num_workers=val_workers,
            pin_memory=is_cuda,
            prefetch_factor=4 if val_workers > 0 else None,
            persistent_workers=(val_workers > 0),
        )'''

new_val_w = '''        val_workers = min(8, os.cpu_count() or 1)
        loader = torch.utils.data.DataLoader(
            val_tokenized,
            batch_size=current_batch_size,
            shuffle=False,
            num_workers=val_workers,
            pin_memory=is_cuda,
            prefetch_factor=8 if val_workers > 0 else None,
            persistent_workers=(val_workers > 0),
        )'''

content = content.replace(old_val_w, new_val_w, 1)

p.write_text(content, encoding="utf-8")
print("Updated run_quality_sweep.py with workers=8, prefetch_factor=8.")
