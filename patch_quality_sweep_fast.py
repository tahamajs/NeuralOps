from pathlib import Path

p = Path("Codes/framework/run_quality_sweep.py")
content = p.read_text(encoding="utf-8")

# 1. Update train_loader creation with num_workers, prefetch_factor, pin_memory
old_loader = '''    train_ds = _PREPARED_DATASET_CACHE.get(train_cache_key)
    if train_ds is not None and len(train_ds) > 0 and not allow_synthetic_local:
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=current_batch_size, shuffle=True, drop_last=True
        )'''

new_loader = '''    train_ds = _PREPARED_DATASET_CACHE.get(train_cache_key)
    if train_ds is not None and len(train_ds) > 0 and not allow_synthetic_local:
        import os
        workers = min(4, os.cpu_count() or 1)
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

content = content.replace(old_loader, new_loader, 1)

# 2. Update val loader creation with num_workers, prefetch_factor, pin_memory
old_val_loader = '''    val_tokenized = _PREPARED_DATASET_CACHE.get(dataset_cache_key)
    if val_tokenized is not None and len(val_tokenized) > 0:
        loader = torch.utils.data.DataLoader(val_tokenized, batch_size=current_batch_size, shuffle=False, pin_memory=is_cuda)'''

new_val_loader = '''    val_tokenized = _PREPARED_DATASET_CACHE.get(dataset_cache_key)
    if val_tokenized is not None and len(val_tokenized) > 0:
        import os
        val_workers = min(4, os.cpu_count() or 1)
        loader = torch.utils.data.DataLoader(
            val_tokenized,
            batch_size=current_batch_size,
            shuffle=False,
            num_workers=val_workers,
            pin_memory=is_cuda,
            prefetch_factor=4 if val_workers > 0 else None,
            persistent_workers=(val_workers > 0),
        )'''

content = content.replace(old_val_loader, new_val_loader, 1)

# 3. Update autocast to float16
content = content.replace(
    "with torch.amp.autocast('cuda'):",
    "with torch.amp.autocast('cuda', dtype=torch.float16):"
)

# 4. Add per-seed completion logging
old_return = '''    return SeedResult(
        seed=seed,'''

new_return = '''    logger.info(
        "    --> Seed %d completed: Acc=%.2f%%, F1=%.4f, Fid=%.4f, Peak VRAM=%.1fMB",
        seed, acc_pct, f1, fidelity, peak_vram_mb
    )
    return SeedResult(
        seed=seed,'''

content = content.replace(old_return, new_return, 1)

# 5. Ensure matmul precision high is set
old_matmul = "torch.set_float32_matmul_precision('high')"
if old_matmul not in content:
    content = content.replace(
        "torch.backends.cudnn.benchmark = True",
        "torch.backends.cudnn.benchmark = True\n        torch.set_float32_matmul_precision('high')"
    )

p.write_text(content, encoding="utf-8")
print("Codes/framework/run_quality_sweep.py patched with fast settings.")
