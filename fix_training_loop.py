import re
from pathlib import Path

file_path = Path('Codes/framework/run_quality_sweep.py')
content = file_path.read_text()

# We need to replace from "# 4. Prepare Training Batch & Validation Set" to "        peak_vram_mb = 0.0"
start_str = "    # 4. Prepare Training Batch & Validation Set"
end_str = "        peak_vram_mb = 0.0"

start_idx = content.find(start_str)
end_idx = content.find(end_str) + len(end_str)

replacement = """    # 4. Prepare Real Training DataLoader
    from framework.data.prepare_datasets import GLUE_TASK_CONFIG
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok_name = MODEL_TOKENIZER_MAPPING.get(model_family.lower(), "bert-base-uncased")
    train_cache_key = (dataset.lower(), tok_name, seq_len, "train")

    if train_cache_key not in _PREPARED_DATASET_CACHE and dataset.lower() in GLUE_TASK_CONFIG:
        tokenizer = AutoTokenizer.from_pretrained(tok_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        config = GLUE_TASK_CONFIG[dataset.lower()]
        k1, k2 = config["keys"]
        try:
            train_data = load_dataset("glue", dataset.lower(), split="train")
            def tok_fn(ex):
                args = (ex[k1],) if k2 is None else (ex[k1], ex[k2])
                return tokenizer(*args, padding="max_length", truncation=True, max_length=seq_len)
            
            train_tokenized = train_data.map(tok_fn, batched=True, desc=f"Tokenizing {dataset} train")
            train_tokenized.set_format(type="torch", columns=[c for c in ["input_ids", "attention_mask", "label"] if c in train_tokenized.column_names])
            _PREPARED_DATASET_CACHE[train_cache_key] = train_tokenized
        except Exception as e:
            logger.warning(f"Failed to load train dataset for {dataset}: {e}")
            _PREPARED_DATASET_CACHE[train_cache_key] = None
    elif dataset.lower() not in GLUE_TASK_CONFIG:
        _PREPARED_DATASET_CACHE[train_cache_key] = None

    train_ds = _PREPARED_DATASET_CACHE.get(train_cache_key)
    if train_ds is not None and len(train_ds) > 0 and not allow_synthetic:
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=current_batch_size, shuffle=True, drop_last=True
        )
    else:
        single_batch = get_dataset_batch(
            dataset, model_family, current_batch_size, seq_len,
            num_labels=num_labels, device=device, allow_synthetic=allow_synthetic
        )
        train_loader = [single_batch]

    actual_steps_per_epoch = len(train_loader)
    if epochs > 0:
        total_steps = actual_steps_per_epoch * epochs
        if total_steps > max_steps:
            max_steps = min(total_steps, 200)  # cap at 200 to prevent infinite sweeps

    def get_infinite_batches(loader):
        while True:
            for b in loader:
                yield {
                    ("labels" if k == "label" else k): (v.to(device, non_blocking=is_cuda) if isinstance(v, torch.Tensor) else v)
                    for k, v in b.items() if k in ["input_ids", "attention_mask", "label", "labels"]
                }
    batch_iter = get_infinite_batches(train_loader)

    # 5. Optimizer & Linear Warmup Scheduler
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

    warmup_steps = max(1, int(max_steps * 0.1))
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, float(max_steps - step) / float(max(1, max_steps - warmup_steps)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 6. Timed Training Loop with Step Trajectory Tracking
    traj_loss, traj_grad_norm, traj_fwd_ms, traj_bwd_ms = [], [], [], []
    
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        fwd_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(max_steps)]
        fwd_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(max_steps)]
        bwd_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(max_steps)]
        bwd_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(max_steps)]

        for step in range(max_steps):
            batch = next(batch_iter)
            filtered_batch = _filter_batch(batch, model_family, model)
            
            optimizer.zero_grad()
            fwd_start_events[step].record()
            with torch.amp.autocast('cuda'):
                out = model(**filtered_batch)
            fwd_end_events[step].record()

            logits = getattr(out, "logits", out[0] if isinstance(out, (tuple, list)) else None)
            labels = batch.get("labels")
            loss = None

            if hasattr(out, "loss") and out.loss is not None:
                loss = out.loss
            elif logits is not None and labels is not None:
                if num_labels == 1:
                    loss = nn.MSELoss()(logits.squeeze(), labels.squeeze().float())
                else:
                    loss = nn.CrossEntropyLoss()(logits.view(-1, num_labels), labels.view(-1))
            
            if loss is None or not torch.is_tensor(loss):
                loss = torch.tensor(0.0, requires_grad=True, device=device)

            traj_loss.append(round(float(loss.detach().item()), 4))

            bwd_start_events[step].record()
            loss.backward()
            bwd_end_events[step].record()

            total_norm_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm_sq += float(p.grad.detach().data.norm(2).item() ** 2)
            grad_norm = math.sqrt(total_norm_sq)
            traj_grad_norm.append(round(grad_norm, 4))
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        bwd_end_events[-1].synchronize()
        for s in range(max_steps):
            traj_fwd_ms.append(round(float(fwd_start_events[s].elapsed_time(fwd_end_events[s])), 2))
            traj_bwd_ms.append(round(float(bwd_start_events[s].elapsed_time(bwd_end_events[s])), 2))
        
        peak_vram_mb = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2)
    else:
        for step in range(max_steps):
            batch = next(batch_iter)
            filtered_batch = _filter_batch(batch, model_family, model)

            optimizer.zero_grad()
            t0 = time.perf_counter()
            out = model(**filtered_batch)
            traj_fwd_ms.append(round((time.perf_counter() - t0) * 1000.0, 2))

            logits = getattr(out, "logits", out[0] if isinstance(out, (tuple, list)) else None)
            labels = batch.get("labels")
            loss = None
            if hasattr(out, "loss") and out.loss is not None:
                loss = out.loss
            elif logits is not None and labels is not None:
                loss = nn.CrossEntropyLoss()(logits.view(-1, num_labels), labels.view(-1))
            if loss is None:
                loss = torch.tensor(0.0, requires_grad=True, device=device)

            traj_loss.append(round(float(loss.detach().item()), 4))
            t2 = time.perf_counter()
            loss.backward()
            traj_bwd_ms.append(round((time.perf_counter() - t2) * 1000.0, 2))
            
            total_norm_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm_sq += float(p.grad.detach().data.norm(2).item() ** 2)
            traj_grad_norm.append(round(math.sqrt(total_norm_sq), 4))
            optimizer.step()
            scheduler.step()
        peak_vram_mb = 0.0"""

new_content = content[:start_idx] + replacement + content[end_idx:]
file_path.write_text(new_content)
print("Updated successfully")
