import re
from pathlib import Path

file_path = Path('Codes/framework/run_quality_sweep.py')
content = file_path.read_text()

start_str = "    val_tokenized = _PREPARED_DATASET_CACHE.get(dataset_cache_key)"
end_str = "        val_batch = get_dataset_batch("

start_idx = content.find(start_str)
end_idx = content.find(end_str) + len(end_str)

replacement = """    if dataset_cache_key not in _PREPARED_DATASET_CACHE and dataset.lower() in GLUE_TASK_CONFIG:
        tokenizer = AutoTokenizer.from_pretrained(tok_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        config = GLUE_TASK_CONFIG[dataset.lower()]
        k1, k2 = config["keys"]
        try:
            val_data = load_dataset("glue", dataset.lower(), split="validation")
            def tok_fn(ex):
                args = (ex[k1],) if k2 is None else (ex[k1], ex[k2])
                return tokenizer(*args, padding="max_length", truncation=True, max_length=seq_len)
            
            val_tokenized = val_data.map(tok_fn, batched=True, desc=f"Tokenizing {dataset} validation")
            val_tokenized.set_format(type="torch", columns=[c for c in ["input_ids", "attention_mask", "label"] if c in val_tokenized.column_names])
            _PREPARED_DATASET_CACHE[dataset_cache_key] = val_tokenized
        except Exception:
            _PREPARED_DATASET_CACHE[dataset_cache_key] = None

    val_tokenized = _PREPARED_DATASET_CACHE.get(dataset_cache_key)
    if val_tokenized is not None and len(val_tokenized) > 0:
        loader = torch.utils.data.DataLoader(val_tokenized, batch_size=current_batch_size, shuffle=False, pin_memory=is_cuda)
        with torch.no_grad():
            for val_batch in loader:
                inputs = {("labels" if k == "label" else k): (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in val_batch.items()}
                filtered_inputs = _filter_batch(inputs, model_family, model)
                with torch.amp.autocast('cuda') if is_cuda else contextlib.nullcontext():
                    val_out = model(**filtered_inputs)
                val_logits = getattr(val_out, "logits", val_out[0] if isinstance(val_out, (tuple, list)) else None)
                if val_logits is not None:
                    all_module_logits.append(val_logits.cpu())
                    if num_labels == 1:
                        preds = val_logits.squeeze()
                    else:
                        preds = torch.argmax(val_logits, dim=-1)
                    all_preds.extend(preds.cpu().numpy())
                    if "labels" in inputs:
                        all_labels.extend(inputs["labels"].cpu().numpy())
    else:
        # Synthetic / fast validation fallback
        val_batch = get_dataset_batch("""

new_content = content[:start_idx] + replacement + content[end_idx:]
file_path.write_text(new_content)
print("Updated successfully")
