#!/usr/bin/env python3
"""BERT fine-tuning on SST-2 with comprehensive validation."""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
LOG = logging.getLogger(__name__)

DEFAULT_MODEL = 'bert-base-uncased'
DEFAULT_TASK = 'sst2'
DEFAULT_EPOCHS = 3
DEFAULT_BATCH = 16
DEFAULT_LR = 2e-5
DEFAULT_MAXLEN = 128
PUBLISHED_BASELINE = 0.927

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    p.add_argument('--batch_size', type=int, default=DEFAULT_BATCH)
    p.add_argument('--lr', type=float, default=DEFAULT_LR)
    p.add_argument('--max_len', type=int, default=DEFAULT_MAXLEN)
    p.add_argument('--validate_only', action='store_true')
    p.add_argument('--device', default='auto')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_dir', default=str(PROJECT_ROOT / 'results/glue/bert_base_sst2'))
    return p.parse_args()

def set_seed(seed):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_device(mode='auto'):
    if mode == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return mode

def prepare_data(tokenizer, max_len, batch_size):
    from datasets import load_dataset
    from neuralops.validation import DataValidator

    validator = DataValidator('glue', 'sst2')
    if not validator.validate():
        raise RuntimeError(f'Data validation failed: {validator.errors}')

    ds = load_dataset('glue', 'sst2')
    def tokenize(examples):
        return tokenizer(examples['sentence'], truncation=True, max_length=max_len)

    ds = ds.map(tokenize, batched=True)
    ds = ds.rename_column('label', 'labels')
    ds = ds.remove_columns([c for c in ds['train'].column_names if c not in ['input_ids', 'attention_mask', 'labels', 'token_type_ids']])
    ds.set_format('torch')

    collator = DataCollatorWithPadding(tokenizer)
    train_dl = DataLoader(ds['train'], batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_dl = DataLoader(ds['validation'], batch_size=batch_size*2, shuffle=False, collate_fn=collator)
    return train_dl, val_dl, ds

def evaluate(model, dl, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            preds = out.logits.argmax(-1)
            correct += (preds == batch['labels']).sum().item()
            total += batch['labels'].size(0)
    return correct / total

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    LOG.info(f'Device: {device}')
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model and tokenizer
    LOG.info(f'Loading {args.model}...')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    # Validation phase
    from neuralops.validation import ModelValidator, TrainingValidator, MemoryTracker, ReportGenerator
    reporter = ReportGenerator(args.output_dir)

    mv = ModelValidator(model, (2, 32), device=device)
    model_ok = mv.validate()
    reporter.add('Model', 'forward+backward', '✅' if model_ok else '❌', notes=str(mv.errors))
    if not model_ok:
        LOG.error(f'Model validation failed: {mv.errors}')
        if args.validate_only:
            sys.exit(1)

    # Prepare data
    LOG.info('Preparing data...')
    train_dl, val_dl, ds = prepare_data(tokenizer, args.max_len, args.batch_size)
    LOG.info(f'Train: {len(train_dl)} batches, Val: {len(val_dl)} batches')
    reporter.add('Data', 'SST-2 dataset', '✅', value=f'{len(ds["train"])} train')

    # Overfit test
    model.to(device)
    sample_batch = next(iter(train_dl))
    sample_batch = {k: v.to(device) for k, v in sample_batch.items()}
    overfit_ok = TrainingValidator.overfit_test(model, sample_batch, n_samples=8, steps=30, lr=1e-3, device=device)
    reporter.add('Training', 'overfit_test', '✅' if overfit_ok else '❌')

    if args.validate_only:
        reporter.save('validation_report.md')
        LOG.info('Validation-only mode: done.')
        return

    # Full training
    LOG.info('Starting full training...')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps//10, num_training_steps=total_steps)

    tracker = MemoryTracker()
    history = {'train_loss': [], 'val_acc': []}
    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for step, batch in enumerate(train_dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            tracker.update()
            if (step+1) % 100 == 0:
                LOG.info(f'Epoch {epoch+1} Step {step+1}/{len(train_dl)} loss={loss.item():.4f}')

        avg_loss = epoch_loss / len(train_dl)
        val_acc = evaluate(model, val_dl, device)
        epoch_time = time.time() - t0
        LOG.info(f'Epoch {epoch+1}: loss={avg_loss:.4f}, val_acc={val_acc:.4f}, time={epoch_time:.1f}s')
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            model.save_pretrained(f'{args.output_dir}/best_model')
            tokenizer.save_pretrained(f'{args.output_dir}/best_model')

    tracker.report()

    # Compare against baseline
    delta = best_acc - PUBLISHED_BASELINE
    status = '✅' if best_acc >= PUBLISHED_BASELINE - 0.03 else '⚠️' if best_acc >= PUBLISHED_BASELINE - 0.05 else '❌'
    LOG.info(f'Best accuracy: {best_acc:.4f} (published: {PUBLISHED_BASELINE:.3f}, delta: {delta:+.4f})')

    reporter.add('Results', 'best_val_accuracy', status, value=f'{best_acc:.4f}', notes=f'Published: {PUBLISHED_BASELINE}, Delta: {delta:+.4f}')
    reporter.add('VRAM', 'peak_vram', '✅', value=f'{tracker.peak:.2f}GB')

    # Save metrics
    metrics = {
        'task': 'sst2', 'model': args.model,
        'best_val_accuracy': best_acc, 'published_baseline': PUBLISHED_BASELINE,
        'delta': delta, 'epochs': args.epochs,
        'history': history, 'peak_vram_gb': tracker.peak
    }
    with open(f'{args.output_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    reporter.save('validation_report.md')
    LOG.info(f'Done! Results in {args.output_dir}')

if __name__ == '__main__':
    main()
