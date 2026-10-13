#!/usr/bin/env python3
"""Multi-task GLUE trainer that iterates over tasks."""
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
LOG = logging.getLogger(__name__)

# GLUE tasks with their configs
GLUE_TASKS = {
    'sst2': {'config': 'sst2', 'text_col': 'sentence', 'text2_col': None, 'num_labels': 2, 'metric': 'accuracy', 'baseline': 0.927},
    'mrpc': {'config': 'mrpc', 'text_col': 'sentence1', 'text2_col': 'sentence2', 'num_labels': 2, 'metric': 'f1', 'baseline': 0.884},
    'cola': {'config': 'cola', 'text_col': 'sentence', 'text2_col': None, 'num_labels': 2, 'metric': 'matthews_correlation', 'baseline': 0.583},
    'rte':  {'config': 'rte',  'text_col': 'sentence1', 'text2_col': 'sentence2', 'num_labels': 2, 'metric': 'accuracy', 'baseline': 0.665},
}

def parse_args():
    p = argparse.ArgumentParser(description='GLUE benchmark trainer')
    p.add_argument('--model', default='bert-base-uncased')
    p.add_argument('--tasks', nargs='+', default=list(GLUE_TASKS.keys()), choices=list(GLUE_TASKS.keys()))
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--max_len', type=int, default=128)
    p.add_argument('--device', default='auto')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--validate_only', action='store_true')
    p.add_argument('--output_dir', default=str(PROJECT_ROOT / 'results/glue'))
    return p.parse_args()

def run_task(args, task):
    """Train a single GLUE task inline."""
    import torch
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        get_linear_schedule_with_warmup,
    )
    from neuralops.validation import DataValidator, MemoryTracker, ReportGenerator

    cfg = GLUE_TASKS[task]
    output_dir = os.path.join(args.output_dir, f'{args.model.replace("/", "_")}_{task}')
    os.makedirs(output_dir, exist_ok=True)

    device = 'cuda' if (args.device == 'auto' and torch.cuda.is_available()) else args.device if args.device != 'auto' else 'cpu'
    LOG.info(f'\n{"="*60}\nTask: {task.upper()} | Device: {device}\n{"="*60}')

    reporter = ReportGenerator(output_dir)

    # Validate data
    validator = DataValidator('glue', cfg['config'])
    data_ok = validator.validate()
    reporter.add('Data', task, '✅' if data_ok else '❌')
    if not data_ok:
        LOG.error(f'Data validation failed for {task}: {validator.errors}')
        return None

    # Load dataset and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ds = load_dataset('glue', cfg['config'])

    def tokenize(examples):
        if cfg['text2_col']:
            return tokenizer(examples[cfg['text_col']], examples[cfg['text2_col']],
                             truncation=True, max_length=args.max_len)
        return tokenizer(examples[cfg['text_col']], truncation=True, max_length=args.max_len)

    ds = ds.map(tokenize, batched=True)
    ds = ds.rename_column('label', 'labels')
    keep_cols = ['input_ids', 'attention_mask', 'labels', 'token_type_ids']
    ds = ds.remove_columns([c for c in ds['train'].column_names if c not in keep_cols])
    ds.set_format('torch')

    collator = DataCollatorWithPadding(tokenizer)
    train_dl = DataLoader(ds['train'], batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_split = 'validation' if 'validation' in ds else 'validation_matched'
    val_dl = DataLoader(ds[val_split], batch_size=args.batch_size * 2, shuffle=False, collate_fn=collator)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=cfg['num_labels'])
    model = model.to(device)

    if args.validate_only:
        # Just do a quick forward pass check
        sample = next(iter(val_dl))
        sample = {k: v.to(device) for k, v in sample.items()}
        with torch.no_grad():
            out = model(**sample)
        LOG.info(f'[{task}] Validation-only forward pass OK, logits shape: {out.logits.shape}')
        reporter.add('Model', 'forward_pass', '✅', value=str(out.logits.shape))
        reporter.save(f'{task}_validation_report.md')
        return {'task': task, 'validate_only': True, 'status': 'ok'}

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10,
                                                num_training_steps=total_steps)
    tracker = MemoryTracker()
    history = {'train_loss': [], 'val_metric': []}
    best_metric = 0.0
    best_epoch = 0

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
            if (step + 1) % 50 == 0:
                LOG.info(f'[{task}] E{epoch+1} S{step+1}/{len(train_dl)} loss={loss.item():.4f}')

        avg_loss = epoch_loss / len(train_dl)

        # Evaluate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                preds = out.logits.argmax(-1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(batch['labels'].cpu().tolist())

        # Compute metric
        if cfg['metric'] == 'accuracy':
            metric_val = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
        elif cfg['metric'] == 'f1':
            from sklearn.metrics import f1_score
            metric_val = f1_score(all_labels, all_preds)
        elif cfg['metric'] == 'matthews_correlation':
            from sklearn.metrics import matthews_corrcoef
            metric_val = matthews_corrcoef(all_labels, all_preds)
        else:
            metric_val = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

        epoch_time = time.time() - t0
        LOG.info(f'[{task}] E{epoch+1}: loss={avg_loss:.4f}, {cfg["metric"]}={metric_val:.4f}, time={epoch_time:.1f}s')
        history['train_loss'].append(avg_loss)
        history['val_metric'].append(metric_val)

        if metric_val > best_metric:
            best_metric = metric_val
            best_epoch = epoch + 1
            model.save_pretrained(f'{output_dir}/best_model')
            tokenizer.save_pretrained(f'{output_dir}/best_model')

    tracker.report()
    delta = best_metric - cfg['baseline']
    status = '✅' if best_metric >= cfg['baseline'] - 0.03 else '⚠️' if best_metric >= cfg['baseline'] - 0.05 else '❌'
    LOG.info(f'[{task}] Best {cfg["metric"]}: {best_metric:.4f} (published: {cfg["baseline"]:.3f}, delta: {delta:+.4f})')

    reporter.add('Results', f'best_{cfg["metric"]}', status, value=f'{best_metric:.4f}',
                 notes=f'Published: {cfg["baseline"]}, Delta: {delta:+.4f}, Epoch: {best_epoch}')
    reporter.add('VRAM', 'peak_vram', '✅', value=f'{tracker.peak:.2f}GB')
    reporter.save(f'{task}_training_report.md')

    result = {
        'task': task, 'model': args.model,
        'metric': cfg['metric'], 'best_metric': best_metric,
        'published_baseline': cfg['baseline'], 'delta': delta,
        'best_epoch': best_epoch, 'epochs': args.epochs,
        'history': history, 'peak_vram_gb': tracker.peak
    }
    with open(f'{output_dir}/metrics.json', 'w') as f:
        json.dump(result, f, indent=2)

    return result

def main():
    args = parse_args()
    LOG.info(f'GLUE Trainer | Model: {args.model} | Tasks: {args.tasks}')
    LOG.info(f'Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}')

    all_results = {}
    for task in args.tasks:
        try:
            result = run_task(args, task)
            if result:
                all_results[task] = result
        except Exception as e:
            LOG.error(f'Task {task} failed: {e}', exc_info=True)
            all_results[task] = {'task': task, 'error': str(e)}

    # Summary
    LOG.info('\n' + '='*60)
    LOG.info('GLUE BENCHMARK SUMMARY')
    LOG.info('='*60)
    for task, res in all_results.items():
        if 'error' in res:
            LOG.info(f'  {task}: ERROR - {res["error"]}')
        elif res.get('validate_only'):
            LOG.info(f'  {task}: VALIDATE-ONLY OK')
        else:
            cfg = GLUE_TASKS[task]
            LOG.info(f'  {task}: {cfg["metric"]}={res["best_metric"]:.4f} (baseline={cfg["baseline"]:.3f}, delta={res["delta"]:+.4f})')

    # Save overall summary
    summary_path = os.path.join(args.output_dir, 'glue_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    LOG.info(f'\nSummary saved to {summary_path}')

if __name__ == '__main__':
    main()
