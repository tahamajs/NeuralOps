Module neuralops.evaluation.full_mnli_validation
================================================
full_mnli_validation.py
-----------------------
Re-evaluates the top PASS modules from the local sweep against the full
9815-sample MNLI validation_matched set.

Usage (from project root):
    python neuralops/full_mnli_validation.py         --model models/distilbert_mnli_full         --sweep_csv results/02_glue_sweeps/distilbert/distilbert_mnli_full_mnli_sweep/summary.csv         --dataset_dir neuralops/data/glue_data/mnli         --out results/02_glue_sweeps/distilbert/distilbert_mnli_full_mnli_sweep/full_mnli_validation.csv

Functions
---------

`encode_batch(tokenizer, texts, seq_len, device)`
:   

`eval_on_dataset(model, tokenizer, val_ds, seq_len: int, batch_size: int, device)`
:   Run evaluation on the provided dataset split. Returns full metrics dict.

`main()`
:   

`wilson_ci(n_correct: int, n_total: int, z: float = 1.96)`
:   Wilson score 95% CI for a proportion.