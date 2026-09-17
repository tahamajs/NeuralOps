# Dataset Hashes

## Commands

```bash
sha256sum Codes/framework/data/glue_data/ag_news/* > env/dataset_hashes.txt
sha256sum results/unified_asvd/*/finetuned_model.pt > env/model_hashes.txt
```

## Output

- `papers/modules/<module>/env/dataset_hashes.txt`
- `papers/modules/<module>/env/model_hashes.txt`
