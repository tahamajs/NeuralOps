Set-Location D:\svd_project
Write-Host "Running DistilBERT..."
python Codes\framework\eval_int8.py models\distilbert_sst2 results\unified_a3\distilbert\checkpoints\a3\hf_export\pytorch_model.bin sst2 128 > distilbert_int8.log 2>&1

Write-Host "Running RoBERTa..."
python Codes\framework\eval_int8.py models\roberta-base results\unified_a3\roberta\checkpoints\a3\hf_export\pytorch_model.bin sst2 128 > roberta_int8.log 2>&1

Write-Host "Running GPT-2..."
python Codes\framework\eval_int8.py models\gpt2 results\unified_a3\gpt2\checkpoints\a3\hf_export\pytorch_model.bin sst2 128 > gpt2_int8.log 2>&1

Write-Host "Done!"
