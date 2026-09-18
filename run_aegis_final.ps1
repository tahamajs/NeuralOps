$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:HF_ORG = "Hooshaai"
$env:PYTHONWARNINGS = "ignore"
$env:OMP_NUM_THREADS = "8"

Set-Location D:\svd_project

Write-Host "=== Aegis Final Run ===" -ForegroundColor Green

# 4 modules with all fixes
$modules = @(
    @{name="aegis_v2"; model="models/distilbert_sst2"; family="bert"; dataset="sst2"},
    @{name="aegis_v2"; model="models/roberta-base"; family="roberta"; dataset="mnli"},
    @{name="aegis_v4"; model="models/gpt2"; family="gpt2"; dataset="ag_news"},
    @{name="aegis_v5"; model="models/distilbert_sst2"; family="bert"; dataset="sst2"}
)

foreach ($m in $modules) {
    Write-Host "Running $($m.name) on $($m.dataset) [$($m.model)]" -ForegroundColor Cyan
    
    python -u Codes/framework/run_quality_sweep.py `
        --modules $m.name `
        --model $m.model `
        --model_family $m.family `
        --dataset $m.dataset `
        --rank 64 `
        --batch_size 128 `
        --recovery_steps 50 `
        --lora_rank 16 `
        --precision fp16 `
        --hf_org "Hooshaai" `
        --upload_to_hf `
        --output_dir "results/fixed_aegis/$($m.name)_$($m.dataset)/" `
        2>&1 | Tee-Object -Append -FilePath "logs/aegis_final.log"
    
    Start-Sleep -Seconds 10
}

Write-Host "=== All Aegis Runs Complete ===" -ForegroundColor Green
