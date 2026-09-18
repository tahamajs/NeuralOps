$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$modules = @("aegis_v2", "aegis_v3", "aegis_v4", "aegis_v5")
$models = @{
    "aegis_v2" = @("models/distilbert_sst2", "models/roberta-base")
    "aegis_v3" = @("models/gpt2")
    "aegis_v4" = @("models/gpt2")
    "aegis_v5" = @("models/distilbert_sst2")
}
$datasets = @{
    "models/distilbert_sst2" = "sst2"
    "models/roberta-base"    = "mnli"
    "models/gpt2"            = "agnews"
}

foreach ($mod in $modules) {
    foreach ($mdl in $models[$mod]) {
        $ds = $datasets[$mdl]
        Write-Host "=== Running $mod on $mdl ($ds) ==="
        python -u Codes/framework/run_quality_sweep.py `
            --modules $mod `
            --model $mdl `
            --dataset $ds `
            --batch_size 256 `
            --rank 64 `
            --max_error 0.50 `
            --output_dir "results/fixed_aegis/${mod}_${ds}" `
            --fresh
        Write-Host "=== Done $mod on $ds ==="
    }
}
