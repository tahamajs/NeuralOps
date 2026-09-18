$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:HF_ORG = "TahaaaaM"

# Only run what hasn't finished yet
$tasks = @(
    @("aegis_v2", "models/roberta-base", "mnli"),
    @("aegis_v3", "models/gpt2", "agnews"),
    @("aegis_v4", "models/gpt2", "agnews"),
    @("aegis_v5", "models/distilbert_sst2", "sst2")
)

foreach ($task in $tasks) {
    $mod = $task[0]
    $mdl = $task[1]
    $ds = $task[2]
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
