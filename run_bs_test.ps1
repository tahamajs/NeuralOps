$batch_sizes = @(32, 64, 128, 256)
foreach ($bs in $batch_sizes) {
    Write-Host "==========================="
    Write-Host "Running BS=$bs..."
    $start = Get-Date
    python -u Codes/framework/run_quality_sweep.py --modules a3 --model models/distilbert_sst2 --dataset sst2 --batch_size $bs --rank 64 --max_error 0.50 --output_dir results/bs_test/bs_$bs
    $end = Get-Date
    Write-Host "BS=$bs took $($end - $start)"
}
