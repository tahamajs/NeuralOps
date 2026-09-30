# PowerShell script to launch 4 parallel sweep workers
Set-Location -Path "D:\SVD_Project"
$workers = 4
for ($i = 0; $i -lt $workers; $i++) {
    $argList = "Codes/framework/run_gpt2_sweep.py --output_dir results/05_perplexity_seqlen/gpt2_wikitext2_sweep --total_workers $workers --worker_id $i --retry_flags FAIL --log_file logs/sweep_worker_$i.log"
    Start-Process -FilePath "python" -ArgumentList $argList -WorkingDirectory "D:\SVD_Project"
    Write-Host "Started worker $i with log logs/sweep_worker_$i.log"
    Start-Sleep -Seconds 5
}
Write-Host "All 4 workers launched successfully."
