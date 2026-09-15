Write-Host "=== 1. adaptive_sparse_attention ==="
if (Test-Path "D:\SVD_Project\Codes\runners\benchmark_logs\benchmark_adaptive_sparse_attention.log") {
    Get-Content "D:\SVD_Project\Codes\runners\benchmark_logs\benchmark_adaptive_sparse_attention.log" -Tail 30
} else {
    Write-Host "No benchmark log found for adaptive_sparse_attention."
}

Write-Host "`n=== 2. gqa ==="
if (Test-Path "D:\SVD_Project\results\unified_gqa\distilbert\console.log") {
    Get-Content "D:\SVD_Project\results\unified_gqa\distilbert\console.log" -Tail 25
} elseif (Test-Path "D:\SVD_Project\Codes\runners\benchmark_logs\benchmark_gqa.log") {
    Get-Content "D:\SVD_Project\Codes\runners\benchmark_logs\benchmark_gqa.log" -Tail 25
}

Write-Host "`n=== 3. simple_ssm ==="
if (Test-Path "D:\SVD_Project\results\unified_simple_ssm\gpt2\console.log") {
    Get-Content "D:\SVD_Project\results\unified_simple_ssm\gpt2\console.log" -Tail 30
} elseif (Test-Path "D:\SVD_Project\Codes\runners\benchmark_logs\benchmark_simple_ssm.log") {
    Get-Content "D:\SVD_Project\Codes\runners\benchmark_logs\benchmark_simple_ssm.log" -Tail 30
}
