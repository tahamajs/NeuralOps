if (Test-Path "D:\SVD_Project\Codes\runners\benchmark_logs\benchmark_svd_compressed.log") {
    Get-Content "D:\SVD_Project\Codes\runners\benchmark_logs\benchmark_svd_compressed.log" -Tail 30
} else {
    Write-Host "No log for svd_compressed"
}
