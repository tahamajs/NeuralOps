Write-Host "=== 1. tpa on GPT2 ==="
if (Test-Path "D:\SVD_Project\results\unified_tpa\gpt2\console.log") {
    Get-Content "D:\SVD_Project\results\unified_tpa\gpt2\console.log" -Tail 25
}

Write-Host "`n=== 2. svd_compressed on DistilBERT / RoBERTa ==="
if (Test-Path "D:\SVD_Project\results\unified_svd_compressed\distilbert\console.log") {
    Get-Content "D:\SVD_Project\results\unified_svd_compressed\distilbert\console.log" -Tail 25
} elseif (Test-Path "D:\SVD_Project\results\unified_svd_compressed\roberta\console.log") {
    Get-Content "D:\SVD_Project\results\unified_svd_compressed\roberta\console.log" -Tail 25
}

Write-Host "`n=== 3. simple_ssm on GPT2 ==="
if (Test-Path "D:\SVD_Project\results\unified_simple_ssm\gpt2\console.log") {
    Get-Content "D:\SVD_Project\results\unified_simple_ssm\gpt2\console.log" -Tail 25
}
