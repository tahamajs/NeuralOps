Write-Host "=== 1. RUNNING PYTHON PROCESSES ==="
Get-WmiObject Win32_Process -Filter "Name='python.exe'" | Select-Object ProcessId, CommandLine

Write-Host "`n=== 2. LATEST AUTO GIT SYNC LOG ==="
if (Test-Path "D:\SVD_Project\results\auto_git_sync.log") {
    Get-Content "D:\SVD_Project\results\auto_git_sync.log" -Tail 15
}

Write-Host "`n=== 3. LATEST AUTO HF UPLOAD LOG ==="
if (Test-Path "D:\SVD_Project\results\auto_upload.log") {
    Get-Content "D:\SVD_Project\results\auto_upload.log" -Tail 15
}

Write-Host "`n=== 4. RERUN BENCHMARK PROGRESS ==="
if (Test-Path "D:\SVD_Project\results\full_benchmark\rerun_stdout.log") {
    Get-Content "D:\SVD_Project\results\full_benchmark\rerun_stdout.log" -Tail 25
}
