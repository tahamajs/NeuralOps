Get-WmiObject Win32_Process -Filter "Name='python.exe'" | Select-Object ProcessId, CommandLine
Write-Host "--- AUTO UPLOAD LOG ---"
if (Test-Path "D:\SVD_Project\results\auto_upload.log") {
    Get-Content "D:\SVD_Project\results\auto_upload.log" -Tail 20
} else {
    Write-Host "No log file found yet."
}
