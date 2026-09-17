$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location "D:\SVD_Project"
$Python = "C:\Users\Hadi Schumacher\anaconda3\python.exe"
$Script = "D:\SVD_Project\auto_git_sync.py"
$Log = "D:\SVD_Project\git_sync.log"

while ($true) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $Log -Value "[$ts] === Git Sync Starting ===" -Encoding UTF8
    & $Python -u $Script *>> $Log
    $exitCode = $LASTEXITCODE
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $Log -Value "[$ts] === Git Sync Exited with $exitCode ===" -Encoding UTF8
    Start-Sleep -Seconds 30
}
