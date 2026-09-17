$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location "D:\SVD_Project"
$Python = "C:\Users\Hadi Schumacher\anaconda3\python.exe"
$Script = "D:\SVD_Project\auto_upload_hf_results.py"
$Log = "D:\SVD_Project\hf_uploader.log"

while ($true) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $Log -Value "[$ts] === HF Uploader Starting ===" -Encoding UTF8
    & $Python -u $Script *>> $Log
    $exitCode = $LASTEXITCODE
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $Log -Value "[$ts] === HF Uploader Exited with $exitCode ===" -Encoding UTF8
    Start-Sleep -Seconds 30
}
