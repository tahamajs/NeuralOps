$Python   = "C:\Users\Hadi Schumacher\anaconda3\python.exe"
$Script   = "D:\SVD_Project\run_paper_artifacts.py"
$WorkDir  = "D:\SVD_Project"
$LogFile  = "D:\SVD_Project\paper_artifacts_forever.log"
Set-Location $WorkDir
for ($i = 1; $i -le 100; $i++) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $LogFile -Value "[$ts] === Attempt $i starting ==="
    & $Python -u $Script 2>&1 | Tee-Object -FilePath $LogFile -Append
    $exitCode = $LASTEXITCODE
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $LogFile -Value "[$ts] === Exited with code $exitCode ==="
    if ($exitCode -eq 0) { break }
    Start-Sleep -Seconds 30
}
