# Auto-restarting background sweep runner
# Invokes the multi-class sweep in an infinite loop. Exits on success (exit code 0).

$RepoDir = "D:\SVD_Project"
$PythonExe = "C:\Users\Hadi Schumacher\anaconda3\python.exe"
$LogFile = "D:\SVD_Project\sweep_forever.log"
$Script = "D:\SVD_Project\run_agnews_multi_class_sweep.py"

[System.IO.File]::AppendAllText($LogFile, "======================================================`n[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting infinite sweep monitor...`n======================================================`n", [System.Text.Encoding]::ASCII)

while ($true) {
    [System.IO.File]::AppendAllText($LogFile, "`n[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Launching sweep script...`n", [System.Text.Encoding]::ASCII)
    
    Set-Location $RepoDir
    & $PythonExe -u $Script --models distilbert,roberta,gpt2 --recovery_steps 25 2>&1 | Out-File -FilePath $LogFile -Append -Encoding ascii
    
    $exitCode = $LASTEXITCODE
    
    if ($exitCode -eq 0) {
        [System.IO.File]::AppendAllText($LogFile, "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Sweep completed successfully (Exit Code 0). Exiting loop.`n", [System.Text.Encoding]::ASCII)
        break
    } else {
        [System.IO.File]::AppendAllText($LogFile, "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Sweep crashed or aborted (Exit Code $exitCode). Restarting in 10 seconds...`n", [System.Text.Encoding]::ASCII)
        Start-Sleep -Seconds 10
    }
}
