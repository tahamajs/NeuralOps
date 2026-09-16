# Auto-restarting background sweep runner
# Invokes the multi-class sweep in an infinite loop. Exits on success (exit code 0).

$RepoDir = "D:\SVD_Project"
$PythonExe = "C:\Users\Hadi Schumacher\anaconda3\python.exe"
$LogFile = "D:\SVD_Project\sweep_forever.log"
$Script = "D:\SVD_Project\run_agnews_multi_class_sweep.py"

Add-Content -Path $LogFile -Value "======================================================"
Add-Content -Path $LogFile -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting infinite sweep monitor..."
Add-Content -Path $LogFile -Value "======================================================"

while ($true) {
    Add-Content -Path $LogFile -Value "`n[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Launching sweep script..."
    
    # Run the sweep and pipe output to log
    Set-Location $RepoDir
    & $PythonExe -u $Script --models distilbert,roberta,gpt2 --recovery_steps 25 | Out-File -FilePath $LogFile -Append
    
    $exitCode = $LASTEXITCODE
    
    if ($exitCode -eq 0) {
        Add-Content -Path $LogFile -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Sweep completed successfully (Exit Code 0). Exiting loop."
        break
    } else {
        Add-Content -Path $LogFile -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Sweep crashed or aborted (Exit Code $exitCode). Restarting in 10 seconds..."
        Start-Sleep -Seconds 10
    }
}
