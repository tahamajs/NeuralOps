$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$WorkDir  = "D:\SVD_Project"
Set-Location $WorkDir

$Python   = "C:\Users\Hadi Schumacher\anaconda3\python.exe"
$AblationScript = "D:\SVD_Project\run_paper_artifacts.py"
$UploadScript   = "D:\SVD_Project\auto_upload_hf_results.py"

$AblationLog    = "D:\SVD_Project\paper_artifacts.log"
$UploadLog      = "D:\SVD_Project\hf_uploader.log"

# Function to launch and monitor a Python daemon detached
$JobScript = {
    param($ScriptPath, $LogPath)
    $Py = "C:\Users\Hadi Schumacher\anaconda3\python.exe"
    while ($true) {
        $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Add-Content -Path $LogPath -Value "[$ts] === Worker starting ===" -Encoding UTF8
        & $Py -u $ScriptPath *>> $LogPath
        $exitCode = $LASTEXITCODE
        $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Add-Content -Path $LogPath -Value "[$ts] === Worker exited with code $exitCode ===" -Encoding UTF8
        if ($exitCode -eq 0) { break }
        Start-Sleep -Seconds 20
    }
}

# Start background uploader (checks every 300 seconds, low CPU, zero GPU VRAM)
Start-Job -ScriptBlock $JobScript -ArgumentList $UploadScript, $UploadLog -Name "HFUploader"

# Run the capped rank ablation in foreground of this script
while ($true) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $AblationLog -Value "[$ts] === Starting Ablation Run ===" -Encoding UTF8
    & $Python -u $AblationScript *>> $AblationLog
    $exitCode = $LASTEXITCODE
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $AblationLog -Value "[$ts] === Ablation Run exited with code $exitCode ===" -Encoding UTF8
    if ($exitCode -eq 0) { break }
    Start-Sleep -Seconds 30
}
