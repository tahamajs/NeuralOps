$rank_psi = New-Object System.Diagnostics.ProcessStartInfo
$rank_psi.FileName = "cmd.exe"
$rank_psi.Arguments = "/c C:\Users\`"Hadi Schumacher`"\SVD_Project\scripts\run_9_modules_rank_sweep.bat"
$rank_psi.WorkingDirectory = "C:\Users\Hadi Schumacher\SVD_Project"
$rank_psi.UseShellExecute = $false
$rank_psi.CreateNoWindow = $true
$rank_process = [System.Diagnostics.Process]::Start($rank_psi)
Write-Host "Started rank sweeper with batch_size 16!"
