# launch_benchmark.ps1 — launches via cmd.exe batch file (avoids PowerShell redirect+hidden crash)
$ErrorActionPreference = 'Continue'
New-Item -ItemType Directory -Force -Path 'D:\SVD_Project\results\full_benchmark' | Out-Null

Start-Process -FilePath 'cmd.exe' `
    -ArgumentList '/c', 'D:\SVD_Project\run_queue.bat' `
    -WindowStyle Hidden `
    -WorkingDirectory 'D:\SVD_Project'

Write-Host 'Launcher executed.'
