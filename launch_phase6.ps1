Set-Location -Path "D:\SVD_Project"
$LogFile = "D:\SVD_Project\phase6_ablation.log"
$Python = "C:\Users\Hadi Schumacher\anaconda3\python.exe"
$Script = "D:\SVD_Project\run_paper_artifacts.py"

Add-Content -Path $LogFile -Value "[$(Get-Date)] Starting Phase 6 parallel ablation queue..."

& $Python -u $Script *>> $LogFile

Add-Content -Path $LogFile -Value "[$(Get-Date)] Phase 6 finished with code $LASTEXITCODE"
