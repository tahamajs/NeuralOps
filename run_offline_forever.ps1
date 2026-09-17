$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
Set-Location "D:\SVD_Project"
& "C:\Users\Hadi Schumacher\anaconda3\python.exe" -u test_all_modules.py *>&1 | Out-File -FilePath "D:\SVD_Project\test_all_modules.log" -Encoding utf8
