Set-Location D:\SVD_Project

# 1) Wait for the current finish_benchmark.py session to end
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'finish_benchmark.py' }) {
    Start-Sleep -Seconds 60
}

# 2) Commit any stray results, then sync latest fixes
git add -A 2>$null
git commit -m 'chore(results): pre-rerun sync' 2>$null
git pull --rebase origin main 2>$null
git push origin main 2>$null

# 3) Clear non-PASS (stale crash/collapse) state entries so fixed modules re-run
python -c "import json; p='finish_benchmark_state.json'; s=json.load(open(p)); keep={k:v for k,v in s.items() if v.get('status')=='PASS'}; json.dump(keep, open(p,'w'), indent=2); print('cleared', len(s)-len(keep), 'of', len(s))"

# 4) Relaunch benchmark (proxy for HF uploads)
$env:http_proxy = 'http://192.168.1.101:10808'
$env:https_proxy = 'http://192.168.1.101:10808'
python -u finish_benchmark.py *>> rerun_failed.log
