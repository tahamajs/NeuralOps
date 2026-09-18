import subprocess
output = subprocess.check_output(
    ['powershell', '-Command', 'Get-CimInstance Win32_Process -Filter "ProcessId = 18020" | Select-Object ProcessId, CommandLine | ConvertTo-Json'],
    text=True
)
print(output)
