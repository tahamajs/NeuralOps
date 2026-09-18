import subprocess
import json

output = subprocess.check_output(
    ['powershell', '-Command', 'Get-CimInstance Win32_Process -Filter "Name = \'python.exe\'" | Select-Object ProcessId, CommandLine | ConvertTo-Json'],
    text=True
)
print(output)
