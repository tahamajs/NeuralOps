Get-WmiObject Win32_Process -Filter "Name='python.exe'" | Select-Object ProcessId, CommandLine
