Get-Process python -ErrorAction SilentlyContinue | ForEach-Object {
    $procId = $_.Id
    $cmd = (Get-CimInstance Win32_Process -Filter "ProcessId = $procId").CommandLine
    [PSCustomObject]@{
        Id = $procId
        CPU = $_.CPU
        WorkingSetMB = [math]::Round($_.WorkingSet64 / 1MB, 2)
        CommandLine = $cmd
    }
} | Format-Table -AutoSize
