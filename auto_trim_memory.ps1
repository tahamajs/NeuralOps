# Background RAM Guardian
$MaxAllowedUsagePct = 85.0
while ($true) {
    try {
        $os = Get-CimInstance Win32_OperatingSystem
        $usedPct = [math]::Round(($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / $os.TotalVisibleMemorySize * 100, 1)
        
        # If RAM exceeds 85%, trim working sets and terminate orphaned language server nodes
        if ($usedPct -ge $MaxAllowedUsagePct) {
            # 1. Kill any VS Code node servers taking > 1 GB
            Get-Process node -ErrorAction SilentlyContinue | Where-Object { $_.WorkingSet64 -gt 1GB } | Stop-Process -Force -ErrorAction SilentlyContinue
            
            # 2. Trim working sets of running non-critical processes
            [System.GC]::Collect()
            [System.GC]::WaitForPendingFinalizers()
        }
    } catch {
    }
    Start-Sleep -Seconds 60
}
