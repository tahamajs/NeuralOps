# ============================================================
# run_all_modules_delta.ps1 (v7 - attention + compressor)
# ============================================================

param(
    [string]$ModelName = "gpt2",
    [int]$Rank = 32,
    [string]$Methods = "C",
    [int]$BatchSize = 4,
    [int]$GradAccumSteps = 1,
    [int]$RecoverySteps = 300,
    [int]$SelfTrainSteps = 200,
    [double]$LearningRate = 3e-4,
    [int]$EvalEvery = 100,
    [string]$Device = "cuda",
    [string]$OutputDir = "results\delta_sweep",
    [int]$TimeoutSec = 300,
    [switch]$RerunFailed,
    [switch]$OnlyAttention,
    [switch]$OnlyCompressor
)

$ErrorActionPreference = "Continue"

# ------------------------------------------------------------
# 1. Collect module list (attention + compressor)
# ------------------------------------------------------------
Write-Host ""
Write-Host "=== Collecting module lists ===" -ForegroundColor Cyan

$collector = @'
import json
from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
import neuralops.attentions
import neuralops.compression_engine

attn_keys = sorted(ATTENTION_REGISTRY.keys())
comp_keys = sorted(COMPRESSOR_REGISTRY.keys())

modules = []
for k in attn_keys:
    modules.append({"name": k, "type": "attention"})
for k in comp_keys:
    modules.append({"name": k, "type": "compressor"})

with open("_modules_tmp.json", "w") as f:
    json.dump(modules, f)

print("Total canonical attention keys:", len(attn_keys))
print("Total canonical compressor keys:", len(comp_keys))
print("Total modules:", len(modules))
'@

$collector | python -
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to collect modules. Aborting." -ForegroundColor Red
    exit 1
}

$allModules = Get-Content "_modules_tmp.json" -Raw | ConvertFrom-Json

# Filter by type if requested
$modules = @()
foreach ($m in $allModules) {
    if ($OnlyAttention -and $m.type -ne "attention") { continue }
    if ($OnlyCompressor -and $m.type -ne "compressor") { continue }
    $modules += $m
}

Write-Host ("Loaded {0} modules after filter" -f $modules.Count) -ForegroundColor Green

if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

# ------------------------------------------------------------
# 2. Load existing results
# ------------------------------------------------------------
$summaryFile = Join-Path $OutputDir "summary.csv"

$existingResults = @{}
if (Test-Path $summaryFile) {
    try {
        $existing = Import-Csv $summaryFile
        foreach ($row in $existing) {
            if ($row.module) {
                $key = $row.type + "::" + $row.module
                $existingResults[$key] = $row.status
            }
        }
        Write-Host ("Loaded {0} existing results" -f $existingResults.Count) -ForegroundColor DarkGray
    } catch {
        Write-Host "Could not parse summary.csv - starting fresh" -ForegroundColor Yellow
        $existingResults = @{}
    }
} else {
    "module,type,status,delta,student_ppl,duration_sec,error,timestamp" | Out-File $summaryFile -Encoding utf8
}

# ------------------------------------------------------------
# 3. Skip decision
# ------------------------------------------------------------
function Test-ShouldSkip {
    param([string]$Mod, [string]$Type)
    $key = $Type + "::" + $Mod
    if (-not $existingResults.ContainsKey($key)) { return $false }
    $status = $existingResults[$key]
    if ($RerunFailed) {
        if ($status -eq "ERROR" -or $status -eq "TIMEOUT") { return $false }
        return $true
    }
    return $true
}

# ------------------------------------------------------------
# 4. Parse python log
# ------------------------------------------------------------
function Get-ParsedLog {
    param([string]$LogFile)
    $result = @{ delta = ""; ppl = "" }
    if (-not (Test-Path $LogFile)) { return $result }
    $content = Get-Content $LogFile -Raw -ErrorAction SilentlyContinue
    if (-not $content) { return $result }
    $m1 = [regex]::Match($content, 'DELTA[^0-9+\-]*([-+]?[0-9]+\.?[0-9]*)')
    if ($m1.Success) { $result.delta = $m1.Groups[1].Value }
    $m2 = [regex]::Match($content, 'Student[^0-9]*([0-9]+\.?[0-9]*)')
    if ($m2.Success) { $result.ppl = $m2.Groups[1].Value }
    return $result
}

# ------------------------------------------------------------
# 5. Run one module
# ------------------------------------------------------------
function Invoke-OneModule {
    param(
        [string]$Mod,
        [string]$Type,
        [int]$Position,
        [int]$TotalCount
    )

    $prefix = $Type
    $logFile  = Join-Path $OutputDir ($prefix + "_" + $Mod + ".log")
    $errFile  = Join-Path $OutputDir ($prefix + "_" + $Mod + ".err")
    $jsonFile = Join-Path $OutputDir ($prefix + "_" + $Mod + ".json")

    Write-Host ("[{0}/{1}] START [{2}]: {3}" -f $Position, $TotalCount, $Type, $Mod) -ForegroundColor Yellow
    $startTime = Get-Date

    $pyArgs = @(
        "neuralops/benchmarks/_legacy/kd_quick_test.py",
        "--modules", $Mod,
        "--type", $Type,
        "--model_name", $ModelName,
        "--rank", "$Rank",
        "--methods", $Methods,
        "--use_cache",
        "--batch_size", "$BatchSize",
        "--grad_accum_steps", "$GradAccumSteps",
        "--recovery_steps", "$RecoverySteps",
        "--self_train_steps", "$SelfTrainSteps",
        "--learning_rate", "$LearningRate",
        "--eval_every", "$EvalEvery",
        "--device", $Device
    )

    $proc = $null
    $timedOut = $false
    $exitCode = 0

    try {
        $proc = Start-Process -FilePath "python" -ArgumentList $pyArgs -RedirectStandardOutput $logFile -RedirectStandardError $errFile -NoNewWindow -PassThru
        $finished = $proc.WaitForExit($TimeoutSec * 1000)
        if (-not $finished) {
            Write-Host ("  TIMEOUT - killing PID {0}" -f $proc.Id) -ForegroundColor Red
            try { $proc.Kill(); $proc.WaitForExit(5000) | Out-Null } catch {}
            $timedOut = $true
            $exitCode = -999
        } else {
            $exitCode = $proc.ExitCode
        }
    }
    catch {
        $exMsg = $_.Exception.Message
        Write-Host ("  EXCEPTION: {0}" -f $exMsg) -ForegroundColor Red
        $exitCode = -1
    }
    finally {
        if ($proc -ne $null) { try { $proc.Dispose() } catch {} }
    }

    $duration = ((Get-Date) - $startTime).TotalSeconds
    $durRounded = [math]::Round($duration, 1)
    $timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")

    $parsed = Get-ParsedLog -LogFile $logFile
    $delta = $parsed.delta
    $ppl = $parsed.ppl

    if ($timedOut) {
        $status = "TIMEOUT"
        $errorMsg = "exceeded " + $TimeoutSec + "s"
    } elseif ($exitCode -ne 0) {
        $status = "ERROR"
        $errorMsg = "exit=" + $exitCode
    } elseif ($delta -ne "" -and [double]$delta -gt 1.0) {
        $status = "PASS"
        $errorMsg = ""
    } elseif ($delta -ne "") {
        $status = "FAIL_DELTA_ZERO"
        $errorMsg = "delta=" + $delta
    } else {
        $status = "UNKNOWN"
        $errorMsg = "no delta parsed"
    }

    $jsonObj = [PSCustomObject]@{
        module       = $Mod
        type         = $Type
        status       = $status
        delta        = if ($delta -ne "") { [double]$delta } else { $null }
        student_ppl  = if ($ppl -ne "") { [double]$ppl } else { $null }
        duration_sec = $durRounded
        error        = $errorMsg
        timestamp    = $timestamp
    }
    $jsonObj | ConvertTo-Json -Depth 3 | Out-File $jsonFile -Encoding utf8

    # Update summary.csv
    $mutex = New-Object System.Threading.Mutex($false, "Global\SummaryCSVLock")
    try {
        $mutex.WaitOne() | Out-Null

        $allRows = @()
        if (Test-Path $summaryFile) {
            $allRows = @(Import-Csv $summaryFile | Where-Object { -not ($_.module -eq $Mod -and $_.type -eq $Type) })
        }

        $newRow = [PSCustomObject]@{
            module       = $Mod
            type         = $Type
            status       = $status
            delta        = $delta
            student_ppl  = $ppl
            duration_sec = $durRounded
            error        = $errorMsg
            timestamp    = $timestamp
        }

        $combined = $allRows + $newRow
        $combined | Export-Csv -Path $summaryFile -NoTypeInformation -Encoding utf8
    }
    finally {
        $mutex.ReleaseMutex()
        $mutex.Dispose()
    }

    if ($status -eq "PASS") {
        Write-Host ("[{0}/{1}] PASS [{2}]: {3} (delta={4}, ppl={5}, {6}s)" -f $Position, $TotalCount, $Type, $Mod, $delta, $ppl, $durRounded) -ForegroundColor Green
    } elseif ($status -eq "FAIL_DELTA_ZERO") {
        Write-Host ("[{0}/{1}] FAIL_DELTA_ZERO [{2}]: {3} (delta={4})" -f $Position, $TotalCount, $Type, $Mod, $delta) -ForegroundColor Red
    } elseif ($status -eq "TIMEOUT") {
        Write-Host ("[{0}/{1}] TIMEOUT [{2}]: {3} (>{4}s)" -f $Position, $TotalCount, $Type, $Mod, $TimeoutSec) -ForegroundColor DarkRed
    } else {
        Write-Host ("[{0}/{1}] {2} [{3}]: {4}" -f $Position, $TotalCount, $status, $Type, $Mod) -ForegroundColor Magenta
    }
}

# ------------------------------------------------------------
# 6. Main loop
# ------------------------------------------------------------
$total = $modules.Count
$idx = 0
$skipped = 0
$ran = 0

foreach ($m in $modules) {
    $idx++

    if (Test-ShouldSkip -Mod $m.name -Type $m.type) {
        $key = $m.type + "::" + $m.name
        $prevStatus = $existingResults[$key]
        Write-Host ("[{0}/{1}] SKIP [{2}] (already {3}): {4}" -f $idx, $total, $m.type, $prevStatus, $m.name) -ForegroundColor DarkGray
        $skipped++
        continue
    }

    Invoke-OneModule -Mod $m.name -Type $m.type -Position $idx -TotalCount $total
    $ran++
    Start-Sleep -Seconds 2
}

# ------------------------------------------------------------
# 7. Final summary
# ------------------------------------------------------------
Write-Host ""
Write-Host "=== FINAL SUMMARY ===" -ForegroundColor Cyan
Write-Host ("Skipped (already done): {0}" -f $skipped)
Write-Host ("Newly run:             {0}" -f $ran)

$summary = Import-Csv $summaryFile
$totalRows = $summary.Count

$attnRows = @($summary | Where-Object { $_.type -eq "attention" })
$compRows = @($summary | Where-Object { $_.type -eq "compressor" })

Write-Host ""
Write-Host "=== ATTENTION ===" -ForegroundColor Cyan
$attnTotal = $attnRows.Count
$attnPass = @($attnRows | Where-Object { $_.status -eq "PASS" }).Count
$attnFail = @($attnRows | Where-Object { $_.status -eq "FAIL_DELTA_ZERO" }).Count
$attnErr  = @($attnRows | Where-Object { $_.status -eq "ERROR" }).Count
$attnTmo  = @($attnRows | Where-Object { $_.status -eq "TIMEOUT" }).Count
Write-Host ("Total:    {0}" -f $attnTotal)
Write-Host ("PASS:     {0}" -f $attnPass) -ForegroundColor Green
Write-Host ("FAIL_D0:  {0}" -f $attnFail) -ForegroundColor Red
Write-Host ("TIMEOUT:  {0}" -f $attnTmo)  -ForegroundColor DarkRed
Write-Host ("ERROR:    {0}" -f $attnErr)  -ForegroundColor Magenta

Write-Host ""
Write-Host "=== COMPRESSOR ===" -ForegroundColor Cyan
$compTotal = $compRows.Count
$compPass = @($compRows | Where-Object { $_.status -eq "PASS" }).Count
$compFail = @($compRows | Where-Object { $_.status -eq "FAIL_DELTA_ZERO" }).Count
$compErr  = @($compRows | Where-Object { $_.status -eq "ERROR" }).Count
$compTmo  = @($compRows | Where-Object { $_.status -eq "TIMEOUT" }).Count
Write-Host ("Total:    {0}" -f $compTotal)
Write-Host ("PASS:     {0}" -f $compPass) -ForegroundColor Green
Write-Host ("FAIL_D0:  {0}" -f $compFail) -ForegroundColor Red
Write-Host ("TIMEOUT:  {0}" -f $compTmo)  -ForegroundColor DarkRed
Write-Host ("ERROR:    {0}" -f $compErr)  -ForegroundColor Magenta

Write-Host ""
Write-Host ("Full report: {0}" -f $summaryFile)