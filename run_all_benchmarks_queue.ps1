# Master Benchmark Orchestrator PowerShell Script for Windows GPU Server
param (
    [string]$Models = "distilbert,roberta,gpt2",
    [int]$RecoverySteps = 50,
    [int]$BatchSize = 0,
    [string]$StartFrom = "",
    [string]$Modules = "",
    [switch]$NoSkip = $false,
    [int]$TimeoutSec = 3600
)

$ErrorActionPreference = "Continue"
Set-Location -Path "D:\SVD_Project"

$argsList = @(
    "run_all_benchmarks_queue.py",
    "--models", $Models,
    "--recovery_steps", $RecoverySteps,
    "--timeout", $TimeoutSec
)

if ($BatchSize -gt 0) {
    $argsList += @("--batch_size", $BatchSize)
}
if ($StartFrom -ne "") {
    $argsList += @("--start_from", $StartFrom)
}
if ($Modules -ne "") {
    $argsList += @("--modules", $Modules)
}
if ($NoSkip) {
    $argsList += @("--no_skip")
}

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "STARTING SEQUENTIAL QUEUE BENCHMARK ORCHESTRATOR ON GPU SERVER" -ForegroundColor Green
Write-Host "Command: python -u $($argsList -join ' ')" -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Cyan

python -u @argsList
