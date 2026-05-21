# PowerShell installer for Windows
Write-Host "=== NeuralOps GPU Provider Agent Installer ==="

# Check Python
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python 3.10+ required. Install from python.org"
    exit 1
}

# Check NVIDIA drivers
$nvidia = & nvidia-smi 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "NVIDIA drivers not found"
    exit 1
}

# Check Docker Desktop
if (!(Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Docker Desktop required. Install from docker.com"
    exit 1
}

# Install agent
$installDir = "$env:USERPROFILE\.neuralops-agent"
New-Item -ItemType Directory -Force -Path $installDir | Out-Null
python -m venv "$installDir\venv"
& "$installDir\venv\Scripts\Activate.ps1"
pip install neuralops-agent

Write-Host ""
Write-Host "Installation complete!"
Write-Host "   Run: neuralops-agent --api-key YOUR_KEY"