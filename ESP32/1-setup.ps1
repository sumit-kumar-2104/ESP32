# =============================================================================
# STEP 1 - One-time setup: installs the build tool (ESPHome) into this folder.
# Right-click this file -> "Run with PowerShell", or run:  .\1-setup.ps1
# =============================================================================
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "==> Checking Python..." -ForegroundColor Cyan
$py = $null
foreach ($c in @("python", "py -3.12", "py -3")) {
    try { & $c.Split(" ")[0] $c.Split(" ")[1..9] --version 2>$null | Out-Null; $py = $c; break } catch {}
}
if (-not $py) {
    Write-Host "Python not found. Install Python 3.12 from https://www.python.org/downloads/ (check 'Add to PATH')." -ForegroundColor Red
    Read-Host "Press Enter to exit"; exit 1
}
Write-Host "Using: $py" -ForegroundColor Green

Write-Host "==> Creating virtual environment (venv)..." -ForegroundColor Cyan
& $py.Split(" ")[0] $py.Split(" ")[1..9] -m venv venv

Write-Host "==> Installing ESPHome (this can take a few minutes)..." -ForegroundColor Cyan
& "$PSScriptRoot\venv\Scripts\python.exe" -m pip install --upgrade pip
& "$PSScriptRoot\venv\Scripts\python.exe" -m pip install esphome

# Create secrets.yaml from the template if it doesn't exist yet
$secrets = Join-Path $PSScriptRoot "firmware\secrets.yaml"
if (-not (Test-Path $secrets)) {
    Copy-Item (Join-Path $PSScriptRoot "firmware\secrets.yaml.example") $secrets
    Write-Host ""
    Write-Host "==> Created firmware\secrets.yaml" -ForegroundColor Yellow
    Write-Host "    OPEN IT NOW and enter your 2.4 GHz WiFi name and password!" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "Next: edit firmware\secrets.yaml with your WiFi, then run  .\2-flash.ps1" -ForegroundColor Green
Read-Host "Press Enter to exit"
