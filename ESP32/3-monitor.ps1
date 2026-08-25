# =============================================================================
# STEP 3 (optional) - Watch the live serial logs / CSI + movement numbers.
# Run:  .\3-monitor.ps1
# =============================================================================
$ErrorActionPreference = "Stop"
$esphome = Join-Path $PSScriptRoot "venv\Scripts\esphome.exe"
if (-not (Test-Path $esphome)) {
    Write-Host "ESPHome not installed. Run  .\1-setup.ps1  first." -ForegroundColor Red
    Read-Host "Press Enter to exit"; exit 1
}
Set-Location (Join-Path $PSScriptRoot "firmware")
& $esphome logs espectre-esp32.yaml
