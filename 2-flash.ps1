# =============================================================================
# STEP 2 - Build the firmware and flash it onto the ESP32 over USB.
# Plug in the ESP32 first, then run:  .\2-flash.ps1
# When asked how to upload, choose the USB/serial COM port of your board.
# =============================================================================
$ErrorActionPreference = "Stop"
$esphome = Join-Path $PSScriptRoot "venv\Scripts\esphome.exe"
if (-not (Test-Path $esphome)) {
    Write-Host "ESPHome not installed. Run  .\1-setup.ps1  first." -ForegroundColor Red
    Read-Host "Press Enter to exit"; exit 1
}
$secrets = Join-Path $PSScriptRoot "firmware\secrets.yaml"
if (-not (Test-Path $secrets)) {
    Write-Host "Missing firmware\secrets.yaml - run  .\1-setup.ps1  and edit your WiFi." -ForegroundColor Red
    Read-Host "Press Enter to exit"; exit 1
}

Set-Location (Join-Path $PSScriptRoot "firmware")
Write-Host "==> Building and flashing (first build takes a few minutes)..." -ForegroundColor Cyan
Write-Host "    Tip: if upload times out, hold the BOOT button, re-run, release at 'Connecting...'." -ForegroundColor DarkGray
& $esphome run espectre-esp32.yaml

Read-Host "Press Enter to exit"
