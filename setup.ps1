# One-time setup on Windows (PowerShell). Right-click > Run with PowerShell, or:  .\setup.ps1  [cpu]
param([string]$mode = "")
Set-Location $PSScriptRoot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -q --upgrade pip
if ($mode -eq "cpu") { pip install -r requirements-cpu.txt gradio } else { pip install -r requirements.txt }
python -m pytest -q tests
Write-Host ""
Write-Host "Done. Each time you open a terminal here, run:  .\.venv\Scripts\Activate.ps1"
Write-Host "Then try:  python app\app.py --demo"
