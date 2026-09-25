# One-time setup: Python 3.12 virtual environment with the scoring sandbox's package versions.
# Run from the project folder:  powershell -ExecutionPolicy Bypass -File .\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    if (Get-Command py -ErrorAction SilentlyContinue) { py -3.12 -m venv .venv }
    else { python -m venv .venv }
    if ($LASTEXITCODE -ne 0) { throw "Could not create a Python 3.12 venv. Install Python 3.12 from python.org, then rerun." }
}

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $python -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)"
if ($LASTEXITCODE -ne 0) { throw ".venv is not Python 3.12. Delete the .venv folder, install Python 3.12, rerun." }

& $python -m pip install --upgrade pip
& $python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "pip install failed (see above)." }

& $python check_setup.py
Write-Host "`nSetup done. Activate later with:  .\.venv\Scripts\Activate.ps1"
