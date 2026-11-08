$ErrorActionPreference = "Stop"

$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $project "venv\Scripts\python.exe"

if (!(Test-Path $python)) {
    Write-Host "Project virtual environment is missing." -ForegroundColor Red
    exit 1
}

Set-Location $project

Write-Host "`n--- TOMATOAI ENVIRONMENT ---" -ForegroundColor Cyan
& $python -c "import sys; print('Python:', sys.executable)"

Write-Host "`n--- STARTING TOMATOAI ---" -ForegroundColor Green
& $python app.py
