# Amazon Bestselling Books MLOps - PowerShell Pipeline Automation
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Starting Automated Amazon MLOps Pipeline in src/ ..." -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

python -m src.run_all

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Pipeline execution failed with exit code $LASTEXITCODE." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n[INFO] Running pytest test suite..." -ForegroundColor Cyan
pytest tests/ -v

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Test suite failed with exit code $LASTEXITCODE." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n======================================================================" -ForegroundColor Green
Write-Host "[SUCCESS] Complete Amazon MLOps Pipeline and Tests executed successfully!" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
